"""VisionMambaBlock module."""

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from einops.layers.torch import Reduce
import torch.nn.functional as F
from functools import reduce
from collections.abc import Iterable

import math


class PScan(torch.autograd.Function):
    """

    An implementation of the parallel scan operation in PyTorch (Blelloch version).
    This code is based on Francois Fleuret’s pscan (all credits to him). However, the keys differences are :
    -it has been written in an iterative way (rather than recursive)
    -the backward pass has been rewritten

    Please see docs/pscan.ipynb for a detailed explanation of what happens here.

    Example:
    pscan = PScan.apply

    x = torch.randn(2, 3, 4, 5, requires_grad=True)
    y = torch.randn(2, 3, 4, 5, requires_grad=True)

    model = pscan(x, y)
    model.sum().backward()

    """

    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep or reduction step
        Aa = A
        Xa = X
        for k in range(num_steps):
            T = 2 * (Xa.size(2) // 2)

            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # down sweep
        for k in range(num_steps - 1, -1, -1):
            Aa = A[:, :, 2**k - 1 : L : 2**k]
            Xa = X[:, :, 2**k - 1 : L : 2**k]

            T = 2 * (Xa.size(2) // 2)

            if T < Xa.size(2):
                Xa[:, :, -1].add_(Aa[:, :, -1].mul(Xa[:, :, -2]))
                Aa[:, :, -1].mul_(Aa[:, :, -2])

            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """

        # clone tensor (in-place ops)
        A = A_in.clone()  # (B, L, D, N)
        X = X_in.clone()  # (B, L, D, N)

        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, L, N)
        X = X.transpose(2, 1)  # (B, D, L, N)

        # parallel scan
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        return X.transpose(2, 1)

    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (B, L, D, N), X : (B, D, L, N)
            grad_output_in : (B, L, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """

        A_in, X = ctx.saved_tensors

        # clone tensors
        A = A_in.clone()
        # grad_output_in will be cloned with flip()

        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, L, N)
        A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
        grad_output_b = grad_output_in.transpose(2, 1)

        # reverse parallel scan
        grad_output_b = grad_output_b.flip(2)
        PScan.pscan(A, grad_output_b)
        grad_output_b = grad_output_b.flip(2)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output_b[:, :, 1:])

        return Q.transpose(2, 1), grad_output_b.transpose(2, 1)


pscan = PScan.apply


def selective_scan(x, delta, A, B, C, D):
    """
    Perform selective scan operation on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, ED).
        delta (torch.Tensor): Delta tensor of shape (B, L, ED).
        A (torch.Tensor): A tensor of shape (ED, N).
        B (torch.Tensor): B tensor of shape (B, L, N).
        C (torch.Tensor): C tensor of shape (B, L, N).
        D (torch.Tensor): D tensor of shape (ED).

    Returns:
        torch.Tensor: Output tensor of shape (B, L, ED).
    """

    _, L, _ = x.shape

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

    BX = deltaB * x.unsqueeze(-1)  # (B, L, ED, N)

    hs = pscan(deltaA, BX)

    y = (
        hs @ C.unsqueeze(-1)
    ).squeeze()  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

    y = y + D * x

    return y


def selective_scan_seq(x, delta, A, B, C, D, dim_inner: int, d_state: int):
    """
    Perform selective scan sequence operation on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, ED).
        delta (torch.Tensor): Delta tensor of shape (B, L, ED).
        A (torch.Tensor): A tensor of shape (ED, N).
        B (torch.Tensor): B tensor of shape (B, L, N).
        C (torch.Tensor): C tensor of shape (B, L, N).
        D (torch.Tensor): D tensor of shape (ED).
        dim_inner (int): Inner dimension size.
        d_state (int): State dimension size.

    Returns:
        torch.Tensor: Output tensor of shape (B, L, ED).
    """

    _, L, _ = x.shape

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

    BX = deltaB * x.unsqueeze(-1)  # (B, L, ED, N)

    h = torch.zeros(
        x.size(0),
        dim_inner,
        d_state,
        device=deltaA.device,
    )  # (B, ED, N)
    hs = []

    for t in range(0, L):
        h = deltaA[:, t] * h + BX[:, t]
        hs.append(h)

    hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

    # y = (C.unsqueeze(2) * hs).sum(3)
    y = (
        hs @ C.unsqueeze(-1)
    ).squeeze()  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

    y = y + D * x

    return y


class SSM(nn.Module):
    def __init__(self, in_features, dt_rank: int, dim_inner: int, d_state: int):
        """
        Initializes the SSM module.

        Args:
            in_features (int): The size of the input features.
            dt_rank (int): The rank of the dt projection.
            dim_inner (int): The inner dimension of the dt projection.
            d_state (int): The dimension of the state.

        """
        super().__init__()
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        # Linear layer expecting 'in_features' as the input size
        self.deltaBC_layer = nn.Linear(
            in_features, dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj_layer = nn.Linear(dt_rank, dim_inner, bias=True)

        # Defining A_log and D as parameters
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
                    dim_inner, 1
                )
            )
        )
        self.D = nn.Parameter(torch.ones(dim_inner))

    def forward(self, x, pscan: bool = True):
        """
        Performs forward pass of the SSM module.

        Args:
            x (torch.Tensor): The input tensor.
            pscan (bool, optional): Whether to use selective_scan or selective_scan_seq. Defaults to True.

        Returns:
            torch.Tensor: The output tensor.

        """
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.deltaBC_layer(x)
        delta, B, C = torch.split(
            deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj_layer(delta))

        # Assuming selective_scan and selective_scan_seq are defined functions
        if pscan:
            y = selective_scan(x, delta, A, B, C, D)
        else:
            y = selective_scan_seq(x, delta, A, B, C, D)

        return y


# Pair
def pair(t, dims=2):
    return t if isinstance(t, Iterable) else (t,) * dims


def output_head(dim: int, num_classes: int):
    """
    Creates a head for the output layer of a model.

    Args:
        dim (int): The input dimension of the head.
        num_classes (int): The number of output classes.

    Returns:
        nn.Sequential: The output head module.
    """
    return nn.Sequential(
        Reduce("b s d -> b d", "mean"),
        nn.LayerNorm(dim),
        nn.Linear(dim, num_classes),
    )


class VisionEncoderMambaBlock(nn.Module):
    """
    VisionMambaBlock is a module that implements the Mamba block from the paper
    Vision Mamba: Efficient Visual Representation Learning with Bidirectional
    State Space Model

    Args:
        dim (int): The input dimension of the input tensor.
        dt_rank (int): The rank of the state space model.
        dim_inner (int): The dimension of the inner layer of the
            multi-head attention.
        d_state (int): The dimension of the state space model.


    Example:
    >>> block = VisionMambaBlock(dim=256, heads=8, dt_rank=32,
            dim_inner=512, d_state=256)
    >>> x = torch.randn(1, 32, 256)
    >>> out = block(x)
    >>> out.shape
    torch.Size([1, 32, 256])
    """

    def __init__(
        self,
        dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state)

        # Linear layer for z and x
        self.proj = nn.Linear(dim, dim)

        # Softplus
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape

        # Skip connection
        skip = x

        # Normalization
        x = self.norm(x)

        # Split x into x1 and x2 with linears
        z1 = self.proj(x)
        x = self.proj(x)

        # forward con1d
        x1 = self.process_direction(
            x,
            self.forward_conv1d,
            self.ssm,
        )

        # backward conv1d
        x2 = self.process_direction(
            x,
            self.backward_conv1d,
            self.ssm,
        )

        # Activation
        z = self.silu(z1)

        # Matmul
        x1 *= z
        x2 *= z

        # Residual connection
        return x1 + x2 + skip

    def process_direction(
        self,
        x: Tensor,
        conv1d: nn.Conv1d,
        ssm: SSM,
    ):
        x = rearrange(x, "b s d -> b d s")
        x = self.softplus(conv1d(x))
        x = rearrange(x, "b d s -> b s d")
        x = ssm(x)
        return x


class Vim(nn.Module):
    """
    Vision Mamba (Vim) model implementation.

    Args:
        dim (int): Dimension of the model.
        dt_rank (int, optional): Rank of the dynamic tensor. Defaults to 32.
        dim_inner (int, optional): Inner dimension of the model. Defaults to None.
        d_state (int, optional): State dimension of the model. Defaults to None.
        num_classes (int, optional): Number of output classes. Defaults to None.
        image_size (int, optional): Size of the input image. Defaults to 224.
        patch_size (int, optional): Size of the image patch. Defaults to 16.
        channels (int, optional): Number of image channels. Defaults to 3.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        depth (int, optional): Number of encoder layers. Defaults to 12.

    Attributes:
        dim (int): Dimension of the model.
        dt_rank (int): Rank of the dynamic tensor.
        dim_inner (int): Inner dimension of the model.
        d_state (int): State dimension of the model.
        num_classes (int): Number of output classes.
        image_size (int): Size of the input image.
        patch_size (int): Size of the image patch.
        channels (int): Number of image channels.
        dropout (float): Dropout rate.
        depth (int): Number of encoder layers.
        to_patch_embedding (nn.Sequential): Sequential module for patch embedding.
        dropout (nn.Dropout): Dropout module.
        cls_token (nn.Parameter): Class token parameter.
        to_latent (nn.Identity): Identity module for latent representation.
        layers (nn.ModuleList): List of encoder layers.
        output_head (output_head): Output head module.

    """

    def __init__(
        self,
        dim: int,
        dt_rank: int = 32,
        dim_inner: int = None,
        d_state: int = None,
        num_classes: int = None,
        image_size: int = 224,
        patch_size: int = 16,
        channels: int = 3,
        dropout: float = 0.1,
        depth: int = 12,
        spatial_dims: int = 3,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.dropout = dropout
        self.depth = depth

        image_size = pair(image_size, spatial_dims)
        patch_size = pair(patch_size, spatial_dims)
        patch_dim = channels * reduce(lambda x, y: x * y, patch_size)

        if spatial_dims == 2:
            self.to_patch_embedding = nn.Sequential(
                Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                    p1=patch_size[0],
                    p2=patch_size[1],
                ),
                nn.Linear(patch_dim, dim),
            )
        elif spatial_dims == 3:
            self.to_patch_embedding = nn.Sequential(
                Rearrange(
                    "b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)",
                    p1=patch_size[0],
                    p2=patch_size[1],
                    p3=patch_size[2],
                ),
                nn.Linear(patch_dim, dim),
            )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Latent
        self.to_latent = nn.Identity()

        # encoder layers
        self.layers = nn.ModuleList()

        # Append the encoder layers
        for _ in range(depth):
            self.layers.append(
                VisionEncoderMambaBlock(
                    dim=dim,
                    dt_rank=dt_rank,
                    dim_inner=dim_inner,
                    d_state=d_state,
                    *args,
                    **kwargs,
                )
            )

        # Output head
        self.output_head = output_head(dim, num_classes)

    def forward(self, x: Tensor):
        # Patch embedding
        b, c, *shp = x.shape
        x = self.to_patch_embedding(x)

        # Shape
        b, n, _ = x.shape

        # Cls tokens
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)

        # Concatenate
        # x = torch.cat((cls_tokens, x), dim=1)

        # Dropout
        x = self.dropout(x)

        # Forward pass with the layers
        for layer in self.layers:
            x = layer(x)

        # Latent
        x = self.to_latent(x)

        # Output head with the cls tokens
        return self.output_head(x)
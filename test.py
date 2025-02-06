import argparse
import torch as th

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.modules.encoders.modules import FrozenBERTEmbedder


def get_model(base, precision='fp32'):
    config = OmegaConf.load(base)
    config.model['target'] = config.model.get('test_target', 'target')
    config.model['params'] = OmegaConf.merge(config.model['params'], config.model.get('test_only_params', {}))
    model = instantiate_from_config(config.model).set_precision(precision)
    return model


def test_guidegen_stage1(model, text_input, **kw):
    if not isinstance(text_input, th.Tensor):
        # default text encoder
        encoder = FrozenBERTEmbedder().cuda()
        text_input = encoder.encode(text_input)[None]
    batch = {getattr(model.__class__.__bases__[0], 'crossattn_key', 'prompt_context'): text_input}
    return getattr(model, 'log_images', lambda _: None)(model, batch, **kw)


def test_guidegen_stage2(model, mask_input, text_input, **kw):
    if not isinstance(text_input, th.Tensor):
        # default text encoder
        encoder = FrozenBERTEmbedder().cuda()
        text_input = encoder.encode(text_input)[None]
    batch = {getattr(model, 'cond_stage_key', {'crossattn': 'prompt_context'})['crossattn']: text_input,
             getattr(model, 'cond_stage_key', {'concat': 'mask'})['crossattn']: mask_input,}
    return getattr(model.__class__.__bases__[0], 'log_images', lambda _: None)(model, batch, **kw)


def test_guidegen_singlecase(model1, model2, text_input, **kw):
    if not isinstance(text_input, th.Tensor):
        # default text encoder
        encoder = FrozenBERTEmbedder().cuda()
        text_input = encoder.encode(text_input)[None]
    batch = {getattr(model1, 'crossattn_key', 'prompt_context'): text_input}
    mask_output = getattr(model1.__class__.__bases__[0], 'test_step', lambda _: None)(model1, batch, **kw)
    batch = {getattr(model2, 'cond_stage_key', {'crossattn': 'prompt_context'})['crossattn']: text_input,
             getattr(model2, 'cond_stage_key', {'concat': 'mask'})['crossattn']: mask_output,}
    image_output = getattr(model2.__class__.__bases__[0], 'test_step', lambda _: None)(model2, batch, **kw)
    return image_output
    

if __name__ == '__main__':
    pass
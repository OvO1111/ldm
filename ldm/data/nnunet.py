import os, json
from torch.utils.data import default_collate, Dataset
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
)


class nnUNetLikeDataset(Dataset):
    def __init__(self, dataset_base, split='train', fold=0, patch_size=(128, 128, 128),
                 **kw):
        nnUNet_raw = os.environ.get('nnUNet_raw')
        nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
        
        dataset_base = os.path.join(nnUNet_raw, dataset_base)
        dataset_json = os.path.join(dataset_base, 'dataset.json')
        splits_file = os.path.join(nnUNet_preprocessed, os.path.basename(dataset_base), "splits_final.json")
        self.train_val_cases = {"images": sorted([os.path.join(dataset_base, "imagesTr", _) for _ in os.listdir(os.path.join(dataset_base, "imagesTr"))]),
                                "labels": sorted([os.path.join(dataset_base, "labelsTr", _) for _ in os.listdir(os.path.join(dataset_base, "labelsTr"))])}
        with open(dataset_json, 'r') as f:
            dataset = json.load(f)
        with open(splits_file, 'r') as f:
            splits = json.load(f)[fold]
        
        self.train_cases = []
        self.val_cases = []
        self.test_cases = []
        for image, label in zip(self.train_val_cases['images'], self.train_val_cases['labels']):
            if os.path.basename(label).split('.')[0] in splits['train']:
                self.train_cases.append({"image": image, "label": label})
            elif os.path.basename(label).split('.')[0] in splits['val']:
                self.val_cases.append({"image": image, "label": label})
                
        if os.path.exists(os.path.join(dataset_base, "imagesTs")):
            self.test_cases = [{"image": os.path.join(dataset_base, "imagesTs", _), 'label': None} for _ in os.listdir(os.path.join(dataset_base, "imagesTs"))]
        
        self.patch_size = patch_size
        self.split_cases = getattr(self, f'{split}_cases')
        self.transforms = Compose([
            LoadImaged(keys=['image', 'label'], allow_missing_keys=True),
            EnsureChannelFirstd(keys=['image', 'label'], allow_missing_keys=True),
            # CropForegroundd(keys=['image', 'label'], source_key='image'),
            ScaleIntensityRanged(keys=['image'], a_min=kw.get('window_min', -150), a_max=kw.get('window_max', 250), b_min=-1, b_max=1, clip=True),
            # RandCropByLabelClassesd(keys=['image', 'label'], label_key='label',
            #                         spatial_size=self.patch_size, ratios=(0,) + (1,) * (len(dataset['labels']) - 1), num_classes=len(dataset['labels'])),
        ])
        
    def __len__(self):
        return len(self.split_cases)
    
    def __getitem__(self, index):
        case = self.split_cases[index]
        processed = self.transforms(case)
        return processed
    
    def collate(self, batch):
        return default_collate(batch)
        
        
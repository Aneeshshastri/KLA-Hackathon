from pathlib import Path

import grain.python as grain
import numpy as np
import torch


# Config
NUM_SAMPLES = 3200
LR_SIZE = 128

BATCH_SIZE = 16
NUM_WORKERS = 4

SEED = 42

# Augmentation probabilities
P_HFLIP = 0.5
P_VFLIP = 0.5
P_ROT90 = 0.75
P_TRANSLATE = 0.5

# Translation in LR pixels
MAX_TRANSLATION = 2


# Data Source

class NpyPairDataSource(grain.sources.RandomAccessDataSource):
    
    def __init__(self, noisy_dir, gt_dir):
        self.noisy_dir = Path(noisy_dir)
        self.gt_dir = Path(gt_dir)
        self.noisy_files = sorted(self.noisy_dir.glob("*.npy"))
        self.gt_files = sorted(self.gt_dir.glob("*.npy"))

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, index):
        noisy_lr = np.load(self.noisy_files[index])
        gt = np.load(self.gt_files[index])
        return {"noisy_lr": noisy_lr,"gt": gt,}

    def __repr__(self):
        return (
            f"NpyPairDataSource("
            f"num_samples={len(self)}, "
            f"noisy_dir='{self.noisy_dir}', "
            f"gt_dir='{self.gt_dir}')"
        )


# Aligned Augmentation
class RandomAlignedAugment(grain.transforms.RandomMap):
    def __init__(
        self,
        p_hflip=P_HFLIP,
        p_vflip=P_VFLIP,
        p_rot90=P_ROT90,
        p_translate=P_TRANSLATE,
        max_translation=MAX_TRANSLATION,
    ):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90
        self.p_translate = p_translate
        self.max_translation = max_translation

    def random_map(self, element, rng):
        noisy = element["noisy_lr"]
        gt = element["gt"]

        # Convert to numpy
        noisy = np.asarray(noisy)
        gt = np.asarray(gt)

        # Horizontal flip
        if rng.random() < self.p_hflip:
            noisy = np.flip(noisy, axis=1)
            gt = np.flip(gt, axis=1)

        # Vertical flip
        if rng.random() < self.p_vflip:
            noisy = np.flip(noisy, axis=0)
            gt = np.flip(gt, axis=0)

        # Random 90-degree rotation
        if rng.random() < self.p_rot90:
            k = rng.integers(1, 4)
            noisy = np.rot90(noisy, k=k, axes=(0, 1))
            gt = np.rot90(gt, k=k, axes=(0, 1))

        # Small translation
        if rng.random() < self.p_translate:
            dx = rng.integers(-self.max_translation,self.max_translation + 1)
            dy = rng.integers(-self.max_translation,self.max_translation + 1)
            noisy = translate_reflect(noisy, dx, dy)
            gt = translate_reflect(gt, dx, dy)

        #Make the array continouos after the geometric alterations
        noisy = np.ascontiguousarray(noisy)
        gt = np.ascontiguousarray(gt)

        return {"noisy_lr": noisy,"gt": gt}


# Translation
def translate_reflect(image, dx, dy):
    if dx == 0 and dy == 0:
        return image

    h, w = image.shape[:2]

    pad_x = abs(dx)
    pad_y = abs(dy)

    padded = np.pad(image,((pad_y, pad_y),(pad_x, pad_x),)+ ((0, 0),) * (image.ndim - 2),mode="reflect",)

    start_y = pad_y - dy
    start_x = pad_x - dx

    return padded[start_y:start_y + h,start_x:start_x + w]

# Tensor Conversion
class ToTensor(grain.transforms.Map):
    def map(self, element):
        noisy = element["noisy_lr"]
        gt = element["gt"]
        noisy = np.asarray(noisy, dtype=np.float32)
        gt = np.asarray(gt, dtype=np.float32)

        # H,W -> 1,H,W
        if noisy.ndim == 2:
            noisy = noisy[None, ...]
        if gt.ndim == 2:
            gt = gt[None, ...]

        return {"noisy_lr": torch.from_numpy(np.ascontiguousarray(noisy)),"gt": torch.from_numpy(np.ascontiguousarray(gt))}

# Create Grain DataLoader
def create_dataloader(noisy_dir,gt_dir,batch_size=BATCH_SIZE,worker_count=NUM_WORKERS,seed=SEED):
    source = NpyPairDataSource(noisy_dir=noisy_dir,gt_dir=gt_dir,)
    sampler = grain.IndexSampler(num_records=len(source),num_epochs=None,shuffle=True,seed=seed,shard_options=grain.NoSharding(),)
    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[
            RandomAlignedAugment(
                p_hflip=P_HFLIP,
                p_vflip=P_VFLIP,
                p_rot90=P_ROT90,
                p_translate=P_TRANSLATE,
                max_translation=MAX_TRANSLATION,
            ),
            ToTensor(),
            grain.transforms.Batch(
                batch_size=batch_size,
                drop_remainder=True,
            ),
        ],
        worker_count=worker_count
    )
    return dataloader


# USE THE LOADER LIKE THIS

# from dataset import create_dataloader


# train_loader = create_dataloader(
#     noisy_dir="noisy images",
#     gt_dir="gt images",

#     batch_size=...,
#     worker_count=...,
#     seed=...,
# )


# for batch in train_loader:

#     noisy_lr = batch["noisy_lr"]
#     gt = batch["gt"]

#     # model(...)
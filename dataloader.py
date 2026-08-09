from pathlib import Path

import grain.python as grain
import numpy as np


# Configuration
BATCH_SIZE = 16
NUM_WORKERS = 1
SEED = 42

P_HFLIP = 0.5
P_VFLIP = 0.5
P_ROT90 = 0.75
P_TRANSLATE = 0.5

MAX_TRANSLATION = 2


# Data Source
class NpyPairDataSource(grain.sources.RandomAccessDataSource):

    def __init__(self, noisy_dir, gt_dir):
        self.noisy_files = sorted(Path(noisy_dir).glob("*.npy"))
        self.gt_files = sorted(Path(gt_dir).glob("*.npy"))
        
        # Load everything once
        self.noisy_data = np.stack([np.load(file)for file in self.noisy_files])
        self.gt_data = np.stack([np.load(file)for file in self.gt_files])
        
    def __len__(self):
        return len(self.noisy_data)
    def __getitem__(self, index):
        return {
            "noisy_lr": self.noisy_data[index],
            "gt": self.gt_data[index]
        }


# Translation
def translate_reflect(image, dx, dy):
    if dx == 0 and dy == 0:
        return image
    h, w = image.shape[:2]

    pad_x = abs(dx)
    pad_y = abs(dy)

    padded = np.pad(
        image,((pad_y, pad_y),(pad_x, pad_x)),
        mode="reflect"
    )
    start_y = pad_y - dy
    start_x = pad_x - dx

    return padded[start_y:start_y + h,start_x:start_x + w]


# Random Aligned Augmentation
class RandomAlignedAugment(grain.transforms.RandomMap):

    def __init__(
        self,
        p_hflip=P_HFLIP,
        p_vflip=P_VFLIP,
        p_rot90=P_ROT90,
        p_translate=P_TRANSLATE,
        max_translation=MAX_TRANSLATION
    ):

        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90
        self.p_translate = p_translate
        self.max_translation = max_translation

    def random_map(self, element, rng):
        noisy = element["noisy_lr"]
        gt = element["gt"]

        # Horizontal flip
        if rng.random() < self.p_hflip:
            noisy = np.flip(noisy, axis=1)
            gt = np.flip(gt, axis=1)

        # Vertical flip
        if rng.random() < self.p_vflip:
            noisy = np.flip(noisy, axis=0)
            gt = np.flip(gt, axis=0)

        # 90 / 180 / 270 degree rotation
        if rng.random() < self.p_rot90:
            k = rng.integers(1, 4)
            noisy = np.rot90(noisy,k=k,axes=(0, 1))
            gt = np.rot90(gt,k=k,axes=(0, 1))

        # Small translation
        if rng.random() < self.p_translate:
            dx = rng.integers(-self.max_translation,self.max_translation + 1)
            dy = rng.integers(-self.max_translation,self.max_translation + 1)
            noisy = translate_reflect(noisy,dx,dy)
            gt = translate_reflect(gt,dx,dy)
            
        # Ensure positive/standard strides
        noisy = np.ascontiguousarray(noisy)
        gt = np.ascontiguousarray(gt)

        return {
            "noisy_lr": noisy,
            "gt": gt
        }


# Add Channel Dimension
class AddChannelDimension(grain.transforms.Map):
    def map(self, element):
        noisy = element["noisy_lr"]
        gt = element["gt"]

        # H,W -> 1,H,W
        if noisy.ndim == 2:
            noisy = noisy[None, ...]

        if gt.ndim == 2:
            gt = gt[None, ...]

        return {
            "noisy_lr": noisy,
            "gt": gt
        }


# Create DataLoader
def create_dataloader(
    noisy_dir,
    gt_dir,
    batch_size=BATCH_SIZE,
    worker_count=NUM_WORKERS,
    seed=SEED,
    shuffle=False,
    augment=False
):
    source = NpyPairDataSource(noisy_dir,gt_dir)
    sampler = grain.IndexSampler(num_records=len(source),num_epochs=None,shuffle=shuffle,seed=seed,shard_options=grain.NoSharding())
    operations = []
    if augment:
        operations.append(RandomAlignedAugment())

    operations.append(AddChannelDimension())
    operations.append(grain.transforms.Batch(batch_size=batch_size,drop_remainder=True))

    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=worker_count
    )
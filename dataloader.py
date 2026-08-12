from pathlib import Path

import grain
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

    def __init__(self, noisy_files: list, gt_files: list):
        self.noisy_files = list(noisy_files)
        self.gt_files = list(gt_files)
        
        # Load everything once as a list of numpy arrays to support mixed resolutions
        self.noisy_data = [np.load(file) for file in self.noisy_files]
        self.gt_data = [np.load(file) for file in self.gt_files]
        
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
            dx = rng.integers(-self.max_translation, self.max_translation + 1)
            dy = rng.integers(-self.max_translation, self.max_translation + 1)
            
            # If gt is larger than noisy (e.g. upsample dataset), scale the translation
            scale_y = gt.shape[0] // noisy.shape[0]
            scale_x = gt.shape[1] // noisy.shape[1]
            
            noisy = translate_reflect(noisy, dx, dy)
            gt = translate_reflect(gt, dx * scale_x, dy * scale_y)
            
        # Ensure positive/standard strides
        noisy = np.ascontiguousarray(noisy)
        gt = np.ascontiguousarray(gt)

        return {
            "noisy_lr": noisy,
            "gt": gt
        }


# Custom DataLoader that handles dynamic shape grouping
class GroupedDataLoader:
    def __init__(self, source, batch_size, shuffle, seed, augment):
        self.source = source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        
        # Group indices by noisy resolution
        self.indices_by_shape = {}
        for idx in range(len(source)):
            shape = source.noisy_data[idx].shape[:2]
            if shape not in self.indices_by_shape:
                self.indices_by_shape[shape] = []
            self.indices_by_shape[shape].append(idx)
            
        # Get count of total batches we will produce
        self.total_batches = 0
        for shape, indices in self.indices_by_shape.items():
            self.total_batches += len(indices) // batch_size

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        # Create batches
        batches = []
        for shape, indices in self.indices_by_shape.items():
            indices_copy = list(indices)
            if self.shuffle:
                self.rng.shuffle(indices_copy)
            
            # Group into batches (drop remainder per group)
            for i in range(0, len(indices_copy) - self.batch_size + 1, self.batch_size):
                batches.append(indices_copy[i:i+self.batch_size])
                
        if self.shuffle:
            self.rng.shuffle(batches)
            
        self.current_batch_idx = 0
        self.batches = batches
        return self

    def __next__(self):
        if self.current_batch_idx >= len(self.batches):
            raise StopIteration
            
        batch_indices = self.batches[self.current_batch_idx]
        self.current_batch_idx += 1
        
        batch_noisy = []
        batch_gt = []
        
        augmenter = RandomAlignedAugment() if self.augment else None
        
        for idx in batch_indices:
            element = self.source[idx]
            if augmenter:
                element = augmenter.random_map(element, self.rng)
                
            noisy = element["noisy_lr"]
            gt = element["gt"]
            
            # Add channel dimensions manually
            if noisy.ndim == 2:
                noisy = noisy[..., None]
            if gt.ndim == 2:
                gt = gt[..., None]
                
            batch_noisy.append(noisy)
            batch_gt.append(gt)
            
        return {
            "noisy_lr": np.stack(batch_noisy),
            "gt": np.stack(batch_gt)
        }


# Create DataLoader
def create_src_dataloader(
    noisy_files,
    gt_files,
    batch_size=BATCH_SIZE,
    worker_count=NUM_WORKERS,
    seed=SEED,
    shuffle=False,
    augment=False
):
    source = NpyPairDataSource(noisy_files, gt_files)
    loader = GroupedDataLoader(source, batch_size, shuffle, seed, augment)
    return source, loader
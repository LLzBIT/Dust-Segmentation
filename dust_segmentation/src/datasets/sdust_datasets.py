from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as transforms
from torchvision.transforms.functional import InterpolationMode

from dust_segmentation.src.utils.io import build_image_mask_paths, load_split_list


@dataclass
class DatasetConfig:
    root: Path
    train_images: str
    train_masks: str
    test_images: str
    test_masks: str
    splits_dir: str
    image_size: Tuple[int, int]
    batch_size: int
    shuffle_buffer: int
    augment_random_flip: bool
    num_classes: int


class SDustDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        config: DatasetConfig,
        training: bool,
    ) -> None:
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = config.image_size
        self.training = training
        self.augment_random_flip = config.augment_random_flip and training

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _load_mask(self, path: Path) -> Image.Image:
        return Image.open(path).convert("L")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self._load_image(self.image_paths[idx])
        mask = self._load_mask(self.mask_paths[idx])

        image = transforms.resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
        mask = transforms.resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST)

        if self.augment_random_flip:
            if torch.rand(1).item() > 0.5:
                image = transforms.hflip(image)
                mask = transforms.hflip(mask)
            if torch.rand(1).item() > 0.5:
                image = transforms.vflip(image)
                mask = transforms.vflip(mask)

        image_tensor = transforms.to_tensor(image)
        mask_tensor = transforms.to_tensor(mask)
        mask_tensor = (mask_tensor >= 0.5).float()

        return image_tensor, mask_tensor


def _make_dataset(
    image_paths: List[Path],
    mask_paths: List[Path],
    config: DatasetConfig,
    training: bool,
) -> DataLoader:
    dataset = SDustDataset(image_paths, mask_paths, config, training=training)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=training,
    )


def _load_split(config: DatasetConfig, split_name: str) -> Tuple[List[Path], List[Path]]:
    split_file = config.root / config.splits_dir / f"{split_name}.txt"
    filenames = load_split_list(split_file)
    if split_name in {"train", "val"}:
        return build_image_mask_paths(config.root, config.train_images, config.train_masks, filenames)
    return build_image_mask_paths(config.root, config.test_images, config.test_masks, filenames)


def create_datasets(config: DatasetConfig):
    train_images, train_masks = _load_split(config, "train")
    val_images, val_masks = _load_split(config, "val")
    test_images, test_masks = _load_split(config, "test")

    train_loader = _make_dataset(train_images, train_masks, config, training=True)
    val_loader = _make_dataset(val_images, val_masks, config, training=False)
    test_loader = _make_dataset(test_images, test_masks, config, training=False)
    return train_loader, val_loader, test_loader


def load_test_filenames(config: DatasetConfig) -> List[str]:
    split_file = config.root / config.splits_dir / "test.txt"
    return load_split_list(split_file)

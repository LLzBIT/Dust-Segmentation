from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf

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


def _decode_image(path: tf.Tensor) -> tf.Tensor:
    data = tf.io.read_file(path)
    image = tf.image.decode_png(data, channels=3)
    return tf.image.convert_image_dtype(image, tf.float32)


def _decode_mask(path: tf.Tensor) -> tf.Tensor:
    data = tf.io.read_file(path)
    mask = tf.image.decode_png(data, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.where(mask >= 0.5, 1.0, 0.0)
    return mask


def _resize(image: tf.Tensor, mask: tf.Tensor, size: Tuple[int, int]) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.resize(image, size, method=tf.image.ResizeMethod.BILINEAR)
    mask = tf.image.resize(mask, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image, mask


def _augment(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    return image, mask


def _make_dataset(
    image_paths: List[Path],
    mask_paths: List[Path],
    config: DatasetConfig,
    training: bool,
) -> tf.data.Dataset:
    image_strs = [str(path) for path in image_paths]
    mask_strs = [str(path) for path in mask_paths]
    dataset = tf.data.Dataset.from_tensor_slices((image_strs, mask_strs))
    if training:
        dataset = dataset.shuffle(config.shuffle_buffer, reshuffle_each_iteration=True)

    def _load_pair(img_path, mask_path):
        image = _decode_image(img_path)
        mask = _decode_mask(mask_path)
        image, mask = _resize(image, mask, config.image_size)
        if training and config.augment_random_flip:
            image, mask = _augment(image, mask)
        return image, mask

    dataset = dataset.map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


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

    train_ds = _make_dataset(train_images, train_masks, config, training=True)
    val_ds = _make_dataset(val_images, val_masks, config, training=False)
    test_ds = _make_dataset(test_images, test_masks, config, training=False)
    return train_ds, val_ds, test_ds


def load_test_filenames(config: DatasetConfig) -> List[str]:
    split_file = config.root / config.splits_dir / "test.txt"
    return load_split_list(split_file)
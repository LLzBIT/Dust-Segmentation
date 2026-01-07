import json
import time
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf
import yaml

from dust_segmentation.src.datasets.sdust_dataset import DatasetConfig, create_datasets
from dust_segmentation.src.losses.bce import binary_crossentropy
from dust_segmentation.src.models.vgg_unet import build_vgg19_unet
from dust_segmentation.src.utils.seed import set_seed


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _create_run_dir(base_dir: Path, name: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = int(time.time())
    run_dir = base_dir / f"{name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def _create_optimizer(name: str, lr: float) -> tf.keras.optimizers.Optimizer:
    if name.lower() == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def _build_dataset_config(config: Dict[str, Any]) -> DatasetConfig:
    data_cfg = config["data"]
    return DatasetConfig(
        root=Path(data_cfg["root"]),
        train_images=data_cfg["train_images"],
        train_masks=data_cfg["train_masks"],
        test_images=data_cfg["test_images"],
        test_masks=data_cfg["test_masks"],
        splits_dir=data_cfg["splits_dir"],
        image_size=tuple(data_cfg["image_size"]),
        batch_size=data_cfg["batch_size"],
        shuffle_buffer=data_cfg["shuffle_buffer"],
        augment_random_flip=data_cfg["augment"]["random_flip"],
        num_classes=data_cfg["num_classes"],
    )


def train(config_path: Path) -> Path:
    config = _load_config(config_path)
    set_seed(config["experiment"]["seed"])

    dataset_config = _build_dataset_config(config)
    train_ds, val_ds, _ = create_datasets(dataset_config)

    input_shape = (*dataset_config.image_size, 3)
    model = build_vgg19_unet(input_shape, dataset_config.num_classes)

    optimizer = _create_optimizer(config["training"]["optimizer"], config["training"]["lr"])
    model.compile(optimizer=optimizer, loss=binary_crossentropy())

    run_dir = _create_run_dir(Path("dust_segmentation/outputs/runs"), config["experiment"]["name"])
    log_path = run_dir / "logs.csv"
    callbacks = [
        tf.keras.callbacks.CSVLogger(str(log_path)),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "checkpoints" / "best.weights.h5"),
            monitor=config["training"]["checkpoint_monitor"],
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["epochs"],
        callbacks=callbacks,
    )

    with (run_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history.history, handle, indent=2)

    return run_dir


if __name__ == "__main__":
    train(Path("dust_segmentation/configs/train.yaml"))
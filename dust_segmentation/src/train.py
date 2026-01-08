import json
import time
from pathlib import Path
from typing import Any, Dict

import yaml
import torch

from dust_segmentation.src.datasets.sdust_datasets import DatasetConfig, create_datasets
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


def _create_optimizer(name: str, lr: float, parameters) -> torch.optim.Optimizer:
    if name.lower() == "adam":
        return torch.optim.Adam(parameters, lr=lr)
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
    train_loader, val_loader, _ = create_datasets(dataset_config)

    input_shape = (*dataset_config.image_size, 3)
    model = build_vgg19_unet(input_shape, dataset_config.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = _create_optimizer(config["training"]["optimizer"], config["training"]["lr"], model.parameters())
    criterion = binary_crossentropy()

    run_dir = _create_run_dir(Path("dust_segmentation/outputs/runs"), config["experiment"]["name"])
    log_path = run_dir / "logs.csv"
    history = {"loss": [], "val_loss": []}
    best_val_loss = float("inf")

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("epoch,loss,val_loss\n")

    for epoch in range(config["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_running_loss += loss.item() * images.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"{epoch + 1},{train_loss:.6f},{val_loss:.6f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best.weights.pt")

    with (run_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    return run_dir


if __name__ == "__main__":
    train(Path("dust_segmentation/configs/train.yaml"))

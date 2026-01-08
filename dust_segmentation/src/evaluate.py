import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from dust_segmentation.src.datasets.sdust_datasets import DatasetConfig, create_datasets, load_test_filenames
from dust_segmentation.src.metrics.segmentation_metrics import compute_segmentation_metrics
from dust_segmentation.src.models.vgg_unet import build_vgg19_unet
from dust_segmentation.src.utils.seed import set_seed
from dust_segmentation.src.utils.visualize import save_prediction_masks


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def evaluate(config_path: Path, weights_path: Path, run_dir: Path) -> None:
    config = _load_config(config_path)
    set_seed(config["experiment"]["seed"])

    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = _build_dataset_config(config)
    _, _, test_loader = create_datasets(dataset_config)

    input_shape = (*dataset_config.image_size, 3)
    model = build_vgg19_unet(input_shape, dataset_config.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    threshold = config["evaluation"]["threshold"]

    predictions = []
    y_true = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            predictions.append(outputs)
            y_true.append(masks.numpy())

    predictions_array = np.concatenate(predictions, axis=0)
    predicted_masks = (predictions_array >= threshold).astype(np.uint8)

    y_true_array = np.concatenate(y_true, axis=0)
    y_pred = predicted_masks

    metrics = compute_segmentation_metrics(y_true_array, y_pred)

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics.__dict__, handle, indent=2)

    filenames = load_test_filenames(dataset_config)
    save_prediction_masks(run_dir / "predictions", filenames, predicted_masks)


if __name__ == "__main__":
    evaluate(
        Path("dust_segmentation/configs/train.yaml"),
        Path("dust_segmentation/outputs/runs/latest/checkpoints/best.weights.pt"),
        Path("dust_segmentation/outputs/runs/latest"),
    )

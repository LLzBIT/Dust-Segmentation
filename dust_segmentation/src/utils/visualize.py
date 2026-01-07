from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def save_prediction_masks(
    output_dir: Path,
    filenames: Iterable[str],
    masks: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in zip(filenames, masks):
        mask_uint8 = (mask.squeeze() * 255).astype(np.uint8)
        image = Image.fromarray(mask_uint8, mode="L")
        image.save(output_dir / name)
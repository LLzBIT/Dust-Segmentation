from pathlib import Path
from typing import List, Tuple


def load_split_list(split_path: Path) -> List[str]:
    items = []
    with split_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(line)
    return items


def build_image_mask_paths(
    root: Path,
    image_dir: str,
    mask_dir: str,
    file_list: List[str],
) -> Tuple[List[Path], List[Path]]:
    images = [root / image_dir / name for name in file_list]
    masks = [root / mask_dir / name for name in file_list]
    return images, masks
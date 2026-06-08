from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


BIT_DATASET_DIRS = ("A", "B", "label", "list")
DATASET_ALIASES = {
    "LEVIR": "LEVIR-CD",
    "LEVIR-CD": "LEVIR-CD",
    "WHU": "WHU-CD",
    "WHU-CD": "WHU-CD",
    "DSIFN": "DSIFN-CD",
    "DSIFN-CD": "DSIFN-CD",
}
DATASET_SLUGS = {
    "LEVIR-CD": "levir",
    "WHU-CD": "whu",
    "DSIFN-CD": "dsifn",
}


@dataclass(frozen=True, slots=True)
class SampleSpec:
    name: str
    source_name: str
    crop_box: tuple[int, int, int, int] | None = None


def ensure_bit_layout(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    for dirname in BIT_DATASET_DIRS:
        (root / dirname).mkdir(parents=True, exist_ok=True)
    return root


def canonical_dataset_name(name: str) -> str:
    key = name.strip().upper()
    if key not in DATASET_ALIASES:
        base_key = key.split("-", 1)[0]
        if base_key not in DATASET_ALIASES:
            raise KeyError(f"Unsupported dataset name: {name}")
        return DATASET_ALIASES[base_key]
    return DATASET_ALIASES[key]


def dataset_slug(name: str) -> str:
    return DATASET_SLUGS[canonical_dataset_name(name)]


def dataset_instance_slug(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "-").replace("_", "-")
    if normalized in {alias.lower() for alias in DATASET_ALIASES}:
        return dataset_slug(name)
    return normalized


def read_split_list(list_file: str | Path) -> list[str]:
    path = Path(list_file)
    names: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        names.append(line.split()[0])
    return names


def label_path_for_name(root: str | Path, image_name: str) -> Path:
    root = Path(root)
    source = Path(image_name)
    for label_folder in _LABEL_FOLDER_ALIASES:
        candidate = root / label_folder / source.name
        if candidate.exists():
            return candidate
        stem = source.stem
        candidate = root / label_folder / f"{stem}.png"
        if candidate.exists():
            return candidate
        matches = sorted((root / label_folder).glob(f"{stem}.*"))
        if matches:
            return matches[0]
    stem = source.stem
    return root / _LABEL_FOLDER_ALIASES[0] / f"{stem}.png"


_FOLDER_ALIASES: list[tuple[str, str]] = [("A", "B"), ("t1", "t2"), ("im1", "im2"), ("img1", "img2")]
_LABEL_FOLDER_ALIASES = ("label", "mask")


def _detect_img_folders(split_root: Path) -> tuple[str, str] | None:
    """Return (folder_a, folder_b) names for the first matching pair found under split_root."""
    for folder_a, folder_b in _FOLDER_ALIASES:
        if (split_root / folder_a).exists() and (split_root / folder_b).exists():
            return folder_a, folder_b
    return None


def discover_split_layout(root: str | Path, split: str = "test") -> tuple[Path, list[str], str, str]:
    """Support BIT layout and split-subdirectory layouts with flexible folder names.

    Supported layouts:

    1. BIT layout
       root/
         A/ (or t1/, im1/, img1/)
         B/ (or t2/, im2/, img2/)
         label/
         list/{split}.txt

    2. Split-subdirectory layout
       root/
         test/A,B,label   (or test/t1,t2,label etc.)

    3. DSIFN split layout
       root/
         test/t1,t2,mask
         train/t1,t2,mask
         val/t1,t2,mask

    Returns (data_root, image_names, folder_a_name, folder_b_name).
    """
    root = Path(root)
    list_file = root / "list" / f"{split}.txt"
    if list_file.exists():
        pair = _detect_img_folders(root)
        folder_a, folder_b = pair if pair else ("A", "B")
        return root, read_split_list(list_file), folder_a, folder_b

    split_root = root / split
    pair = _detect_img_folders(split_root)
    if pair is not None:
        folder_a, folder_b = pair
        names = sorted(path.name for path in (split_root / folder_a).iterdir() if path.is_file())
        return split_root, names, folder_a, folder_b

    raise FileNotFoundError(
        f"Could not resolve dataset split '{split}' under {root}. "
        f"Expected {list_file} or a split subdirectory with one of: "
        + ", ".join(f"{a}/{b}" for a, b in _FOLDER_ALIASES)
        + f" and a label folder in {list(_LABEL_FOLDER_ALIASES)}."
    )


def _load_rgb(path: Path, img_size: int, crop_box: tuple[int, int, int, int] | None = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if crop_box is not None:
        image = image.crop(crop_box)
    if img_size is not None:
        image = TF.resize(image, [img_size, img_size], interpolation=InterpolationMode.BICUBIC)
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def _load_label(path: Path, img_size: int, crop_box: tuple[int, int, int, int] | None = None) -> torch.Tensor:
    image = Image.open(path)
    if crop_box is not None:
        image = image.crop(crop_box)
    if img_size is not None:
        image = TF.resize(image, [img_size, img_size], interpolation=InterpolationMode.NEAREST)
    array = np.array(image, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor


def _compute_patch_offsets(length: int, patch_size: int, stride: int) -> list[int]:
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and patch_stride must be positive integers.")
    if length <= patch_size:
        return [0]
    offsets = list(range(0, max(length - patch_size, 0) + 1, stride))
    last = length - patch_size
    if offsets[-1] != last:
        offsets.append(last)
    return offsets


def _build_sample_specs(
    data_root: Path,
    names: list[str],
    folder_a: str,
    patch_size: int | None,
    patch_stride: int | None,
) -> list[SampleSpec]:
    if patch_size is None or patch_stride is None:
        return [SampleSpec(name=name, source_name=name) for name in names]

    samples: list[SampleSpec] = []
    for name in names:
        with Image.open(data_root / folder_a / name) as image:
            width, height = image.size
        y_offsets = _compute_patch_offsets(height, patch_size, patch_stride)
        x_offsets = _compute_patch_offsets(width, patch_size, patch_stride)
        for y0 in y_offsets:
            for x0 in x_offsets:
                crop_box = (x0, y0, min(x0 + patch_size, width), min(y0 + patch_size, height))
                sample_name = f"{Path(name).name}::y{y0:04d}_x{x0:04d}"
                samples.append(SampleSpec(name=sample_name, source_name=name, crop_box=crop_box))
    return samples


class BitStyleChangeDataset(Dataset):
    """Minimal BIT/ChangeFormer-compatible inference dataset."""

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        img_size: int = 256,
        with_labels: bool = True,
        patch_size: int | None = None,
        patch_stride: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.with_labels = with_labels
        self.data_root, self.names, self.folder_a, self.folder_b = discover_split_layout(self.root, split=split)
        self.samples = _build_sample_specs(self.data_root, self.names, self.folder_a, patch_size, patch_stride)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_spec = self.samples[index]
        a_path = self.data_root / self.folder_a / sample_spec.source_name
        b_path = self.data_root / self.folder_b / sample_spec.source_name
        sample: dict[str, torch.Tensor | str] = {
            "name": sample_spec.name,
            "A": _load_rgb(a_path, self.img_size, crop_box=sample_spec.crop_box),
            "B": _load_rgb(b_path, self.img_size, crop_box=sample_spec.crop_box),
        }
        if self.with_labels:
            sample["L"] = _load_label(
                label_path_for_name(self.data_root, sample_spec.source_name),
                self.img_size,
                crop_box=sample_spec.crop_box,
            )
        return sample


def make_inference_loader(
    dataset_root: str | Path,
    split: str = "test",
    img_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 0,
    with_labels: bool = True,
    patch_size: int | None = None,
    patch_stride: int | None = None,
) -> DataLoader:
    dataset = BitStyleChangeDataset(
        dataset_root,
        split=split,
        img_size=img_size,
        with_labels=with_labels,
        patch_size=patch_size,
        patch_stride=patch_stride,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def iter_image_names(dataset_root: str | Path, split: str = "test") -> Iterator[str]:
    _, names, *_ = discover_split_layout(Path(dataset_root), split=split)
    yield from names

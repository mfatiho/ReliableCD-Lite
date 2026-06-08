from __future__ import annotations

from pathlib import Path

from reliable.data.bit_format import ensure_bit_layout
from reliable.data.preprocessing_manifest import PreprocessingManifest


def prepare_whu_dataset(src: str | Path, dst: str | Path) -> Path:
    """Prepare WHU-CD directory layout without mutating third-party code."""
    src_path = Path(src)
    dst_path = ensure_bit_layout(dst)
    manifest = PreprocessingManifest(
        dataset="WHU-CD",
        source_root=str(src_path),
        output_root=str(dst_path),
        image_size_original=[0, 0],
        patch_size=[0, 0],
        overlap=0,
        label_conversion="existing binary mask preserved",
        normalization="ImageNet mean/std",
        num_test_patches=0,
        script="scripts/prepare_whu.py",
    )
    manifest.save(dst_path / "preprocessing_manifest.json")
    return dst_path

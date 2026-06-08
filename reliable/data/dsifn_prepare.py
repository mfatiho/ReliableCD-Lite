from __future__ import annotations

from pathlib import Path

from reliable.data.bit_format import ensure_bit_layout
from reliable.data.preprocessing_manifest import PreprocessingManifest


def prepare_dsifn_dataset(src: str | Path, dst: str | Path) -> Path:
    """Prepare DSIFN-CD directory layout and record binary-label conversion."""
    src_path = Path(src)
    dst_path = ensure_bit_layout(dst)
    manifest = PreprocessingManifest(
        dataset="DSIFN-CD",
        source_root=str(src_path),
        output_root=str(dst_path),
        image_size_original=[0, 0],
        patch_size=[0, 0],
        overlap=0,
        label_conversion="mask > 0 -> 255",
        normalization="ImageNet mean/std",
        num_test_patches=0,
        script="scripts/prepare_dsifn.py",
    )
    manifest.save(dst_path / "preprocessing_manifest.json")
    return dst_path

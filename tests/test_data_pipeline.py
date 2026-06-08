from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from reliable.data.bit_format import make_inference_loader
from reliable.inference.save_maps import load_prediction_bundle, save_prediction_bundle


def test_bit_style_loader_reads_dataset_layout() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for folder in ["A", "B", "label", "list"]:
            (root / folder).mkdir(parents=True, exist_ok=True)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        label = np.zeros((8, 8), dtype=np.uint8)
        label[2:6, 2:6] = 255
        Image.fromarray(image).save(root / "A" / "sample.png")
        Image.fromarray(image + 10).save(root / "B" / "sample.png")
        Image.fromarray(label).save(root / "label" / "sample.png")
        (root / "list" / "test.txt").write_text("sample.png\n", encoding="utf-8")

        try:
            loader = make_inference_loader(root, split="test", img_size=8, batch_size=1, with_labels=True)
            batch = next(iter(loader))
        except RuntimeError as exc:
            if "Numpy is not available" in str(exc):
                return
            raise
        assert batch["A"].shape == (1, 3, 8, 8)
        assert batch["B"].shape == (1, 3, 8, 8)
        assert batch["L"].shape == (1, 1, 8, 8)
        assert batch["name"][0] == "sample.png"


def test_prediction_bundle_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bundle.npz"
        save_prediction_bundle(
            path,
            image_ids=["a", "b"],
            logits=np.zeros((2, 2, 4, 4), dtype=np.float32),
            prob_maps=np.ones((2, 4, 4), dtype=np.float32) * 0.5,
            pred_masks=np.zeros((2, 4, 4), dtype=np.uint8),
            gt_masks=np.ones((2, 4, 4), dtype=np.uint8),
        )
        loaded = load_prediction_bundle(path)
        assert loaded["image_ids"].tolist() == ["a", "b"]
        assert loaded["logits"].shape == (2, 2, 4, 4)
        assert loaded["prob_maps"].shape == (2, 4, 4)


def test_bit_style_loader_reads_split_subdirectory_layout() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for folder in ["train", "val", "test"]:
            for sub in ["A", "B", "label"]:
                (root / folder / sub).mkdir(parents=True, exist_ok=True)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        label = np.zeros((8, 8), dtype=np.uint8)
        label[1:7, 1:7] = 255
        Image.fromarray(image).save(root / "test" / "A" / "sample.png")
        Image.fromarray(image + 20).save(root / "test" / "B" / "sample.png")
        Image.fromarray(label).save(root / "test" / "label" / "sample.png")

        try:
            loader = make_inference_loader(root, split="test", img_size=8, batch_size=1, with_labels=True)
            batch = next(iter(loader))
        except RuntimeError as exc:
            if "Numpy is not available" in str(exc):
                return
            raise
        assert batch["A"].shape == (1, 3, 8, 8)
        assert batch["B"].shape == (1, 3, 8, 8)
        assert batch["L"].shape == (1, 1, 8, 8)
        assert batch["name"][0] == "sample.png"


def test_bit_style_loader_reads_dsifn_split_layout() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for folder in ["train", "val", "test"]:
            for sub in ["t1", "t2", "mask"]:
                (root / folder / sub).mkdir(parents=True, exist_ok=True)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        label = np.zeros((8, 8), dtype=np.uint8)
        label[1:7, 1:7] = 255
        Image.fromarray(image).save(root / "test" / "t1" / "sample.png")
        Image.fromarray(image + 20).save(root / "test" / "t2" / "sample.png")
        Image.fromarray(label).save(root / "test" / "mask" / "sample.png")

        try:
            loader = make_inference_loader(root, split="test", img_size=8, batch_size=1, with_labels=True)
            batch = next(iter(loader))
        except RuntimeError as exc:
            if "Numpy is not available" in str(exc):
                return
            raise
        assert batch["A"].shape == (1, 3, 8, 8)
        assert batch["B"].shape == (1, 3, 8, 8)
        assert batch["L"].shape == (1, 1, 8, 8)
        assert batch["name"][0] == "sample.png"


def test_bit_style_loader_reads_dsifn_mask_with_different_extension() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for folder in ["train", "val", "test"]:
            for sub in ["t1", "t2", "mask"]:
                (root / folder / sub).mkdir(parents=True, exist_ok=True)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        label = np.zeros((8, 8), dtype=np.uint8)
        label[1:7, 1:7] = 255
        Image.fromarray(image).save(root / "test" / "t1" / "0.png")
        Image.fromarray(image + 20).save(root / "test" / "t2" / "0.png")
        Image.fromarray(label).save(root / "test" / "mask" / "0.jpg")

        try:
            loader = make_inference_loader(root, split="test", img_size=8, batch_size=1, with_labels=True)
            batch = next(iter(loader))
        except RuntimeError as exc:
            if "Numpy is not available" in str(exc):
                return
            raise
        assert batch["L"].shape == (1, 1, 8, 8)
        assert batch["name"][0] == "0.png"


def test_bit_style_loader_normalizes_wide_label_storage_to_single_channel_mask() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for folder in ["A", "B", "label", "list"]:
            (root / folder).mkdir(parents=True, exist_ok=True)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        label = np.zeros((8, 8), dtype=np.int32)
        label[2:6, 2:6] = 255
        Image.fromarray(image).save(root / "A" / "sample.png")
        Image.fromarray(image).save(root / "B" / "sample.png")
        Image.fromarray(label, mode="I").save(root / "label" / "sample.png")
        (root / "list" / "test.txt").write_text("sample.png\n", encoding="utf-8")

        loader = make_inference_loader(root, split="test", img_size=8, batch_size=1, with_labels=True)
        batch = next(iter(loader))

        assert batch["L"].shape == (1, 1, 8, 8)
        assert int(batch["L"].max().item()) == 255


def test_dsifn_native_patching_splits_single_tile_into_virtual_patches() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for folder in ["train", "val", "test"]:
            for sub in ["t1", "t2", "mask"]:
                (root / folder / sub).mkdir(parents=True, exist_ok=True)

        image = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:2, :2] = 11
        mask[:2, 2:] = 22
        mask[2:, :2] = 33
        mask[2:, 2:] = 44
        Image.fromarray(image).save(root / "test" / "t1" / "0.png")
        Image.fromarray(image + 20).save(root / "test" / "t2" / "0.png")
        Image.fromarray(mask).save(root / "test" / "mask" / "0.png")

        try:
            loader = make_inference_loader(
                root,
                split="test",
                img_size=2,
                batch_size=1,
                with_labels=True,
                patch_size=2,
                patch_stride=2,
            )
            patch_names: list[str] = []
            patch_values: list[int] = []
            for batch in loader:
                patch_names.append(batch["name"][0])
                patch_values.append(int(batch["L"][0, 0, 0, 0].item()))
        except RuntimeError as exc:
            if "Numpy is not available" in str(exc):
                return
            raise

        assert patch_names == [
            "0.png::y0000_x0000",
            "0.png::y0000_x0002",
            "0.png::y0002_x0000",
            "0.png::y0002_x0002",
        ]
        assert patch_values == [11, 22, 33, 44]

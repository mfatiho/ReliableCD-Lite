from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from reliable.data import dataset_instance_slug
from reliable.utils.experiment import resolve_dataset_loader_kwargs, resolve_dataset_root, resolve_device


def test_resolve_device_auto_prefers_cuda_when_available() -> None:
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu() -> None:
    with patch("torch.cuda.is_available", return_value=False):
        assert resolve_device("auto") == "cpu"


def test_resolve_device_rejects_cuda_when_unavailable() -> None:
    with patch("torch.cuda.is_available", return_value=False):
        try:
            resolve_device("cuda")
        except RuntimeError as exc:
            assert "CUDA was requested" in str(exc)
        else:
            raise AssertionError("Expected unavailable CUDA request to fail.")


def test_resolve_dataset_loader_kwargs_enables_dsifn_native_patching_from_config() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "device: auto",
                    "threshold: 0.5",
                    "paths: {}",
                    "datasets:",
                    "  dsifn-256:",
                    "    path: /tmp/dsifn",
                    "    preprocessing:",
                    "      patching:",
                    "        enabled: true",
                    "        patch_size: 256",
                    "        stride: 128",
                ]
            ),
            encoding="utf-8",
        )
        assert resolve_dataset_loader_kwargs("dsifn-256", config_path) == {
            "patch_size": 256,
            "patch_stride": 128,
        }


def test_resolve_dataset_loader_kwargs_ignores_entries_without_patching() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "device: auto",
                    "threshold: 0.5",
                    "paths: {}",
                    "datasets:",
                    "  levir-256:",
                    "    path: /tmp/levir",
                    "    preprocessing:",
                    "      patching:",
                    "        enabled: false",
                    "        patch_size: 256",
                    "        stride: 256",
                ]
            ),
            encoding="utf-8",
        )
        assert resolve_dataset_loader_kwargs("LEVIR", config_path) == {}


def test_resolve_dataset_root_supports_mapping_and_variant_lookup() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "device: auto",
                    "threshold: 0.5",
                    "paths: {}",
                    "datasets:",
                    "  dsifn:",
                    "    path: /tmp/dsifn-default",
                    "  dsifn-256:",
                    "    path: /tmp/dsifn-patched",
                    "    preprocessing:",
                    "      patching:",
                    "        enabled: true",
                    "        patch_size: 256",
                    "        stride: 256",
                ]
            ),
            encoding="utf-8",
        )
        assert resolve_dataset_root("DSIFN", config_path) == Path("/tmp/dsifn-default")
        assert resolve_dataset_root("dsifn-256", config_path) == Path("/tmp/dsifn-patched")


def test_resolve_dataset_root_falls_back_to_256_variant_when_base_entry_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "device: auto",
                    "threshold: 0.5",
                    "paths: {}",
                    "datasets:",
                    "  levir-256:",
                    "    path: /tmp/levir-patched",
                    "    preprocessing:",
                    "      patching:",
                    "        enabled: true",
                    "        patch_size: 256",
                    "        stride: 256",
                    "  whu-256:",
                    "    path: /tmp/whu-patched",
                ]
            ),
            encoding="utf-8",
        )
        assert resolve_dataset_root("LEVIR", config_path) == Path("/tmp/levir-patched")
        assert resolve_dataset_root("WHU", config_path) == Path("/tmp/whu-patched")


def test_resolve_dataset_root_rejects_windows_drive_relative_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "device: auto",
                    "threshold: 0.5",
                    "paths: {}",
                    "datasets:",
                    "  levir-256:",
                    r"    path: D:datasets\LEVIR-CD",
                ]
            ),
            encoding="utf-8",
        )
        try:
            resolve_dataset_root("LEVIR", config_path)
        except ValueError as exc:
            assert "drive-relative" in str(exc)
            assert r"D:\datasets\LEVIR-CD" in str(exc)
        else:
            raise AssertionError("Expected drive-relative Windows dataset path to fail.")


def test_dataset_instance_slug_preserves_family_slug_for_canonical_names_and_variant_slug_for_instances() -> None:
    assert dataset_instance_slug("DSIFN") == "dsifn"
    assert dataset_instance_slug("DSIFN-CD") == "dsifn"
    assert dataset_instance_slug("dsifn-256") == "dsifn-256"

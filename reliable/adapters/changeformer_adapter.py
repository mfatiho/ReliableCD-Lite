from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable
import importlib
import sys

import torch
from torch import nn

from reliable.adapters.base import AdapterModule, ensure_repo_path, extract_logits_tensor, validate_binary_logits


def _default_changeformer_factory(
    repo_root: Path,
    checkpoint_path: Path,
    device: str,
    config_path: Path | None,
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net_name, embed_dim = _infer_changeformer_config(checkpoint, config_path)
    args = SimpleNamespace(
        net_G=net_name,
        embed_dim=embed_dim,
        lr_policy="linear",
        max_epochs=200,
        gpu_ids=[],
    )
    _purge_third_party_modules()
    sys.path.insert(0, str(repo_root))
    try:
        networks = importlib.import_module("models.networks")
        model = networks.define_G(args=args, gpu_ids=[])
    finally:
        if sys.path and sys.path[0] == str(repo_root):
            sys.path.pop(0)
    state_dict = checkpoint["model_G_state_dict"] if isinstance(checkpoint, dict) and "model_G_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    return model


def _infer_changeformer_config(checkpoint: object, config_path: Path | None) -> tuple[str, int]:
    if config_path is not None and config_path.exists():
        import json

        if config_path.suffix.lower() == ".json":
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            return str(payload.get("net_G", "ChangeFormerV6")), int(payload.get("embed_dim", 256))
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_G_state_dict", checkpoint)
        if isinstance(state_dict, dict):
            if any("TDec_x2" in key for key in state_dict):
                return "ChangeFormerV6", 256
    return "ChangeFormerV6", 256


def _purge_third_party_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "models" or module_name.startswith(("models.", "datasets.", "misc.", "data_config", "utils")):
            sys.modules.pop(module_name, None)


class ChangeFormerAdapter(AdapterModule):
    """Adapter for ChangeFormer LEVIR-CD sanity-check inference."""

    model_name = "ChangeFormer"

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        threshold: float = 0.5,
        repo_root: str = "third_party/ChangeFormer",
        config_path: str | None = None,
        dataset_name: str = "LEVIR-CD",
        model: nn.Module | None = None,
        model_factory: Callable[[Path, Path, str, Path | None], nn.Module] | None = None,
    ) -> None:
        if dataset_name.upper() not in {"LEVIR", "LEVIR-CD"}:
            raise ValueError("ChangeFormerAdapter is restricted to LEVIR-CD sanity check scope.")
        super().__init__(device=device, threshold=threshold)
        self.checkpoint_path = Path(checkpoint_path)
        self.repo_root = Path(repo_root)
        self.config_path = Path(config_path) if config_path is not None else None
        self.dataset_name = dataset_name
        factory = model_factory or _default_changeformer_factory
        self.model = model if model is not None else factory(
            ensure_repo_path(self.repo_root),
            self.checkpoint_path,
            self.device,
            self.config_path,
        )
        self.model.to(self.device)
        self.model.eval()

    def predict_logits(self, img_A: torch.Tensor, img_B: torch.Tensor) -> torch.Tensor:
        img_A, img_B = self.preprocess(img_A, img_B)
        logits = extract_logits_tensor(self.model(img_A, img_B))
        return validate_binary_logits(logits)

    def predict_prob(self, img_A: torch.Tensor, img_B: torch.Tensor) -> torch.Tensor:
        logits = self.predict_logits(img_A, img_B)
        return self.postprocess_prob(torch.softmax(logits, dim=1)[:, 1:2])

    def predict_mask(self, img_A: torch.Tensor, img_B: torch.Tensor) -> torch.Tensor:
        return (self.predict_prob(img_A, img_B) >= self.threshold).to(dtype=torch.float32)

    @property
    def supports_mc_dropout(self) -> bool:
        return False

    def enable_dropout_only(self) -> None:
        raise NotImplementedError("MC Dropout is not required for ChangeFormer sanity check.")

    def disable_dropout(self) -> None:
        self.model.eval()

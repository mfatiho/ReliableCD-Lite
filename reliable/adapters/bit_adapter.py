from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable
import importlib
import sys

import torch
from torch import nn

from reliable.adapters.base import AdapterModule, enable_dropout_only, ensure_repo_path, extract_logits_tensor, validate_binary_logits


def _default_bit_factory(repo_root: Path, checkpoint_path: Path, device: str) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net_name = _infer_bit_net_name(checkpoint)
    args = SimpleNamespace(
        net_G=net_name,
        lr_policy="linear",
        max_epochs=200,
        gpu_ids=[],
    )
    _purge_third_party_modules()
    sys.path.insert(0, str(repo_root))
    try:
        with _patch_torchvision_resnets():
            networks = importlib.import_module("models.networks")
            model = networks.define_G(args=args, gpu_ids=[])
    finally:
        if sys.path and sys.path[0] == str(repo_root):
            sys.path.pop(0)
    state_dict = checkpoint["model_G_state_dict"] if isinstance(checkpoint, dict) and "model_G_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model


def _infer_bit_net_name(checkpoint: object) -> str:
    if not isinstance(checkpoint, dict):
        return "base_transformer_pos_s4_dd8_dedim8"
    state_dict = checkpoint.get("model_G_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        return "base_transformer_pos_s4_dd8_dedim8"
    keys = list(state_dict.keys())
    if any("resnet.layer4" in key for key in keys) and not any("transformer_decoder" in key for key in keys):
        return "base_resnet18"
    decoder_q_key = "transformer_decoder.layers.0.0.fn.fn.to_q.weight"
    if decoder_q_key in state_dict:
        decoder_q_weight = state_dict[decoder_q_key]
        if isinstance(decoder_q_weight, torch.Tensor):
            first_dim = int(decoder_q_weight.shape[0])
            if first_dim == 64:
                return "base_transformer_pos_s4_dd8_dedim8"
            if first_dim == 512:
                return "base_transformer_pos_s4_dd8"
    if any("transformer_decoder" in key for key in keys):
        out_key = "transformer_decoder.layers.0.0.fn.fn.to_out.0.weight"
        if out_key in state_dict and isinstance(state_dict[out_key], torch.Tensor):
            first_dim = int(state_dict[out_key].shape[1])
            if first_dim == 64:
                return "base_transformer_pos_s4_dd8_dedim8"
            if first_dim == 512:
                return "base_transformer_pos_s4_dd8"
        return "base_transformer_pos_s4_dd8_dedim8"
    if any("resnet.layer4" in key for key in keys):
        return "base_resnet18"
    return "base_transformer_pos_s4_dd8_dedim8"


class _patch_torchvision_resnets:
    """Two compatibility fixes for the third-party BIT_CD repo:

    1. Strip deprecated `pretrained=` kwarg so newer torchvision doesn't error.
    2. Inject a `torchvision.models.utils` shim when the module no longer exists
       (removed in torchvision ≥ 0.13; BIT_CD still imports from it).
    """

    def __enter__(self) -> None:
        import torchvision.models as tv_models

        self._tv_models = tv_models
        self._originals = {
            "resnet18": tv_models.resnet18,
            "resnet34": tv_models.resnet34,
            "resnet50": tv_models.resnet50,
        }

        def wrap(fn):
            def inner(*args, **kwargs):
                kwargs.pop("pretrained", None)
                kwargs.setdefault("weights", None)
                return fn(*args, **kwargs)
            return inner

        tv_models.resnet18 = wrap(tv_models.resnet18)
        tv_models.resnet34 = wrap(tv_models.resnet34)
        tv_models.resnet50 = wrap(tv_models.resnet50)

        # Inject torchvision.models.utils shim if the real module is absent.
        # BIT_CD/models/resnet.py does:
        #   from torchvision.models.utils import load_state_dict_from_url
        # This was moved to torch.hub in torchvision 0.13+.
        self._injected_utils = "torchvision.models.utils" not in sys.modules
        if self._injected_utils:
            import types
            from torch.hub import load_state_dict_from_url  # noqa: F401
            shim = types.ModuleType("torchvision.models.utils")
            shim.load_state_dict_from_url = load_state_dict_from_url
            sys.modules["torchvision.models.utils"] = shim

    def __exit__(self, *_args: object) -> None:
        for name, fn in self._originals.items():
            setattr(self._tv_models, name, fn)
        if self._injected_utils:
            sys.modules.pop("torchvision.models.utils", None)


def _purge_third_party_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "models" or module_name.startswith(("models.", "datasets.", "misc.", "data_config", "utils")):
            sys.modules.pop(module_name, None)


class BITAdapter(AdapterModule):
    """Adapter for the official BIT_CD implementation."""

    model_name = "BIT"

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        threshold: float = 0.5,
        repo_root: str = "third_party/BIT_CD",
        model: nn.Module | None = None,
        model_factory: Callable[[Path, Path, str], nn.Module] | None = None,
    ) -> None:
        super().__init__(device=device, threshold=threshold)
        self.checkpoint_path = Path(checkpoint_path)
        self.repo_root = Path(repo_root)
        factory = model_factory or _default_bit_factory
        self.model = model if model is not None else factory(ensure_repo_path(self.repo_root), self.checkpoint_path, self.device)
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
        return True

    def enable_dropout_only(self) -> None:
        enable_dropout_only(self.model)

    def disable_dropout(self) -> None:
        self.model.eval()

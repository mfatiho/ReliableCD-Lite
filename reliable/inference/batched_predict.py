from __future__ import annotations

from collections.abc import Iterable

import torch

from reliable.inference.deterministic import deterministic_predict


@torch.no_grad()
def batched_predict(adapter, batches: Iterable[tuple[torch.Tensor, torch.Tensor, list[str]]]) -> dict[str, dict[str, torch.Tensor]]:
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    for img_A, img_B, image_ids in batches:
        pred = deterministic_predict(adapter, img_A, img_B)
        for idx, image_id in enumerate(image_ids):
            outputs[image_id] = {
                "logits": pred.logits[idx].detach().cpu(),
                "prob": pred.prob[idx].detach().cpu(),
                "mask": pred.mask[idx].detach().cpu(),
            }
    return outputs

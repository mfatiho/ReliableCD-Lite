from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from reliable.utils.progress import make_progress


def measure_runtime(
    adapter,
    dataloader,
    mode: str,
    warmup_batches: int = 5,
    measure_batches: int = 50,
    device: str = "cuda",
    progress_desc: str | None = None,
    progress_position: int = 0,
) -> pd.DataFrame:
    """Measure runtime and peak GPU memory for Table 4."""

    from reliable.runtime.modes import RUNTIME_MODES
    from reliable.runtime.profiler import Timer
    from reliable.uq.entropy import binary_entropy
    from reliable.uq.mc_dropout import MCDropoutEstimator
    from reliable.uq.shift_sensitivity import ShiftSensitivity
    from reliable.uq.tta import TTAEstimator

    config = RUNTIME_MODES[mode]
    device_str = str(device)
    use_cuda = device_str.startswith("cuda") and torch.cuda.is_available()
    tta_estimator = TTAEstimator() if mode in {"balanced", "full"} else None
    mc_estimator = MCDropoutEstimator(adapter, n_passes=20) if mode == "full" and adapter.supports_mc_dropout else None
    shift_estimator = ShiftSensitivity() if mode == "full" else None

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device_str)

    warmup_progress = None
    if progress_desc:
        warmup_progress = make_progress(
            total=warmup_batches,
            desc=f"{progress_desc} warmup",
            unit="batch",
            leave=False,
            position=progress_position,
        )
    batch_count = 0
    for batch in dataloader:
        if batch_count >= warmup_batches:
            break
        img_A, img_B = _extract_inputs(batch, device_str)
        _run_mode_forward(
            adapter=adapter,
            img_A=img_A,
            img_B=img_B,
            mode=mode,
            tta_estimator=tta_estimator,
            mc_estimator=mc_estimator,
            shift_estimator=shift_estimator,
            binary_entropy_fn=binary_entropy,
        )
        batch_count += 1
        if warmup_progress is not None:
            warmup_progress.update(1)
    if warmup_progress is not None:
        warmup_progress.close()

    times_ms: list[float] = []
    batch_size = 1
    input_size = ""
    measure_progress = None
    if progress_desc:
        measure_progress = make_progress(
            total=measure_batches,
            desc=f"{progress_desc} measure",
            unit="batch",
            leave=False,
            position=progress_position,
        )
    batch_count = 0
    for batch in dataloader:
        if batch_count >= measure_batches:
            break
        img_A, img_B = _extract_inputs(batch, device_str)
        if batch_count == 0:
            batch_size = img_A.shape[0]
            input_size = f"{img_A.shape[-2]}x{img_A.shape[-1]}"
        if use_cuda:
            torch.cuda.synchronize(device_str)
        with Timer() as t:
            _run_mode_forward(
                adapter=adapter,
                img_A=img_A,
                img_B=img_B,
                mode=mode,
                tta_estimator=tta_estimator,
                mc_estimator=mc_estimator,
                shift_estimator=shift_estimator,
                binary_entropy_fn=binary_entropy,
            )
            if use_cuda:
                torch.cuda.synchronize(device_str)
        times_ms.append(t.elapsed_ms)
        batch_count += 1
        if measure_progress is not None:
            measure_progress.update(1)
    if measure_progress is not None:
        measure_progress.close()

    peak_mb = 0.0
    gpu_name = ""
    if use_cuda:
        peak_mb = float(torch.cuda.max_memory_allocated(device_str) / 1024**2)
        gpu_name = torch.cuda.get_device_name(device_str)

    return pd.DataFrame(
        [
            {
                "model_name": getattr(adapter, "model_name", ""),
                "dataset": "",
                "mode": mode,
                "signals": ",".join(config["signals"]),
                "effective_forward_passes": config["effective_forward_passes"],
                "input_size": input_size,
                "batch_size": batch_size,
                "gpu_name": gpu_name,
                "runtime_ms_mean": float(np.mean(times_ms)) if times_ms else 0.0,
                "runtime_ms_std": float(np.std(times_ms)) if times_ms else 0.0,
                "cost_multiplier": 1.0,  # placeholder; recompute_cost_multipliers() sets correct wall-time ratios
                "peak_memory_mb": peak_mb,
                "num_images": len(times_ms) * batch_size,
            }
        ]
    )


def recompute_cost_multipliers(df: pd.DataFrame) -> pd.DataFrame:
    """Replace placeholder cost_multiplier=1.0 with wall-time ratio vs. deterministic mode.

    Groups by model_name + dataset, divides each mode's runtime_ms_mean by the
    deterministic runtime.  If no deterministic row exists for a group the values
    are left as-is.
    """
    out = df.copy()
    for _, grp in out.groupby(["model_name", "dataset"], sort=False):
        det_rows = grp[grp["mode"] == "deterministic"]
        if det_rows.empty:
            continue
        det_ms = float(det_rows["runtime_ms_mean"].iloc[0])
        if det_ms <= 0:
            continue
        out.loc[grp.index, "cost_multiplier"] = (grp["runtime_ms_mean"] / det_ms).round(2)
    return out


def summarize_runtime_samples(samples: list[dict], cost_multiplier: float) -> pd.DataFrame:
    return pd.DataFrame(samples).assign(cost_multiplier=cost_multiplier)


def _extract_inputs(batch, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        img_A = batch["A"]
        img_B = batch["B"]
    else:
        img_A = batch[0]
        img_B = batch[1]
    return img_A.to(device), img_B.to(device)


@torch.inference_mode()
def _run_mode_forward(
    adapter,
    img_A,
    img_B,
    mode: str,
    tta_estimator,
    mc_estimator,
    shift_estimator,
    binary_entropy_fn,
) -> None:
    prob = adapter.predict_prob(img_A, img_B)
    if mode in {"fast", "balanced", "full"}:
        _ = binary_entropy_fn(prob)
        _ = torch.abs(prob - 0.5)
    if mode in {"balanced", "full"} and tta_estimator is not None:
        _ = tta_estimator(adapter, img_A, img_B)
    if mode == "full":
        if mc_estimator is not None:
            _ = mc_estimator(img_A, img_B)
        if shift_estimator is not None:
            _ = shift_estimator(adapter, img_A, img_B)

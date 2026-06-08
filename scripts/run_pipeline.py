#!/usr/bin/env python3
"""
scripts/run_pipeline.py — ReliableCD-Lite paper pipeline (README §18, Adım 6-15)

Kurulum adımları (§18 Adım 1-5: ortam, repolar, checkpoint'ler, veri hazırlığı)
manuel yapılmış olmalıdır. Bu script Adım 6'dan başlar.

Temel kullanım:
  python scripts/run_pipeline.py \\
    --bit-checkpoint checkpoints/bit/BIT_LEVIR_best_ckpt.pt

BIT + ChangeFormer paralel:
  python scripts/run_pipeline.py \\
    --bit-checkpoint checkpoints/bit/BIT_LEVIR_best_ckpt.pt \\
    --changeformer \\
    --changeformer-checkpoint checkpoints/changeformer/ChangeFormer_LEVIR.pth \\
    --parallel

Adım atlama / seçme:
  python scripts/run_pipeline.py ... --skip uq calibration runtime
  python scripts/run_pipeline.py ... --only scoring figures

Komutu göster, çalıştırma:
  python scripts/run_pipeline.py ... --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Step registry — sıra önemli
# ---------------------------------------------------------------------------

ALL_STEPS = [
    "baseline",
    "uq",
    "calibration",
    "components",
    "normalizer_validation",
    "scoring",
    "missed_change",
    "referral",
    "referral_ablation",
    "sensitivity",
    "runtime",
    "threshold_sensitivity",
    "figures",
]

_LABELS = {
    "normalizer_validation": "Step 9b - LEVIR validation normalizer components",
    "baseline":      "Adım 6  — Deterministic baseline",
    "uq":            "Adım 7  — UQ maps (entropy / TTA / shift)",
    "calibration":   "Adım 8  — Temperature calibration",
    "components":    "Adım 9  — Component extraction",
    "scoring":       "Adım 10 — CRS + baseline scoring",
    "missed_change": "Adım 11 — Missed-change analysis",
    "referral":      "Adım 12 — Referral experiments",
    "referral_ablation": "Adım 12b — WHU referral ablation",
    "sensitivity":   "Adım 13 — Analyst sensitivity",
    "runtime":       "Adım 14 — Runtime profiling",
    "threshold_sensitivity": "Adım 14b — Threshold sensitivity",
    "figures":       "Adım 15 — Figure generation",
}

BIT_DATASETS = ["levir-256", "whu-256", "dsifn-256"]
CF_DATASETS  = ["levir-256"]


def _include_bit(args: argparse.Namespace) -> bool:
    return args.model_scope in {"bit", "both"}


def _include_changeformer(args: argparse.Namespace) -> bool:
    return args.model_scope in {"changeformer", "both"}

# ---------------------------------------------------------------------------
# GPU detection + device resolution
# ---------------------------------------------------------------------------

def _detect_gpu_count() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return len([l for l in r.stdout.strip().splitlines() if l.strip()])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def _resolve_devices(args: argparse.Namespace) -> None:
    """Set args.bit_device and args.cf_device.

    Parallel + 2 GPU + device is auto/cuda  →  cuda:0 / cuda:1
    Everything else                          →  both get the same args.device
    """
    if (
        args.parallel
        and _include_changeformer(args)
        and args.device in ("auto", "cuda")
        and not args.dry_run
    ):
        n = _detect_gpu_count()
        if n >= 2:
            args.bit_device = "cuda:0"
            args.cf_device  = "cuda:1"
            args.gpu_split   = True
            return

    args.bit_device = args.device
    args.cf_device  = args.device
    args.gpu_split  = False


# ---------------------------------------------------------------------------
# Command builders — her biri list[list[str]] döndürür
# Aynı adımdaki tüm komutlar bağımsızdır ve paralel çalıştırılabilir.
# ---------------------------------------------------------------------------

def _py(*args) -> list[str]:
    # Use unbuffered Python so parallel subprocess output is streamed immediately.
    return [sys.executable, "-u", *[str(a) for a in args]]


def _cmds_baseline(a: argparse.Namespace) -> list[list[str]]:
    cmds = []
    if _include_bit(a):
        cmds.append(_py(
            "scripts/eval_bit_cross_domain.py",
            "--datasets", *BIT_DATASETS,
            "--checkpoint", a.bit_checkpoint,
            "--out-dir", Path(a.out_dir) / "baseline",
            "--empty-policy", "nan",
            "--device", a.bit_device,
        ))
    if _include_changeformer(a):
        cmds.append(_py(
            "scripts/eval_changeformer_levir.py",
            "--dataset", "levir-256",
            "--checkpoint", a.changeformer_checkpoint,
            "--out-dir", Path(a.out_dir) / "baseline",
            "--empty-policy", "nan",
            "--device", a.cf_device,
        ))
    return cmds


def _cmds_uq(a: argparse.Namespace) -> list[list[str]]:
    r = Path(a.out_dir)
    cmds = []
    if _include_bit(a):
        cmds.append(_py(
            "scripts/run_uq.py",
            "--model", "BIT",
            "--datasets", *BIT_DATASETS,
            "--checkpoint", a.bit_checkpoint,
            "--out-dir", r / "uq",
            "--modes", "entropy", "tta", "shift",
            "--device", a.bit_device,
            "--batch-size", a.batch_size,
            "--num-workers", a.num_workers,
        ))
    if _include_changeformer(a):
        cmds.append(_py(
            "scripts/run_uq.py",
            "--model", "ChangeFormer",
            "--datasets", *CF_DATASETS,
            "--checkpoint", a.changeformer_checkpoint,
            "--out-dir", r / "uq",
            "--modes", "entropy", "tta", "shift",
            "--device", a.cf_device,
            "--batch-size", a.batch_size,
            "--num-workers", a.num_workers,
        ))
    return cmds


def _cmds_calibration(a: argparse.Namespace) -> list[list[str]]:
    r = Path(a.out_dir)
    cmds = []
    if _include_bit(a):
        cmds.append(_py(
            "scripts/calibrate_temperature.py",
            "--logits",  r / "baseline" / "bit_levir-256_predictions.npz",
            "--out-dir", r / "calibration",
            "--device",  a.bit_device,
        ))
    if _include_changeformer(a):
        cmds.append(_py(
            "scripts/calibrate_temperature.py",
            "--logits",  r / "baseline" / "changeformer_levir-256_predictions.npz",
            "--out-dir", r / "calibration" / "changeformer",
            "--device",  a.cf_device,
        ))
    return cmds


def _cmds_components(a: argparse.Namespace) -> list[list[str]]:
    r = Path(a.out_dir)
    cmds = []
    if _include_bit(a):
        for dataset in BIT_DATASETS:
            slug = f"bit_{dataset}"
            cmds.append(_py(
                "scripts/extract_components.py",
                "--prediction-bundle", r / "baseline"   / f"{slug}_predictions.npz",
                "--entropy-bundle",    r / "uq"          / f"{slug}_entropy_maps.npz",
                "--tta-bundle",        r / "uq"          / f"{slug}_tta_maps.npz",
                "--shift-bundle",      r / "uq"          / f"{slug}_shift_maps.npz",
                "--out",               r / "components"  / f"{slug}_components.parquet",
                "--model-name", "BIT",
                "--dataset", dataset,
            ))
    if _include_changeformer(a):
        cmds.append(_py(
            "scripts/extract_components.py",
            "--prediction-bundle", r / "baseline"   / "changeformer_levir-256_predictions.npz",
            "--entropy-bundle",    r / "uq"          / "changeformer_levir-256_entropy_maps.npz",
            "--tta-bundle",        r / "uq"          / "changeformer_levir-256_tta_maps.npz",
            "--shift-bundle",      r / "uq"          / "changeformer_levir-256_shift_maps.npz",
            "--out",               r / "components"  / "changeformer_levir-256_components.parquet",
            "--model-name", "ChangeFormer",
            "--dataset", "levir-256",
        ))
    return cmds


def _cmds_normalizer_validation(a: argparse.Namespace) -> list[list[str]]:
    if not _include_bit(a):
        return []
    r = Path(a.out_dir)
    slug = "bit_levir-256"
    return [
        _py(
            "scripts/eval_bit_cross_domain.py",
            "--datasets", "levir-256",
            "--checkpoint", a.bit_checkpoint,
            "--out-dir", r / "validation" / "baseline",
            "--split", "val",
            "--empty-policy", "nan",
            "--device", a.bit_device,
            "--batch-size", a.batch_size,
            "--num-workers", a.num_workers,
        ),
        _py(
            "scripts/run_uq.py",
            "--model", "BIT",
            "--datasets", "levir-256",
            "--checkpoint", a.bit_checkpoint,
            "--out-dir", r / "validation" / "uq",
            "--split", "val",
            "--modes", "entropy", "tta", "shift",
            "--device", a.bit_device,
            "--batch-size", a.batch_size,
            "--num-workers", a.num_workers,
        ),
        _py(
            "scripts/extract_components.py",
            "--prediction-bundle", r / "validation" / "baseline" / f"{slug}_predictions.npz",
            "--entropy-bundle", r / "validation" / "uq" / f"{slug}_entropy_maps.npz",
            "--tta-bundle", r / "validation" / "uq" / f"{slug}_tta_maps.npz",
            "--shift-bundle", r / "validation" / "uq" / f"{slug}_shift_maps.npz",
            "--out", r / "validation" / "components" / f"{slug}_components.parquet",
            "--model-name", "BIT",
            "--dataset", "levir-256-val",
        ),
    ]


def _cmds_scoring(a: argparse.Namespace) -> list[list[str]]:
    r = Path(a.out_dir)
    targets: list[tuple[str, str, Path | None]] = []
    if _include_bit(a):
        bit_normalizer_components = (
            Path(a.bit_normalizer_components)
            if getattr(a, "bit_normalizer_components", None)
            else r / "validation" / "components" / "bit_levir-256_components.parquet"
        )
        targets.extend(
            (f"bit_{ds}_components.parquet", f"bit_{ds}", bit_normalizer_components)
            for ds in BIT_DATASETS
        )
    if _include_changeformer(a):
        targets.append(("changeformer_levir-256_components.parquet", "changeformer_levir-256", None))
    return [
        _py(
            "scripts/run_crs_baselines.py",
            "--components",  r / "components" / parquet,
            "--out-dir",     r / "scoring" / out_subdir,
            *(["--fit-normalizer-from", normalizer_components] if normalizer_components is not None else []),
            "--supplementary-all",
            "--n-boot", a.n_boot,
            "--n-jobs", a.n_jobs,
        )
        for parquet, out_subdir, normalizer_components in targets
    ]


def _cmds_missed_change(a: argparse.Namespace) -> list[list[str]]:
    r = Path(a.out_dir)
    targets = [("BIT", ds, f"bit_{ds}") for ds in BIT_DATASETS] if _include_bit(a) else []
    if _include_changeformer(a):
        targets.append(("ChangeFormer", "levir-256", "changeformer_levir-256"))
    return [
        _py(
            "scripts/run_missed_change.py",
            "--gt",         r / "baseline"      / f"{slug}_predictions.npz",
            "--pred",       r / "baseline"      / f"{slug}_predictions.npz",
            "--out",        r / "missed_change" / f"{slug}_missed_change.csv",
            "--model-name", model_name,
            "--dataset",    dataset,
        )
        for model_name, dataset, slug in targets
    ]


def _cmds_referral(a: argparse.Namespace) -> list[list[str]]:
    r = Path(a.out_dir)
    budgets = ["0.01", "0.03", "0.05", "0.10", "0.20"]
    cmds = []
    if _include_bit(a):
        cmds.append(_py(
            "scripts/run_referral.py",
            "--model", "BIT",
            "--datasets", *BIT_DATASETS,
            "--budgets", *budgets,
            "--component-score",  "score_crs4",
            "--baseline-dir",     r / "baseline",
            "--uq-dir",           r / "uq",
            "--components-dir",   r / "components",
            "--pixel-map-source", "entropy",
            "--out",              r / "referral" / "bit_referral.csv",
        ))
    if _include_changeformer(a):
        cmds.append(_py(
            "scripts/run_referral.py",
            "--model", "ChangeFormer",
            "--datasets", *CF_DATASETS,
            "--budgets", *budgets,
            "--component-score",  "score_crs4",
            "--baseline-dir",     r / "baseline",
            "--uq-dir",           r / "uq",
            "--components-dir",   r / "components",
            "--pixel-map-source", "entropy",
            "--out",              r / "referral" / "changeformer_referral.csv",
        ))
    return cmds


def _cmds_sensitivity(a: argparse.Namespace) -> list[list[str]]:
    r = Path(a.out_dir)
    common = [
        "--budgets", "0.05", "0.10",
        "--pixel-source",    "entropy",
        "--component-score", "score_crs4",
        "--gain-source",     "component",
        "--alphas", "1.0", "0.9", "0.7", "0.5",
    ]
    cmds = [
        # Tüm BIT dataset'leri — ana paper tablosu
        _py(
            "scripts/run_analyst_sensitivity.py",
            "--referral-wide", r / "referral" / "bit_referral_wide.csv",
            "--datasets", *BIT_DATASETS,
            *common,
            "--per-setting-dir", r / "referral" / "analyst_sensitivity",
            "--out",             r / "referral" / "analyst_sensitivity_table.csv",
        ),
        # Sadece WHU-256 — birincil operasyonel tablo
        _py(
            "scripts/run_analyst_sensitivity.py",
            "--referral-wide", r / "referral" / "bit_referral_wide.csv",
            "--datasets", "whu-256",
            *common,
            "--per-setting-dir", r / "referral" / "analyst_sensitivity",
            "--out",             r / "referral" / "analyst_sensitivity_whu_main.csv",
        ),
    ]
    if _include_changeformer(a):
        cmds.append(_py(
            "scripts/run_analyst_sensitivity.py",
            "--referral-wide", r / "referral" / "changeformer_referral_wide.csv",
            "--datasets", "levir-256",
            *common,
            "--per-setting-dir", r / "referral" / "analyst_sensitivity" / "changeformer",
            "--out",             r / "referral" / "changeformer_analyst_sensitivity_table.csv",
        ))
    if not _include_bit(a):
        cmds = cmds[2:]
    return cmds


def _cmds_referral_ablation(a: argparse.Namespace) -> list[list[str]]:
    r = Path(a.out_dir)
    if not _include_bit(a):
        return []
    return [_py(
        "scripts/run_referral_component_ablation.py",
        "--model", "BIT",
        "--dataset", "whu-256",
        "--budgets", "0.05", "0.10",
        "--component-scores", "score_crs4", "score_margin", "score_entropy",
        "--baseline-dir", r / "baseline",
        "--components-dir", r / "components",
        "--out", r / "referral" / "bit_whu-256_component_ablation.csv",
        "--n-boot", a.n_boot,
    )]


def _cmds_runtime(a: argparse.Namespace) -> list[list[str]]:
    r = Path(a.out_dir)
    warmup  = "1" if a.runtime_mode == "smoke" else "5"
    measure = "5" if a.runtime_mode == "smoke" else "50"
    cmds = [_py(
        "scripts/measure_runtime.py",
        "--model", "BIT",
        "--dataset", "levir-256",
        "--checkpoint", a.bit_checkpoint,
        "--modes", "deterministic", "fast", "balanced", "full",
        "--out",             r / "runtime" / "bit_levir-256_runtime.csv",
        "--device",          a.bit_device,
        "--batch-size",      a.batch_size,
        "--num-workers",     a.num_workers,
        "--warmup-batches",  warmup,
        "--measure-batches", measure,
    )]
    if _include_changeformer(a):
        cmds.append(_py(
            "scripts/measure_runtime.py",
            "--model", "ChangeFormer",
            "--dataset", "levir-256",
            "--checkpoint", a.changeformer_checkpoint,
            "--modes", "deterministic", "fast", "balanced", "full",
            "--out",             r / "runtime" / "changeformer_levir-256_runtime.csv",
            "--device",          a.cf_device,
            "--batch-size",      a.batch_size,
            "--num-workers",     a.num_workers,
            "--warmup-batches",  warmup,
            "--measure-batches", measure,
        ))
    if not _include_bit(a):
        cmds = cmds[1:]
    return cmds


def _cmds_threshold_sensitivity(a: argparse.Namespace) -> list[list[str]]:
    r = Path(a.out_dir)
    cmds = [_py(
        "scripts/run_threshold_sensitivity.py",
        "--model", "BIT",
        "--datasets", *BIT_DATASETS,
        "--thresholds", "0.4", "0.5", "0.6",
        "--baseline-dir", r / "baseline",
        "--out", r / "analysis" / "bit_threshold_sensitivity.csv",
    )]
    if _include_changeformer(a):
        cmds.append(_py(
            "scripts/run_threshold_sensitivity.py",
            "--model", "ChangeFormer",
            "--datasets", *CF_DATASETS,
            "--thresholds", "0.4", "0.5", "0.6",
            "--baseline-dir", r / "baseline",
            "--out", r / "analysis" / "changeformer_threshold_sensitivity.csv",
        ))
    if not _include_bit(a):
        cmds = cmds[1:]
    return cmds


def _cmds_figures(a: argparse.Namespace) -> list[list[str]]:
    r = Path(a.out_dir)
    cmds: list[list[str]] = []
    if _include_bit(a):
        cmds.append(_py(
            "scripts/make_figures.py",
            "--components",    r / "components",
            "--referral-wide", r / "referral" / "bit_referral_wide.csv",
            "--referral-per-image", r / "referral" / "bit_referral_per_image.csv",
            "--missed-change-dir", r / "missed_change",
            "--out-dir",       r / "figures" / "bit",
        ))
        cmds.append(_py(
            "scripts/make_figure2_panel.py",
            "--config",        "configs/default.yaml",
            "--candidates",    r / "figures" / "bit" / "figure2_qualitative_case_candidates.csv",
            "--components-dir", r / "components",
            "--baseline-dir",   r / "baseline",
            "--uq-dir",         r / "uq",
            "--out",            r / "figures" / "bit" / "figure2_rs_panel.pdf",
        ))
    return cmds


_STEP_CMDS = {
    "baseline":      _cmds_baseline,
    "uq":            _cmds_uq,
    "calibration":   _cmds_calibration,
    "components":    _cmds_components,
    "normalizer_validation": _cmds_normalizer_validation,
    "scoring":       _cmds_scoring,
    "missed_change": _cmds_missed_change,
    "referral":      _cmds_referral,
    "referral_ablation": _cmds_referral_ablation,
    "sensitivity":   _cmds_sensitivity,
    "runtime":       _cmds_runtime,
    "threshold_sensitivity": _cmds_threshold_sensitivity,
    "figures":       _cmds_figures,
}

# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _label_of(cmd: list[str]) -> str:
    """Short tag for prefixing output lines: script stem + first distinguishing arg."""
    stem = Path(cmd[1]).stem
    for arg in cmd[2:]:
        if not arg.startswith("-") and not str(arg).startswith("results"):
            return f"{stem}:{arg}"
    return stem


def _run_sequential(cmds: list[list[str]], dry_run: bool) -> None:
    for cmd in cmds:
        parts = [str(c) for c in cmd]
        print("    $", " ".join(parts))
        if not dry_run:
            subprocess.run(parts, check=True)


def _run_parallel(cmds: list[list[str]], dry_run: bool) -> None:
    """Launch all commands simultaneously; stream output with per-process prefix."""
    for cmd in cmds:
        print("    $", " ".join(str(c) for c in cmd))
    if dry_run:
        return

    lock = threading.Lock()

    def stream(proc: subprocess.Popen, prefix: str) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw if isinstance(raw, str) else raw.decode(errors="replace")
            with lock:
                print(f"  [{prefix}] {line}", end="", flush=True)

    procs: list[subprocess.Popen] = []
    threads: list[threading.Thread] = []

    for cmd in cmds:
        parts = [str(c) for c in cmd]
        prefix = _label_of(parts)
        proc = subprocess.Popen(
            parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        t = threading.Thread(target=stream, args=(proc, prefix), daemon=True)
        t.start()
        procs.append(proc)
        threads.append(t)

    for t in threads:
        t.join()

    failed = [p for p in procs if p.wait() != 0]
    if failed:
        raise subprocess.CalledProcessError(failed[0].returncode, failed[0].args)


def _execute(cmds: list[list[str]], dry_run: bool, parallel: bool) -> None:
    if parallel and len(cmds) > 1:
        _run_parallel(cmds, dry_run)
    else:
        _run_sequential(cmds, dry_run)


def _step_allows_parallel(step: str) -> bool:
    # The Figure 2 RS panel consumes the candidate CSV emitted by BIT make_figures.py.
    # The validation-normalizer step has an internal baseline -> UQ -> components dependency.
    # Scoring reuses one source normalizer across BIT datasets.
    return step not in {"figures", "normalizer_validation", "scoring"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ReliableCD-Lite paper pipeline (README §18, Adım 6-15)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    ck = parser.add_argument_group("checkpoint")
    ck.add_argument("--bit-checkpoint", default="checkpoints/bit/BIT_LEVIR_best_ckpt.pt")
    ck.add_argument("--changeformer", action="store_true",
                    help="ChangeFormer sanity-check adımlarını da çalıştır")
    ck.add_argument("--changeformer-checkpoint",
                    default="checkpoints/changeformer/ChangeFormer_LEVIR.pth")
    ck.add_argument(
        "--model-scope",
        choices=["bit", "changeformer", "both"],
        default=None,
        help="Model scope: bit, changeformer, or both. Default is bit; legacy --changeformer means both.",
    )

    io = parser.add_argument_group("output / hardware")
    io.add_argument("--out-dir", default="results")
    io.add_argument("--device", default="auto",
                    help="GPU-bound adımlar için cihaz: auto | cpu | cuda | cuda:N (default: auto)")
    io.add_argument("--batch-size",  type=int, default=4)
    io.add_argument("--num-workers", type=int, default=4)
    io.add_argument("--parallel", action="store_true",
                    help="Her adımda BIT ve ChangeFormer komutlarını eş zamanlı çalıştır")

    sc = parser.add_argument_group("scoring (Adım 10)")
    sc.add_argument("--n-boot", type=int, default=1000)
    sc.add_argument(
        "--bit-normalizer-components",
        default=None,
        help=(
            "Reference component parquet used to fit the BIT CRS z-score normalizer. "
            "Default: <out-dir>/validation/components/bit_levir-256_components.parquet, "
            "generated by the normalizer_validation step."
        ),
    )
    sc.add_argument("--n-jobs", type=int, default=1,
                    help="CRS için paralel thread sayısı. -1 = tüm CPU")

    rt = parser.add_argument_group("runtime (Adım 14)")
    rt.add_argument("--runtime-mode", choices=["paper", "smoke"], default="paper",
                    help="paper: warmup=5/measure=50  smoke: warmup=1/measure=5")

    sel = parser.add_argument_group("step selection")
    sel.add_argument("--skip", nargs="*", metavar="STEP", default=[], choices=ALL_STEPS,
                     help=f"Atlanacak adımlar: {', '.join(ALL_STEPS)}")
    sel.add_argument("--only", nargs="*", metavar="STEP", default=[], choices=ALL_STEPS,
                     help="Yalnızca bu adımları çalıştır (--skip'i geçersiz kılar)")

    parser.add_argument("--dry-run", action="store_true",
                        help="Komutları yazdır, çalıştırma")

    args = parser.parse_args()
    if args.model_scope is None:
        args.model_scope = "both" if args.changeformer else "bit"

    steps = (
        [s for s in ALL_STEPS if s in set(args.only)]
        if args.only
        else [s for s in ALL_STEPS if s not in set(args.skip)]
    )
    if not steps:
        print("Çalıştırılacak adım yok.", file=sys.stderr)
        sys.exit(1)

    if _include_changeformer(args) and not args.dry_run:
        cf_ck = Path(args.changeformer_checkpoint)
        if not cf_ck.exists():
            print(f"HATA: ChangeFormer checkpoint bulunamadı: {cf_ck}", file=sys.stderr)
            sys.exit(1)

    _resolve_devices(args)

    sep = "=" * 66
    print(f"\n{sep}")
    print("  ReliableCD-Lite Paper Pipeline")
    print(sep)
    print(f"  Adımlar ({len(steps)}): {', '.join(steps)}")
    print(f"  Çıktı  : {Path(args.out_dir).resolve()}")
    if _include_bit(args) and _include_changeformer(args):
        exec_tag = "paralel" if args.parallel else "sıralı"
        print(f"  Model  : BIT + ChangeFormer ({exec_tag})")
        if args.gpu_split:
            print(f"  GPU    : BIT → {args.bit_device}  |  ChangeFormer → {args.cf_device}")
        else:
            print(f"  Cihaz  : {args.bit_device}")
    elif _include_changeformer(args):
        print("  Model  : ChangeFormer")
        print(f"  Cihaz  : {args.cf_device}")
    else:
        print("  Model  : BIT")
        print(f"  Cihaz  : {args.bit_device}")
    if args.dry_run:
        print("  Mod    : DRY RUN")
    print(sep)

    timings: dict[str, float] = {}
    pipeline_start = time.perf_counter()

    for step in steps:
        label = _LABELS[step]
        print(f"\n{sep}\n  {label}\n{sep}")
        t0 = time.perf_counter()
        try:
            cmds = _STEP_CMDS[step](args)
            _execute(cmds, dry_run=args.dry_run, parallel=args.parallel and _step_allows_parallel(step))
        except subprocess.CalledProcessError as exc:
            elapsed = time.perf_counter() - t0
            print(f"\n  HATA: {label} başarısız oldu "
                  f"(çıkış kodu {exc.returncode}, {elapsed:.1f}s)", file=sys.stderr)
            print("  Pipeline durduruluyor.", file=sys.stderr)
            sys.exit(exc.returncode)
        elapsed = time.perf_counter() - t0
        timings[step] = elapsed
        print(f"\n  OK  {label}  ({_fmt_time(elapsed)})")

    total = time.perf_counter() - pipeline_start
    print(f"\n{sep}\n  Tüm adımlar tamamlandı\n{sep}")
    for step, t in timings.items():
        print(f"  OK  {_LABELS[step]:<44}  {_fmt_time(t):>8}")
    print(f"  {'':3} {'TOPLAM':<44}  {_fmt_time(total):>8}")
    print(sep)


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


if __name__ == "__main__":
    main()

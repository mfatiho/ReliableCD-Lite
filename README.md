# ReliableCD-Lite

ReliableCD-Lite is a training-free, post-hoc **component-level triage / referral** layer for frozen remote sensing change detection models. It does not propose a new backbone. It ranks the predicted change components of an existing CD model so an analyst can review the highest-risk ones first, under a fixed reviewed-area budget.

This README is self-contained: it covers understanding, install, configuration, verification, and running the project end-to-end.

## Contents

- [1. Purpose and scope](#purpose)
- [2. Method summary](#method)
- [3. Project structure](#structure)
- [4. Models and datasets](#models)
- [5. Setup](#setup)
- [6. Configuration](#config)
- [7. Verifying the installation](#verify)
- [8. Running the full pipeline](#pipeline)
- [9. Step-by-step workflow](#workflow)
- [10. Output files](#outputs)

## <a id="purpose"></a>1. Purpose and scope

ReliableCD-Lite ranks predicted change components by a reliability score and directs a limited human review budget toward the highest-risk components, without changing the model architecture or retraining.

**Scope:**

- Main backbone: `BIT` on `levir-256` (in-domain), `whu-256` (near-domain), `dsifn-256` (far-domain stress).
- Cross-backbone sanity check: `ChangeFormer`, `levir-256` only.
- No fine-tuning on `whu-256` or `dsifn-256`. The whole layer is applied post-hoc.

**Not claimed:** SOTA accuracy, a new backbone, a universal model-agnostic method, calibrated uncertainty, conformal/coverage guarantees, real analyst performance, or an XAI framework.

## <a id="method"></a>2. Method summary

```text
Image pair (A, B)
  → frozen CD model (BIT, or ChangeFormer in a limited scope)
  → probability map / predicted mask
  → risk signals: entropy, TTA disagreement, output-level shift sensitivity,
    boundary uncertainty (+ optional supplementary stochastic baseline, BIT only)
  → connected component extraction (8-connectivity, min area 16 px)
  → CRS / baseline ranking
  → area-budgeted human review simulation
  → Error Recall @ Review Budget + oracle upper-bound gains
```

### Composite Reliability Score (CRS)

Components are ranked by a heuristic, equal-weight **Composite Reliability Score (CRS)**. `z(.)` is z-score normalization with statistics fit once on the LEVIR-CD validation split and applied unchanged to the other sets; `M = |mean_prob - 0.5|` is the probability margin.

| Variant | Formula | Role |
|---|---|---|
| CRS-1 | `z(entropy)` | Single-signal entropy baseline |
| CRS-2 | `z(entropy) + z(boundary)` | + boundary uncertainty |
| CRS-3 | `z(entropy) + z(boundary) - z(M)` | **Default fast score** — one forward pass |
| CRS-4 | `CRS-3 + z(tta) + z(shift)` | Optional full score — adds TTA + shift |

CRS-3 is the recommended default (single deterministic forward pass) and matches or slightly exceeds CRS-4 in-domain. CRS-4 is used for component ranking in the main referral table. No variant uses an MI term. Equal weights keep the method training-free.

### Review budget

The budget is a fraction of total image area, not a component count:

```text
review_budget = reviewed_pixels / total_image_pixels
```

Components are added in descending score order until cumulative area reaches the budget; overshoot on the last component is allowed and reported. Component-level and pixel-level (entropy-ranked) referral are compared under the same budget. Grid: `[0.01, 0.03, 0.05, 0.10, 0.20]`; main report points: 5% and 10%. Component referral is a review-ordering tool; pixel-level entropy referral gives larger oracle correction gains.

## <a id="structure"></a>3. Project structure

```text
ReliableCD/
├── configs/        # YAML config files
├── reliable/       # all project code (see below)
├── scripts/        # runnable entry points
├── tests/          # unit + smoke tests
├── docs/           # paper sources
├── data/           # prepared by the user
├── checkpoints/    # placed by the user
├── third_party/    # official BIT / ChangeFormer repos (never modified)
└── results/        # all generated artifacts
```

Modules under `reliable/`: `adapters/` (BIT, ChangeFormer), `inference/`, `uq/` (entropy, TTA, shift, MC dropout), `calibration/`, `components/` (extraction, features, oracle labeling, missed-change), `scoring/` (z-score, CRS, baselines), `referral/` (pixel, component, oracle correction, analyst sensitivity), `metrics/`, `stats/`, `runtime/`, `utils/`, `visualization/`.

## <a id="models"></a>4. Models and datasets

| Model | Role | Datasets |
|---|---|---|
| BIT | Main model | `levir-256`, `whu-256`, `dsifn-256` |
| ChangeFormer | Cross-backbone sanity check | `levir-256` only |

| Dataset | Role |
|---|---|
| `levir-256` | In-domain validation; normalizer fit source |
| `whu-256` | Near-domain cross-domain test (primary cross-domain evidence) |
| `dsifn-256` | Far-domain stress / breakdown evidence |

DSIFN semantic labels are binarized: `binary_mask = (multiclass_mask > 0).astype(np.uint8) * 255`.

## <a id="setup"></a>5. Setup

**Environment** (the project runs in the `bitcd` conda env):

```bash
conda create -n bitcd python=3.10 -y
conda activate bitcd
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -e .[dev]
```

**Third-party repos** (core files are never modified):

```bash
mkdir -p third_party checkpoints data results
git clone https://github.com/justchenhao/BIT_CD.git third_party/BIT_CD
git clone https://github.com/wgcban/ChangeFormer.git third_party/ChangeFormer
```

**Checkpoints:**

```text
checkpoints/bit/BIT_LEVIR_best_ckpt.pt
checkpoints/changeformer/ChangeFormer_LEVIR.pth
```

**Data.** The loader accepts two layouts: a BIT-style `A/ B/ label/ list/{split}.txt`, or split subdirectories `train|val|test/{A,B,label}/`. It first looks for `list/{split}.txt`, then falls back to the split subdirectories. LEVIR-CD usually ships ready; prepare the others:

```bash
python scripts/prepare_whu.py   --src /path/raw/WHU-CD   --dst /data/WHU-CD-256
python scripts/prepare_dsifn.py --src /path/raw/DSIFN-CD --dst /data/DSIFN-CD
```

## <a id="config"></a>6. Configuration

`configs/default.yaml` holds the global seed, device, threshold, repo/data paths, and the three paper datasets with their preprocessing. Edit each `path` to your machine:

```yaml
seed: 42
device: auto          # auto | cpu | cuda | cuda:0
threshold: 0.5

paths:
  bit_repo: third_party/BIT_CD
  changeformer_repo: third_party/ChangeFormer
  data_root: data
  results_root: results

datasets:
  levir-256:
    path: D:\datasets\LEVIR-CD
    preprocessing:
      patching: { enabled: true, patch_size: 256, stride: 256 }
  whu-256:
    path: D:\datasets\WHU-CD
    preprocessing:
      patching: { enabled: true, patch_size: 256 }
  dsifn-256:
    path: D:\datasets\DSIFN-CD
    preprocessing:
      patching: { enabled: true, patch_size: 256, stride: 256 }
```

Other config files:

| File | Contents |
|---|---|
| `bit.yaml` | `model_name`, `threshold`, `repo_root` |
| `changeformer.yaml` | same fields + `dataset_name: LEVIR-CD` (LEVIR scope) |
| `crs.yaml` | `uncertainty_source` per model (BIT → `mean_mi`, ChangeFormer → `mean_entropy`); BIT falls back to entropy when the supplementary MI column is absent |
| `uq.yaml` | `mc_dropout` (BIT only, 20 passes, `inject_p` 0.1), `tta` (6 fixed augmentations), `shift_sensitivity` (1 px, 4 cardinal directions, shifts image A) |
| `referral.yaml` | `budgets`, `budget_definition: image_area`, `component_selection: cumulative_area`, overshoot flag, `oracle_gain_label: upper_bound`, main/supplementary baseline lists |
| `runtime.yaml` | signal sets for `deterministic` / `fast` / `balanced` / `full` modes |

## <a id="verify"></a>7. Verifying the installation

All 108 tests must pass before running experiments. No checkpoints or data are required.

```bash
python -m pytest tests/        # add --basetemp=.tmp_pytest on Windows if temp dir is locked
bash scripts/smoke_test.sh
```

## <a id="pipeline"></a>8. Running the full pipeline

After Sections 5–7, the entire workflow runs with one command. The orchestrator runs the stages in order (baseline → UQ → calibration → components → scoring → missed-change → referral → analyst sensitivity → runtime → figures) and links each stage's output to the next.

```bash
python scripts/run_pipeline.py --bit-checkpoint checkpoints/bit/BIT_LEVIR_best_ckpt.pt
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--bit-checkpoint` | required | BIT checkpoint file |
| `--changeformer` / `--changeformer-checkpoint` | off | Also run ChangeFormer (LEVIR only) |
| `--device` | `cuda` | `cpu`, `cuda`, `cuda:0`, … |
| `--parallel` | off | Run BIT and ChangeFormer in parallel (auto GPU split if ≥2 GPUs) |
| `--skip` / `--only` | — | Skip / restrict to listed steps |
| `--dry-run` | off | Print commands without running |
| `--out-dir` | `results` | Output root |

Step names: `baseline`, `uq`, `calibration`, `components`, `scoring`, `missed_change`, `referral`, `sensitivity`, `runtime`, `figures`.

```bash
# BIT + ChangeFormer in parallel
python scripts/run_pipeline.py \
  --bit-checkpoint checkpoints/bit/BIT_LEVIR_best_ckpt.pt \
  --changeformer --changeformer-checkpoint checkpoints/changeformer/ChangeFormer_LEVIR.pth --parallel

# Regenerate only the figures
python scripts/run_pipeline.py --bit-checkpoint checkpoints/bit/BIT_LEVIR_best_ckpt.pt --only figures
```

## <a id="workflow"></a>9. Step-by-step workflow

The same stages, one command at a time, for inspecting or rerunning a single stage. Commands below use BIT `levir-256`; repeat per dataset by substituting the paths, `--datasets`/`--dataset`, and `--out`. Keep the stage order.

**1. Baseline** — checkpoint-based inference; saves metrics, prob maps, masks.

```bash
python scripts/eval_bit_cross_domain.py \
  --datasets levir-256 whu-256 dsifn-256 \
  --checkpoint checkpoints/bit/BIT_LEVIR_best_ckpt.pt \
  --out-dir results/baseline --empty-policy nan --device cuda
```
ChangeFormer: `scripts/eval_changeformer_levir.py --dataset levir-256 ...`.

**2. UQ maps** — entropy, TTA, shift (GPU strongly recommended). `--device auto` picks GPU when available.

```bash
python scripts/run_uq.py --model BIT --datasets levir-256 whu-256 dsifn-256 \
  --checkpoint checkpoints/bit/BIT_LEVIR_best_ckpt.pt \
  --out-dir results/uq --modes entropy tta shift --device auto --batch-size 4 --num-workers 4
```
The optional stochastic baseline (BIT only) uses `--modes mi`; treat it as supplementary.

**3. Calibration** — temperature scaling fit on LEVIR val logits.

```bash
python scripts/calibrate_temperature.py \
  --logits results/baseline/bit_levir-256_predictions.npz --out-dir results/calibration --device auto
```

**4. Component table** — components + lifted maps + oracle labels (once per dataset).

```bash
python scripts/extract_components.py \
  --prediction-bundle results/baseline/bit_levir-256_predictions.npz \
  --entropy-bundle results/uq/bit_levir-256_entropy_maps.npz \
  --tta-bundle results/uq/bit_levir-256_tta_maps.npz \
  --shift-bundle results/uq/bit_levir-256_shift_maps.npz \
  --out results/components/bit_levir-256_components.parquet --model-name BIT --dataset levir-256
```

**5. CRS and baseline tables** — z-score normalizer, `crs1-4`, `score_*` columns.

```bash
python scripts/run_crs_baselines.py \
  --components results/components/bit_levir-256_components.parquet \
  --out-dir results/scoring/bit_levir-256 --supplementary-all
```

**6. Missed-change** — mandatory; quantifies the false-negative blind spot. Keep it as a limitation, separate from the referral metric.

```bash
python scripts/run_missed_change.py \
  --gt results/baseline/bit_levir-256_predictions.npz \
  --pred results/baseline/bit_levir-256_predictions.npz \
  --out results/missed_change/bit_levir-256_missed_change.csv --model-name BIT --dataset levir-256
```

**7. Referral** — pixel vs component under the same area budget. Image-level mean is the main point estimate.

```bash
python scripts/run_referral.py --model BIT --datasets levir-256 whu-256 dsifn-256 \
  --budgets 0.01 0.03 0.05 0.10 0.20 --component-score score_crs4 \
  --baseline-dir results/baseline --uq-dir results/uq --components-dir results/components \
  --pixel-map-source entropy --out results/referral/bit_referral.csv
```
The `f1_gain_upper_bound` / `iou_gain_upper_bound` column names must be preserved.

**8. Analyst sensitivity** — scales oracle upper-bound gains under imperfect correction; not a user study. Fed from the referral wide CSV.

```bash
python scripts/run_analyst_sensitivity.py \
  --referral-wide results/referral/bit_referral_wide.csv \
  --datasets whu-256 levir-256 dsifn-256 --budgets 0.05 0.10 \
  --pixel-source entropy --component-score score_crs4 --gain-source component \
  --alphas 1.0 0.9 0.7 0.5 --per-setting-dir results/referral/analyst_sensitivity \
  --out results/referral/analyst_sensitivity_table.csv
```

**9. Runtime** — real timing for `deterministic`, `fast`, `balanced`, `full`. Use `--device auto`/`cuda`; keep `--warmup-batches 5 --measure-batches 50` for paper-grade runs.

```bash
python scripts/measure_runtime.py --model BIT --dataset levir-256 \
  --checkpoint checkpoints/bit/BIT_LEVIR_best_ckpt.pt \
  --modes deterministic fast balanced full --out results/runtime/bit_levir-256_runtime.csv \
  --device auto --batch-size 4 --num-workers 4 --warmup-batches 5 --measure-batches 50
```

**10. Figures** — `matplotlib`-based; finalize referral/scoring CSVs first. `make_figures.py` produces seven figures plus the qualitative candidate CSV (AURC is reported as inline text, no separate figure).

```bash
python scripts/make_figures.py \
  --components results/components --referral-wide results/referral/bit_referral_wide.csv \
  --out-dir results/figures/bit
```
The qualitative RS panel (paper Fig 7) is built by `make_figure2_panel.py` from that candidate CSV; paper-facing PDFs are copied into `docs/assets/figures/bit/pdf/`.

## <a id="outputs"></a>10. Output files

A complete run produces, under `results/`:

```text
baseline/        bit_<dataset>_metrics.csv, _predictions.npz, _manifest.json
uq/              bit_<dataset>_{entropy,tta,shift}_maps.npz, _uq_summary.csv
calibration/     temperature_scaling.json, ece_brier_before_after.csv, calibration_before_after.pdf
components/      bit_<dataset>_components.parquet, _components_summary.csv
scoring/         bit_<dataset>/{main_baselines_table2.csv, supplementary_full_baselines.csv}
missed_change/   bit_<dataset>_missed_change.csv
referral/        bit_referral.csv, bit_referral_wide.csv, _review_budget_curves.pdf,
                 analyst_sensitivity_table.csv, analyst_sensitivity/
runtime/         bit_<dataset>_runtime.csv, _runtime.pdf
figures/bit/     pdf/{figure1_framework, figure3_review_budget_curves,
                 figure5b_spearman_supplementary, figure5c_feature_error_correlation,
                 figure7_score_separation, figure_protocol_walkthrough, figure_cost_accuracy}.pdf,
                 figure2_rs_panel.pdf, figure2_qualitative_case_candidates.csv
```

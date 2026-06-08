from __future__ import annotations

import pandas as pd

from reliable.visualization.panels import (
    PAPER_RC_PARAMS,
    display_dataset_name,
    paper_style,
    plot_dataset_difficulty_dashboard,
    plot_framework_diagram,
    select_failure_mode_cases,
    select_review_case_images,
)


def test_paper_rc_params_uses_serif_font() -> None:
    assert PAPER_RC_PARAMS["font.family"] == "serif"
    assert PAPER_RC_PARAMS["savefig.dpi"] == 300


def test_paper_style_returns_rc_context() -> None:
    ctx = paper_style()
    assert hasattr(ctx, "__enter__")


def test_paper_style_enables_latex_for_pgf() -> None:
    import matplotlib as mpl

    with paper_style("figure.pgf"):
        assert mpl.rcParams["text.usetex"] is False
        assert mpl.rcParams["pgf.texsystem"] == "xelatex"
        assert mpl.rcParams["pgf.rcfonts"] is False


def test_display_dataset_name_uses_shared_labels() -> None:
    assert display_dataset_name("levir-256") == "LEVIR-CD"
    assert display_dataset_name("whu-256") == "WHU-CD"
    assert display_dataset_name("dsifn-256") == "DSIFN-CD†"


def test_framework_diagram_writes_compact_pdf(tmp_path) -> None:
    out_path = tmp_path / "figure1_framework.pdf"

    plot_framework_diagram(out_path, model_name="BIT")

    assert out_path.exists()


def test_dataset_difficulty_dashboard_supports_single_lowercase_dataset(tmp_path) -> None:
    components_df = pd.DataFrame(
        [
            {
                "model_name": "ChangeFormer",
                "dataset": "levir-256",
                "image_id": "img1",
                "component_id": 1,
                "mean_entropy": 0.12,
                "mean_tta": 0.05,
                "mean_shift": 0.03,
                "is_error_component": False,
            },
            {
                "model_name": "ChangeFormer",
                "dataset": "levir-256",
                "image_id": "img2",
                "component_id": 2,
                "mean_entropy": 0.22,
                "mean_tta": 0.07,
                "mean_shift": 0.04,
                "is_error_component": True,
            },
        ]
    )
    referral_df = pd.DataFrame(
        [
            {
                "model_name": "ChangeFormer",
                "dataset": "levir-256",
                "budget": 0.05,
                "component_f1_gain_upper_bound": 0.08,
            },
            {
                "model_name": "ChangeFormer",
                "dataset": "levir-256",
                "budget": 0.10,
                "component_f1_gain_upper_bound": 0.12,
            },
        ]
    )

    out_path = tmp_path / "figure6_dataset_difficulty_dashboard.pdf"
    plot_dataset_difficulty_dashboard(
        components_df,
        referral_df,
        out_path,
        model_name="ChangeFormer",
    )
    assert out_path.exists()


def test_select_failure_mode_cases_returns_labeled_examples() -> None:
    components_df = pd.DataFrame(
        [
            {
                "dataset": "levir-256",
                "image_id": "img1",
                "component_id": 1,
                "is_error_component": True,
                "score_crs4": 4.0,
                "best_gt_iou": 0.1,
                "error_pixel_ratio": 0.8,
                "area": 20,
            },
            {
                "dataset": "levir-256",
                "image_id": "img2",
                "component_id": 2,
                "is_error_component": False,
                "score_crs4": 3.0,
                "best_gt_iou": 0.9,
                "error_pixel_ratio": 0.1,
                "area": 10,
            },
            {
                "dataset": "levir-256",
                "image_id": "img3",
                "component_id": 3,
                "is_error_component": True,
                "score_crs4": 2.0,
                "best_gt_iou": 0.2,
                "error_pixel_ratio": 0.7,
                "area": 50,
            },
        ]
    )
    out = select_failure_mode_cases(components_df)
    assert set(out["case_type"]) == {"high_risk_error", "high_risk_correct", "large_area_error"}


def test_select_review_case_images_returns_component_pixel_and_miss_cases() -> None:
    referral_per_image_df = pd.DataFrame(
        [
            {"dataset": "levir-256", "image_id": "img1", "method": "component", "budget": 0.05, "error_pixel_recall": 0.8, "f1_gain_upper_bound": 0.02, "iou_gain_upper_bound": 0.01},
            {"dataset": "levir-256", "image_id": "img1", "method": "pixel", "budget": 0.05, "error_pixel_recall": 0.2, "f1_gain_upper_bound": 0.01, "iou_gain_upper_bound": 0.02},
            {"dataset": "levir-256", "image_id": "img2", "method": "component", "budget": 0.05, "error_pixel_recall": 0.1, "f1_gain_upper_bound": 0.01, "iou_gain_upper_bound": 0.01},
            {"dataset": "levir-256", "image_id": "img2", "method": "pixel", "budget": 0.05, "error_pixel_recall": 0.7, "f1_gain_upper_bound": 0.03, "iou_gain_upper_bound": 0.04},
        ]
    )
    missed_change_df = pd.DataFrame(
        [
            {"dataset": "levir-256", "image_id": "img3", "is_missed": True, "gt_area": 120.0, "pred_overlap_ratio": 0.0},
        ]
    )
    out = select_review_case_images(referral_per_image_df, missed_change_df)
    assert set(out["case_type"]) == {"component_better", "pixel_better", "complete_miss"}

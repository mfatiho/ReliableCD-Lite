from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.make_figure2_panel import asset_pdf_path_for_pdf
from scripts.make_figure2_panel import asset_pgf_path_for_pdf
from scripts.make_figure2_panel import asset_png_path_for_pdf
from scripts.make_figure2_panel import copy_pgf_asset_bundle
from scripts.make_figure2_panel import format_risk_rank_label
from scripts.make_figure2_panel import render_panel_outputs


def test_panel_asset_png_path_mirrors_results_figures_subdirs() -> None:
    pdf_path = Path("results/figures/bit/figure2_rs_panel.pdf")

    png_path = asset_png_path_for_pdf(pdf_path)

    assert png_path == Path("docs/assets/figures/bit/png/figure2_rs_panel.png")


def test_panel_asset_pgf_path_mirrors_results_figures_subdirs() -> None:
    pdf_path = Path("results/figures/bit/figure2_rs_panel.pdf")

    pgf_path = asset_pgf_path_for_pdf(pdf_path)

    assert pgf_path == Path("docs/assets/figures/bit/pgf/figure2_rs_panel.pgf")


def test_panel_asset_pdf_path_mirrors_results_figures_subdirs() -> None:
    pdf_path = Path("results/figures/bit/figure2_rs_panel.pdf")

    asset_pdf_path = asset_pdf_path_for_pdf(pdf_path)

    assert asset_pdf_path == Path("docs/assets/figures/bit/pdf/figure2_rs_panel.pdf")


def test_panel_copy_pgf_asset_bundle_copies_main_file_and_companions(tmp_path) -> None:
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()
    src_pgf = src_dir / "panel.pgf"
    companion = src_dir / "panel-img0.png"
    src_pgf.write_text("pgf", encoding="utf-8")
    companion.write_bytes(b"png")

    dst_pgf = copy_pgf_asset_bundle(src_pgf, dst_dir / "panel.pgf")

    assert dst_pgf == dst_dir / "panel.pgf"
    assert dst_pgf.read_text(encoding="utf-8") == "pgf"
    assert (dst_dir / "panel-img0.png").read_bytes() == b"png"


def test_render_panel_outputs_skips_pgf_when_xelatex_is_unavailable(tmp_path) -> None:
    calls: list[Path] = []

    def render_panel(_cases, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(target.suffix, encoding="utf-8")
        calls.append(target)

    expected_png = tmp_path / "asset" / "png" / "figure2_rs_panel.png"
    with (
        patch("scripts.make_figure2_panel.can_render_pgf", return_value=False),
        patch("scripts.make_figure2_panel.render_panel", side_effect=render_panel),
        patch("scripts.make_figure2_panel.asset_png_path_for_pdf", return_value=expected_png),
        patch("scripts.make_figure2_panel.copy2"),
    ):
        render_panel_outputs([], tmp_path / "figure2_rs_panel.pdf")

    assert calls == [
        tmp_path / "pdf" / "figure2_rs_panel.pdf",
        expected_png,
    ]
    assert not (tmp_path / "pgf" / "figure2_rs_panel.pgf").exists()


def test_format_risk_rank_label_uses_descending_score_rank() -> None:
    dataset_scores = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)

    label = format_risk_rank_label(3.0, dataset_scores)

    assert label == "CRS rank: Top 50%"


def test_format_risk_rank_label_returns_none_without_reference_scores() -> None:
    assert format_risk_rank_label(3.0, None) is None

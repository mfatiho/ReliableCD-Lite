from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts.make_figures import (
    _asset_pdf_path_for_pdf,
    _asset_pgf_path_for_pdf,
    _asset_png_path_for_pdf,
    _copy_pgf_asset_bundle,
    _export_png_asset,
    _pgf_path_for_pdf,
    _render_figure_with_png,
)


def test_asset_png_path_mirrors_results_figures_subdirs() -> None:
    pdf_path = Path("results/figures/bit/figure4_aurc_summary.pdf")
    out_dir = Path("results/figures/bit")

    png_path = _asset_png_path_for_pdf(pdf_path, out_dir)

    assert png_path == Path("docs/assets/figures/bit/png/figure4_aurc_summary.png")


def test_asset_png_path_handles_root_results_figures_dir() -> None:
    pdf_path = Path("results/figures/figure1_framework.pdf")
    out_dir = Path("results/figures")

    png_path = _asset_png_path_for_pdf(pdf_path, out_dir)

    assert png_path == Path("docs/assets/figures/png/figure1_framework.png")


def test_asset_pgf_path_mirrors_results_figures_subdirs() -> None:
    pdf_path = Path("results/figures/bit/figure4_aurc_summary.pdf")
    out_dir = Path("results/figures/bit")

    pgf_path = _asset_pgf_path_for_pdf(pdf_path, out_dir)

    assert pgf_path == Path("docs/assets/figures/bit/pgf/figure4_aurc_summary.pgf")


def test_asset_pdf_path_mirrors_results_figures_subdirs() -> None:
    pdf_path = Path("results/figures/bit/figure4_aurc_summary.pdf")
    out_dir = Path("results/figures/bit")

    asset_pdf_path = _asset_pdf_path_for_pdf(pdf_path, out_dir)

    assert asset_pdf_path == Path("docs/assets/figures/bit/pdf/figure4_aurc_summary.pdf")


def test_export_png_asset_skips_when_pdftoppm_is_unavailable(tmp_path) -> None:
    pdf_path = tmp_path / "figure.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with patch("scripts.make_figures.which", return_value=None), patch("scripts.make_figures.subprocess.run") as run:
        png_path = _export_png_asset(pdf_path, tmp_path)

    run.assert_not_called()
    assert png_path == Path("docs/assets/figures") / tmp_path.name / "png" / "figure.png"


def test_pgf_path_for_pdf_uses_same_stem() -> None:
    pdf_path = Path("results/figures/bit/figure4_aurc_summary.pdf")

    pgf_path = _pgf_path_for_pdf(pdf_path)

    assert pgf_path == Path("results/figures/bit/pgf/figure4_aurc_summary.pgf")


def test_copy_pgf_asset_bundle_copies_main_file_and_companions(tmp_path) -> None:
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()
    src_pgf = src_dir / "figure.pgf"
    companion = src_dir / "figure-img0.png"
    src_pgf.write_text("pgf", encoding="utf-8")
    companion.write_bytes(b"png")

    dst_pgf = _copy_pgf_asset_bundle(src_pgf, dst_dir / "figure.pgf")

    assert dst_pgf == dst_dir / "figure.pgf"
    assert dst_pgf.read_text(encoding="utf-8") == "pgf"
    assert (dst_dir / "figure-img0.png").read_bytes() == b"png"


def test_render_figure_with_png_renders_pdf_and_png_without_converter(tmp_path) -> None:
    calls: list[Path] = []

    def render(path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(target.suffix, encoding="utf-8")
        if target.suffix == ".pgf":
            (target.parent / f"{target.stem}-img0.png").write_bytes(b"png")
        calls.append(target)
        return target

    pdf_path = tmp_path / "figure.pdf"

    with patch("scripts.make_figures.which", return_value="xelatex"):
        rendered_pdf = _render_figure_with_png(render, pdf_path, tmp_path)

    expected_pdf = tmp_path / "pdf" / "figure.pdf"
    expected_asset_pdf = Path("docs/assets/figures") / tmp_path.name / "pdf" / "figure.pdf"
    expected_png = Path("docs/assets/figures") / tmp_path.name / "png" / "figure.png"
    expected_pgf = tmp_path / "pgf" / "figure.pgf"
    expected_asset_pgf = Path("docs/assets/figures") / tmp_path.name / "pgf" / "figure.pgf"
    assert rendered_pdf == expected_pdf
    assert calls == [expected_pdf, expected_pgf, expected_png]
    assert expected_pdf.exists()
    assert expected_asset_pdf.exists()
    assert expected_pgf.exists()
    assert expected_png.exists()
    assert expected_asset_pgf.exists()
    assert (expected_asset_pgf.parent / "figure-img0.png").exists()


def test_render_figure_with_png_skips_pgf_when_xelatex_is_unavailable(tmp_path) -> None:
    calls: list[Path] = []

    def render(path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(target.suffix, encoding="utf-8")
        calls.append(target)
        return target

    pdf_path = tmp_path / "figure.pdf"
    expected_asset_pdf = tmp_path / "asset" / "pdf" / "figure.pdf"
    expected_png = tmp_path / "asset" / "png" / "figure.png"

    with (
        patch("scripts.make_figures.which", return_value=None),
        patch("scripts.make_figures._asset_pdf_path_for_pdf", return_value=expected_asset_pdf),
        patch("scripts.make_figures._asset_png_path_for_pdf", return_value=expected_png),
    ):
        rendered_pdf = _render_figure_with_png(render, pdf_path, tmp_path)

    expected_pdf = tmp_path / "pdf" / "figure.pdf"
    assert rendered_pdf == expected_pdf
    assert calls == [expected_pdf, expected_png]
    assert not (tmp_path / "pgf" / "figure.pgf").exists()

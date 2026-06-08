from __future__ import annotations

from argparse import Namespace

from scripts import run_pipeline


def _args(model_scope: str) -> Namespace:
    return Namespace(
        model_scope=model_scope,
        bit_checkpoint="checkpoints/bit/BIT_LEVIR_best_ckpt.pt",
        changeformer_checkpoint="checkpoints/changeformer/ChangeFormer_LEVIR.pth",
        out_dir="results",
        bit_device="auto",
        cf_device="auto",
        batch_size=4,
        num_workers=4,
        n_boot=1000,
        n_jobs=1,
        bit_normalizer_components=None,
        runtime_mode="smoke",
    )


def _command_texts(cmds: list[list[str]]) -> list[str]:
    return [" ".join(str(part).replace("\\", "/") for part in cmd) for cmd in cmds]


def test_changeformer_model_scope_excludes_bit_commands() -> None:
    args = _args("changeformer")

    for builder in [
        run_pipeline._cmds_baseline,
        run_pipeline._cmds_uq,
        run_pipeline._cmds_calibration,
        run_pipeline._cmds_components,
        run_pipeline._cmds_scoring,
        run_pipeline._cmds_missed_change,
        run_pipeline._cmds_referral,
        run_pipeline._cmds_sensitivity,
        run_pipeline._cmds_runtime,
        run_pipeline._cmds_threshold_sensitivity,
    ]:
        commands = _command_texts(builder(args))
        assert commands
        assert all("ChangeFormer" in command or "changeformer" in command for command in commands)
        assert all(" BIT " not in command and "bit_" not in command for command in commands)

    assert run_pipeline._cmds_figures(args) == []


def test_changeformer_model_scope_skips_bit_only_referral_ablation() -> None:
    assert run_pipeline._cmds_referral_ablation(_args("changeformer")) == []


def test_figures_step_includes_bit_rs_panel_after_make_figures() -> None:
    commands = run_pipeline._cmds_figures(_args("both"))

    command_texts = _command_texts(commands)

    assert any("scripts/make_figures.py" in command and "results/figures/bit" in command for command in command_texts)
    assert any("scripts/make_figure2_panel.py" in command and "figure2_rs_panel.pdf" in command for command in command_texts)
    make_figures_idx = next(i for i, command in enumerate(command_texts) if "scripts/make_figures.py" in command and "results/figures/bit" in command)
    figure2_panel_idx = next(i for i, command in enumerate(command_texts) if "scripts/make_figure2_panel.py" in command)
    assert make_figures_idx < figure2_panel_idx


def test_figures_step_is_not_parallelized_because_panel_depends_on_candidates() -> None:
    assert run_pipeline._step_allows_parallel("baseline") is True
    assert run_pipeline._step_allows_parallel("figures") is False


def test_scoring_step_is_not_parallelized_because_it_reuses_source_normalizer() -> None:
    assert run_pipeline._step_allows_parallel("scoring") is False


def test_normalizer_validation_step_is_not_parallelized_because_commands_are_dependent() -> None:
    assert run_pipeline._step_allows_parallel("normalizer_validation") is False


def test_normalizer_validation_builds_levir_val_component_reference() -> None:
    commands = _command_texts(run_pipeline._cmds_normalizer_validation(_args("bit")))

    assert len(commands) == 3
    assert "scripts/eval_bit_cross_domain.py" in commands[0]
    assert "--split val" in commands[0]
    assert "results/validation/baseline" in commands[0]
    assert "scripts/run_uq.py" in commands[1]
    assert "--split val" in commands[1]
    assert "results/validation/uq" in commands[1]
    assert "scripts/extract_components.py" in commands[2]
    assert "results/validation/components/bit_levir-256_components.parquet" in commands[2]
    assert "--dataset levir-256-val" in commands[2]


def test_bit_scoring_reuses_single_levir_validation_reference_normalizer() -> None:
    commands = _command_texts(run_pipeline._cmds_scoring(_args("bit")))

    assert len(commands) == 3
    assert all("--fit-normalizer-from results/validation/components/bit_levir-256_components.parquet" in command for command in commands)

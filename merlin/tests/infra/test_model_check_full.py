"""check --mode full: step counts, the model gates of contract §4.4, and the judged rows it compares."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.model_check import chain as CH
from merlin.model_check import cli
from merlin.model_check import full as F
from merlin.model_check import full_build as FB
from merlin.model_check.models import ModelsFileError, load


def _models():
    return load(repo_root() / "examples" / "gemmini" / "numerics" / "models.yaml")


def _causal():
    return next(m for m in _models().models.values() if m.chain["kind"] == "causal")


def test_decode_steps_caps_only_a_causal_chains_recurrent_programs():
    model = _causal()
    full = F.step_counts(model, None)
    capped = F.step_counts(model, 4)
    recurrent = [p.name for p in model.programs if p.steps > 1]
    assert recurrent and all(capped[name] == 4 for name in recurrent)
    assert all(capped[p.name] == full[p.name] for p in model.programs if p.steps == 1)
    flow = next(m for m in _models().models.values() if m.chain["kind"] == "flow")
    with pytest.raises(ModelsFileError):
        F.step_counts(flow, 4)
    with pytest.raises(ModelsFileError):
        F.step_counts(model, 0)


def test_every_model_declares_how_its_full_mode_image_is_built():
    models = _models()
    for model in models.models.values():
        settings = FB.BuildSettings.of(models, model, Path("/host"))
        assert settings.arena_mb > 0 and settings.board and settings.dedup_weights
        assert settings.board_catalog.is_file()


def test_full_mode_and_binding_select_the_fire_aligned_16gib_board():
    models = _models()
    settings = FB.BuildSettings.of(models, models.model("resnet50"), Path("/host"))
    binding = yaml.safe_load((repo_root() / "examples/gemmini/numerics/binding.yaml").read_text())
    catalog = yaml.safe_load(settings.board_catalog.read_text())["boards"]
    assert settings.board == binding["board"] == "firesim_gemmini_rocket_u250_30mhz"
    assert catalog[settings.board]["dram_bytes"] == "16 GiB"
    assert settings.host_dts.name == "firechip.chip.FireSim.FireSimGemminiRocketConfig.dts"
    assert binding["elaboration"]["dts"].endswith("GemminiGsimFireSimAlignedConfig.dts")
    assert binding["fpga"]["dts"].endswith(settings.host_dts.name)


def test_the_check_parser_takes_an_input_filter():
    args = cli.build_parser().parse_args(
        [
            "check",
            "--submission",
            "S",
            "--model",
            "m",
            "--split",
            "visible",
            "--mode",
            "full",
            "--input",
            "v00",
            "--input",
            "v03",
            "--decode-steps",
            "4",
            "--out",
            "o.json",
        ]
    )
    assert args.inputs == ["v00", "v03"] and args.decode_steps == 4


def _ratio(value, reference):
    return {"rel_rmse": value, "worst": value, "top1_agree": 1.0, "top5_agree": 1.0}, {
        "rel_rmse": reference,
        "worst": reference,
        "top1_agree": 1.0,
        "top5_agree": 1.0,
    }


def test_the_logits_gate_is_the_ratio_rule_plus_topk_agreement():
    ours, ref = _ratio(0.10, 0.10)
    assert F.model_gate("logits", ours, ref)["pass"]
    ours, ref = _ratio(0.12, 0.10)
    gate = F.model_gate("logits", ours, ref)
    assert not gate["pass"] and not gate["checks"]["rel_rmse"]["pass"]
    ours, ref = _ratio(0.10, 0.10)
    ours["top1_agree"] = 0.5
    assert not F.model_gate("logits", ours, ref)["checks"]["top1_agree"]["pass"]


def test_the_distribution_gate_holds_the_worst_kl_and_top1_agreement():
    ref = {"kl_worst": 0.2, "top1_agree": 0.9}
    assert F.model_gate("next_token_distribution", {"kl_worst": 0.21, "top1_agree": 0.9}, ref)["pass"]
    assert not F.model_gate("next_token_distribution", {"kl_worst": 0.23, "top1_agree": 0.9}, ref)["pass"]
    assert not F.model_gate("next_token_distribution", {"kl_worst": 0.1, "top1_agree": 0.8}, ref)["pass"]
    assert F.model_gate(
        "next_token_distribution", {"kl_worst": 1e-7, "top1_agree": 1.0}, {"kl_worst": 0.0, "top1_agree": 1.0}
    )["pass"], "the floor admits a reference with no divergence"


def test_the_per_channel_gate_is_per_channel():
    ref = {"per_channel": {"0": {"rel_rmse": 0.1, "worst": 0.1}, "1": {"rel_rmse": 0.2, "worst": 0.2}}}
    ours = {"per_channel": {"0": {"rel_rmse": 0.1, "worst": 0.1}, "1": {"rel_rmse": 0.25, "worst": 0.2}}}
    gate = F.model_gate("per_channel", ours, ref)
    assert not gate["pass"] and [k for k, c in gate["checks"].items() if not c["pass"]] == ["1.rel_rmse"]


def test_a_partial_decode_is_judged_on_as_many_fp_rows_as_it_ran():
    model = _causal()
    vocab = 7
    rng = np.random.default_rng(0)
    run = CH.ChainRun()
    judged = model.judged["outputs"]
    for ref in judged:
        run.steps.setdefault(ref["program"], [])
    for ref, steps in zip(judged, (1, 3), strict=True):
        for _ in range(steps):
            outputs = [None] * (int(ref["output"]) + 1)
            outputs[int(ref["output"])] = rng.normal(size=(1, vocab)).astype(np.float32)
            run.steps[ref["program"]].append(outputs)
    fp = [rng.normal(size=(33, vocab)).astype(np.float32)]
    metrics = F.judged_metrics(model, run, fp)
    assert metrics["distributions"] == 4


def test_step_outputs_compare_integers_exactly_and_record_float_distance():
    row = F._compare(np.array([1, 2, 3], np.int64), np.array([1, 2, 4], np.int64), "i64")
    assert not row["exact"] and row["mismatched"] == 1
    row = F._compare(np.array([1.0, 2.0], np.float32), np.array([1.0, 2.0], np.float32), "f32")
    assert row["exact"] and row["rel_rmse_vs_int8"] == 0.0
    row = F._compare(np.array([True, False]), np.array([True, True]), "i1")
    assert row["mismatched"] == 1


def test_npz_arrays_are_read_in_numeric_key_order(tmp_path):
    np.savez(tmp_path / "a.npz", **{f"out{i}": np.array([i]) for i in range(12)})
    assert [int(a[0]) for a in F._ordered(tmp_path / "a.npz")] == list(range(12))


def test_inputs_run_concurrently_by_default_but_never_more_than_there_are():
    assert F.default_jobs(1) == 1
    assert 1 <= F.default_jobs(64) <= 64


def test_the_differential_holds_every_output_to_every_bit():
    ours = [np.array([1.0, -0.0], np.float32), np.array([3], np.int64), np.array([True])]
    assert F._same(ours, [a.copy() for a in ours]) == []
    flipped = [np.array([1.0, 0.0], np.float32), np.array([3], np.int64), np.array([True])]
    assert F._same(ours, flipped) == [0], "a sign bit is a difference"
    assert F._same(ours, [ours[0], np.array([3], np.int32), ours[2]]) == [1], "so is a dtype"


def test_images_are_built_below_the_out_root_whatever_the_work_directory(tmp_path, monkeypatch):
    """A harness passes ``--work`` outside this checkout; a whole-model build may only write below out/."""
    out = tmp_path / "out"
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(out))
    inside = out / "build" / "model_check" / "w"
    assert F.build_root(inside, "m") == inside
    elsewhere = F.build_root(tmp_path / "run" / "check.work", "m")
    assert elsewhere.is_relative_to(out) and elsewhere.is_dir() and elsewhere.name.startswith("m-")
    assert F.build_root(tmp_path / "run" / "check.work", "m") != elsewhere, "each check its own directory"


def test_full_mode_takes_the_differential_and_refuses_a_program():
    import argparse

    F.validate(argparse.Namespace(differential=True, program=None))
    with pytest.raises(ModelsFileError):
        F.validate(argparse.Namespace(differential=False, program="decode"))

"""Certification cannot publish prefix-gated or unpaired point wins."""

import importlib.util
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


SCRIPT = repo_root() / "build_tools" / "scripts" / "k1_int8_fair_compare.py"


def _module():
    spec = importlib.util.spec_from_file_location("k1_int8_fair_compare_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dump_cap_parser_has_an_explicit_full_output_mode():
    module = _module()
    assert module._dump_cap("full") is None
    assert module._dump_cap("none") is None
    assert module._dump_cap("4096") == 4096
    with pytest.raises(Exception):
        module._dump_cap("0")


def test_certification_threads_full_output_and_uses_paired_confidence():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "dump_cap=dump_cap" in source
    assert "min_coverage=1.0 if shared_bar is not None" in source
    assert "shared_accuracy_verdict" in source
    assert "paired_speedup_confidence" in source
    assert '"--min-pairs", type=int, default=7' in source
    assert '"--max-pairs", type=int, default=15' in source
    assert '"--confidence-margin", type=float, default=1.05' in source
    assert "pair_index % 2" in source
    assert 'reference.get("ok")' in source
    assert '"look_adjustment": "Bonferroni"' in source


def test_shared_bar_not_the_stricter_internal_tier_controls_certification(monkeypatch, tmp_path):
    module = _module()
    gate = {
        "ok": False,
        "cos": 0.8,
        "rel": 0.2,
        "fp32_cos": 0.995,
        "fp32_rel": 0.04,
        "comparison_complete": True,
        "tiers": ["fp32", "w8a8"],
        "tier_ok": None,
    }
    monkeypatch.setattr(module, "_conditions", lambda: {})
    monkeypatch.setattr(module.zm, "_gate", lambda *_args, **_kwargs: gate)
    monkeypatch.setattr(
        module.k1,
        "run_on_k1",
        lambda *_args, **_kwargs: {"prefix": object(), "metrics": {"wall_ns": 100}},
    )
    pkg = type("Package", (), {"run_id": "fixture"})()
    result = module.ours_arm(
        tmp_path, pkg, {}, tmp_path / "work", n=1, warmup=1, iters=3,
        dump_cap=None,
        shared_bar={"cos_threshold": 0.99, "rel_threshold": 0.05, "basis": "fixture"},
    )
    assert result["ok"] is True
    assert result["median_wall_ns"] == 100
    assert result["shared_accuracy"]["passes"] is True


def test_prepared_certification_executes_the_same_binary_without_rebuilding(monkeypatch, tmp_path):
    module = _module()
    gate = {
        "fp32_cos": 0.995, "fp32_rel": 0.04, "comparison_complete": True,
        "cos": 0.995, "rel": 0.04, "tiers": ["fp32"], "tier_ok": "fp32",
    }
    calls = []
    monkeypatch.setattr(module, "_conditions", lambda: {})
    monkeypatch.setattr(module.zm, "_gate", lambda *_args, **_kwargs: gate)
    monkeypatch.setattr(
        module.k1, "run_on_k1",
        lambda *_args, **_kwargs: pytest.fail("a prepared campaign must not rebuild"),
    )

    def run_binary(*args, **kwargs):
        calls.append((args, kwargs))
        return {"prefix": object(), "metrics": {"wall_ns": 100}}

    monkeypatch.setattr(module.k1, "run_binary_on_k1", run_binary)
    pkg = type("Package", (), {"run_id": "fixture"})()
    prepared = {"binary": str(tmp_path / "same.elf"), "work": str(tmp_path / "build"),
                "multi_program": False}
    result = module.ours_arm(
        tmp_path, pkg, {}, tmp_path / "pairs", n=2, warmup=1, iters=3,
        dump_cap=None, prepared=prepared,
        shared_bar={"cos_threshold": 0.99, "rel_threshold": 0.05, "basis": "fixture"},
    )

    assert result["ok"] is True
    assert len(calls) == 2
    assert all(call[0][3] == Path(prepared["binary"]) for call in calls)
    assert all(call[1]["capture_full_output"] is True for call in calls)
    assert all(call[1]["env"] == {"MERLIN_ITERS": "3", "MERLIN_WARMUP": "1"}
               for call in calls)


def test_reference_warm_slope_reuses_the_export(monkeypatch):
    module = _module()
    calls = []

    class Result:
        e2e_wall_ns = 100
        cos = 1.0
        rel = 0.0
        load_ns = 10
        accuracy_reference = "capture_golden_fp32"
        quant_recipe = "pt2e_qd8"
        bundle_id = "fixture"
        gap_reason = ""

        @staticmethod
        def status():
            return "pass"

    def run_model(*args, **kwargs):
        calls.append(kwargs)
        return Result()

    monkeypatch.setattr(module, "_conditions", lambda: {})
    monkeypatch.setattr(module.et, "run_model", run_model)
    result = module.et_arm("fixture", qd8=True, n_lo=1, n_hi=3, cpu_threads=1)

    assert result["ok"] is True
    assert len(calls) == 2
    assert all(call["reuse_export"] is True for call in calls)


def _verdict_fixture(*, ours_bundle="fixture", ref_bundle="fixture",
                     ref_recipe="pt2e_qd8", ref_accuracy="capture_golden_fp32"):
    module = _module()
    ours = {
        "min_wall_ns": 90,
        "gate": {"fp32_cos": 0.9999, "fp32_rel": 0.001},
        "accuracy_reference_by_tier": {"fp32": "capture_golden_fp32"},
    }
    arm = {
        "warm_ns": 100,
        "runs": [{
            "bundle_id": ref_bundle,
            "quant_recipe": ref_recipe,
            "accuracy_reference": ref_accuracy,
            "cos": 0.9998,
            "rel": 0.002,
            "load_ns": 5,
        }],
    }
    return module.verdict(ours, arm, ours_bundle)


def test_deployment_comparison_refuses_a_bundle_mismatch_without_a_nested_number():
    result = _verdict_fixture(ours_bundle="model_layout_rewrite", ref_bundle="other_model")
    assert result["status"] == "not_comparable"
    assert result["deployment_comparison"]["status"] == "not_comparable"
    assert "speedup" not in result["deployment_comparison"]


def test_deployment_comparison_allows_distinct_recipes_only_at_shared_fp32_quality():
    result = _verdict_fixture()
    assert result["status"] == "not_comparable"  # arithmetic recipes are distinct
    assert "quant" in result["reason"].lower()
    deployment = result["deployment_comparison"]
    assert deployment["status"] == "measured"
    assert deployment["beats_executorch"] is True
    assert deployment["speedup"] == pytest.approx(100 / 90)


def test_deployment_comparison_refuses_different_accuracy_references():
    result = _verdict_fixture(ref_accuracy="some_other_golden")
    assert result["status"] == "not_comparable"
    assert result["deployment_comparison"]["status"] == "not_comparable"
    assert "speedup" not in result["deployment_comparison"]

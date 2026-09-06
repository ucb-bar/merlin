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

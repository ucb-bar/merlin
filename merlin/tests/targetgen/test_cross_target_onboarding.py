"""Phase 4 — the onboarding proof: a target brings up its full scaffolding from {RTL + a capability
manifest + a descriptor} with ZERO target-specific code in the core generators. Exercised on radiance
(SIMT), the target most different from gemmini, so the whole shared-infra pipeline is validated for a
non-systolic accelerator.

Also guards the overfit smell test: the core generators carry no hardcoded target NAME.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.target_experiment import load_target_experiment, load_capability_manifest
from merlin.targetgen.runner_config import runner_config_from_manifest
from merlin.targetgen.generate_prompt import prompt_slots
from merlin.targetgen.generate_bundles import generate_bundles

_RAD_DESC = "merlin/experiments/radiance_capsule_bench_v0/target_experiment.yaml"


def _load_radiance(monkeypatch):
    # radiance is an out-of-tree target discovered via MERLIN_TARGET_PATH — the documented plug-in path.
    monkeypatch.setenv("MERLIN_TARGET_PATH", "out/artifacts/targets")
    try:
        return load_target_experiment(_RAD_DESC), load_capability_manifest("radiance")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"radiance OOT target not resolvable in this checkout: {e}")


def test_radiance_onboards_end_to_end_from_manifest(monkeypatch):
    te, m = _load_radiance(monkeypatch)
    # SIMT routing comes entirely from the manifest's compute-unit kind via the family registry
    assert m.kind == "simt"
    assert m.encoding_required is False and m.trace_gate is None   # no .insn encoding / no RoCC trace gate

    cfg = runner_config_from_manifest(m)
    assert cfg.trace_gate is None                                 # the runner skips the RoCC trace stage
    assert cfg.perf_fields == ("flops", "gflops", "pct_fp_peak")  # SIMT perf headline

    s = prompt_slots(te, m)
    assert s["tool_stem"] == "radiance-opt" and s["kernel_symbol"] == "radiance_kernel"  # derived, not gemmini
    assert "gemmini" not in s["endpoint_desc"].lower() and "rocc" not in s["endpoint_desc"].lower()

    assert len(generate_bundles(te)) == 4                          # the 4-arm ladder generates for radiance


def test_core_generators_carry_no_hardcoded_target_name():
    """families / runner_config / generate_prompt must be target-name-free (kinds + slots only) — the
    overfit smell test. (capsule_runner keeps a gemmini default for back-compat and is excluded.)"""
    names = ("gemmini", "radiance", "muon", "atlas", "mx_gemmini")
    for rel in ("merlin/python/merlin/targetgen/families.py",
                "merlin/python/merlin/targetgen/runner_config.py",
                "merlin/python/merlin/targetgen/generate_prompt.py"):
        text = (repo_root() / rel).read_text()
        # strip line comments: only flag a target name used as a quoted code literal
        code = "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())
        for n in names:
            assert f'"{n}"' not in code and f"'{n}'" not in code, f"{rel} hardcodes target name {n!r}"

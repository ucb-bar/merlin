"""Gated end-to-end proof of the lite debugger against a real external_backend target + its functional
model. Skips cleanly when the model venv / mlc dir / an external_backend descriptor is absent (CI without
the heavy deps), so it never blocks the hermetic suite. When it runs it asserts the three contracts that
make the debugger safe + useful: the OUTPUT region is refused, an INPUT window comes back populated (the
canonical preload landed), and `run_to=N` stops early with a value-free on-chip populated map.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir


def _atlas_like_descriptor() -> Path | None:
    """Any capsule-bench target descriptor whose backend is external_backend (self-hosted ISA). Returns the
    first one found under the experiments tree, or None (skip) — no target name is hardcoded."""
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin.targetgen import capsule_runner as CR
    root = merlin_dir() / "experiments" / "capsule_bench" / "targets"
    if not root.is_dir():
        return None
    for desc in sorted(root.glob("*/target_experiment.yaml")):
        try:
            te = load_target_experiment(desc)
            if CR._endpoint_of(te.target)[0] == "external_backend":
                return desc
        except Exception:  # noqa: BLE001 — unresolvable descriptor: keep looking
            continue
    return None


def _skip_unless_ready():
    from merlin.targetgen.rtl import mlc_bridge
    if mlc_bridge.mlc_dir() is None:
        pytest.skip("mlc dir unavailable (MERLIN_MLC_DIR) — no functional program runner")
    desc = _atlas_like_descriptor()
    if desc is None:
        pytest.skip("no external_backend target descriptor present")
    return desc


def test_debug_e2e_refuses_output_and_maps_state():
    desc = _skip_unless_ready()
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin.targetgen.contract.materialize import public_capsules_for
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import program_oracle as PO

    te = load_target_experiment(desc)
    target = te.target
    _, model_ext = CR._endpoint_of(target)
    caps = sorted((public_capsules_for(te)).glob("*/capsule.yaml"))
    if not caps:
        pytest.skip("no public capsules materialized for this target")
    cap_dir = caps[0].parent

    try:
        cb = PO.build_debug_cb(target, cap_dir)
    except PO.OracleUnavailable as e:
        pytest.skip(f"cb build unavailable: {e}")

    inp = next((t for t in cb["tensors"].values() if t["role"] in ("input", "weight")), None)
    spec = PO._resolve_out_spec(target, cb, {})
    out_base, out_n = int(spec["base"]), PO._out_nbytes(spec)
    if inp is None:
        pytest.skip("capsule has no input/weight tensor to dump")

    # a trivial kernel is enough — we assert plumbing + redaction, not numeric correctness
    with tempfile.TemporaryDirectory() as td:
        ks = Path(td) / "kernel.S"
        ks.write_text(".text\n.word 0x00000000\n")
        try:
            r = PO.run_program_debug(
                target, model_ext=model_ext, cb=cb, kernel_s=ks,
                dump_regions=[[inp["base"], 32], [out_base, min(32, out_n)]],   # input (allowed) + output (refused)
                run_to=4, state_summary=True, workdir=Path(td), timeout=300)
        except PO.OracleUnavailable as e:
            pytest.skip(f"functional runner unavailable: {e}")

    # 1) the OUTPUT region is never dumped, and is explicitly rejected
    assert all(reg["base"] != out_base for reg in r["regions"])
    assert any(reg["base"] == out_base for reg in r["rejected_regions"])

    # 2) the INPUT window came back (the canonical preload staged into DRAM)
    got = [reg for reg in r["regions"] if reg["base"] == inp["base"]]
    assert got and got[0]["returned_bytes"] == 32

    # 3) the stop is one of the three defined reasons, bounded by run_to, with a value-free on-chip map.
    # (A trivial kernel may halt ("finished") before instruction 4; a real one stops at "run_to". Both are
    # correct — is_finished() is checked before the run_to bound.)
    assert r["halt_reason"] in ("run_to", "finished", "max_cycles")
    assert r["instr_count"] <= 4
    oc = r["on_chip"]
    assert oc is not None and set(("vmem_populated", "mrf_populated", "acc_populated")) <= set(oc)
    assert all(isinstance(b, bool) for b in oc["mrf_populated"])   # booleans only — no values leak

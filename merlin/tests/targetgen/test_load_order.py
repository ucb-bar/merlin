"""The opt-in memory-order tier: a static trigger, a seed sweep, and what a positive does to the certificate.

Four properties, each with the mutation that would break it:

* the static check fires on two inbound transfers into overlapping accumulator rows with no barrier
  between them, and on anything it cannot resolve, and on nothing else (a fence, disjoint rows, a
  scratchpad destination, an outbound transfer, a command that stays on chip);
* the tier is OFF unless asked for: unset, a grade through the real ladder is byte-identical to one
  from before the tier existed, and the sweep is never invoked;
* switched on, a clear trace is recorded as clear and never swept, and a target without the facts is
  recorded as not applicable rather than checked against guessed ones;
* an order-sensitive sweep fails the certificate, names the reproducing seeds, and the promotion ledger
  records that as a failed certificate -- the path whose silence has hidden five defects before.
"""

from __future__ import annotations

import json
import sys

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import load_order as LO

# A synthetic address layout. Shape only: the tier reads these from the target's derived constants.
ACC, ACCUM, FULL = 1 << 31, 1 << 30, 1 << 29
FACTS = LO.OrderFacts(ACC, ACCUM, ACC | ACCUM | FULL, 16)
UNKNOWN = {"raw": None, "kind": "unknown"}


def _ld(i, cls, addr=None, rows=16, cols=16):
    """An inbound transfer: an off-chip operand and, when resolved, the local address it defines."""
    dec = {"dram": UNKNOWN} if addr is None else {"dram": UNKNOWN, "spad_addr": addr, "rows": rows, "cols": cols}
    return {"index": i, "class": cls, "decoded": dec}


def _trace(*ins):
    return {"instructions": list(ins)}


OVERWRITE_THEN_ACCUMULATE = _trace(
    _ld(0, "LD_A", ACC | 32),
    _ld(1, "LD_B", ACC | ACCUM | 32),
    {"index": 2, "class": "ST_X", "decoded": {"dram": UNKNOWN, "acc_addr": ACC | 32}},
)


@pytest.fixture
def switched_on(monkeypatch):
    monkeypatch.setenv(LO.SEEDS_ENV, "8")


def test_overwrite_then_accumulate_into_the_same_rows_is_a_hazard():
    got = LO.static_check(OVERWRITE_THEN_ACCUMULATE, FACTS)
    assert got["verdict"] == "hazard" and got["hazard_pairs"] == [
        {"first": 0, "second": 1, "classes": ["LD_A", "LD_B"]}
    ]
    assert got["n_unresolved_pairs"] == 0, "a resolved OUTBOUND transfer writes no accumulator row"


def test_a_fence_between_them_clears_it():
    t = _trace(_ld(0, "LD_A", ACC | 32), {"index": 1, "class": "FENCE"}, _ld(2, "LD_B", ACC | ACCUM | 32))
    assert LO.static_check(t, FACTS)["verdict"] == "clear"


def test_disjoint_rows_and_scratchpad_destinations_are_clear():
    assert (
        LO.static_check(_trace(_ld(0, "LD_A", ACC | 0), _ld(1, "LD_B", ACC | ACCUM | 16)), FACTS)["verdict"] == "clear"
    )
    assert LO.static_check(_trace(_ld(0, "LD_A", 32), _ld(1, "LD_B", 32)), FACTS)["verdict"] == "clear"


def test_a_partial_overlap_is_a_hazard():
    t = _trace(_ld(0, "LD_A", ACC | 0, rows=16), _ld(1, "LD_B", ACC | ACCUM | 8, rows=4))
    assert LO.static_check(t, FACTS)["verdict"] == "hazard"


def test_a_multi_block_transfer_or_an_unknown_block_width_is_unbounded():
    """A transfer wider than one block strides by a value configured elsewhere, so its rows are not known."""
    wide = _trace(_ld(0, "LD_A", ACC | 0, cols=48), _ld(1, "LD_B", ACC | ACCUM | 64))
    assert LO.static_check(wide, FACTS)["verdict"] == "unresolved"
    blind = LO.OrderFacts(ACC, ACCUM, ACC | ACCUM | FULL, None)
    assert LO.static_check(_trace(_ld(0, "LD_A", ACC | 0), _ld(1, "LD_B", ACC | ACCUM | 64)), blind)["verdict"] == (
        "unresolved"
    )


def test_an_unresolved_transfer_or_an_opaque_command_is_never_clear():
    assert LO.static_check(_trace(_ld(0, "LD_A", ACC), _ld(1, "LD_B")), FACTS)["verdict"] == "unresolved"
    # a command whose memory effects the check does not model (a hardware loop unroller) may issue its
    # own loads: two of them with no barrier is unresolved, not clear
    t = _trace({"index": 0, "class": "LOOP_X"}, {"index": 1, "class": "LOOP_X"})
    assert LO.static_check(t, FACTS)["verdict"] == "unresolved"


def test_a_command_that_stays_on_chip_neither_triggers_nor_clears():
    """Configuration and the compute/preload pair issue no memory request; a FENCE is what clears."""
    on_chip = [{"index": 1, "class": c, "decoded": {"c_addr": ACC | 32}} for c in ("PRELOAD", "COMPUTE_PRELOADED")]
    t = _trace(_ld(0, "LD_A", ACC | 32), *on_chip, _ld(3, "LD_B", ACC | ACCUM | 32))
    got = LO.static_check(t, FACTS)
    assert got["verdict"] == "hazard" and got["hazard_pairs"][0]["second"] == 3
    assert LO.static_check(_trace(*on_chip), FACTS)["verdict"] == "clear"


def test_a_single_load_or_an_empty_trace_is_clear():
    assert LO.static_check(_trace(_ld(0, "LD_A")), FACTS)["verdict"] == "clear"
    assert LO.static_check(None, FACTS)["verdict"] == "clear"


def _fake_sweep(fail_seeds):
    calls = []

    def sweep(engine, elf, seeds, judge, **kw):
        calls.append({"seeds": list(seeds), **kw})
        runs = [{"seed": None, "rc": 0, "wall_s": 1.0, "outcome": "pass"}] if kw.get("include_in_order", True) else []
        runs += [{"seed": s, "rc": 0, "wall_s": 1.0, "outcome": "fail" if s in fail_seeds else "pass"} for s in seeds]
        return {"runs": runs, "elapsed_s": 1.0, "emulator_sha256": "e", "elf_sha256": "f"}

    return sweep, calls


def _check(tmp_path, trace, sweep, **kw):
    elf = tmp_path / "p.elf"
    elf.write_bytes(b"x")
    kw.setdefault("facts", FACTS)
    kw.setdefault("engine", (tmp_path / "emu", "emu"))
    return LO.run_order_check(target="t", trace=trace, elf=elf, judge=lambda rc, t: "pass", sweep=sweep, **kw)


def test_the_tier_is_off_unless_asked_for(tmp_path, monkeypatch):
    monkeypatch.delenv(LO.SEEDS_ENV, raising=False)
    sweep, calls = _fake_sweep({1})
    assert _check(tmp_path, OVERWRITE_THEN_ACCUMULATE, sweep) is None and calls == []
    monkeypatch.setenv(LO.SEEDS_ENV, "0")
    assert _check(tmp_path, OVERWRITE_THEN_ACCUMULATE, sweep) is None and calls == []


def test_a_failing_seed_fails_and_is_named(tmp_path, switched_on):
    sweep, calls = _fake_sweep({3, 5})
    out = _check(tmp_path, OVERWRITE_THEN_ACCUMULATE, sweep)
    assert out.failed and out.seeds == [3, 5] and out.record["status"] == "order_sensitive"
    assert "seed 3" in out.reason and "seed 5" in out.reason


def test_a_clear_trace_is_recorded_and_never_swept(tmp_path, switched_on):
    sweep, calls = _fake_sweep(set())
    out = _check(tmp_path, _trace(), sweep)
    assert out.record["status"] == "static_clear" and not out.failed and calls == []


def test_a_target_without_the_facts_is_not_applicable(tmp_path, switched_on, monkeypatch):
    monkeypatch.setattr(LO, "facts_for", lambda target: (None, "no derived ISA constants for 't'"))
    sweep, calls = _fake_sweep({1})
    out = LO.run_order_check(
        target="t", trace=OVERWRITE_THEN_ACCUMULATE, elf=tmp_path / "p.elf", judge=None, sweep=sweep
    )
    assert out.record == {"status": "not_applicable", "detail": "no derived ISA constants for 't'"} and calls == []


def test_no_engine_is_recorded_as_unavailable_not_as_stable(tmp_path, switched_on):
    sweep, calls = _fake_sweep(set())
    out = _check(tmp_path, OVERWRITE_THEN_ACCUMULATE, sweep, engine=(None, "none declared"))
    assert out.record["status"] == "unavailable" and not out.failed and calls == []


def test_the_judge_holds_a_run_to_the_golden():
    j = LO.console_judge(lambda outs: outs == {"Y": [1]}, lambda text: {"Y": [int(text)]})
    assert (j(0, "1"), j(0, "2"), j(3, "1"), j(0, "x").startswith("unjudged")) == (
        "pass",
        "fail",
        "unjudged:rc=3",
        True,
    )


def test_seeds_are_dealt_across_the_profiles_and_the_in_order_run_is_paid_once(tmp_path, switched_on):
    """Two races, two settings: every profile gets seeds, and only the first sweep re-runs in order."""
    sweep, calls = _fake_sweep({2})
    out = _check(tmp_path, OVERWRITE_THEN_ACCUMULATE, sweep)
    assert [c["seeds"] for c in calls] == [[1, 3, 5, 7], [2, 4, 6, 8]]
    assert [c["include_in_order"] for c in calls] == [True, False]
    assert [c["max_latency"] for c in calls] == [p["max_latency"] for p in LO.DEFAULT_PROFILES]
    assert all(c["via"] == "env" for c in calls), "the engine must run with its backend's own command line"
    assert out.seeds == [2] and [r["profile"] for r in out.record["runs"] if r["seed"] == 2] == [1]
    assert sum(1 for r in out.record["runs"] if r["seed"] is None) == 1


def test_naming_a_knob_in_the_environment_replaces_the_profiles(tmp_path, switched_on, monkeypatch):
    monkeypatch.setenv(LO.LATENCY_ENV, "300")
    sweep, calls = _fake_sweep(set())
    _check(tmp_path, OVERWRITE_THEN_ACCUMULATE, sweep)
    assert len(calls) == 1 and calls[0]["max_latency"] == 300 and calls[0]["seeds"] == list(range(1, 9))


def test_the_engine_is_found_by_role_and_verified_by_content(tmp_path, monkeypatch):
    from merlin.common import provenance as P

    emu = tmp_path / "emulator"
    emu.write_bytes(b"engine")
    registry = tmp_path / "pins.yaml"
    body = {"path": str(emu), "target": "t", "role": LO.ENGINE_ROLE, "digest": P.file_digest(emu)}
    registry.write_text(json.dumps({"artifacts": {"t_engine": body}}))
    monkeypatch.setattr(P, "pins_path", lambda: registry)
    path, why = LO.engine_for("t")
    assert path == emu and why.startswith("t_engine")
    assert LO.engine_for("other")[0] is None
    emu.write_bytes(b"rebuilt in place")
    path, why = LO.engine_for("t")
    assert path is None and "digest" in why, "an engine whose bytes moved is not the engine that was pinned"


# --------------------------------------------------------------------------------------------------
# through the real ladder
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def base_on_path():
    here = str(merlin_dir() / "tests" / "targetgen")
    sys.path.insert(0, here)
    yield
    sys.path.remove(here)


def _ladder_env(monkeypatch, tmp_path, trace, fail_seeds):
    import test_tier_certificate_cache as base  # the sibling module's helpers, reused as plain functions

    from merlin.runtime import reference, simulator
    from merlin.targetgen import tier_cache as TC

    monkeypatch.setenv("MERLIN_TIER_CERT_CACHE", "0")
    TC._INSTRUMENT_MEMO.clear()
    cb = {
        "abi_version": "0.1",
        "target": "cachetest",
        "commands": [],
        "tensors": {"Y0": {"role": "output", "shape": [32, 32], "dtype": "bf16"}},
    }
    outputs = {"Y0": [0] * (32 * 32)}
    monkeypatch.setattr(CR.CG, "golden", lambda *_a, **_k: outputs)
    monkeypatch.setattr(reference, "reference_outputs", lambda *_a, **_k: outputs)
    monkeypatch.setattr(simulator, "simulate", lambda *_a, **_k: {"outputs": outputs})
    monkeypatch.setattr(CR, "run_entrypoints", lambda *a, **k: (object(), cb, "# kernel.S\n"))
    monkeypatch.setattr(base._TP, "tier_order", lambda target, tiers: [t for t in ("L2", "L3") if t in set(tiers)])
    monkeypatch.setattr(CR, "_match_by_policy", lambda *a, **k: True)
    ok = {"status": "pass", "policy": "p", "max_abs_error": 0, "max_rel_error": 0, "mismatch_count": 0,
          "first_mismatch": None, "per_output": {}}  # fmt: skip
    monkeypatch.setattr(CR.CG, "compare_declared_outputs", lambda *a, **k: dict(ok))
    from merlin.targetgen import provenance as _PROV

    monkeypatch.setattr(_PROV, "toolchain_shas", lambda *a, **k: dict(base.SHAS))
    f = tmp_path / "grader.py"
    f.write_text("judge\n")
    base._fixed_instrument(monkeypatch, [f])
    # The stubbed front half decodes no real program, so the static check is handed the trace under
    # test in place of whatever the ladder holds (the check itself is the real one).
    real = LO.static_check
    monkeypatch.setattr(LO, "static_check", lambda _decoded, facts: real(trace, facts))
    monkeypatch.setattr(LO, "facts_for", lambda target: (FACTS, "test"))
    monkeypatch.setattr(LO, "engine_for", lambda target: (tmp_path / "emu", "emu (test)"))
    sweep, calls = _fake_sweep(fail_seeds)
    from merlin.targetgen import mem_perturb as MP

    monkeypatch.setattr(MP, "sweep", sweep)
    return base, calls


def _grade(base, tmp_path, run_id):
    return CR.run_capsule(
        base._capsule(), "unused-package", runs_root=tmp_path / run_id, run_id=run_id,
        config=base._two_tier_config(), oracle_adapters=base._adapters(b"\x7fELF-program", {}),
    )  # fmt: skip


def _norm(result):
    """A result with the fields that legitimately differ between two grades removed."""
    r = json.loads(json.dumps(result))
    for t in r.get("tiers", {}).values():
        for k in ("timing", "concurrency", "submission", "console_log", "console_bytes"):
            t.pop(k, None)
    return r


def test_off_leaves_the_grade_byte_identical_and_never_sweeps(tmp_path, monkeypatch, base_on_path):
    base, calls = _ladder_env(monkeypatch, tmp_path, OVERWRITE_THEN_ACCUMULATE, {1, 2})
    monkeypatch.delenv(LO.SEEDS_ENV, raising=False)
    off = _grade(base, tmp_path, "off")
    monkeypatch.setenv(LO.SEEDS_ENV, "0")
    zero = _grade(base, tmp_path, "zero")
    assert calls == [] and off["status"] == "pass", off.get("failure")
    assert "order_check" not in off and _norm(off) == _norm(zero)


def test_on_and_clear_records_the_static_verdict_and_never_sweeps(tmp_path, monkeypatch, base_on_path):
    base, calls = _ladder_env(monkeypatch, tmp_path, {"instructions": []}, set())
    monkeypatch.setenv(LO.SEEDS_ENV, "8")
    res = _grade(base, tmp_path, "clear")
    assert calls == [] and res["status"] == "pass"
    assert res["order_check"]["status"] == "static_clear" and res["order_check"]["cert_tier"] == "L3"


def test_the_sweep_runs_only_when_the_static_check_fires(tmp_path, monkeypatch, base_on_path):
    base, calls = _ladder_env(monkeypatch, tmp_path, OVERWRITE_THEN_ACCUMULATE, set())
    monkeypatch.setenv(LO.SEEDS_ENV, "8")
    res = _grade(base, tmp_path, "fires")
    assert len(calls) == len(LO.DEFAULT_PROFILES) and res["status"] == "pass", res.get("failure")
    assert res["order_check"]["status"] == "order_stable"
    assert res["order_check"]["cert_tier"] == "L3" and res["order_check"]["static"]["verdict"] == "hazard"


def test_an_order_sensitive_program_fails_its_certificate(tmp_path, monkeypatch, base_on_path):
    base, calls = _ladder_env(monkeypatch, tmp_path, OVERWRITE_THEN_ACCUMULATE, {2, 7})
    monkeypatch.setenv(LO.SEEDS_ENV, "8")
    res = _grade(base, tmp_path, "sensitive")
    assert res["status"] == "fail"
    assert res["tiers"]["L3"]["status"] == "fail" and "seed 2" in res["tiers"]["L3"]["reason"]
    assert res["failure"]["plane"] == "L3_order" and "seed 7" in res["failure"]["detail"]
    assert res["order_check"]["failing_seeds"] == [2, 7]


def test_the_promotion_ledger_records_an_order_failure_as_a_failed_cert(tmp_path, monkeypatch, base_on_path):
    """A promoted cert job whose L3 passed in order but failed the order check must land as FAIL --
    not as a pass, and not left pending."""
    from merlin_experiments.phase1.feedback import promotion as B
    from phase1_feedback import feedback_context

    base, _ = _ladder_env(monkeypatch, tmp_path, OVERWRITE_THEN_ACCUMULATE, {4})
    monkeypatch.setenv(LO.SEEDS_ENV, "8")
    res = _grade(base, tmp_path, "promoted")
    assert B.row_status({"capsule": "A", "status": res["status"]}) == "fail"

    ws = tmp_path / "ws"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("x: 1")
    (ws / ".qa_channel").mkdir()
    loop = {"per_capsule": [{"capsule": "A", "pass": True, "execution_digest": "a" * 64}]}
    assert B.promote(ws, ws / ".qa_channel", loop, "L2", "L3", None, sys.stderr, context=feedback_context()) == ["A"]
    assert B._tier_state(ws)["A"]["L3"]["status"] == "pending"
    cert = {"per_capsule": [{"capsule": "A", "status": res["status"], "execution_digest": "a" * 64}]}
    B.record_cert(ws, cert, "L3", sys.stderr)
    assert B._tier_state(ws)["A"]["L3"]["status"] == "fail"


def test_which_classes_stay_on_chip_is_the_shared_vocabulary(monkeypatch):
    """The local-only set is read from the shared class table at use: a class that table declares
    plumbing is skipped, a loop class (whose unroller issues loads the trace cannot see) is not, and a
    class the table does not know is opaque."""
    from merlin.targetgen import semantic_families as SF

    local = SF.local_only_classes()
    assert {"CONFIG_LD", "PRELOAD"} <= local and "LOOP_WS" not in local
    trace = _trace({"index": 0, "class": "CONFIG_LD", "decoded": {}}, {"index": 1, "class": "LOOP_WS", "decoded": {}})
    loads = [LO._as_load(row, FACTS) for row in trace["instructions"]]
    assert loads[0] is None and loads[1] is not None and loads[1].rows is None
    monkeypatch.setattr(SF, "_ISA_PLUMBING_CLASSES", SF._ISA_PLUMBING_CLASSES | {"LOOP_WS"})
    assert LO._as_load(trace["instructions"][1], FACTS) is None

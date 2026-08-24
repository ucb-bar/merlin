"""The ``capacity_fit`` obligation has to work on a SECOND target, not only the one it was built against.

Measured on the first attempt to carry the gemmini whole-model certification over to atlas:

* the operand-store capacity was read from an RTL fact keyed on one memory ROLE, and the second target
  declares no memories at all in its facts, so the obligation returned ``holds: None`` -- silently
  undecidable while looking exactly like a target with no problem. It is STILL undecidable there (mlc
  discovers the SRAMs and refuses to classify them, and the one candidate store that could be summed
  was refuted by measurement), but it now says so instead of going quiet;
* the obligation was evaluated inside the RoCC/oot-cert path only, and the second target routes its
  layers through the PROGRAM ORACLE, so nothing was evaluated on its path at all;
* the element width came from scraping every digit out of the dtype token, and ``fp8_e4m3`` has three
  digit runs -- it concatenated to 843, sizing a 64 KiB operand file as 624 elements.
"""
from __future__ import annotations

import pytest

from merlin.compile_cli import _dtype_bits, _operand_store_bytes, capacity_fit


@pytest.mark.parametrize("tok,bits", [
    ("i8", 8), ("int8", 8), ("i32", 32), ("bf16", 16), ("f32", 32),
    ("fp8_e4m3", 8),      # the bug: digit-scraping read "843"
    ("fp8_e5m2", 8),
    ("fp4_e2m1", 4),      # sub-byte: must not round up to 8 here
    ("e4m3", 8),          # an alias spelling still resolves through the registry
    (None, 8),
])
def test_element_width_comes_from_the_format_not_the_spelling(tok, bits):
    assert _dtype_bits(tok) == bits


def test_a_float_format_token_does_not_concatenate_its_digit_runs():
    """The specific failure: 'fp8_e4m3' scraped to 843 bits, pricing a 65536-byte store at 624 elements
    instead of 65536. Asserted on the arithmetic, since which store a float target declares is a
    separate question (see the module docstring)."""
    assert _dtype_bits("fp8_e4m3") == 8
    from merlin.compile_cli import _operand_store_capacity_elems
    import merlin.compile_cli as cc
    real = cc._operand_store_bytes
    try:
        cc._operand_store_bytes = lambda _t: 65536
        assert _operand_store_capacity_elems("any", "fp8_e4m3") == 65536
    finally:
        cc._operand_store_bytes = real


def test_a_sub_byte_format_is_counted_in_bits():
    """fp4 packs two per byte; counting it a byte apiece halves the capacity for free."""
    from merlin.compile_cli import _operand_store_capacity_elems
    b = _operand_store_bytes("gemmini")
    if not b:
        pytest.skip("gemmini operand store not derivable in this checkout")
    assert _operand_store_capacity_elems("gemmini", "fp4_e2m1") == b * 2


def test_the_mapped_target_still_derives_its_store():
    """gemmini's comes from mlc's discovered memory MAP — no label declared anywhere."""
    b = _operand_store_bytes("gemmini")
    if b is None:
        pytest.skip("gemmini RTL discovery unavailable in this checkout")
    assert b == 262144


def test_an_unclassified_target_reports_undecidable_rather_than_guessing():
    """atlas: mlc discovers every SRAM and then refuses to classify them. Two candidate stores exist and
    the tempting one is REFUTED BY MEASUREMENT — a 32x256x256 layer needs 73728 elements against the
    64 KiB matrix register file and runs unblocked on the arc cosim anyway. So nothing is declared, and
    the obligation reports itself undecidable. This test exists to keep someone from 'fixing' that by
    declaring the register file, which would split layers the device handles and, on this bf16
    accumulator, change the reduction order while doing it."""
    from merlin.targetgen.rtl import mlc_bridge as mb
    if not mb.discovered_memories("atlas"):
        pytest.skip("atlas RTL discovery unavailable in this checkout")
    assert _operand_store_bytes("atlas") is None
    assert capacity_fit("atlas", 32, 256, 256, "fp8_e4m3", 32)["holds"] is None


def test_the_biggest_memory_is_never_taken_as_the_operand_store():
    """Why a store is declared or left unknown, never inferred: this device's INSTRUCTION memory
    (32768 x 4 B = 128 KiB) is its largest SRAM, so 'take the biggest' would answer with IMEM."""
    from merlin.targetgen.rtl import mlc_bridge as mb
    mems = mb.discovered_memories("atlas")
    if not mems:
        pytest.skip("atlas RTL discovery unavailable in this checkout")
    biggest = max(int(m["depth"]) * int(m["row_bytes"]) for m in mems)
    assert biggest == 131072 and _operand_store_bytes("atlas") != biggest


def test_an_undeclared_capacity_stays_unknown_on_every_target():
    assert capacity_fit("definitely_not_a_target", 16, 512, 512, "int8", 16)["holds"] is None


def test_run_paths_do_not_depend_on_the_process_cwd(tmp_path, monkeypatch):
    """A grading thread that chdirs (mlc resolves its arc artifacts relative to its own root) must not
    be able to relocate another thread's run directory. Measured: a suite died writing
    capsule_result.json into a path that existed, because it was created in one cwd and written in
    another."""
    from merlin.targetgen.capsule_common import make_run_paths
    (tmp_path / "here").mkdir()
    monkeypatch.chdir(tmp_path / "here")
    paths = make_run_paths("relative_runs", "cap0", suite="s", target="t",
                           dtype="int8", benchmark="b")
    assert paths.run_path.is_absolute(), paths.run_path
    monkeypatch.chdir(tmp_path)
    assert paths.run_path.is_dir(), "the run dir must still resolve after a chdir"


def _probe(monkeypatch, target, dtype, m, k, n):
    """Run one layer through the mesh entry point with every real oracle path stubbed to 'declined', so
    the assertion is about what the CALLER records, not about any simulator being present."""
    import merlin.compile_cli as cc
    for fn in ("_matmul_via_oot_cert", "_matmul_via_program_oracle", "_matmul_via_bespoke_sim"):
        monkeypatch.setattr(cc, fn, lambda *a, **kw: None, raising=True)
    obs: dict = {}
    cc.run_matmul_on_mesh(target, [[0.0] * k for _ in range(m)], [[0.0] * n for _ in range(k)],
                          operand_dtype=dtype, timeout=5, observed=obs)
    return obs


@pytest.mark.parametrize("target,dtype,path,holds", [
    ("gemmini", "int8", "oot_cert", True),          # declares a store: the obligation has a verdict
    ("atlas", "fp8_e4m3", "program_oracle", None),  # declares none: evaluated, and the answer is unknown
])
def test_the_obligation_is_evaluated_on_every_mesh_path(monkeypatch, target, dtype, path, holds):
    """It used to live inside the RoCC/oot-cert path. A self-hosted-ISA target leaves through the
    program oracle, so its layers had no obligation evaluated at all -- an oversized layer there
    declined as 'the oracle returned nothing', which is the failure mode this check exists to end.

    What is asserted is that the obligation is EVALUATED on both paths. Whether it can answer is a
    separate matter: an undecidable ``holds: None`` is a legitimate outcome, and getting one recorded
    is the whole difference from the silence this replaced.
    """
    obs = _probe(monkeypatch, target, dtype, 32, 64, 64)
    if obs.get("path") is None:
        pytest.skip(f"{target} has no reachable mesh path in this checkout")
    assert obs["path"] == path
    assert "capacity_fit_check" in obs, "no obligation was evaluated on this endpoint"
    assert obs["capacity_fit_check"]["holds"] is holds, obs["capacity_fit_check"]


@pytest.mark.parametrize("target,dtype,exact,accum", [("gemmini", "int8", True, "i32")])
def test_an_oversized_layer_is_blocked_and_charged(monkeypatch, target, dtype, exact, accum):
    obs = _probe(monkeypatch, target, dtype, 32, 512, 512)
    cf = obs.get("capacity_fit")
    if cf is None:
        pytest.skip(f"{target} declares no operand-store capacity in this checkout")
    assert cf["holds"] is False and cf["required_elems"] > cf["capacity_elems"]
    assert cf["discharged_by"] == "merlin runtime (host-side residency tiling)"
    assert obs["blocked"]["n_subtiles"] > 1
    # WHETHER THE SPLIT IS FREE. Over an integer accumulator the K-partials sum exactly; over a float
    # accumulator the reduction order changes (per-block on the device, cross-block on the host), and a
    # verdict that does not say so reads as "this is what the accelerator computes".
    assert cf["split_exact"] is exact and cf["accum_dtype"] == accum
    assert ("CHANGES THE REDUCTION ORDER" in cf["note"]) is (not exact)


def test_an_undecidable_obligation_says_so_instead_of_going_quiet(monkeypatch):
    """`holds: None` is the correct answer for a target that declares no capacity -- and a silent None
    is how a whole class of residency failures came to be reported as an unreachable oracle."""
    import merlin.compile_cli as cc
    monkeypatch.setattr(cc, "_operand_store_bytes", lambda _t: None, raising=True)
    obs = _probe(monkeypatch, "gemmini", "int8", 32, 512, 512)
    if obs.get("path") is None:
        pytest.skip("no reachable mesh path in this checkout")
    assert "contract_violation" not in obs, "an undecidable obligation must not be charged to anyone"
    assert obs["capacity_fit_unevaluable"]["holds"] is None
    assert "UNATTRIBUTED" in obs["capacity_fit_unevaluable"]["detail"]


def test_the_unclassified_target_declines_with_the_reason_named(monkeypatch):
    """End to end on the real target: no declared store, so a decline must carry
    ``capacity_fit_unevaluable`` -- naming the missing fact -- and must charge nobody."""
    obs = _probe(monkeypatch, "atlas", "fp8_e4m3", 32, 512, 512)
    if obs.get("path") is None:
        pytest.skip("atlas has no reachable mesh path in this checkout")
    assert "blocked" not in obs, "nothing to block against: no capacity is declared"
    assert "contract_violation" not in obs
    assert obs["capacity_fit_unevaluable"]["holds"] is None
    assert "UNATTRIBUTED" in obs["capacity_fit_unevaluable"]["detail"]


def test_a_capsule_that_chdirs_cannot_relocate_its_siblings(tmp_path, monkeypatch):
    """The stronger form of the cwd bug: resolving per-capsule is not enough.

    Some capsule paths enter a context that chdirs the process (mlc resolves its arc artifacts relative
    to its own root). A relative runs-root resolved INSIDE that window becomes an absolute path under
    the wrong tree. Measured: of 26 op capsules run with 8 workers, 18 wrote their whole run directory
    into the mlc checkout -- the suite still scored, because results come back in memory, but the trace
    collection reads from this root, so those 18 contributed no coverage at all.
    """
    from merlin.targetgen import capsule_runner as CR

    seen: list[str] = []
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "here").mkdir()

    def _fake_run_capsule(cap, package_dir, *, runs_root, **kw):
        seen.append(runs_root)
        import os
        os.chdir(tmp_path / "elsewhere")          # the sibling-thread chdir, made deterministic
        return {"capsule": cap["name"], "status": "pass", "kind": "op", "tiers": {}}

    monkeypatch.setattr(CR, "load_package", lambda *a, **k: object())
    monkeypatch.setattr(CR, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(CR, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(CR, "_split_ineligible", lambda caps, t: (caps, []))
    monkeypatch.setattr(CR, "run_capsule", _fake_run_capsule)
    monkeypatch.chdir(tmp_path / "here")

    CR.run_suite([{"name": f"c{i}", "kind": "op"} for i in range(4)], "pkg",
                 runs_root="relative_runs", target="gemmini", max_workers=1)

    assert len(set(seen)) == 1, f"the runs root moved mid-suite: {sorted(set(seen))}"
    assert seen[0] == str((tmp_path / "here" / "relative_runs")), seen[0]


def test_a_crashed_op_capsule_cannot_leave_the_model_gate_denominator(monkeypatch):
    """An error is not evidence the op suite works -- it is the absence of evidence.

    The gate counted only pass/fail, so every capsule that died with a runner crash left the
    denominator with it. Measured on atlas: 18 of 26 op capsules crashed (a relative package path
    resolved inside another thread's chdir), and the whole-model capstone cleared its 0.8 gate on the
    surviving 7/8 = 0.88 while two thirds of the suite had not run.
    """
    from merlin.targetgen import capsule_runner as CR

    ran: list[str] = []
    caps = ([{"name": f"ok{i}", "kind": "op"} for i in range(7)]
            + [{"name": "bad", "kind": "op"}]
            + [{"name": f"crash{i}", "kind": "op"} for i in range(18)]
            + [{"name": "M0", "kind": "model", "gate": {"after_op_pass_fraction": 0.8}}])

    def _fake(cap, package_dir, **kw):
        ran.append(cap["name"])
        status = ("pass" if cap["name"].startswith("ok")
                  else "error" if cap["name"].startswith("crash") else "fail")
        return {"capsule": cap["name"], "status": status, "kind": cap.get("kind"), "tiers": {}}

    monkeypatch.setattr(CR, "load_package", lambda *a, **k: object())
    monkeypatch.setattr(CR, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(CR, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(CR, "_split_ineligible", lambda c, t: (c, []))
    monkeypatch.setattr(CR, "_gate_counts", lambda r, c, t: True)
    monkeypatch.setattr(CR, "run_capsule", _fake)

    out = CR.run_suite(caps, "pkg", runs_root="runs", target="gemmini", max_workers=1)
    m0 = next(r for r in out if r["capsule"] == "M0")
    assert m0["status"] == "gated", "7 of 26 passing must not clear a 0.8 gate"
    assert "0.27" in m0["failure"]["detail"], m0["failure"]["detail"]
    assert "M0" not in ran, "a gated capstone must not have been executed"

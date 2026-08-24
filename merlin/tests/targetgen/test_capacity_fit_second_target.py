"""The ``capacity_fit`` obligation has to work on a SECOND target, not only the one it was built against.

Measured on the first attempt to carry the gemmini whole-model certification over to atlas:

* the operand-store capacity was read from an RTL fact keyed on one memory ROLE, and the second target
  declares no memories at all in its facts, so the obligation returned ``holds: None`` -- silently
  undecidable while looking exactly like a target with no problem;
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
    """The specific failure: 'fp8_e4m3' -> 843 bits -> a 65536-byte store priced at 624 elements."""
    assert _dtype_bits("fp8_e4m3") != 843
    v = capacity_fit("atlas", 1, 4608, 512, "fp8_e4m3", 32)
    if v["capacity_elems"] is None:
        pytest.skip("atlas operand store not derivable in this checkout")
    assert v["capacity_elems"] == 65536, v


def test_a_sub_byte_format_is_counted_in_bits():
    """fp4 packs two per byte; counting it a byte apiece halves the capacity for free."""
    from merlin.compile_cli import _operand_store_capacity_elems
    b = _operand_store_bytes("gemmini")
    if not b:
        pytest.skip("gemmini operand store not derivable in this checkout")
    assert _operand_store_capacity_elems("gemmini", "fp4_e2m1") == b * 2


@pytest.mark.parametrize("target,expect_bytes", [("gemmini", 262144), ("atlas", 65536)])
def test_both_targets_derive_an_operand_store(target, expect_bytes):
    """Two independent sources agree on each: gemmini via mlc's discovered memory MAP, atlas via the
    descriptor-declared bank-group prefix summed over mlc's discovered memory LIST (and corroborated by
    its shipped ISA model's num_m_registers x mrf_depth x mrf_width)."""
    b = _operand_store_bytes(target)
    if b is None:
        pytest.skip(f"{target} RTL discovery unavailable in this checkout")
    assert b == expect_bytes


def test_the_biggest_memory_is_not_the_operand_store():
    """Why the label is declared rather than guessed: this device's INSTRUCTION memory (32768 x 4 B =
    128 KiB) is larger than its operand register file (64 KiB), so 'take the largest SRAM' picks IMEM
    and prices the obligation at twice the real capacity."""
    from merlin.targetgen.rtl import mlc_bridge as mb
    mems = mb.discovered_memories("atlas")
    if not mems:
        pytest.skip("atlas RTL discovery unavailable in this checkout")
    biggest = max(int(m["depth"]) * int(m["row_bytes"]) for m in mems)
    store = _operand_store_bytes("atlas")
    assert store is not None and biggest > store, (biggest, store)


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


@pytest.mark.parametrize("target,dtype,path", [
    ("gemmini", "int8", "oot_cert"),
    ("atlas", "fp8_e4m3", "program_oracle"),
])
def test_the_obligation_is_evaluated_on_every_mesh_path(monkeypatch, target, dtype, path):
    """It used to live inside the RoCC/oot-cert path. A self-hosted-ISA target leaves through the
    program oracle, so its layers had no obligation evaluated at all -- an oversized layer there
    declined as 'the oracle returned nothing', which is the failure mode this check exists to end."""
    obs = _probe(monkeypatch, target, dtype, 32, 64, 64)
    if obs.get("path") is None:
        pytest.skip(f"{target} has no reachable mesh path in this checkout")
    assert obs["path"] == path
    assert obs["capacity_fit_check"]["holds"] is True, obs["capacity_fit_check"]


@pytest.mark.parametrize("target,dtype,exact,accum", [
    ("gemmini", "int8", True, "i32"),
    ("atlas", "fp8_e4m3", False, "bf16"),
])
def test_an_oversized_layer_is_blocked_and_charged_on_both_targets(monkeypatch, target, dtype,
                                                                   exact, accum):
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

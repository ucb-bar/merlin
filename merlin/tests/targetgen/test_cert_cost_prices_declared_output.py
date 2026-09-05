"""A certification is priced by what a capsule WRITES, read from the module's own declaration.

THE REGRESSION THIS PINS. ``cert_cost.capsule_output_elements`` used to take the extent of the operand
a terminal command READ, under a fixed list of operand keys. That inference is correct only for a
shape-preserving op, and this corpus is full of ops that are not one. Measured against the corpus's own
goldens, it disagreed for 43 of 452 capsules and raised on a 44th:

* a convolution was priced by its INPUT IMAGE. The convolution-window axis solves the image from the
  kernel for a fixed output, so the operand grows with the window while the result stays 4x4 -- and
  ``SY_conv_k16x16_s16x16`` was priced at 16,384 elements and 7,177 predicted seconds against a
  256-element result worth 81 s. Its k8x8 and k4x4 siblings were over budget for the same reason, and
  four other conv members were priced UNDER their true cost by the same inference.
* a batched matmul was priced by its stacked operand, and seven flash-attention capsules at ZERO,
  because the op they end on reads an operand key the list did not enumerate.

The declared result types agree with the goldens for 450 of those 452. So the fix is not a different
guess: it is the extent the module is under contract to produce.

AND THE SECOND HALF, in ``check_cert_affordability``: an over-budget capsule was advised to declare
``max_oracle_tier: L2`` or an ``extends``, and both are a cap onto a tier BELOW the cert tier. A target
whose adapter registry offers only the cert tier has none to cap onto, so that advice is a fix its
runner would refuse. Which remedies are open is therefore derived per row.
"""
from __future__ import annotations

import importlib.util

import pytest
import yaml

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen import cert_cost as CC
from merlin.targetgen.contract import interface_emit as IE

_CAPSULES = merlin_dir() / "contract" / "capsules"


def _gate():
    path = repo_root() / "build_tools" / "scripts" / "check_cert_affordability.py"
    if not path.is_file():
        pytest.skip("affordability gate absent")
    spec = importlib.util.spec_from_file_location("_cert_affordability_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: A convolution whose ACTIVATION is 64x lts result: exactly the shape the window axis produces, since
#: it solves the image from the kernel for a fixed output. Written out rather than read from the corpus
#: so the property is pinned even in a checkout whose corpus has not been regenerated.
_CONV = '''module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t", merlin_iface.abi_version = "0.1"} {
  %IFM = merlin_iface.tensor {name = "IFM", role = "input"} : tensor<1x64x64x4xi8>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<1024x16xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_conv_rhs"} : (tensor<1024x16xi8>) -> !merlin_iface.resident
  %Y0 = merlin_iface.conv2d %IFM, %W_res {kernel = [16, 16, 4, 16], stride = [16, 16], padding = [0, 0, 0, 0], dilation = [1, 1], name = "Y0", epilogue = [], output_dtype = "i32", layout = "nhwc"} : (tensor<1x64x64x4xi8>, !merlin_iface.resident) -> tensor<16x16xi32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
'''


def test_a_convolution_is_priced_by_what_it_writes_not_by_the_image_it_reads():
    """The measured defect: 16,384 operand elements against a 256-element result, 7,177s against 81s."""
    got = CC.capsule_output_elements(_CONV)
    assert got == 256, (
        f"a 4x4x16 convolution result priced at {got} elements; the activation it reads is 16,384, "
        f"and pricing by that predicts {CC.predict_seconds_from_output(16384)[0]:,.0f}s of "
        f"cycle-accurate time for a capsule worth {CC.predict_seconds_from_output(256)[0]:,.0f}s")
    secs, extrapolated = CC.predict_seconds_from_output(got)
    assert not extrapolated and secs < 300.0, (
        "a fixed-4x4-output convolution must be affordable at the corpus's own budget")


def test_the_declared_extent_is_read_from_the_module_not_recomputed():
    """The mapping keys on what ``parse_interface_mlir`` calls a command's ``dst``."""
    declared = IE.declared_result_elements(_CONV)
    assert declared["Y0"] == 256
    # A residency handle is device state, not a program output: zero, and PRESENT, so a caller can
    # tell "writes nothing" from "unknown".
    assert declared["W_res"] == 0
    assert "W_res" in declared


def test_a_terminal_write_whose_result_type_cannot_be_read_is_refused_not_guessed():
    """FAIL CLOSED. Pricing an unreadable write by an operand it read is what this refusal replaces."""
    untyped = _CONV.replace(
        ' : (tensor<1x64x64x4xi8>, !merlin_iface.resident) -> tensor<16x16xi32>', '')
    assert "-> tensor<16x16xi32>" not in untyped, "the fixture did not actually drop the result type"
    with pytest.raises(ValueError) as exc:
        CC.capsule_output_elements(untyped)
    assert "Y0" in str(exc.value), "the refusal must name the write it could not price"


def test_the_corpus_prices_agree_with_the_goldens_it_ships():
    """The independent check: the golden is the elements an engine actually has to produce.

    Bounded to the families the operand inference got wrong -- convolution, batched contraction and
    the per-channel sweep -- so it stays a fast unit test rather than a corpus walk.
    """
    def _nelem(v):
        if isinstance(v, list):
            return sum(_nelem(x) for x in v) if v and isinstance(v[0], list) else len(v)
        return 1

    checked = 0
    for cy in sorted(_CAPSULES.rglob("capsule.yaml")):
        name = cy.parent.name
        if not (name.startswith("SY_conv") or "conv2d" in name or "batched" in name):
            continue
        ifc = cy.parent / "capsule.interface.mlir"
        golden = cy.parent / "golden.yaml"
        if not (ifc.is_file() and golden.is_file()):
            continue
        outs = (yaml.safe_load(golden.read_text(encoding="utf-8")) or {}).get("outputs") or {}
        if not outs:
            continue
        try:
            got = CC.capsule_output_elements(ifc.read_text(encoding="utf-8"))
        except IE.InterfaceGrammarError:           # a linalg capsule is priced by another path
            continue
        want = sum(_nelem(v) for v in outs.values())
        assert got == want, (
            f"{cy.parent.relative_to(_CAPSULES)}: priced at {got} written elements while its own "
            f"golden carries {want} -- the price and the answer key disagree about what it writes")
        checked += 1
    if not checked:
        pytest.skip("no convolution or batched capsule with a golden in this checkout")
    assert checked >= 3, f"only {checked} capsule(s) checked; this establishes little"


def test_a_capsule_is_not_advised_a_cap_its_target_cannot_take():
    """``max_oracle_tier``/``extends`` are a cap onto a CHEAPER tier. Some targets declare none."""
    gate = _gate()
    from merlin.targetgen.conformance import _declared_oracle_tiers  # noqa: PLC2701

    descs = repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets"
    if not descs.is_dir():
        pytest.skip("no target descriptors in this checkout")
    single, laddered = [], []
    for d in sorted(descs.iterdir()):
        if not (d / "target_experiment.yaml").is_file():
            continue
        tiers = [str(t) for t in (_declared_oracle_tiers(d.name) or ())]
        (single if len(tiers) <= 1 else laddered).append(d.name)
    if not single or not laddered:
        pytest.skip("this roster does not contain both a single-tier and a laddered target, so the "
                    "distinction cannot be exercised here")

    cache: dict = {}
    for name in single:
        cheaper = gate._cheaper_tier(name, cache)
        assert cheaper is None or cheaper == gate._TIER_UNKNOWN, (
            f"{name} declares no tier below its cert tier, so no cap is available to it")
        remedies = gate._remedies({"needs_cycle_accurate": False}, cheaper)
        assert "cap_onto_cheaper_tier" not in remedies, (
            f"{name} was advised to cap onto a cheaper tier it does not declare: {remedies}")
        assert "smaller_shape" in remedies and "accepted_cost" in remedies, (
            "a target with no cheaper tier keeps the other two remedies; reporting none would read "
            "as an unfixable capsule")
    for name in laddered:
        cheaper = gate._cheaper_tier(name, cache)
        assert cheaper, f"{name} declares a tier ladder, so a cap onto its cheaper tier IS available"
        assert "cap_onto_cheaper_tier" in gate._remedies({"needs_cycle_accurate": False}, cheaper)


def test_a_cycle_counted_capsule_is_never_advised_a_cap():
    """Capping it at the loop tier would delete the measurement the capsule exists to take."""
    gate = _gate()
    remedies = gate._remedies({"needs_cycle_accurate": True}, "L2")
    assert "cap_onto_cheaper_tier" not in remedies, remedies
    assert remedies == ["smaller_shape", "accepted_cost"]


def test_an_unresolvable_tier_ladder_is_unknown_and_not_an_absent_cap():
    """The two license opposite actions: accept the cost, versus go and resolve the adapters."""
    gate = _gate()
    cache = {"nobody": gate._TIER_UNKNOWN}
    remedies = gate._remedies({"needs_cycle_accurate": False}, gate._cheaper_tier("nobody", cache))
    assert "cap_onto_cheaper_tier_UNRESOLVED" in remedies, remedies
    assert "cap_onto_cheaper_tier" not in remedies, (
        "an unresolved ladder must not read as a cap that is available")

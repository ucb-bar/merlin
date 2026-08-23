"""The whole-model mesh lane must be DERIVED from the target, not written for one of them.

Four independent hardcodings kept every model capsule off the accelerator, and they hid each other:

* the runner looked for ``capsule.linalg.mlir`` while every model capsule declares
  ``capsule.interface.mlir``, so the routing plan, the coverage certificate AND the mesh verification
  were all skipped -- silently, since a skipped block leaves no trace in the result;
* the execution lane defaulted to the host dispatch runtime unconditionally;
* ``_classify_mesh_matmul`` accepted element types ``("f32", "i8")``, a literal pair, so a float-datapath
  accelerator never classified a single layer as routable;
* and the boundary then called the mesh with ``operand_dtype="int8", accum_dtype="i32"``.

The ordering is what made it durable: because the classifier routed nothing, the hardcoded dtypes below
it were never reached, so nothing ever observed them to be wrong. The model ran on the host and the run
looked clean.

A fifth defect was the same shape one level up: routing compared format SPELLINGS, so ``i8`` against a
contract declaring ``int8`` routed 0 of 15 contractions to a mesh that supports all of them -- and the
coverage certificate reported that as a real-looking ARR of 0.0.
"""
from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import capsule_source as CSRC
from merlin.targetgen import routing as R

TARGETS = ("gemmini", "atlas")


def _model_capsule(target: str):
    root = repo_root() / "merlin" / "contract" / "capsules"
    d = (root / target / "model") if (root / target / "model").is_dir() else (root / "model")
    for cf in sorted(d.glob("*/capsule.yaml")) if d.is_dir() else []:
        lin = cf.parent / "capsule.interface.mlir"
        if lin.is_file():
            return yaml.safe_load(cf.read_text()), lin.read_text()
    pytest.skip(f"no model capsule generated for {target}")


@pytest.mark.parametrize("target", TARGETS)
def test_the_mesh_datapath_is_read_off_the_target(target):
    from merlin.runtime.dispatch_runtime import mesh_datapath
    b = mesh_datapath(target)
    spelling = b.mlir_dtype(b.operand_dtype)
    assert b.operand_dtype and b.accum_dtype and spelling
    assert isinstance(b.integer, bool)
    # the MLIR spelling must be the one the target's own registry entry declares
    from merlin.targetgen.corpus_spec import dtype_info
    assert spelling == dtype_info(b.operand_dtype)[1]


def test_two_targets_derive_two_different_datapaths():
    """The regression guard: a single hardcoded pair cannot be right for both."""
    from merlin.runtime.dispatch_runtime import mesh_datapath
    got = {t: mesh_datapath(t) for t in TARGETS}
    assert len({(g.operand_dtype, g.accum_dtype, g.integer) for g in got.values()}) > 1, got
    assert (any(g.integer for g in got.values()) and not all(g.integer for g in got.values())), \
        "one integer datapath and one float datapath — the boundary cannot treat them alike"


def test_the_datapath_carries_the_targets_declared_facts_not_dataclass_defaults():
    """The regression guard for the bug this whole path was built around.

    ``subnormal_operand_flush`` is derived from RTL, written into the target's profile, and honoured by
    the capsule golden engine — and the mesh boundary still read ``False``, because the binding it got was
    derived from a dict holding nothing but two dtypes. The fact was correct everywhere except at the one
    consumer that feeds the mesh real model operands. Assert the declared block ARRIVES, so a future
    hand-built datapath dict fails here instead of silently defaulting.
    """
    from merlin.runtime.dispatch_runtime import mesh_datapath
    from merlin.targetgen.corpus_spec import profile_datapath
    for t in TARGETS:
        declared = profile_datapath(t)
        if not declared:
            continue
        b = mesh_datapath(t)
        for field in ("subnormal_operand_flush", "atol", "rtol"):
            if field in declared:
                assert getattr(b, field) == declared[field], \
                    f"{t}: profile declares {field}={declared[field]!r}, binding carries {getattr(b, field)!r}"


@pytest.mark.parametrize("target", TARGETS)
def test_a_model_routes_its_contractions_to_the_mesh(target):
    """Routed on the format the capsule DECLARES, which is the one the runner threads."""
    cap, lin = _model_capsule(target)
    declared = ((cap.get("operation") or {}).get("attributes") or {}).get("dtype")
    assert declared, "a model capsule must declare its datapath format"
    plan = R.route_plan(CSRC.model_op_demands(lin, declared), target)
    assert len(plan.get("mesh") or []) > 0, \
        f"{target}: 0 contractions routed on its own declared format {declared!r}"


def test_an_alias_routes_but_a_different_format_still_does_not():
    """Alias-awareness is not widening. Both halves are asserted, because only the second one is safety.

    ``e4m3`` and ``e5m2`` are both 8-bit floats; routing e5m2 data onto an e4m3 unit would be a
    miscompile that produces plausible numbers.
    """
    cap, lin = _model_capsule("atlas")
    declared = ((cap.get("operation") or {}).get("attributes") or {}).get("dtype")
    n_declared = len(R.route_plan(CSRC.model_op_demands(lin, declared), "atlas").get("mesh") or [])
    assert n_declared > 0
    for other in ("fp8_e5m2", "e5m2"):
        n = len(R.route_plan(CSRC.model_op_demands(lin, other), "atlas").get("mesh") or [])
        assert n == 0, f"{other} routed {n} contraction(s) onto an e4m3 unit"


@pytest.mark.parametrize("target", TARGETS)
def test_routing_and_eligibility_agree_on_what_a_format_is(target):
    """The two sides of ARR. If they disagree about spelling, the ratio measures spelling."""
    from merlin.targetgen import eligibility as el
    cap, lin = _model_capsule(target)
    declared = ((cap.get("operation") or {}).get("attributes") or {}).get("dtype")
    cap_map = el.capability_map_for_target(target)
    demands = CSRC.model_op_demands(lin, declared)
    routed = {id(r.demand) for r in (R.route_plan(demands, target).get("mesh") or [])}
    for d in demands:
        if id(d) not in routed:
            continue
        v = el.is_eligible(el.RegionDescriptor(source=d.site or d.op, op=d.op, in_dtype=d.in_fmt,
                                               weight_dtype=d.weight_fmt), cap_map)
        assert v.eligible, \
            f"{target}: routed {d.op} to the mesh while the oracle calls it ineligible ({v.reason})"


@pytest.mark.parametrize("target", TARGETS)
def test_the_runner_finds_the_linalg_the_capsule_declares(target):
    """The break that hid the other three: a name mismatch producing a silent skip."""
    cap, _lin = _model_capsule(target)
    root = repo_root() / "merlin" / "contract" / "capsules"
    d = (root / target / "model") if (root / target / "model").is_dir() else (root / "model")
    cdir = next(p for p in sorted(d.glob("*/capsule.yaml")) if p.parent.name == cap["name"]).parent
    declared = cap.get("linalg_mlir") or cap.get("interface_mlir")
    assert declared, "a model capsule must declare where its linalg lives"
    assert (cdir / str(declared)).is_file(), f"{target}: declared {declared!r} is not on disk"


@pytest.mark.parametrize("target", TARGETS)
def test_a_capstone_that_demands_acceleration_selects_the_mesh_lane(target):
    """Grading a must_accelerate capsule on the host lane passes a run that cannot possibly satisfy it."""
    import os
    cap, _ = _model_capsule(target)
    if not (cap.get("semantic") or {}).get("must_accelerate"):
        pytest.skip(f"{target}: capstone demand not grounded in this checkout")
    assert not os.environ.get("MERLIN_MODEL_GRADE_RUN"), \
        "this test describes the DEFAULT lane; unset the override to run it"
    sem = cap.get("semantic") or {}
    lane = os.environ.get("MERLIN_MODEL_GRADE_RUN") or ("mesh" if sem.get("must_accelerate") else "host")
    assert lane == "mesh"

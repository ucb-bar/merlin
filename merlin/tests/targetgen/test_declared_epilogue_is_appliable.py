"""No capsule may declare an epilogue the target's readout does not apply at the width it commits.

MEASURED TWICE, and the second time is why this file scans the INTERFACE rather than the YAML.

First: twelve capsules declared ``relu`` while committing ``i32``. On this target the activation exists
only on the narrowing readout; the full-width readout writes the raw accumulator and applies nothing.
The grade-time check (``merlin.verify.epilogue_applicability``) refused them one at a time, hours into
an agent run, having read the very declaration the capsule writer never consulted.

Second, after that was fixed: four more slipped through, because the first audit keyed on the capsule
YAML's top-level ``operation.attributes.epilogue``. A ``resident_reuse`` capsule carries its stages
INSIDE ``attributes.matmuls``, and two hand-authored capsules had a corrected YAML beside an
interface that still said ``i32``. The compiler reads the interface, and so does the grade. So this
scans every ``epilogue = [...]`` / ``output_dtype`` pair in every emitted interface, which is the one
place the question is always askable however the capsule is shaped.

Parsed structurally with ``str.split`` -- no regex, per the repo's derive-don't-overfit rule.
"""

from __future__ import annotations

import pathlib

import pytest
import selected_driver

from merlin.common.paths import repo_root
from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.targetgen.readout_facet import epilogue_readouts, epilogue_stage_routes
from merlin.verify.epilogue_applicability import assess, selectors_applying

pytestmark = pytest.mark.target("gemmini")

CORPUS = repo_root() / "merlin/contract/capsules"

#: Targets whose corpora live under this root but whose readouts are their own. A capsule for another
#: target must never be judged against this one's declaration.
OTHER_TARGET_DIRS = ("/radiance/", "/atlas/", "/saturn_opu/")


def _commit_sites(path: pathlib.Path):
    """``(stages, committed_dtype)`` for every commit in one interface that declares an epilogue."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if "epilogue = [" not in line or "output_dtype" not in line:
            continue
        inner = line.split("epilogue = [", 1)[1].split("]", 1)[0]
        stages = [t.strip().strip('"') for t in inner.split(",") if t.strip()]
        if not stages:
            continue
        yield stages, line.split('output_dtype = "', 1)[1].split('"', 1)[0]


def _gemmini_interfaces():
    for f in sorted(CORPUS.rglob("capsule.interface.mlir")):
        if any(d in str(f) for d in OTHER_TARGET_DIRS):
            continue
        yield f


@selected_driver.requires_support("gemmini")
def test_every_declared_epilogue_is_applied_by_the_readout_it_commits_at():
    readouts = epilogue_readouts("gemmini")
    routes = epilogue_stage_routes("gemmini")
    assert readouts, "gemmini declares no readouts; this check would be vacuous"

    violations, scanned = [], 0
    for f in _gemmini_interfaces():
        sites = list(_commit_sites(f))
        if not sites:
            continue
        scanned += len(sites)
        verdict = assess(parse_interface_mlir(f.read_text(encoding="utf-8")), readouts, routes=routes)
        if verdict.refusing:
            violations.append((f.parent.name, verdict.status, [s.stage for s in verdict.discarded]))

    # Commit sites carrying a NON-EMPTY epilogue. Measured at 40 for the gemmini corpus; the floor
    # guards against a parse change silently scanning nothing and reporting a clean sweep.
    assert scanned >= 30, f"only {scanned} epilogue-bearing commit sites scanned; the parse changed shape"
    assert not violations, (
        "these capsules declare an epilogue that neither readout nor a witnessed, composition-scoped "
        f"route applies — the grade will refuse them as protocol violations: {violations}"
    )


@selected_driver.requires_support("gemmini")
@pytest.mark.parametrize("stage", ["relu", "acc_scale", "bias_add", "maxpool"])
def test_the_target_applies_the_stages_its_requirement_demands(stage):
    """The requirement and the readout declaration must not drift apart again.

    The conformance spec's epilogue axis is intersected with this declaration, so a stage it requires
    must be one some readout applies. If this fails, the spec is asking for something unbuildable.
    """
    readouts = epilogue_readouts("gemmini")
    routes = epilogue_stage_routes("gemmini")
    assert selectors_applying(readouts, [stage], routes=routes, composition="contraction"), (
        f"the requirement demands a fused {stage!r} but no readout or contraction route applies it"
    )


@selected_driver.requires_support("gemmini")
def test_bias_route_selects_a_contraction_commit_width_without_narrow_readout_bias():
    from merlin.targetgen.corpus_spec import CorpusBinding, _resolve_output_dtype, build_matmul

    binding = CorpusBinding(
        target="gemmini",
        tile_dim=16,
        operand_dtype="int8",
        accum_dtype="i32",
        integer=True,
        tiers=[],
        compare="exact_int",
    )
    roles = frozenset({"bias"})
    assert _resolve_output_dtype(binding, ["bias_add"], {}, available_operand_roles=roles) == "i8"
    assert _resolve_output_dtype(binding, ["bias_add"], {"output_dtype": "i32"}, available_operand_roles=roles) == "i32"
    assert _resolve_output_dtype(binding, ["bias_add", "relu"], {}, available_operand_roles=roles) == "i8"
    with pytest.raises(ValueError, match="no readout or contraction route"):
        _resolve_output_dtype(binding, ["bias_add"], {})
    with pytest.raises(ValueError, match="no readout or contraction route"):
        _resolve_output_dtype(binding, ["relu", "bias_add"], {}, available_operand_roles=roles)
    assert not selectors_applying(epilogue_readouts("gemmini"), ["bias_add"])
    capsule, interface = build_matmul(
        {
            "name": "derived_bias_probe",
            "kind": "layer",
            "source_role": "derived_sweep",
            "source_reference": "contraction stage route",
            "op": "matmul",
            "epilogue": ["bias_add"],
            "M": 16,
            "K": 16,
            "N": 16,
        },
        binding,
    )
    [bias] = [row for row in capsule["inputs"] if row["role"] == "bias"]
    assert bias["dtype"] == "i32" and bias["shape"] == [16]
    cb = parse_interface_mlir(interface)
    commit = next(command for command in cb["commands"] if command["opcode"] == "COMMIT")
    assert commit["attributes"]["bias"] == bias["name"]
    assert assess(cb, epilogue_readouts("gemmini"), routes=epilogue_stage_routes("gemmini")).status == "applied"


@selected_driver.requires_support("gemmini")
def test_captured_readout_facet_keeps_bias_outside_readout_applies():
    from merlin.targetgen.readout_facet import capture_inputs, derive

    inputs = capture_inputs("gemmini", facts={}, include_taxonomy=False)
    facet = derive("gemmini", facts={}, readouts=inputs["readouts"], stage_routes=inputs["stage_routes"])
    record = facet.to_dict()
    assert all("bias_add" not in row["applies"] for row in record["readouts"])
    assert any("bias_add" in row["stages"] for row in record["stage_routes"])


def test_no_capsule_declares_its_output_dtype_twice_in_disagreement():
    """A capsule may state its output element type twice; the two must agree.

    The interface scan above cannot see this: it reads the emitted MLIR, while `numeric_policy.dtype`
    lives only in the YAML. MEASURED — correcting two hand-authored capsules' attributes and interface
    to i8 left `numeric_policy.dtype: i32` behind, and the runner refused both as RUNNER_CRASH mid-run.
    Refusing is right (the two size the same DRAM slot, so one would silently mis-size it); being
    refused for the first time inside a graded run is not.
    """
    import yaml

    from merlin.targetgen.capsule_dram import declared_output_dtype

    unresolved, checked = [], 0
    for f in sorted(CORPUS.rglob("capsule.yaml")):
        try:
            doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 -- a malformed capsule is another test's business
            continue
        outs = [o.get("name") for o in (doc.get("outputs") or []) if o.get("name")]
        out = (doc.get("operation") or {}).get("attributes", {}).get("out") or (outs[0] if outs else "Y0")
        checked += 1
        try:
            declared_output_dtype(doc, out)
        except Exception as exc:  # noqa: BLE001 -- the refusal IS the finding
            unresolved.append((f.parent.name, f"{type(exc).__name__}: {exc}"[:160]))

    assert checked > 300, f"only {checked} capsules checked; the corpus or the walk changed shape"
    assert not unresolved, f"these capsules cannot resolve their output dtype: {unresolved}"

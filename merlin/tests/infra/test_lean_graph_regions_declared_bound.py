"""Additional ABI/full-effect/actual-after-buffer qualification over frozen v1."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import artifacts_dir, build_dir
from merlin.verify import lean_graph_regions as base
from merlin.verify import lean_graph_regions_declared_bound as bound
from merlin.verify.lean_graph_regions_declared_check import save
from merlin.verify.lean_graph_regions_declared_verify import audit, oracle


@pytest.fixture(scope="module")
def run():
    name = os.environ.get("MERLIN_GRAPH_DECLARED_BOUND_RUN", "bound-tests-development-v1")
    root = artifacts_dir() / "verification/graph-regions-declared" / name
    root.mkdir(parents=True, exist_ok=False)
    lean, environment, identity = bound.build_bound(build_dir() / "graph-regions-declared" / name)
    prior = artifacts_dir() / "verification/graph-regions-declared/qualification-v1-originals"
    fixed = json.loads((prior / "FIXED-ORACLES.json").read_text())
    return dict(
        root=root,
        lean=lean,
        environment=environment,
        identity_file=build_dir() / "graph-regions-declared" / name / "build/IDENTITY.json",
        prior=prior,
        fixed=fixed,
        receipts={},
        failed={},
    )


def positive(run, label):
    if label in run["failed"]:
        pytest.fail("retained first positive failure; no retry: " + run["failed"][label])
    if label not in run["receipts"]:
        try:
            run["receipts"][label] = bound.verify_bound(
                run["prior"] / ("actual-" + label),
                run["root"] / ("verified-" + label),
                identity_file=run["identity_file"],
                fixed_oracles=run["fixed"],
            )
        except Exception as error:
            run["failed"][label] = repr(error)
            raise
    return run["receipts"][label]


def query(run, label, extra, names):
    receipt = positive(run, "i8")
    destination = run["root"] / label
    destination.mkdir()
    proof = destination / "Check.lean"
    proof.write_text(base.fresh_ref(receipt["proof"]).read_text() + extra)
    result = subprocess.run(
        [str(run["lean"]), "-j1", str(proof.resolve())],
        env=run["environment"],
        capture_output=True,
        text=True,
        timeout=90,
    )
    save(destination / "process.json", dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    assert result.returncode == 0, result.stdout
    return audit(result.stdout, bound.ROOTS + names)


@pytest.mark.parametrize("label", ["i8", "i16shared"])
def test_complete_stronger_boundary(run, label):
    assert bound.validate_bound(positive(run, label))


@pytest.mark.parametrize(
    "field",
    [
        "read_flag",
        "write_flag",
        "copy_flag",
        "readout",
        "stages",
        "waits",
        "token",
        "encoding",
        "byte_length",
        "byte_order",
    ],
)
def test_full_effect_metadata_refused(run, field):
    change = {
        "read_flag": "accesses := [⟨bound0.reads[0]?.getD bound0.output,true⟩,"
        "⟨bound0.reads[1]?.getD bound0.output,false⟩,⟨bound0.reads[2]?.getD bound0.output,false⟩,"
        "⟨bound0.output,true⟩]",
        "write_flag": "accesses := completeEffect0.accesses.map (fun access => {access with writes := false})",
        "copy_flag": "exactCopy := true",
        "readout": 'readout := some "extra readout"',
        "stages": 'stages := ["wrong stage"]',
        "waits": 'waits := ["unexpected wait"]',
        "token": 'token := some "async token"',
        "encoding": 'scalarEncoding := "unsupported"',
        "byte_length": "scalarLength := completeEffect0.scalarLength + 1",
        "byte_order": 'byteOrders := ["big","little","little","little"]',
    }[field]
    extra = (
        f"\ndef badEffect : CompleteEffect := {{completeEffect0 with {change}}}\n"
        + """
theorem full_effect_refused : checkCompleteEffect 141 bound0 badEffect afterAdapterBytes0 = false := by decide
#print axioms full_effect_refused
"""
    )
    if field == "byte_length":
        extra = extra.replace(
            "= false := by decide",
            """= false := by
  apply Bool.eq_false_iff.mpr
  intro accepted
  have wrong := (checkCompleteEffect_iff.mp accepted).2.2.2.2.2.2.1
  have original := (checkCompleteEffect_iff.mp actual_effect0).2.2.2.2.2.2.1
  change completeEffect0.scalarLength + 1 = completeEffect0.scalarBytes.length at wrong
  have impossible := wrong.trans original.symm
  exact (Nat.ne_of_gt (Nat.lt_succ_self _)) impossible""",
        )
    query(run, "effect-" + field, extra, ["full_effect_refused"])


@pytest.mark.parametrize("kind", ["inputs", "outputs"])
def test_independent_original_windows_refused(run, kind):
    field = "1" if kind == "inputs" else "2"
    original_list = "actualFused.inputs" if kind == "inputs" else "actualFused.outputs"
    extra = (
        f"\ndef badBindings : ExternalBindings := {{originalBindings with {kind} := originalBindings.{kind}.reverse}}\n"
    )
    extra += f"""
theorem original_windows_refused : checkOriginalBindings
    ((actualGraph.nodes.drop 4).take 3) (actualGraph.nodes.drop 7) actualGraph.observations
    addresses [3,47] stage0.program.output originalKernel.arguments badBindings = false := by
  apply Bool.eq_false_iff.mpr
  intro accepted
  have facts := checkOriginalBindings_sound accepted
  have checked := actual_realization
  simp only [checkRealization, checkFusion, Bool.and_eq_true, decide_eq_true_eq] at checked
  rw [← checked.2.1.2, ← checked.2.2] at facts
  have different : {original_list}.map addresses ≠ badBindings.{kind} := by decide
  exact different facts.{field}
#print axioms original_windows_refused
"""
    query(run, "original-" + kind, extra, ["original_windows_refused"])


def test_coherent_wrong_after_scalar_refused(run):
    from merlin.verify import lean_graph_regions_declared_ast as ast

    receipt = positive(run, "i8")
    original = base.fresh_ref(receipt["after_buffers"][0])
    mutant = run["root"] / "coherent-after-scalar.mlir"
    mutant.write_text(original.read_text().replace("%r=arith.addi", "%r=arith.subi"))
    base_receipt = json.loads(base.fresh_ref(receipt["base"]).read_text())
    identity = json.loads(base.fresh_ref(base_receipt["identity"]).read_text())
    oracle(
        "mlir",
        mutant,
        run["root"] / "coherent-after-scalar-oracle",
        python=Path(identity["context"]["oracle_invocation"]),
        fixed=run["fixed"]["mlir-scalar"],
    )
    document = json.loads((run["root"] / "coherent-after-scalar-oracle/export.json").read_text())
    extra = (
        f"\ndef changedAfter : Module := {ast.module(document)}\n"
        + """
theorem changed_after_refused : checkScalar scalar0_source changedAfter
    "source" "forward" none none program0 = false := by
  apply checkScalar_refuse_trace
  decide
#print axioms changed_after_refused
"""
    )
    query(run, "coherent-after-scalar-lean", extra, ["changed_after_refused"])


@pytest.mark.parametrize("kind", ["swapped", "missing", "substituted", "rehashed_proof"])
def test_strong_receipt_associations(run, kind):
    receipt = copy.deepcopy(positive(run, "i8"))
    if kind == "swapped":
        receipt["after_buffers"][1], receipt["after_buffers"][2] = (
            receipt["after_buffers"][2],
            receipt["after_buffers"][1],
        )
    elif kind == "missing":
        receipt["after_buffers"][0]["path"] = str((run["root"] / "missing-after.mlir").resolve())
    elif kind == "substituted":
        path = run["root"] / "substituted-after.mlir"
        path.write_bytes(base.fresh_ref(receipt["after_buffers"][1]).read_bytes())
        receipt["after_buffers"][0] = base.file_ref(path)
    else:
        other = positive(run, "i16shared")
        for field in ("proof", "process", "audit"):
            receipt[field] = other[field]
    save(run["root"] / (kind + "-rehashed-receipt.json"), receipt)
    with pytest.raises((ValueError, FileNotFoundError)):
        bound.validate_bound(receipt)


def test_new_soundness_roots(run):
    extra = "\n#print axioms Merlin.GraphRegionsDeclared.checkOriginalBindings_sound\n"
    extra += "#print axioms Merlin.GraphRegionsDeclared.checkCompleteEffect_sound\n"
    query(
        run,
        "new-exact-root-audit",
        extra,
        [
            "Merlin.GraphRegionsDeclared.checkOriginalBindings_sound",
            "Merlin.GraphRegionsDeclared.checkCompleteEffect_sound",
        ],
    )

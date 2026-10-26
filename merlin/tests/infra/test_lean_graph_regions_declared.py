"""Adversarial actual-buffer tests for the additive declared-call graph leaves.

Development runs retain every query. No native numerical/contract approval follows
from this suite. A separately frozen run is needed for a qualification label.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import artifacts_dir, build_dir
from merlin.verify import lean_graph_regions as base
from merlin.verify import lean_graph_regions_declared_ast as ast
from merlin.verify.lean_graph_regions_declared_capture import capture
from merlin.verify.lean_graph_regions_declared_check import save, scalar_pair
from merlin.verify.lean_graph_regions_declared_context import build, select
from merlin.verify.lean_graph_regions_declared_packet import raw_kernel
from merlin.verify.lean_graph_regions_declared_verify import (
    ROOTS,
    audit,
    oracle,
    validate,
    validate_associations,
    verify,
)


@pytest.fixture(scope="module")
def run():
    name = os.environ.get("MERLIN_GRAPH_DECLARED_RUN", "development-tests-v1")
    root = artifacts_dir() / "verification/graph-regions-declared" / name
    root.mkdir(parents=True, exist_ok=False)
    selected = select(root / "context")
    output = build_dir() / "graph-regions-declared" / name
    lean, env, identity = build(
        output,
        selected,
        [
            "Merlin.GraphRegionsDeclaredBridge",
            "Merlin.GraphRegionsDeclaredRaw",
            "Merlin.GraphRegionsDeclaredProjection",
            "Merlin.GraphRegionsDeclaredCountermodels",
        ],
    )
    reference = Path("/home/jeremy/Projects/merlin-graph-proof-intake")
    captures = {}
    for label, shape, width, shared in (("i8", (3, 47), 1, False), ("i16shared", (2, 23), 2, True)):
        destination = root / ("actual-" + label)
        capture(reference, destination, shape=shape, width=width, shared=shared, physical=True)
        captures[label] = destination
    fixed = {}
    for profile, mode, python, source in (
        ("mlir-full", "mlir", Path(selected["oracle_invocation"]), captures["i8"] / "source.mlir"),
        ("mlir-scalar", "mlir", Path(selected["oracle_invocation"]), captures["i8"] / "stage-0-source.mlir"),
        ("kernel", "kernel", Path(os.sys.executable), captures["i8"] / "actual-output-kernel.json"),
    ):
        fixed[profile] = oracle(mode, source, root / ("host-intake-" + profile), python=python, reference=reference)[
            "host"
        ]
    save(root / "FIXED-ORACLES.json", fixed)
    return dict(
        root=root,
        captures=captures,
        selected=selected,
        lean=lean,
        environment=env,
        identity_file=output / "IDENTITY.json",
        fixed=fixed,
        receipts={},
        failed_positives={},
    )


def compile_query(run, label, text, roots):
    destination = run["root"] / label
    destination.mkdir()
    proof = destination / "Check.lean"
    proof.write_text(text)
    result = subprocess.run(
        [str(run["lean"]), "-j1", str(proof.resolve())],
        env=run["environment"],
        capture_output=True,
        text=True,
        timeout=90,
    )
    save(destination / "process.json", dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    assert result.returncode == 0, result.stdout
    return audit(result.stdout, roots)


@pytest.mark.parametrize("label", ["i8", "i16shared"])
def test_complete_actual_boundary(run, label):
    receipt = positive(run, label)
    assert validate(receipt)


def positive(run, label):
    if label in run["failed_positives"]:
        pytest.fail("retained earlier positive failure; not rerun: " + run["failed_positives"][label])
    if label not in run["receipts"]:
        try:
            run["receipts"][label] = verify(
                run["captures"][label],
                run["root"] / ("verified-" + label),
                identity_file=run["identity_file"],
                fixed_oracles=run["fixed"],
            )
        except Exception as error:
            run["failed_positives"][label] = repr(error)
            raise
    return run["receipts"][label]


SCALAR_MUTATIONS = {
    "wrong_operand": ("%p,%s", "%p,%x"),
    "swapped_operands": ("%x,%w", "%w,%x"),
    "wrong_return": ("func.return %r", "func.return %p"),
    "wrong_opcode": ("%r=arith.addi", "%r=arith.subi"),
    "overflow_flag": ("%r=arith.addi %p,%s:i8", "%r=arith.addi %p,%s overflow<nsw>:i8"),
    "hidden_operation": ("func.return %r", "%unused=arith.addi %p,%s:i8\nfunc.return %r"),
    "unsupported_min": ("%p=arith.muli", "%p=arith.minsi"),
}


@pytest.mark.parametrize("label", list(SCALAR_MUTATIONS))
def test_compiler_valid_scalar_refusal(run, label):
    old, new = SCALAR_MUTATIONS[label]
    original = run["captures"]["i8"] / "stage-0-selected.mlir"
    assert old in original.read_text()
    mutant = run["root"] / (label + ".mlir")
    mutant.write_text(original.read_text().replace(old, new))
    receipt = scalar_pair(
        run["captures"]["i8"] / "stage-0-source.mlir",
        mutant,
        run["root"] / ("scalar-" + label),
        selected=run["selected"],
        lean=run["lean"],
        environment=run["environment"],
        expected=False,
    )
    process = json.loads(Path(receipt["process"]["path"]).read_text())
    audit(process["stdout"], ["actual_acceptance"])


@pytest.mark.parametrize("label", ["sub", "constant", "trunc", "signed", "unsigned"])
def test_independent_scalar_variants(run, label):
    bodies = {
        "sub": ("%a:i8,%b:i8", "i8", "%r=arith.subi %a,%b:i8"),
        "constant": ("%a:i16", "i16", "%c=arith.constant 6:i16\n%r=arith.addi %a,%c:i16"),
        "trunc": ("%a:i32", "i8", "%r=arith.trunci %a:i32 to i8"),
        "signed": ("%a:i8", "i16", "%r=arith.extsi %a:i8 to i16"),
        "unsigned": ("%a:i8", "i16", "%r=arith.extui %a:i8 to i16"),
    }
    arguments, result, body = bodies[label]
    # Independently retained buffers; only their literal structural relation is claimed.
    source = run["root"] / (label + "-source.mlir")
    selected = run["root"] / (label + "-selected.mlir")
    source.write_text(f"module {{func.func @source({arguments})->{result} {{{body}\nfunc.return %r:{result}}}}}")
    selected.write_text(
        f"builtin.module {{func.func @forward({arguments})->{result} {{\n{body}\nfunc.return %r:{result}\n}}}}"
    )
    scalar_pair(
        source,
        selected,
        run["root"] / ("variant-" + label),
        selected=run["selected"],
        lean=run["lean"],
        environment=run["environment"],
    )


@pytest.mark.parametrize(
    "label",
    [
        "wrong_source",
        "wrong_output",
        "stale_offset",
        "wrong_extent",
        "wrong_width",
        "missing_second",
        "swapped_outputs",
        "extra_call",
        "wrong_role",
    ],
)
def test_structurally_valid_kernel_refusal(run, label):
    original = json.loads((run["captures"]["i8"] / "actual-output-kernel.json").read_text())
    mutant = copy.deepcopy(original)

    def argument(call, key, value):
        for name, field in mutant["body"][call]["args"]:
            if name == key:
                field["value"] = value

    if label == "wrong_source":
        argument(0, "a", 3)
    elif label == "wrong_output":
        argument(1, "dst", 7)
    elif label == "stale_offset":
        argument(2, "ao", 1024)
    elif label == "wrong_extent":
        argument(0, "n", 140)
    elif label == "wrong_width":
        argument(0, "width", 2)
    elif label == "missing_second":
        mutant["body"].pop()
    elif label == "swapped_outputs":
        argument(1, "dst", 7)
        argument(2, "dst", 6)
    elif label == "extra_call":
        mutant["body"].append(copy.deepcopy(mutant["body"][-1]))
    else:
        mutant["body"][1]["instr"] = "multiply"
    path = run["root"] / ("kernel-" + label + ".json")
    save(path, mutant)
    oracle(
        "kernel",
        path,
        run["root"] / ("kernel-" + label + "-oracle"),
        python=Path(os.sys.executable),
        reference=Path("/home/jeremy/Projects/merlin-graph-proof-intake"),
        fixed=run["fixed"]["kernel"],
    )
    base_query = Path(positive(run, "i8")["proof"]["path"]).read_text()
    text = (
        base_query
        + f"\ndef mutatedKernel : RawKernel := {raw_kernel(mutant)}\n"
        + """
theorem actual_refusal : checkKernel mutatedKernel originalKernel.name originalKernel.arguments
  [schema0,schema1,schema2] [bound0,bound1,bound2] = false := by decide
#print axioms actual_refusal
"""
    )
    compile_query(
        run,
        "kernel-" + label + "-lean",
        text,
        ROOTS + ["actual_refusal"],
    )


@pytest.mark.parametrize("label", ["substitute", "swap", "missing", "stale", "bool_size"])
def test_original_snapshot_associations(run, label):
    root = run["root"] / ("association-" + label)
    root.mkdir()
    first = root / "first"
    first.write_bytes(b"before")
    second = root / "second"
    second.write_bytes(b"after changed")
    original = {"before": base.file_ref(first)}
    snapshot = {"before": base.file_ref(first)}
    assert validate_associations(original, snapshot) is None
    if label == "substitute":
        original["before"] = base.file_ref(second)
    elif label == "swap":
        snapshot["before"] = base.file_ref(second)
    elif label == "missing":
        original["before"]["path"] = str((root / "absent").resolve())
    elif label == "stale":
        first.write_bytes(b"stale")
    else:
        original["before"]["bytes"] = True
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_associations(original, snapshot)


def test_exact_root_audit_refuses_added_axiom():
    with pytest.raises(ValueError, match="non-standard"):
        audit("'root' depends on axioms: [sorryAx]", ["root"])


def test_serializer_rejects_dropped_operation_field():
    with pytest.raises(ValueError, match="seven-field"):
        ast.operation({"name": "arith.addi"})


@pytest.mark.parametrize("label", ["byte_order", "shape", "role", "element_width", "completion"])
def test_bound_effect_refusal(run, label):
    base_query = Path(positive(run, "i8")["proof"]["path"]).read_text()
    replacement = {
        "byte_order": 'byteOrders := ["big","little","little","little"]',
        "shape": "shape := [141,1]",
        "role": 'role := "wrong same-typed role"',
        "element_width": "elementBits := 16",
        "completion": 'completion := "asynchronous"',
    }[label]
    text = (
        base_query
        + f"\ndef mutatedEffect : BoundCall := {{ bound0 with {replacement} }}\n"
        + """
theorem effect_refused : checkCall 141 stage0 addresses mutatedEffect bound0 = false := by decide
#print axioms effect_refused
"""
    )
    compile_query(run, "bound-" + label, text, ROOTS + ["effect_refused"])


@pytest.mark.parametrize("label", ["returned_observer", "suffix_operand", "nested_suffix"])
def test_compiler_valid_observer_refusal(run, label):
    from merlin.verify.lean_graph_regions_declared_check import export
    from merlin.verify.lean_graph_regions_mlir import compiler_verify

    source = run["captures"]["i8"] / "source.mlir"
    text = source.read_text()
    old, new = {
        "returned_observer": ('"func.return"(%c0, %c1', '"func.return"(%c1, %c0'),
        "suffix_operand": ('"linalg.generic"(%c0, %c1, %position', '"linalg.generic"(%c0, %c0, %position'),
        "nested_suffix": ('"arith.addi"(%a, %b)', '"arith.addi"(%a, %a)'),
    }[label]
    assert old in text
    mutated = run["root"] / (label + ".mlir")
    mutated.write_text(text.replace(old, new))
    validity = compiler_verify(
        mutated, run["root"] / (label + "-compiler"), python=Path(run["selected"]["oracle_invocation"])
    )
    assert validity["oracle"]["valid"]
    document, _ = export(mutated, run["root"] / (label + ".json"), Path(run["selected"]["oracle_invocation"]))
    base_query = Path(positive(run, "i8")["proof"]["path"]).read_text()
    text = (
        base_query
        + f"\ndef mutatedRetained : Module := {ast.module(document)}\n"
        + """
theorem observer_refused : ¬SourceBinding actualBefore mutatedRetained sites actualGraph := by
  intro accepted
  have traced := congrArg moduleRefusalTrace accepted.1
  have different : moduleRefusalTrace actualBefore ≠ moduleRefusalTrace mutatedRetained := by decide
  exact different traced
#print axioms observer_refused
"""
    )
    compile_query(run, "observer-" + label, text, ROOTS + ["observer_refused"])


@pytest.mark.parametrize("label", ["constant", "cast_kind", "cast_type"])
def test_compiler_valid_constant_cast_refusal(run, label):
    if label == "constant":
        source = (
            "module {func.func @source(%x:i16)->i16 {%c=arith.constant 6:i16\n"
            "%r=arith.addi %x,%c:i16\nfunc.return %r:i16}}"
        )
        candidate = (
            "module {func.func @forward(%x:i16)->i16 {%c=arith.constant 7:i16\n"
            "%r=arith.addi %x,%c:i16\nfunc.return %r:i16}}"
        )
    elif label == "cast_kind":
        source = "module {func.func @source(%x:i8)->i16 {%r=arith.extsi %x:i8 to i16\nfunc.return %r:i16}}"
        candidate = "module {func.func @forward(%x:i8)->i16 {%r=arith.extui %x:i8 to i16\nfunc.return %r:i16}}"
    else:
        source = "module {func.func @source(%x:i32)->i8 {%r=arith.trunci %x:i32 to i8\nfunc.return %r:i8}}"
        candidate = "module {func.func @forward(%x:i32)->i16 {%r=arith.trunci %x:i32 to i16\nfunc.return %r:i16}}"
    original = run["root"] / (label + "-refusal-source.mlir")
    mutant = run["root"] / (label + "-refusal-candidate.mlir")
    original.write_text(source)
    mutant.write_text(candidate)
    scalar_pair(
        original,
        mutant,
        run["root"] / ("refused-" + label),
        selected=run["selected"],
        lean=run["lean"],
        environment=run["environment"],
        expected=False,
    )


@pytest.mark.parametrize(
    "label", ["substituted_original", "swapped_originals", "missing_original", "rehashed_other_proof"]
)
def test_complete_receipt_association_attacks(run, label):
    receipt = copy.deepcopy(positive(run, "i8"))
    if label == "substituted_original":
        path = run["root"] / "substituted-valid-original.mlir"
        path.write_bytes((run["captures"]["i8"] / "stage-0-selected.mlir").read_bytes())
        receipt["originals"]["source.mlir"] = base.file_ref(path)
    elif label == "swapped_originals":
        a, b = "stage-0-source.mlir", "stage-0-selected.mlir"
        receipt["originals"][a], receipt["originals"][b] = receipt["originals"][b], receipt["originals"][a]
    elif label == "missing_original":
        receipt["originals"]["source.mlir"]["path"] = str((run["root"] / "missing-original.mlir").resolve())
    else:
        other = positive(run, "i16shared")
        for field in ("proof", "process", "model", "audit"):
            receipt[field] = other[field]
    save(run["root"] / (label + "-rehashed-receipt.json"), receipt)
    with pytest.raises((ValueError, FileNotFoundError)):
        validate(receipt)


def test_per_instruction_countermodels_and_roots(run):
    names = [
        "checkScalar_sound",
        "checkScalar_refuse_trace",
        "generic_execution",
        "source_binding_retained",
        "checkRawCall_sound",
        "checkKernel_sound",
        "instruction_refines",
        "sequence_refines",
        "derived_local_contract",
        "checked_calls_preserve_observers",
        "nofault_premise_required",
        "completion_premise_required",
        "alias_changes_consumer",
        "async_changes_consumer",
        "private_observer_required",
        "suppressed_transfer_fault_required",
    ]
    imports = "import Merlin.GraphRegionsDeclaredBridge\nimport Merlin.GraphRegionsDeclaredRaw\n"
    imports += "import Merlin.GraphRegionsDeclaredProjection\nimport Merlin.GraphRegionsDeclaredCountermodels\n"
    qualified = ["Merlin.GraphRegionsDeclared." + name for name in names]
    compile_query(
        run, "exact-root-audit", imports + "\n".join("#print axioms " + name for name in qualified), qualified
    )

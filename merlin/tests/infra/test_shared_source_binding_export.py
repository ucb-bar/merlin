"""Explicit V2 compiler helpers execute without private core/experiment imports."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.compile.scheduling.direct_source import bind_direct_source_contraction
from merlin.compile.scheduling.export import (
    PUBLIC_MODULES,
    SOURCE_BINDING_MODULES,
    export_shared_scheduling,
)
from merlin.compile.scheduling.library_sources import LIBRARY_MODULES
from merlin.compile.scheduling.source_tokens import llvm_tokens
from merlin.targetgen.compiler_library import CompilerLibraryError, CompilerLibrarySelection, freeze_compiler_library
from merlin.targetgen.package_runtime import CertFailure, integrity_scan
from merlin.targetgen.tool_profile_shared import _release_inventory, reviewed_shared_manifest

SOURCE = """module {
func.func @forward(%a: tensor<5x4xi8>, %b: tensor<4x5xi8>) -> tensor<5x5xi32> {
%z = arith.constant 0 : i32
%s = tensor.splat %z : tensor<5x5xi32>
%r = linalg.generic {
indexing_maps = [affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
iterator_types = ["parallel","parallel","reduction"]
} ins(%a,%b : tensor<5x4xi8>,tensor<4x5xi8>) outs(%s : tensor<5x5xi32>) {
^bb0(%x:i8,%y:i8,%acc:i32):
%u = arith.extsi %x : i8 to i32
%v = arith.extsi %y : i8 to i32
%p = arith.muli %u,%v : i32
%sum = arith.addi %acc,%p : i32
linalg.yield %sum : i32
} -> tensor<5x5xi32>
func.return %r : tensor<5x5xi32>
}}
"""

CANDIDATE = """from merlin.compile.scheduling.direct_source import bind_direct_source_contraction
from merlin.compile.scheduling.source_tokens import llvm_tokens
from merlin.compile.scheduling.staged_dense import (AccumulatorReadout, AccumulatorZero, CompletionFence,
    ComputeConsume, DescriptorIssue, OperandLoad, StagedDenseContract, StagedDenseProblem,
    StagedDenseRewrite, order_output_tiles_n_before_m, eliminate_redundant_rhs_loads, rewrite_staged_dense)

def execute(source, cb, analysis):
    binding = bind_direct_source_contraction(source, cb, analysis)
    problem = StagedDenseProblem(binding['m'], binding['k'], binding['n'])
    contract = StagedDenseContract('neutral-facts', 'neutral-effects', 'raw-signed', 'neutral-control',
        4, 8, 4, 8, 18, 32, 'read_at_compute_consume', 'data_commands_preserve_pending_descriptor',
        'raw_accumulator_identity', True, True, False)
    before = []
    for m in range(0, problem.m, 4):
        for n in range(0, problem.n, 4):
            mr, nr = min(4, problem.m-m), min(4, problem.n-n)
            before.extend((AccumulatorZero(0,mr,nr), OperandLoad('lhs',m,0,0,mr,4),
                OperandLoad('rhs',0,n,4,4,nr), DescriptorIssue(4,0,mr,4,nr,True), CompletionFence(),
                ComputeConsume(0,mr,4), CompletionFence(), AccumulatorReadout(0,m,n,mr,nr), CompletionFence()))
    after, report = rewrite_staged_dense(tuple(before), problem=problem, contract=contract,
        rewrites=(StagedDenseRewrite('columns',order_output_tiles_n_before_m),
            StagedDenseRewrite('reuse',eliminate_redundant_rhs_loads)))
    return dict(binding=binding, removed_rhs_loads=len(before)-len(after),
        outputs=report['checks'][-1]['outputs'], products=report['checks'][-1]['products'],
        token_text=[token.text for token in llvm_tokens('%a = add i32 %b, 1 ; comment')])
"""


def inputs(tmp_path):
    path = tmp_path / "source.mlir"
    path.write_text(SOURCE)
    cb = dict(
        compiler_pointer_abi=dict(version=1, arguments=["a", "b", "out"]),
        tensors={
            "a": dict(shape=[5, 4], dtype="i8", role="input"),
            "b": dict(shape=[4, 5], dtype="i8", role="input"),
            "out": dict(shape=[5, 5], dtype="i32", role="output"),
        },
    )
    analysis = dict(
        interface_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        operations=[
            dict(ordinal=2, kind="multiply_accumulate", initial_interval=[0, 0], reduction_extent=4, result_bits=32)
        ],
    )
    return path, cb, analysis


def review(root, document):
    return freeze_compiler_library(
        root,
        review_id="explicit-neutral-source-binding-review",
        public_modules=tuple(document["public_modules"]),
        sources=tuple((row["path"], row["module"]) for row in document["members"]) + (("shared-library.json", None),),
    )


def test_v2_augments_only_exact_explicit_apis_without_changing_v1(tmp_path):
    first = export_shared_scheduling(tmp_path / "v1")
    second = export_shared_scheduling(tmp_path / "v2", source_binding=True)
    assert first["schema"] == "merlin.shared_scheduling_library.v1"
    assert first["public_modules"] == list(PUBLIC_MODULES)
    assert [row["module"] for row in first["members"]] == [*LIBRARY_MODULES, None, None]
    assert second["schema"] == "merlin.shared_scheduling_library.v2"
    assert _release_inventory(first) == (LIBRARY_MODULES, PUBLIC_MODULES)
    assert _release_inventory(second) == (
        (*LIBRARY_MODULES, *SOURCE_BINDING_MODULES),
        (*PUBLIC_MODULES, *SOURCE_BINDING_MODULES),
    )
    assert reviewed_shared_manifest(tmp_path / "v2") == second
    for row in first["members"]:
        assert (tmp_path / "v1" / row["path"]).read_bytes() == (tmp_path / "v2" / row["path"]).read_bytes()
    assert review(tmp_path / "v2", second).permits(SOURCE_BINDING_MODULES[0])
    assert not review(tmp_path / "v1", first).permits(SOURCE_BINDING_MODULES[0])


@pytest.mark.parametrize("change", ["protocol", "public", "extra", "schema"])
def test_versioned_inventory_cannot_upgrade_a_previous_or_foreign_grant(tmp_path, change):
    document = export_shared_scheduling(tmp_path / "v2", source_binding=True)
    if change == "protocol":
        document["source_binding_protocol"] = "foreign"
    elif change == "public":
        document["public_modules"].append("merlin.runtime.reference")
    elif change == "extra":
        document["members"].insert(-2, dict(module="merlin.private", path="merlin/private.py", sha256="a" * 64))
    else:
        document["schema"] = "merlin.shared_scheduling_library.v1"
    with pytest.raises(ValueError):
        _release_inventory(document)


@pytest.mark.parametrize(
    "fault", ["source", "unsigned", "nonzero", "shape", "analysis", "duplicate", "layout", "bool-version"]
)
def test_public_source_proposal_checks_actual_source_and_complete_pointer_shape(tmp_path, fault):
    path, cb, analysis = inputs(tmp_path)
    if fault == "source":
        path.write_text(SOURCE + "\n")
    elif fault in ("unsigned", "nonzero"):
        path.write_text(
            SOURCE.replace("arith.extsi", "arith.extui")
            if fault == "unsigned"
            else SOURCE.replace("arith.constant 0", "arith.constant 1")
        )
        analysis["interface_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif fault == "shape":
        cb["tensors"]["b"]["shape"] = [5, 4]
    elif fault == "analysis":
        analysis["operations"][0]["initial_interval"] = [1, 1]
    elif fault == "duplicate":
        analysis["operations"].append(copy.deepcopy(analysis["operations"][0]))
    elif fault == "layout":
        cb["tensors"]["b"]["layout"] = "transposed"
    else:
        cb["compiler_pointer_abi"]["version"] = True
    with pytest.raises(ValueError):
        bind_direct_source_contraction(path, cb, analysis)


def test_public_reader_is_only_compile_proposal_data_not_source_authority(tmp_path):
    path, cb, analysis = inputs(tmp_path)
    before = path.read_bytes(), copy.deepcopy(cb), copy.deepcopy(analysis)
    analysis["qualified"] = True  # Such caller data issues no owner or qualification.
    assert bind_direct_source_contraction(path, cb, analysis) == dict(
        ordinal=2, lhs="a", rhs="b", output="out", m=5, k=4, n=5
    )
    assert path.read_bytes() == before[0] and cb == before[1]
    analysis.pop("qualified")
    assert analysis == before[2]
    with pytest.raises(ValueError, match="one-MiB"):
        path.write_bytes(b" " * ((1 << 20) + 1))
        bind_direct_source_contraction(path, cb, analysis)


def test_lexer_preserves_exact_spans_and_rejects_unbounded_input():
    source = '%a = add i32 %b, 1 ; ignored\n@"quoted\\20symbol"'
    tokens = llvm_tokens(source)
    assert all(source[token.start : token.end] == token.text for token in tokens)
    assert tokens[-1].text == '@"quoted\\20symbol"'
    with pytest.raises(ValueError):
        llvm_tokens('"unterminated')
    with pytest.raises(ValueError):
        llvm_tokens("oversized", max_bytes=1)


@pytest.mark.parametrize("selected", [False, True])
def test_original_import_policy_and_isolated_interpreter_execute_only_selected_release(tmp_path, selected):
    path, cb, analysis = inputs(tmp_path)
    root = tmp_path / "sdk"
    document = export_shared_scheduling(root, source_binding=selected)
    library = review(root, document)
    selection = CompilerLibrarySelection(library, root)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "proposal.py").write_text(CANDIDATE)
    package = SimpleNamespace(directory=candidate, integrity_exempt=False)
    with pytest.raises(CertFailure):
        integrity_scan(package)
    if selected:
        integrity_scan(package, compiler_library=selection.contract, compiler_library_root=selection.root)
    else:
        with pytest.raises(CertFailure):
            integrity_scan(package, compiler_library=selection.contract, compiler_library_root=selection.root)
    for module in (
        "merlin.llvmlower.direct_source_contraction",
        "merlin.llvmlower.late_quant_rne",
        "merlin_experiments.phase0.connected_integer_bounds",
        "merlin.compile.scheduling.unreviewed",
    ):
        (candidate / "forbidden.py").write_text("import " + module + "\n")
        with pytest.raises(CertFailure):
            integrity_scan(package, compiler_library=selection.contract, compiler_library_root=selection.root)
    (candidate / "forbidden.py").unlink()
    config = tmp_path / "config.json"
    result = tmp_path / "result.json"
    dependencies = sorted(
        {str(Path(importlib.util.find_spec(name).origin).parent.parent) for name in ("xdsl", "numpy", "yaml", "z3")}
    )
    config.write_text(
        json.dumps(
            dict(
                bundle=str(root),
                dependencies=dependencies,
                selected=selected,
                source=str(path),
                command_buffer=cb,
                analysis=analysis,
                candidate=str(candidate / "proposal.py"),
                result=str(result),
            )
        )
    )
    worker = repo_root() / "merlin/tests/fixtures/shared_source_binding_worker.py"
    run = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(worker), str(config)], capture_output=True, text=True, timeout=30
    )
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(result.read_bytes())
    assert value["removed_rhs_loads"] == 2 if selected else value["unselected_import"] == "refused"
    selection.record()
    (root / "merlin/compile/scheduling/staged_dense.py").write_text("# changed after review\n")
    with pytest.raises(CompilerLibraryError, match="bytes changed"):
        selection.record()


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "tail",
        "unsigned",
        "noninteger",
        "extra_operation",
        "nonzero",
        "order",
        "extra_tensor",
        "analysis",
        "source_hash",
    ],
)
def test_proposal_matches_original_private_reader_on_closed_source_boundaries(tmp_path, fault):
    from merlin.llvmlower.direct_source_contraction import bind_direct_source_contraction as original

    path, cb, analysis = inputs(tmp_path)
    if fault == "tail":
        # This already-tailed source also exercises a partial K panel.
        path.write_text(SOURCE.replace("5x4", "5x3").replace("4x5", "3x5"))
        cb["tensors"]["a"]["shape"] = [5, 3]
        cb["tensors"]["b"]["shape"] = [3, 5]
        analysis["operations"][0]["reduction_extent"] = 3
    elif fault == "unsigned":
        path.write_text(SOURCE.replace("arith.extsi", "arith.extui"))
    elif fault == "noninteger":
        path.write_text(SOURCE.replace("i8", "f32"))
    elif fault == "extra_operation":
        path.write_text(SOURCE.replace("func.return %r", "%extra = arith.constant 1 : i32\nfunc.return %r"))
    elif fault == "nonzero":
        path.write_text(SOURCE.replace("arith.constant 0", "arith.constant 1"))
    elif fault == "order":
        cb["compiler_pointer_abi"]["arguments"].reverse()
    elif fault == "extra_tensor":
        cb["tensors"]["foreign"] = dict(cb["tensors"]["a"])
    elif fault == "analysis":
        analysis["operations"][0]["result_bits"] = 64
    analysis["interface_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if fault == "source_hash":
        analysis["interface_sha256"] = "0" * 64
    outcomes = []
    for read in (original, bind_direct_source_contraction):
        try:
            outcomes.append((True, read(path, copy.deepcopy(cb), copy.deepcopy(analysis))))
        except Exception:
            outcomes.append((False, None))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] is (fault in (None, "tail"))


def test_v2_profile_and_additional_selection_transport_stay_explicit(tmp_path):
    from merlin_experiments.phase2.compiler_library_state import restore_selection

    from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V6, attach_common_substrate, verify_tool_profile
    from merlin.targetgen.tool_profile_shared import prepare_shared_source_release

    root = tmp_path / "release"
    document = prepare_shared_source_release(
        root, python_source_root=repo_root() / "src", license_directory=repo_root(), source_binding=True
    )
    selected = dict(arm="raw_baseline", tools=[], allowed=[], denied=[], host_inputs=[])
    attach_common_substrate(selected, version=COMMON_SUBSTRATE_V6, python_source_root=root, host_index_bits=64)
    profile = verify_tool_profile(selected)
    assert profile["shared_scheduling_library"] == document
    assert set(SOURCE_BINDING_MODULES) <= set(profile["common_public_modules"])
    selection = CompilerLibrarySelection(review(root, document), root)
    restored = restore_selection(selection.record())
    assert restored.record() == selection.record()
    for module in (
        "merlin.llvmlower.direct_source_contraction",
        "merlin.llvmlower.late_quant_rne",
        "merlin_experiments.phase0.connected_integer_bounds",
    ):
        assert not restored.contract.permits(module)


@pytest.fixture(params=[False, True], ids=["default-v6", "source-binding-v2"])
def frozen_public_profile(tmp_path, monkeypatch, request):
    from merlin.targetgen.common_transforms import CommonTransformService
    from merlin.targetgen.sandbox import bwrap as BW
    from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V6, attach_common_substrate
    from merlin.targetgen.tool_profile_shared import prepare_shared_source_release

    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    root = tmp_path / "release"
    prepare_shared_source_release(
        root, python_source_root=repo_root() / "src", license_directory=repo_root(), source_binding=request.param
    )
    bundle = dict(arm="raw_baseline", tools=[], allowed=[], denied=[], host_inputs=[])
    attach_common_substrate(bundle, version=COMMON_SUBSTRATE_V6, python_source_root=root, host_index_bits=64)
    for row in bundle["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(root / row["path"].removeprefix("merlin/python/"))

    def materialize(selected, name):
        import yaml

        workspace = tmp_path / name
        archive = tmp_path / (name + ".yaml")
        archive.write_text(yaml.safe_dump(selected))
        BW.materialize_bundle_inputs(workspace, selected, repo=repo_root())
        owner = CommonTransformService(workspace=workspace, bundle_manifest=archive, repo=repo_root())
        owner._verify()
        return owner

    return root, bundle, materialize, request.param


def test_shared_profile_mounts_every_selected_public_source(frozen_public_profile):
    from merlin_experiments.phase1.feedback.compiler_profile_runtime import _public_python_roots

    from merlin.targetgen.sandbox import bwrap as BW

    root, bundle, materialize, selected = frozen_public_profile
    owner = materialize(bundle, "workspace")
    roots = _public_python_roots(owner)
    assert roots
    _, grants = BW._snapshot_grants(owner.workspace, owner.bundle, owner.repo)
    for module in SOURCE_BINDING_MODULES:
        path = root / (module.replace(".", "/") + ".py")
        matches = [(destination, source) for declared, destination, source in grants if declared == str(path)]
        assert len(matches) == int(selected)
        assert (module in owner.profile["common_public_modules"]) is selected


@pytest.mark.parametrize("fault", ["missing", "changed"])
def test_shared_profile_rejects_missing_or_changed_public_grant(frozen_public_profile, fault):
    from merlin_experiments.phase1.feedback.compiler_profile_runtime import _public_python_roots

    root, bundle, materialize, selected = frozen_public_profile
    module = SOURCE_BINDING_MODULES[0] if selected else "merlin.compile.scheduling.staged_dense"
    original = root / (module.replace(".", "/") + ".py")
    changed = copy.deepcopy(bundle)
    row = next(row for row in changed["allowed"] if row["path"] == str(original))
    if fault == "missing":
        changed["allowed"].remove(row)
    else:
        replacement = root.parent / "changed" / original.relative_to(root)
        replacement.parent.mkdir(parents=True)
        replacement.write_bytes(original.read_bytes() + b"\n# changed selected public source\n")
        row["path"] = str(replacement)
    owner = materialize(changed, fault)
    with pytest.raises(ValueError, match="unique pinned public destination|API bytes differ"):
        _public_python_roots(owner)


@pytest.mark.parametrize(
    "text",
    [
        "%x = add i32 %a, 1 ; ignored\n%x2 = sub i32 %x, 2",
        '@"quoted\\20symbol"(%a, i32 1)',
        'call void asm sideeffect "fence", "~{memory}"()',
        "%0 = getelementptr i8, ptr %arg0, i64 17",
        "",
    ],
)
def test_public_token_spans_match_original_private_lexer(text):
    from merlin.llvmlower.late_quant_rne import _tokens

    def spans(values):
        return [(token.text, token.start, token.end) for token in values]

    assert spans(llvm_tokens(text)) == spans(_tokens(text))

"""Pending rank-one qualification: raw buffers, fresh proofs and actual consumers.

These tests target the installed additive Lean bridge. Their SDK/native
qualification remains pending execution; host assertions establish no admission.
"""

import copy
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from buffer_rank_one_inputs import chain, mutate_target, source

from merlin.common.owned_process import run_owned
from merlin.common.paths import repo_root
from merlin.verify import buffer_reuse_gate as gate
from merlin.verify.buffer_reuse_bridge import reconstruct_chain
from merlin.verify.buffer_reuse_intake import ALLOCATOR, ROOT, digest, prepare, prepare_chain, revalidate


@pytest.mark.parametrize("tiled", [False, True], ids=["untiled3", "tiled4"])
@pytest.mark.parametrize("extent", [1, 3, 65536])
def test_rank_one_actual_producer_preparation_binds_complete_buffers(extent, tiled):
    buffers = chain(extent, tiled=tiled)
    candidate = prepare_chain(buffers, allocator=ALLOCATOR)
    assert candidate.source == buffers[0] and candidate.final == buffers[-1]
    assert json.loads(candidate.raw_chain_json) == list(buffers)
    assert len(buffers) == (4 if tiled else 3)
    assert ".initializeLine link0" in candidate.query
    assert (".tiledLine link0" if tiled else ".untiledLine link0") in candidate.query
    assert ": Epoch.UniformLine :=" in candidate.query
    assert ": Epoch.LineInitializerLink :=" in candidate.query
    if tiled:
        assert buffers[-1] != buffers[-2]
    hint = json.loads(candidate.hint_json)
    links = hint["adapter"]["initializer_links"]
    assert len(links) == 2 and all(row["extent"] == extent for row in links)
    assert all(row["config"]["program"]["shape"] == [extent] for row in links)
    assert all(row["config"]["program"]["outerPrefix"] == [] for row in links)
    assert all(len(row["config"]["program"]["ivs"]) == 1 for row in links)
    assert all("rows" not in row and "columns" not in row for row in links)
    assert revalidate(candidate) == {"fresh": True, "authority": False, "kernel_outcome": "UNOBSERVED"}


@pytest.mark.parametrize("rank,computed", [(3, False), (1, True)])
def test_other_rank_and_computed_initializer_profiles_remain_explicitly_unsupported(rank, computed):
    with pytest.raises(ValueError):
        prepare_chain(chain(rank=rank, computed=computed), allocator=ALLOCATOR)


def test_zero_extent_is_an_actual_storage_producer_refusal():
    with pytest.raises(ValueError, match="actual storage producer declined"):
        chain(0)


@pytest.mark.parametrize("tiled", [False, True], ids=["untiled3", "tiled4"])
def test_rank_two_existing_producer_and_constructor_regression(tiled):
    from test_compiler_buffer_identity import chain as rectangular_chain

    buffers = rectangular_chain(normalized=False, epochs=2)
    if not tiled:
        buffers = buffers[:3]
    candidate = prepare_chain(buffers, allocator=ALLOCATOR)
    assert ".initialize link0" in candidate.query
    assert (".tiled link0" if tiled else ".untiled link0") in candidate.query
    assert ".initializeLine" not in candidate.query and ".tiledLine" not in candidate.query
    assert revalidate(candidate)["authority"] is False


def _check(candidate, lean, directory):
    result = gate.check(candidate, lean=lean, destination=directory, timeout=300)
    assert result.status == "checked", result.reason
    assert result.completion is not None
    delivered = []
    gate.deliver(result.completion, candidate, delivered.append)
    assert delivered == [candidate.final]
    report = json.loads(Path(result.completion.report).read_text())
    expected_schema = (
        "merlin.buffer-reuse.checked.v3"
        if len(json.loads(candidate.raw_chain_json)) == 3
        else "merlin.buffer-reuse.checked.v2"
    )
    assert report["schema"] == expected_schema
    assert ROOT in report["statements"] and set(report["statements"]) == set(candidate.roots)
    assert report["audit"]["violations"] == [] and set(report["audit"]["axioms_used"]) <= gate.STD3
    assert report["relation"] == "equal_possible_successful_public_observations"
    for reference in json.loads(candidate.sources_json):
        path = Path(reference["path"])
        assert digest(path.read_bytes()) == reference["sha256"]
    return result, report


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("tiled", [False, True], ids=["untiled3", "tiled4"])
@pytest.mark.parametrize("extent", [1, 3, 65536])
def test_actual_rank_one_kernel_delivery_and_fresh_process_replay(tmp_path, pass_gate_tools, extent, tiled):
    buffers = chain(extent, tiled=tiled)
    candidate = prepare_chain(buffers, allocator=ALLOCATOR)
    lean = Path(pass_gate_tools.sdk_bin) / "lean"
    _result, report = _check(candidate, lean, tmp_path / "proof")
    assert len(report.get("actual_buffers", report.get("actual4", []))) == len(buffers)
    if extent != 3:
        return  # Fresh-process replay covers one representative of each route.
    request = tmp_path / "raw-replay-request.json"
    request.write_text(json.dumps({"buffers": buffers, "allocator": ALLOCATOR, "index_bits": 64}))
    output = tmp_path / "fresh-process"
    command = [
        sys.executable,
        str(Path(__file__).with_name("buffer_rank_one_replay.py")),
        "--request",
        str(request),
        "--output",
        str(output),
        "--lean",
        str(lean),
    ]
    process = run_owned(
        command,
        cwd=repo_root(),
        env={**os.environ, "PYTHONPATH": str(repo_root() / "src")},
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    (tmp_path / "replay.stdout").write_text(process.stdout)
    (tmp_path / "replay.stderr").write_text(process.stderr)
    assert process.returncode == 0, process.stdout + process.stderr
    replay = json.loads((output / "RESULT.json").read_text())
    assert replay["pid"] != os.getpid() and replay["incoming_live_ledger"] == 0
    assert replay["saved_acceptance_consumed"] is False
    assert replay["actual_sha256"] == [digest(raw) for raw in buffers]
    assert replay["final_sha256"] == digest(buffers[-1])
    assert (output / "delivered.mlir").read_text() == buffers[-1]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("tiled", [False, True], ids=["untiled3", "tiled4"])
def test_public_rank_one_compiler_gate_llvm_and_native_wrap(tmp_path, pass_gate_tools, tiled):
    from test_compiler_formal_transforms import shared_library

    from merlin.llvmlower import lower
    from merlin.verify import compiler_transforms as transforms
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    raw = source(3)
    options = transforms.Options(tile_size=2 if tiled else None, buffer_allocator=ALLOCATOR)
    original = lower.lower_model(
        raw, tmp_path / "original", targets=(), formal_transforms=transforms.Options(memory_source=True)
    )
    baseline = lower.lower_model(raw, tmp_path / "baseline", targets=(), formal_transforms=options)
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=1800)
    selected = {transforms.BUFFER, *([transforms.TILE] if tiled else [])}
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset(selected))
    checked = lower.lower_model(
        raw, tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=validation
    )
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    detail = worker.verdicts[0]["detail"]
    assert detail["full_error_or_world_equivalence"] is False and detail["native_allocator_fidelity"] is False
    report = json.loads(Path(detail["receipt"]).read_text())
    assert len(report.get("actual_buffers", report.get("actual4", []))) == (4 if tiled else 3)
    summary = checked.stats["formal_transforms"]
    assert transforms.BUFFER in summary["changed"]
    assert (transforms.TILE in summary["changed"]) == tiled

    class Descriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("size", ctypes.c_int64),
            ("stride", ctypes.c_int64),
        ]

    class Result(ctypes.Structure):
        _fields_ = [("value", ctypes.c_int32), ("dimension", ctypes.c_int64)]

    observations = []
    for arm, result in (("original", original), ("baseline", baseline), ("checked", checked)):
        directory = tmp_path / arm
        library = shared_library(result, directory)
        function = library._mlir_ciface_kernel
        function.argtypes = [ctypes.POINTER(Result), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(Descriptor)]
        function.restype = None
        for x, y, initial in ((7, -3, 11), (0, 0, 5), (2**31 - 4, 17, 2**31 - 1), (-(2**31), -23, -9)):
            storage = (ctypes.c_int32 * 1)(initial)
            pointer = ctypes.cast(storage, ctypes.c_void_p)
            descriptor, output = Descriptor(pointer, pointer, 0, 1, 1), Result()
            function(ctypes.byref(output), x, y, ctypes.byref(descriptor))
            first = ctypes.c_int32(initial + x).value
            last = ctypes.c_int32(first + y).value
            observed = (output.value, output.dimension, storage[0])
            assert observed == (ctypes.c_int32(first + last).value, 3, last)
            observations.append({"arm": arm, "inputs": [x, y, initial], "observed": observed})
        library._mlir_ciface_untouched.argtypes = [ctypes.c_int32]
        library._mlir_ciface_untouched.restype = ctypes.c_int32
        assert library._mlir_ciface_untouched(-(2**31)) == -(2**31)
    (tmp_path / "NATIVE-OBSERVATIONS.json").write_text(json.dumps(observations, indent=2) + "\n")


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mutation",
    [
        "point_lower",
        "point_upper",
        "point_stride",
        "outer_stride",
        "add_operand",
        "store_seed",
        "store_buffer",
        "store_index",
        "callee",
        "shape",
    ],
)
def test_actual_tiled_rank_one_controls_have_classified_host_or_kernel_refusals(tmp_path, pass_gate_tools, mutation):
    buffers = chain(3, tiled=True)
    changed = mutate_target(buffers, mutation)
    expected_kernel = mutation in {"point_lower", "point_upper", "add_operand"}
    try:
        candidate = prepare_chain(changed, allocator=ALLOCATOR)
    except ValueError as error:
        outcome = {"mutation": mutation, "status": "host_refusal", "reason": str(error), "verifier_valid": True}
        assert not expected_kernel, outcome
    else:
        result = gate.check(
            candidate, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "proof", timeout=300
        )
        assert result.status == "not_established" and result.completion is None, (result.status, result.reason)
        processes = list((tmp_path / "proof").glob("query-*.process.json"))
        failed = [path for path in processes if json.loads(path.read_text())["returncode"] == 1]
        assert failed
        outcome = {
            "mutation": mutation,
            "status": "genuine_lean_refusal",
            "reason": result.reason,
            "processes": list(map(str, failed)),
            "verifier_valid": True,
        }
    (tmp_path / "MUTATION-RESULT.json").write_text(json.dumps(outcome, indent=2) + "\n")


_COMPONENT_PREFIXES = {}


@pytest.fixture
def rank_one_checked_prefix(tmp_path_factory, pass_gate_tools):
    # Share only immutable, freshly compiled component dependencies on the same
    # coherent source/tool cut. This cache is never used to issue gate authority.
    candidate = prepare_chain(chain(3), allocator=ALLOCATOR)
    lean = Path(pass_gate_tools.sdk_bin) / "lean"
    key = (candidate.identity, digest(lean.read_bytes()))
    if key not in _COMPONENT_PREFIXES:
        folder = tmp_path_factory.mktemp("rank-one-current-component-prefix")
        _result, report = _check(candidate, lean, folder / "proof")
        _COMPONENT_PREFIXES[key] = (candidate, folder / "proof/lib", report["library"])
    baseline, library, references = _COMPONENT_PREFIXES[key]
    assert baseline == candidate
    revalidate(baseline)
    for reference in references:
        assert digest(Path(reference["path"]).read_bytes()) == reference["sha256"]
    return baseline, library


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mutation", ["raw", "canonical", "config"])
def test_untrusted_line_link_fields_are_genuine_component_refusals(
    tmp_path, pass_gate_tools, rank_one_checked_prefix, mutation
):
    baseline, library = rank_one_checked_prefix
    hint = copy.deepcopy(json.loads(baseline.hint_json))
    link, other = hint["adapter"]["initializer_links"]
    if mutation in ("raw", "canonical"):
        link[mutation]["seed"] = other[mutation]["seed"]
    else:
        link["config"]["program"]["leaf"]["value"] = other["raw"]["seed"]["name"]
    candidate = prepare(baseline.source, baseline.final, hint, allocator=ALLOCATOR)
    assert candidate.raw_chain_json == ""
    result = gate.check(candidate, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "no-gate")
    assert result.status == "unavailable" and result.completion is None
    assert not (tmp_path / "no-gate").exists()
    before = json.loads(baseline.query_parts_json)
    after = json.loads(candidate.query_parts_json)
    at = next(i for i, part in enumerate(after) if part["module"] == "EpochSyntax")
    assert after[:at] == before[:at]
    part = after[at]
    query = tmp_path / "ChangedEpochSyntax.lean"
    query.write_text(part["source"])
    assert "theorem link0_checked" in query.read_text()
    command = [
        str(Path(pass_gate_tools.sdk_bin) / "lean"),
        "-j1",
        "-o",
        str(tmp_path / "ChangedEpochSyntax.olean"),
        str(query),
    ]
    process = run_owned(
        command,
        cwd=tmp_path,
        env={**os.environ, "LEAN_PATH": str(library), "LEAN_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    (tmp_path / "query.stdout").write_text(process.stdout)
    (tmp_path / "query.stderr").write_text(process.stderr)
    (tmp_path / "PROCESS.json").write_text(
        json.dumps(
            {"command": command, "returncode": process.returncode, "query_sha256": digest(query.read_bytes())}, indent=2
        )
        + "\n"
    )
    with pytest.raises(gate.NotEstablished):
        gate.require_query_success(
            subprocess.CompletedProcess(command, process.returncode, process.stdout, process.stderr)
        )
    assert process.returncode == 1


def test_actual_middle_buffer_is_not_replaceable_by_a_saved_hint():
    buffers = chain(3, tiled=True)
    hint = reconstruct_chain(buffers)
    changed = (*buffers[:1], buffers[2], buffers[1], buffers[3])
    with pytest.raises(ValueError):
        prepare_chain(changed, allocator=ALLOCATOR)
    # The source/final pair alone may still emit a non-authoritative certificate;
    # it cannot enter the gate without the exact complete actual raw chain.
    prepared = prepare(buffers[0], buffers[-1], hint, allocator=ALLOCATOR)
    assert prepared.raw_chain_json == "" and prepared.describe()["authority"] is False

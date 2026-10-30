"""The exact sixteen-kernel handoff through real Merlin gates and host execution.

Full sizes, every output/head, and the pinned deterministic NumPy oracle are
mandatory. Run the campaign one case per fresh process with the retained runner;
these are functional regression cases, never TVM/IREE or timing benchmarks.
An unverified source exclusion is recorded; an admitted gate failure fails.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import re
import shutil
import traceback
from pathlib import Path
from time import perf_counter

import pytest
from kernel_benchmark_handoff_inputs import DATA_SHA256, PINNED_COMMIT, case_ids, fixture

from merlin.common.paths import repo_root
from merlin.verify import pass_validation
from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

CASES = case_ids()
HERE = repo_root() / "merlin/tests/ir/test_kernel_benchmark_formal_gates.py"
INVENTORY_DEFAULT = repo_root() / "merlin/tests/data/kernel_benchmark_handoff_contracts.json"
ORACLE_DEFAULT = repo_root() / "merlin/tests/ir/kernel_benchmark_handoff_oracle.py"


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _ref(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _oracle():
    """Import only hash-checked pinned source; this creates no kernel arrays."""
    inventory_path = Path(os.environ.get("MERLIN_KERNEL_HANDOFF_INVENTORY", INVENTORY_DEFAULT)).resolve()
    if inventory_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("pinned inventory exceeds the bounded4MiB metadata policy")
    inventory_bytes = inventory_path.read_bytes()
    if len(inventory_bytes) > 4 * 1024 * 1024:
        raise ValueError("pinned inventory changed beyond its bounded metadata policy")
    inventory = json.loads(inventory_bytes)
    inventory_ref = {
        "path": str(inventory_path),
        "sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "bytes": len(inventory_bytes),
    }
    if inventory["local_ref_tip"] != PINNED_COMMIT or [row["id"] for row in inventory["cases"]] != list(CASES):
        raise ValueError("the exact pinned sixteen-case handoff inventory is required")
    path = (
        inventory_path.with_name("kernel_benchmark_data.py")
        if os.environ.get("MERLIN_KERNEL_HANDOFF_INVENTORY")
        else ORACLE_DEFAULT
    )
    reference = _ref(path)
    source_bytes = path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != reference["sha256"]:
        raise ValueError("canonical reference changed while reading")
    if reference["sha256"] != DATA_SHA256:
        raise ValueError("canonical reference helper differs from the pinned handoff source")
    spec = importlib.util.spec_from_file_location("merlin_kernel_handoff_pinned_numpy", path)
    if spec is None or spec.loader is None:
        raise ValueError("canonical reference source cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    # Execute the exact bytes that were checked, rather than reopening a file
    # through an import loader after its digest was validated.
    exec(compile(source_bytes, str(path), "exec"), module.__dict__)
    if tuple(module.CASES) != CASES:
        raise ValueError("canonical reference case roster differs")
    return module, inventory, {"inventory": inventory_ref, "reference_helper": reference}


def _fixture_contract(case, value, oracle, archived):
    if value["index_bits"] != 64 or value["case_spec"]["id"] != case:
        raise ValueError("exact source case and index64 fixture required")
    if hashlib.sha256(value["source"].encode()).hexdigest() != value["source_sha256"]:
        raise ValueError("actual fixture source differs from its declared bytes")
    if value["provenance"]["commit"] != PINNED_COMMIT or value["provenance"]["generator"]["sha256"] != DATA_SHA256:
        raise ValueError("fixture translation lost its pinned mathematical input provenance")
    expected = oracle.case_spec(case)
    if _canonical(value["case_spec"]) != _canonical(expected) or _canonical(archived["specification"]) != _canonical(
        expected
    ):
        raise ValueError("fixture changed the original mathematical kernel specification")
    if [row["canonical_input_name"] for row in value["arguments"]] != [row["name"] for row in archived["inputs"]]:
        raise ValueError("historical input roster/order changed")
    for actual, original in zip(value["arguments"], archived["inputs"], strict=True):
        if list(actual["shape"]) != original["shape"] or actual["dtype"] != original["dtype"]:
            raise ValueError("fixture changed original input shape or dtype")
    for actual, original in zip(value["outputs"], archived["outputs"], strict=True):
        if any(actual[key] != original[key] for key in ("name", "dtype")) or list(actual["shape"]) != original["shape"]:
            raise ValueError("fixture dropped or changed an original full output")
    if case in ("T01", "T02") and len(value["outputs"]) != 32:
        raise ValueError("complete attention case requires all32 heads")


def _array_rows(values, oracle):
    return [
        {
            "name": name,
            "shape": list(array.shape),
            "dtype": array.dtype.name,
            "sha256": oracle.array_sha256(array),
            "bytes": array.nbytes,
        }
        for name, array in values.items()
    ]


def _check_archived_arrays(rows, archived):
    for row, expected in zip(rows, archived, strict=True):
        if any(row[key] != expected[key] for key in ("name", "shape", "dtype", "sha256")):
            raise ValueError("fresh canonical array differs from archived handoff: " + row["name"])


def _coverage(path):
    value = json.loads(Path(path).read_bytes())
    if value.get("schema") != "merlin.compiler-coverage.v1" or value.get("finalized") is not True:
        raise ValueError("formal compiler did not finalize its actual coverage")
    selectors = {}
    rows = value["boundaries"]
    for name in sorted(pass_validation.CHECKS):
        observed = [row for row in rows if name in row["pass"].split("+")]
        selectors[name] = {
            "status": "not_observed" if not observed else "+".join(sorted({row["status"] for row in observed})),
            "proof_credit": any(row["proof_credit"] for row in observed),
            "observed_boundaries": observed,
        }
    return {
        "record": value,
        "selectors": selectors,
        "artifact": _ref(path),
        "claim": "individual observed bounded pass boundaries; no whole-kernel or LLVM proof",
    }


class _Guarded:
    """Page-aligned caller-owned dense storage with checked guard/padding bytes."""

    def __init__(self, np, shape, dtype, *, initial=None):
        import math

        self.np = np
        self.dtype = np.dtype(dtype)
        self.shape = tuple(shape)
        self.nbytes = math.prod(self.shape) * self.dtype.itemsize
        self.storage = np.full(self.nbytes + 4096 + 128, 0xA5, dtype=np.uint8)
        self.start = ((self.storage.ctypes.data + 64 + 4095) // 4096) * 4096 - self.storage.ctypes.data
        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.storage, offset=self.start)
        if initial is None:
            self.array.view(np.uint8).reshape(-1).fill(0xCD)
        else:
            self.array[...] = initial
        self.before = hashlib.sha256(self.array.tobytes(order="C")).hexdigest()

    def check(self, *, input_buffer):
        end = self.start + self.nbytes
        if not bool(self.np.all(self.storage[: self.start] == 0xA5)) or not bool(
            self.np.all(self.storage[end:] == 0xA5)
        ):
            raise AssertionError("native call overwrote a guarded buffer boundary")
        after = hashlib.sha256(self.array.tobytes(order="C")).hexdigest()
        if input_buffer and after != self.before:
            raise AssertionError("native call modified canonical input bytes")
        return after


def _abi_arguments(llvm, expected_arity):
    """Inspect actual emitted pointer ABI, including nested LLVM attributes."""
    definitions = list(re.finditer(r"(?m)^define [^\n]*?@_mlir_ciface_forward\(", llvm))
    if len(definitions) != 1 or not re.search(r"\bvoid\s+@_mlir_ciface_forward\($", definitions[0].group()):
        raise ValueError("actual LLVM does not expose the expected void ranked-memref out-param ABI")
    start = definitions[0].end()
    depth, offset = 1, start
    while offset < len(llvm) and depth:
        depth += (llvm[offset] == "(") - (llvm[offset] == ")")
        offset += 1
    if depth:
        raise ValueError("unterminated actual LLVM C-interface signature")
    raw = llvm[start : offset - 1]
    stack, parts, beginning = [], [], 0
    for i, character in enumerate(raw):
        if character in "([{<":
            stack.append(character)
        elif character in ")]}>":
            if not stack:
                raise ValueError("unbalanced actual LLVM argument attribute")
            stack.pop()
        elif character == "," and not stack:
            parts.append(raw[beginning:i].strip())
            beginning = i + 1
    if raw.strip():
        parts.append(raw[beginning:].strip())
    if len(parts) != expected_arity or any(not argument.startswith("ptr ") for argument in parts):
        raise ValueError("actual C-interface argument roster differs from complete input/output descriptors")
    return parts


def _native(result, directory, value, inputs, outputs, oracle):
    """Use the actual standard out-param ABI; no allocated return descriptors."""
    import numpy as np
    from test_compiler_formal_transforms import shared_library

    from merlin.llvmlower.abi import descriptor

    directory.mkdir()
    llvm = result.ll_path.read_text()
    expected_arity = len(value["arguments"]) + len(value["outputs"])
    abi_arguments = _abi_arguments(llvm, expected_arity)
    input_buffers = [
        _Guarded(np, row["shape"], row["dtype"], initial=inputs[row["canonical_input_name"]])
        for row in value["arguments"]
    ]
    output_buffers = [_Guarded(np, row["shape"], row["dtype"]) for row in value["outputs"]]
    buffers = [*input_buffers, *output_buffers]
    descriptors = [descriptor(buf.array.ctypes.data, buf.shape) for buf in buffers]
    descriptor_bytes = [bytes(desc) for desc in descriptors]
    library = shared_library(result, directory)
    function = library._mlir_ciface_forward
    function.argtypes = [ctypes.c_void_p] * len(descriptors)
    function.restype = None
    started = perf_counter()
    function(*(ctypes.byref(desc) for desc in descriptors))
    seconds = perf_counter() - started
    if any(bytes(actual) != before for actual, before in zip(descriptors, descriptor_bytes, strict=True)):
        # These are borrowed/caller-owned descriptors. Freeing an unexpected
        # pointer is unsafe: this ABI violation fails instead of guessing ownership.
        raise AssertionError("native call changed borrowed descriptor ownership/shape/strides")
    for buf in input_buffers:
        buf.check(input_buffer=True)
    observed = {}
    for row, buf in zip(value["outputs"], output_buffers, strict=True):
        buf.check(input_buffer=False)
        expected = outputs[row["name"]]
        if (
            buf.array.dtype != expected.dtype
            or buf.array.shape != expected.shape
            or not np.array_equal(buf.array, expected)
        ):
            np.save(directory / (row["name"] + "-actual.npy"), buf.array, allow_pickle=False)
            np.save(directory / (row["name"] + "-expected.npy"), expected, allow_pickle=False)
            raise AssertionError("complete native output differs from canonical reference: " + row["name"])
        observed[row["name"]] = buf.array
    result_record = {
        "scope": "complete native functional comparison, no LLVM proof or timing claim",
        "seconds_observed": seconds,
        "inputs_unchanged": True,
        "all_guards_intact": True,
        "outputs": _array_rows(observed, oracle),
        "library": _ref(directory / "execution.so"),
        "abi": "void _mlir_ciface_forward(input descriptors..., caller-owned output descriptors...)",
        "actual_abi_arguments": abi_arguments,
        "allocation_ownership": "NumPy owns all guarded buffers; compiler owns/frees its private intermediates",
    }
    # Caller-owned buffers are released by their Python/NumPy owners when this
    # invocation ends; there is no malloc-return output in the selected ABI.
    return result_record


def _host_index_observation(directory):
    """Observe native pointer64 and its raw layout; explicitly select MLIR index64.

    This test-only probe does not infer an omitted default-pointer layout field
    or supply the cross-target admission performed by selected_index_bits.
    """
    from merlin.common.owned_process import run
    from merlin.llvmlower import target_data_layout, toolchain

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    requested = str(toolchain.clang())
    compiler = Path(requested)
    if not compiler.is_file():
        compiler = Path(shutil.which(requested) or requested)
    compiler = compiler.resolve(strict=True)
    before = _ref(compiler)
    source, llvm = directory / "host-pointer.c", directory / "host-pointer.ll"
    source.write_text(
        '_Static_assert(sizeof(void *) == 8, "handoff needs native pointer64");\n'
        '_Static_assert(__CHAR_BIT__ == 8, "handoff needs eight-bit bytes");\n'
        "const unsigned merlin_kernel_handoff_pointer_bits = sizeof(void *) * __CHAR_BIT__;\n"
    )
    command = [str(compiler), "-S", "-emit-llvm", "-x", "c", str(source), "-o", str(llvm)]
    _write(directory / "COMMAND.json", {"argv": command, "cwd": str(directory), "compiler": before})
    observation = {
        "schema": "merlin.kernel-handoff-native-host-index-selection.v1",
        "compiler_requested": requested,
        "compiler": before,
        "source": _ref(source),
        "command": _ref(directory / "COMMAND.json"),
        "scope": (
            "native host pointer ABI observed64; MLIR index64 explicitly selected; "
            "raw clang layout unchanged; no inferred default p0, cross-target admission or LLVM proof"
        ),
    }
    process_record = {"argv": command}
    try:
        try:
            process = run(command, cwd=directory, capture_output=True, text=True, timeout=30)
        except BaseException as error:
            process_record.update(
                failure_type=type(error).__name__,
                failure=str(error),
                owned_process_cleanup=getattr(error, "owned_process_cleanup", None),
            )
            raise
        process_record.update(
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
            owned_process_cleanup=getattr(process, "owned_process_cleanup", {}),
        )
        cleanup = process_record["owned_process_cleanup"]
        if process.returncode or not cleanup.get("complete") or not cleanup.get("guardian_reaped"):
            raise ValueError("native host ABI probe failed or descendant cleanup was not confirmed")
        observation["compiler_after"] = _ref(compiler)
        if observation["compiler_after"] != before:
            raise ValueError("selected native compiler changed during ABI observation")
        raw = llvm.read_text()
        observation["llvm"] = _ref(llvm)
        bits = re.findall(r"(?m)^@merlin_kernel_handoff_pointer_bits\s*=.*\bconstant i32 ([0-9]+)\b", raw)
        pointer_bits = int(bits[0]) if len(bits) == 1 else None
        ctypes_bits = ctypes.sizeof(ctypes.c_void_p) * 8
        if pointer_bits != 64 or ctypes_bits != 64:
            raise ValueError("the exact handoff fixtures require observed native and ctypes pointer64")
        layout = target_data_layout.parse(raw)
        if not layout:
            raise ValueError("selected native compiler did not report its raw target data layout")
        observation.update(
            native_pointer_bits=pointer_bits,
            ctypes_pointer_bits=ctypes_bits,
            data_layout=layout,
            index_bits=64,
            index_selection="explicit MLIR index64 for the handoff ranked-memref ABI",
            status="PASS",
        )
    except BaseException as error:
        observation.update(status="FAILED", failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        _write(directory / "PROCESS-RESULT.json", process_record)
        observation["process"] = _ref(directory / "PROCESS-RESULT.json")
        _write(directory / "OBSERVATION.json", observation)
    return observation


def _lower_host(source, directory, observation, *, selected=None):
    """Both real builds use the same compiler-owned layout/index selection."""
    from merlin.llvmlower.lower import lower_model
    from merlin.verify.compiler_transforms import Options

    return lower_model(
        source,
        directory,
        targets=(),
        data_layout=observation["data_layout"],
        index_bits=observation["index_bits"],
        formal_transforms=Options(),
        pass_validation=selected,
    )


def execute_case(case, work, tools, *, compare_baseline=True):
    """One real full-size case; always retain its report, including failures."""
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    report_path = work / "CASE-REPORT.json"
    report = {
        "schema": "merlin.kernel-handoff-formal-regression.v1",
        "case": case,
        "status": "RUNNING",
        "phase": "source",
        "all_requested_checks": sorted(pass_validation.CHECKS),
        "whole_kernel_formally_verified": False,
        "input_sizes_reduced": False,
    }
    started = perf_counter()
    worker = None
    try:
        oracle, inventory, reference_pins = _oracle()
        value = fixture(case)
        archived = next(row for row in inventory["cases"] if row["id"] == case)
        _fixture_contract(case, value, oracle, archived)
        source = value["source"]
        source_path = work / "source.mlir"
        source_path.write_bytes(source.encode())
        report.update(
            source=_ref(source_path),
            fixture={key: item for key, item in value.items() if key != "source"},
            provenance=reference_pins,
            driver=_ref(HERE),
            options={
                "index_bits": 64,
                "targets": [],
                "formal_transforms": {},
                "ordinary_features": "compiler defaults",
                "compare_baseline": compare_baseline,
            },
        )
        if pass_validation.DEFAULT_CHECKS != frozenset(pass_validation.CHECKS) or len(pass_validation.CHECKS) != 10:
            raise ValueError("this campaign requires all ten actual canonical compiler selectors")
        report["phase"] = "native_index_observation"
        observation = _host_index_observation(work / "host-index-observation")
        report["host_index_observation"] = observation
        report["options"].update(data_layout=observation["data_layout"], index_bits=observation["index_bits"])
        worker = SpeculativeGate(work / "pass-validation", interrupt=False, job_timeout=900, workers=1, nice=19)
        selected = pass_validation.PassValidation(worker, tools, pass_validation.DEFAULT_CHECKS)
        report["phase"] = "checked_compile"
        _write(report_path, report)
        checked = _lower_host(source, work / "checked", observation, selected=selected)
        report["gate_coverage"] = _coverage(worker.directory / "coverage.json")
        if not report["gate_coverage"]["record"]["compiler_completed"]:
            raise AssertionError("formal compiler returned without successful coverage finalization")
        if report["gate_coverage"]["record"]["requested_passes"] != sorted(pass_validation.CHECKS):
            raise AssertionError("formal compiler did not retain all ten requested selectors")
        report["gate_verdicts"] = worker.verdicts
        report["checked_llvm"] = _ref(checked.ll_path)
        if compare_baseline:
            report["phase"] = "ordinary_baseline_compile"
            baseline = _lower_host(source, work / "baseline", observation)
            report["baseline_llvm"] = _ref(baseline.ll_path)
            report["checked_matches_ordinary_llvm"] = checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
            if not report["checked_matches_ordinary_llvm"]:
                raise AssertionError("gate selection changed the ordinary compiler LLVM output")
        report["phase"] = "canonical_full_size_reference"
        _write(report_path, report)
        inputs, outputs = oracle.fixtures(oracle.case_spec(case))
        input_rows, output_rows = _array_rows(inputs, oracle), _array_rows(outputs, oracle)
        _check_archived_arrays(input_rows, archived["inputs"])
        _check_archived_arrays(output_rows, archived["outputs"])
        report["canonical_arrays"] = {
            "inputs": input_rows,
            "outputs": output_rows,
            "all_archived_hashes_match": True,
            "seed": "historical_i8_arena seed0x5EED",
            "bias_seed": value["case_spec"].get("bias_seed"),
        }
        report["phase"] = "native_full_size_comparison"
        _write(report_path, report)
        report["native"] = _native(checked, work / "native", value, inputs, outputs, oracle)
        coverage = report["gate_coverage"]["record"]
        verification = (
            ("partially_verified" if coverage["verified_boundaries"] else "unverified")
            if coverage["unverified_boundaries"]
            else "verified_observed_boundaries"
            if coverage["verified_boundaries"]
            else "no_rewrite_proofs"
        )
        report.update(
            status="PASS",
            phase="complete",
            verification_status=verification,
            proof_credit=coverage["verified_boundaries"],
        )
    except BaseException as error:
        report.update(
            status="FAILED", failure_type=type(error).__name__, failure=str(error), traceback=traceback.format_exc()
        )
        if isinstance(error, GateFailure):
            report["gate_failure"] = error.report
        if worker is not None:
            report["gate_verdicts"] = worker.verdicts
            coverage = worker.directory / "coverage.json"
            if coverage.is_file():
                try:
                    report["gate_coverage"] = _coverage(coverage)
                except (OSError, ValueError, KeyError) as coverage_error:
                    report["coverage_read_error"] = str(coverage_error)
        raise
    finally:
        if worker is not None:
            worker.close()
        report["elapsed_seconds"] = perf_counter() - started
        _write(report_path, report)
    return report


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("MERLIN_RUN_KERNEL_HANDOFF") != "1" and os.environ.get("MERLIN_KERNEL_HANDOFF_REQUIRED") != "1",
    reason="full16 kernel compilation/execution requires explicit MERLIN_RUN_KERNEL_HANDOFF=1",
)
@pytest.mark.parametrize("case", CASES, ids=CASES)
def test_kernel_benchmark_handoff_formal_gate_and_native(
    case, tmp_path, pass_gate_tools, upstream_host_tools, monkeypatch
):
    if os.environ.get("MERLIN_RUN_KERNEL_HANDOFF") != "1":
        pytest.fail("required kernel campaign must explicitly set MERLIN_RUN_KERNEL_HANDOFF=1")
    # Keep the baseline unvalidated and both builds free of environment-selected
    # transformations; the checked build explicitly requests all ten selectors.
    from merlin.verify import compiler_transforms

    for name in (
        pass_validation.ENV_PASSES,
        compiler_transforms.ENV_UNROLL,
        compiler_transforms.ENV_TILE,
        compiler_transforms.ENV_MACC,
        compiler_transforms.ENV_ALLOCATOR,
    ):
        monkeypatch.delenv(name, raising=False)
    report = execute_case(case, tmp_path / case, pass_gate_tools)
    assert report["status"] == "PASS"


@pytest.mark.parametrize("case", CASES, ids=CASES)
def test_kernel_handoff_fixture_metadata_matches_pinned_contract_without_arrays(case, monkeypatch):
    oracle, inventory, _ = _oracle()
    monkeypatch.setattr(oracle, "fixtures", lambda *a: pytest.fail("host metadata test generated full arrays"))
    archived = next(row for row in inventory["cases"] if row["id"] == case)
    _fixture_contract(case, fixture(case), oracle, archived)


def test_kernel_handoff_fixture_cannot_drop_attention_heads():
    oracle, inventory, _ = _oracle()
    value = fixture("T01")
    value["outputs"] = value["outputs"][:-1]
    with pytest.raises(ValueError):
        _fixture_contract("T01", value, oracle, next(row for row in inventory["cases"] if row["id"] == "T01"))


def test_kernel_both_builds_use_observed_native_layout_without_compiling(monkeypatch, tmp_path):
    """Argument-spy control only: no compiler, gate or native success is supplied."""
    from merlin.llvmlower import lower

    observation = {"index_bits": 64, "data_layout": "observed-layout-control", "compiler_sha256": "a" * 64}
    lower_calls = []

    class CallObserved(Exception):
        pass

    def lower_spy(*args, **kwargs):
        lower_calls.append((args, kwargs))
        raise CallObserved("argument capture only; no lowering result")

    monkeypatch.setattr(lower, "lower_model", lower_spy)
    selected = object()
    for name, validation in (("checked", selected), ("baseline", None)):
        with pytest.raises(CallObserved):
            _lower_host("source-control", tmp_path / name, observation, selected=validation)
    assert [args for args, _ in lower_calls] == [
        ("source-control", tmp_path / "checked"),
        ("source-control", tmp_path / "baseline"),
    ]
    for (_, kwargs), validation in zip(lower_calls, (selected, None), strict=True):
        assert kwargs["targets"] == ()
        assert kwargs["data_layout"] == observation["data_layout"] and kwargs["index_bits"] == 64
        assert kwargs["pass_validation"] is validation


@pytest.mark.parametrize("fault", [None, "compiler_changed", "pointer_bits", "ctypes_width", "cleanup"])
def test_kernel_native_probe_protocol_controls_without_invoking_compiler(fault, monkeypatch, tmp_path):
    """Mock probe responses qualify guards only; they supply no ABI/proof credit."""
    from types import SimpleNamespace

    from merlin.common import owned_process
    from merlin.llvmlower import toolchain

    compiler = tmp_path / "selected-clang"
    compiler.write_bytes(b"compiler-control-before")
    monkeypatch.setattr(toolchain, "clang", lambda: compiler)
    original_sizeof = ctypes.sizeof
    if fault == "ctypes_width":
        monkeypatch.setattr(ctypes, "sizeof", lambda value: 4 if value is ctypes.c_void_p else original_sizeof(value))
    layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-n8:16:32:64-S128"

    def fake_probe(command, **kwargs):
        assert command[:5] == [str(compiler), "-S", "-emit-llvm", "-x", "c"]
        assert kwargs == {"cwd": tmp_path / "probe", "capture_output": True, "text": True, "timeout": 30}
        assert "#include" not in Path(command[5]).read_text()
        Path(command[-1]).write_text(
            'target datalayout = "' + layout + '"\n'
            "@merlin_kernel_handoff_pointer_bits = dso_local constant i32 "
            + ("32" if fault == "pointer_bits" else "64")
            + ", align 4\n"
        )
        if fault == "compiler_changed":
            compiler.write_bytes(b"compiler-control-after")
        return SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
            owned_process_cleanup={"complete": fault != "cleanup", "guardian_reaped": True},
        )

    monkeypatch.setattr(owned_process, "run", fake_probe)
    if fault is None:
        observation = _host_index_observation(tmp_path / "probe")
        assert observation["data_layout"] == layout and observation["index_bits"] == 64
        assert observation["native_pointer_bits"] == observation["ctypes_pointer_bits"] == 64
        assert observation["compiler"] == observation["compiler_after"]
    else:
        with pytest.raises(ValueError):
            _host_index_observation(tmp_path / "probe")
    retained = json.loads((tmp_path / "probe/OBSERVATION.json").read_bytes())
    assert retained["status"] == ("PASS" if fault is None else "FAILED")
    assert Path(retained["process"]["path"]).is_file()


def test_kernel_abi_reads_nested_attributes_and_refuses_return_or_arity_changes():
    text = "define void @_mlir_ciface_forward(ptr noundef align 8 dereferenceable(16) %x, ptr %y) {"
    assert len(_abi_arguments(text, 2)) == 2
    with pytest.raises(ValueError):
        _abi_arguments(text.replace("define void", "define ptr"), 2)
    with pytest.raises(ValueError):
        _abi_arguments(text, 3)


def test_kernel_tiny_guard_control_detects_input_or_boundary_writes():
    import numpy as np

    value = _Guarded(np, (2, 3), "int8", initial=np.arange(6, dtype=np.int8).reshape(2, 3))
    assert value.array.ctypes.data % 4096 == 0
    value.check(input_buffer=True)
    value.array[0, 0] = 99
    with pytest.raises(AssertionError, match="modified canonical input"):
        value.check(input_buffer=True)
    value.storage[0] = 0
    with pytest.raises(AssertionError, match="guarded buffer"):
        value.check(input_buffer=False)


def test_kernel_coverage_does_not_fabricate_credit_for_unobserved_selectors(tmp_path):
    document = {
        "schema": "merlin.compiler-coverage.v1",
        "finalized": True,
        "boundaries": [
            {"pass": "prune_dead_pure_tensor_ops", "status": "unverified", "proof_credit": False},
            {"pass": "cse", "status": "unchanged", "proof_credit": False},
        ],
    }
    path = tmp_path / "coverage.json"
    _write(path, document)
    result = _coverage(path)
    assert len(result["selectors"]) == 10
    assert result["selectors"]["ordered_loop_unrolling"] == {
        "status": "not_observed",
        "proof_credit": False,
        "observed_boundaries": [],
    }
    assert result["selectors"]["prune_dead_pure_tensor_ops"]["status"] == "unverified"
    assert not any(row["proof_credit"] for row in result["selectors"].values())

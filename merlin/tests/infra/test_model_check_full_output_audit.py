"""Synthetic protected receipts exercise all-output offline checks; no model or simulator runs."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from merlin.common import invocation_record as I
from merlin.common.digest import sha256_file
from merlin.common.tree_hash import hash_tree
from merlin.llvmlower import compilation_recipe as CR
from merlin.model_check import full_build as FB
from merlin.model_check import full_output_audit as A
from merlin.model_check.cli import tree_sha256
from merlin.runtime import linalg_numpy as LN
from merlin.runtime.out_bin import _checksum


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _pin(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def _npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"out{i}": value for i, value in enumerate(values)})


def _console(values, build_hash="fixture-build"):
    data = f"METRIC build_hash {build_hash}\nMETRIC memref_rank_mismatch 0\nMETRIC cycles 10\n".encode()
    for index, value in enumerate(values):
        payload = value.tobytes()
        sign = "s" if value.dtype.kind == "i" else "u"
        cols = value.shape[-1] if value.ndim else 1
        data += (
            f"OUT_BIN_BEGIN v1 out{index} {value.size // cols} {cols} {value.dtype.itemsize} {sign} {len(payload)}\n"
        ).encode()
        data += payload + f"OUT_BIN_END v1 {_checksum(payload):016x}\n".encode()
    return data + b"DONE\n"


@pytest.fixture
def case(tmp_path, monkeypatch):
    stage, bundle, candidate = (tmp_path / name for name in ("stage", "bundle", "candidate"))
    for path in (stage, bundle, candidate):
        path.mkdir()
    (candidate / "compiler.py").write_text("fixture compiler source\n")
    stage_program = stage / "model.mlir"
    stage_program.write_text("""module {
      func.func @forward(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xi1>, tensor<2xi32>) {
        %b = arith.constant dense<[true, false]> : tensor<2xi1>
        %i = arith.constant dense<[16777217, -4]> : tensor<2xi32>
        return %arg0, %arg0, %b, %i : tensor<2xf32>, tensor<2xf32>, tensor<2xi1>, tensor<2xi32>
      }
    }""")
    np.savez(stage / "inputs.npz", in0=np.zeros(2, np.float32))
    model_config = bundle / "configuration/models.yaml"
    model_config.parent.mkdir()
    model_config.write_text(
        yaml.safe_dump(
            {
                "schema": "merlin.model_check.models.v1",
                "target": "fixture",
                "models": {
                    "toy": {
                        "workload": "fixture",
                        "chain": {"kind": "single"},
                        "programs": [{"name": "p", "steps": 2, "inputs": [{"from": "input", "key": "x", "carry": 1}]}],
                        "judged": {"kind": "logits", "outputs": [{"program": "p", "output": 0}]},
                    }
                },
            }
        )
    )
    _write(bundle / "MANIFEST.json", {"fixture": True})
    (bundle / "SHA256SUMS").write_text("fixture seal\n")
    catalog_path = _write(bundle / "toy/programs/p/catalog.json", {})
    ref_program = tmp_path / "ref.mlir"
    ref_program.write_bytes(stage_program.read_bytes())
    input_path = tmp_path / "inputs.npz"
    np.savez(input_path, x=np.array([1, 2], np.float32))
    numeric = LN.numeric_contract()
    spec = [
        {"output": i, "dtype": np.dtype(dtype).str, "shape": [2], "element_type": elem}
        for i, (dtype, elem) in enumerate((("float32", "f32"), ("float32", "f32"), ("bool", "i1"), ("int32", "i32")))
    ]
    values = [
        [
            np.array([2 + step, 3 + step], np.float32),
            np.array([7 + 2 * step, 8 + 2 * step], np.float32),
            np.array([True, False]),
            np.array([16777217, -4], np.int32),
        ]
        for step in range(2)
    ]
    sealed = bundle / "toy/visible/v00/int8.npz"
    _npz(sealed, [row[0] for row in values])
    numeric_receipt = _write(
        sealed.with_name("reference_numeric_contract.json"),
        {
            "schema": "merlin.model_check.reference_numeric_contract.v1",
            "numeric_contract": numeric,
            "evaluator_sha256": sha256_file(LN.__file__),
            "outputs_sha256": sha256_file(sealed),
        },
    )
    (bundle / "SHA256SUMS").write_text(
        f"{sha256_file(numeric_receipt)}  {numeric_receipt.relative_to(bundle).as_posix()}\n"
    )
    expected = {
        "source_revision": "a" * 40,
        "models_config_sha256": sha256_file(model_config),
        "bundle_manifest_sha256": sha256_file(bundle / "MANIFEST.json"),
        "bundle_sums_sha256": sha256_file(bundle / "SHA256SUMS"),
        "runtime_pins": {},
    }
    spike, extension, compiler = (tmp_path / name for name in ("spike", "libgemmini.so", "cross-compiler"))
    for path in (spike, extension, compiler):
        path.write_text(path.name)
    expected["runtime_pins"] = {"spike": sha256_file(spike), "libgemmini": sha256_file(extension)}
    submission = {
        "path": str(candidate),
        "hash_tree_sha256": hash_tree(candidate)["sha256"],
        "model_check_tree_sha256": tree_sha256(candidate),
    }
    job_work = tmp_path / "work"
    work = job_work / submission["model_check_tree_sha256"][:16] / "toy"
    runs = work / "runs/visible-fixture"
    build = {
        "status": "linked",
        "key": {
            "stage": str(stage),
            "target": "fixture",
            "label": "toy:p",
            "submission_sha256": submission["model_check_tree_sha256"],
            "source": {"model_check/full_build.py": sha256_file(FB.__file__)},
        },
        "elf_sha256": "f" * 64,
        "build_hash": "fixture-build",
    }
    _write(work / "programs/p/build.json", build)
    refsteps, stepdirs = [], []
    for step, output in enumerate(values):
        refpath = tmp_path / f"reference/p.{step}.npz"
        _npz(refpath, output)
        refsteps.append(
            {
                "step": step,
                "outputs_file": _pin(refpath),
                "outputs": [
                    {**schema, "sha256_of_array_bytes": hashlib.sha256(value.tobytes()).hexdigest()}
                    for schema, value in zip(spec, output, strict=True)
                ],
            }
        )
        directory = runs / "visible/v00" / f"p.{step}"
        (directory / "capture").mkdir(parents=True)
        np.savez(directory / "capture/inputs.npz", in0=np.array([1, 2], np.float32) if step == 0 else values[0][1])
        _npz(directory / "outputs.npz", output)
        elf = {"path": str(directory / "model.elf"), "sha256": "e" * 64, "bytes": 1}
        _write(
            directory / "compilation_recipe.json",
            {
                "schema": "merlin.compilation_recipe.v1",
                "status": "completed",
                "producer": _pin(Path(FB.__file__)),
                "executable": elf,
                "commands": [{"status": "returned", "output": elf, "inputs": [], "executable": _pin(compiler)}],
            },
        )
        invdir = directory / "invocations/original"
        invdir.mkdir(parents=True)
        (invdir / "stdout.bin").write_bytes(_console(output))
        (invdir / "stderr.bin").write_bytes(b"")
        _write(
            invdir / "invocation.json",
            {
                "schema": "merlin.invocation_record.v1",
                "kind": "subprocess",
                "stage": "selected_rocc_process",
                "status": "completed",
                "returncode": 0,
                "inputs_unchanged": True,
                "dependencies_unchanged": True,
                "executable_unchanged": True,
                "executable": _pin(spike),
                "argv": [str(spike), str(directory / "model.elf")],
                "dependencies": [_pin(extension)],
                "inputs": [{k: elf[k] for k in ("path", "sha256")}],
                "stdout": _pin(invdir / "stdout.bin"),
                "stderr": _pin(invdir / "stderr.bin"),
            },
        )
        stepdirs.append(directory)
    reference = {
        "schema": A.REFERENCE_SCHEMA,
        "status": "complete",
        "representation": "declared_storage",
        "model": "toy",
        "input_id": "visible/v00",
        "selected_identity": {
            k: v
            for k, v in expected.items()
            if k in {"models_config_sha256", "bundle_manifest_sha256", "bundle_sums_sha256"}
        },
        "input": _pin(input_path),
        "implementation": {"source_pins": [_pin(stage_program)], "numeric_contract": numeric},
        "programs": [
            {
                "program": "p",
                "steps_declared": 2,
                "steps": refsteps,
                "declared_outputs": spec,
                "stage": str(stage),
                "catalog": _pin(catalog_path),
                "reference_program": _pin(ref_program),
                "capture_program": _pin(stage_program),
            }
        ],
        "judged_crossjoin": {"sealed_npz": _pin(sealed), "outputs": 2, "bit_exact": True},
    }
    reference_path = _write(tmp_path / "reference.json", reference)
    report_path = _write(
        tmp_path / "report.json",
        {
            "model": "toy",
            "submission": {"path": str(candidate), "tree_sha256": submission["model_check_tree_sha256"]},
            "comparison": {"float_policy": A.FLOAT_POLICY},
            "programs": [
                {
                    "program": "p",
                    "build": build,
                    "runs": [
                        {"step": i, "input": "visible/v00", "status": "completed", "engine": "spike"} for i in range(2)
                    ],
                }
            ],
        },
    )
    argv = []
    for key, value in {
        "--models": model_config,
        "--model": "toy",
        "--input": "visible/v00",
        "--out": report_path,
        "--mode": "full",
        "--engine": "spike",
        "--full-comparison": "quantized-reference",
        "--bundle": bundle,
        "--quantized-reference-atol": "2e-6",
        "--quantized-reference-rtol": "1e-5",
        "--submission": candidate,
        "--work": job_work,
    }.items():
        argv.extend((key, str(value)))
    job_path = _write(
        tmp_path / "job.json",
        {"selected_identity": expected, "submission": submission, "argv": argv, "work": str(job_work)},
    )
    requirement = tmp_path / "requirement.txt"
    requirement.write_text("Every declared output must match the selected quantized reference.")
    monkeypatch.setattr(A.B.Bundle, "catalog", lambda *a: SimpleNamespace(stage=stage, ref_prepared=ref_program))
    monkeypatch.setattr(A.F, "derived_input", lambda *a: input_path)
    monkeypatch.setattr(
        A.FQ,
        "sealed_reference",
        lambda *a: (
            [row[0] for row in values],
            {"path": str(sealed), "sha256": sha256_file(sealed), "numeric_contract": {"numeric_contract": numeric}},
        ),
    )
    args = {
        "reference_path": reference_path,
        "expected_reference_sha256": sha256_file(reference_path),
        "job_pin_path": job_path,
        "report_path": report_path,
        "models_path": model_config,
        "run_root": runs,
        "expected_identity": expected,
        "requirement_path": requirement,
        "extension_runtime_key": "libgemmini",
    }
    return SimpleNamespace(
        args=args,
        reference=reference,
        reference_path=reference_path,
        job_path=job_path,
        stepdirs=stepdirs,
        values=values,
        input_path=input_path,
        candidate=candidate,
    )


def _replace_actual(case, step, values, *, update_stream=True):
    directory = case.stepdirs[step]
    _npz(directory / "outputs.npz", values)
    if update_stream:
        invdir = directory / "invocations/original"
        (invdir / "stdout.bin").write_bytes(_console(values))
        record = json.loads((invdir / "invocation.json").read_text())
        record["stdout"] = _pin(invdir / "stdout.bin")
        _write(invdir / "invocation.json", record)


def test_every_declared_output_and_raw_readback_is_reopened(case):
    result = A.verify(**case.args)
    assert result["status"] == "pass" and result["unchanged"] is True
    assert result["steps_compared"] == result["steps_expected"] == 2
    assert result["outputs_compared"] == result["n_declared_outputs"] == 8
    assert result["n_judged_outputs"] == 2
    assert all(row["raw_readback_reopened"] is True for row in result["steps"])
    assert all(row["transient_elf_current_reopen"] == "unavailable_deleted" for row in result["steps"])
    assert all(row["pass"] is True for row in result["comparisons"] if row["dtype"] == "|b1")


def _relocated_build_records(case, tmp_path):
    job = json.loads(case.job_path.read_text())
    original = Path(job["work"]) / job["submission"]["model_check_tree_sha256"][:16] / "toy/programs/p/build.json"
    relocated = tmp_path / "independent-out/programs/p/build.json"
    relocated.parent.mkdir(parents=True)
    original.rename(relocated)
    locator = _write(
        tmp_path / "build-records.json",
        {
            "schema": A.BUILD_RECORDS_SCHEMA,
            "job": _pin(case.job_path),
            "report": _pin(case.args["report_path"]),
            "programs": {"p": _pin(relocated)},
        },
    )
    return relocated, locator


def _audit_with_build_records(case, locator):
    return A.verify(
        **case.args,
        program_build_records_path=locator,
        expected_program_build_records_sha256=sha256_file(locator),
    )


def test_pinned_original_build_records_can_live_outside_the_original_run_root(case, tmp_path):
    relocated, locator = _relocated_build_records(case, tmp_path)
    with pytest.raises(A.AuditError, match="missing absolute artifact"):
        A.verify(**case.args)  # The default still requires the historical location.
    result = _audit_with_build_records(case, locator)
    assert result["status"] == "pass" and result["outputs_compared"] == 8 and result["steps_compared"] == 2
    assert result["program_build_records"] == _pin(locator)
    assert _pin(relocated) in result["artifacts"]
    assert result["run_root"] == str(case.args["run_root"])
    values = copy.deepcopy(case.values[1])
    values[1][0] += 1
    _replace_actual(case, 1, values)
    assert _audit_with_build_records(case, locator)["status"] == "fail"  # Every unjudged output is still checked.


@pytest.mark.parametrize("mutation", ["program", "source", "elf", "build_hash", "native_partition"])
def test_relocated_record_cannot_select_another_original_build(case, tmp_path, mutation):
    relocated, locator = _relocated_build_records(case, tmp_path)
    build = json.loads(relocated.read_text())
    if mutation == "program":
        build["key"]["label"] = "toy:another"
    elif mutation == "source":
        build["key"]["source"]["model_check/full_build.py"] = "0" * 64
    elif mutation == "elf":
        build["elf_sha256"] = "0" * 64
    elif mutation == "build_hash":
        build["build_hash"] = "another-build"
    else:
        build["native_qualification"] = {"program": "another"}
    _write(relocated, build)
    selected = json.loads(locator.read_text())
    selected["programs"]["p"] = _pin(relocated)
    _write(locator, selected)  # Even an explicitly selected, freshly pinned wrong record must fail.
    with pytest.raises(A.AuditError):
        _audit_with_build_records(case, locator)


@pytest.mark.parametrize("mutation", ["job", "report", "roster", "record_digest", "locator_digest"])
def test_build_record_locator_binds_the_original_job_report_and_exact_program_roster(case, tmp_path, mutation):
    _, locator = _relocated_build_records(case, tmp_path)
    selected = json.loads(locator.read_text())
    if mutation in ("job", "report"):
        selected[mutation] = _pin(_write(tmp_path / f"another-{mutation}.json", {}))
    elif mutation == "roster":
        selected["programs"]["another"] = selected["programs"].pop("p")
    else:
        selected["programs"]["p"]["sha256"] = "0" * 64
    _write(locator, selected)
    with pytest.raises(A.AuditError):
        if mutation == "locator_digest":
            A.verify(**case.args, program_build_records_path=locator, expected_program_build_records_sha256="0" * 64)
        else:
            _audit_with_build_records(case, locator)


@pytest.mark.parametrize("include_digest", [False, True])
def test_offline_cli_requires_and_uses_the_selected_build_record_locator_digest(
    case, tmp_path, monkeypatch, include_digest
):
    _, locator = _relocated_build_records(case, tmp_path)
    identity = _write(tmp_path / "selected-identity.json", case.args["expected_identity"])
    output = tmp_path / "audit.json"
    options = {
        "reference": case.args["reference_path"],
        "reference-sha256": case.args["expected_reference_sha256"],
        "job-pin": case.args["job_pin_path"],
        "report": case.args["report_path"],
        "models": case.args["models_path"],
        "run-root": case.args["run_root"],
        "expected-identity": identity,
        "requirement-source": case.args["requirement_path"],
        "extension-runtime-key": case.args["extension_runtime_key"],
        "program-build-records": locator,
        "out": output,
    }
    if include_digest:
        options["program-build-records-sha256"] = sha256_file(locator)
    monkeypatch.setattr(
        "sys.argv",
        ["full-output-audit", *[item for key, value in options.items() for item in ("--" + key, str(value))]],
    )
    assert A.main() == (0 if include_digest else 1)
    result = json.loads(output.read_text())
    assert result["status"] == ("pass" if include_digest else "error")
    if include_digest:
        assert result["outputs_compared"] == 8 and result["program_build_records"] == _pin(locator)


def _use_timed_recipes(case):
    with I.observe_timings() as events:
        with I.timed_event("phase", "compilation command"):
            pass
    timing = I.timing_document(events)
    for directory in case.stepdirs:
        path = directory / "compilation_recipe.json"
        recipe = json.loads(path.read_text())
        recipe["schema"] = CR.SCHEMA
        recipe["commands"][0]["timing"] = copy.deepcopy(timing)
        _write(path, recipe)


def test_v2_recipes_reopen_all_outputs_without_reconstructing_discarded_products(case):
    _use_timed_recipes(case)
    result = A.verify(**case.args)
    assert result["status"] == "pass" and result["outputs_compared"] == 8
    assert all(step["transient_elf_current_reopen"] == "unavailable_deleted" for step in result["steps"])
    assert _pin(Path(CR.__file__)) in result["operator_sources"]
    values = copy.deepcopy(case.values[1])
    values[1][0] += 1
    _replace_actual(case, 1, values)
    assert A.verify(**case.args)["status"] == "fail"  # Unjudged carry still has its original gate.


@pytest.mark.parametrize("mutation", ["unknown_schema", "legacy_timing", "malformed", "failed", "phase"])
def test_offline_v2_recipe_timing_is_not_admitted_from_a_schema_label(case, mutation):
    _use_timed_recipes(case)
    path = case.stepdirs[0] / "compilation_recipe.json"
    recipe = json.loads(path.read_text())
    if mutation == "unknown_schema":
        recipe["schema"] = "unknown"
    elif mutation == "legacy_timing":
        recipe["schema"] = CR.LEGACY_SCHEMA
    elif mutation == "malformed":
        recipe["commands"][0]["timing"]["events"][0]["finished_ns"] = -1
    elif mutation == "failed":
        recipe["commands"][0]["timing"]["events"][0]["status"] = "failed"
    else:
        recipe["phases"] = [{"name": "another phase", "timing": recipe["commands"][0]["timing"]}]
    _write(path, recipe)
    with pytest.raises(ValueError):
        A.verify(**case.args)


def test_raw_decoder_matches_full_runtime_without_repeated_signature_parsing(case, monkeypatch):
    from merlin.llvmlower import c_runtime

    stage = Path(case.reference["programs"][0]["stage"])
    specs = c_runtime._out_specs(stage / "model.mlir")
    console = _console(case.values[0])
    original, _ = FB.read(console, stage, "fixture-build")
    decoded = A._decode_readback(console, specs, "fixture-build")
    assert all(A._same(a, b) for a, b in zip(original, decoded, strict=True))
    calls, read_specs = [], c_runtime._out_specs

    def once(path):
        calls.append(path)
        return read_specs(path)

    monkeypatch.setattr(c_runtime, "_out_specs", once)
    result = A.verify(**case.args)
    assert result["status"] == "pass"
    assert calls == [stage / "model.mlir"]  # One program, independently of its step count.


def test_declared_scalar_vector_and_rank_six_match_the_selected_c_readback_producer():
    from merlin.runtime.whole_model_readback import render_outputs_header

    specs = [([], "i64"), ([3], "f32"), ([2, 1, 1, 2, 1, 2], "f32")]
    header = render_outputs_header(specs)
    assert "MERLIN_OUTPUT_ROWS[3] = {1, 1, 4}" in header
    assert "MERLIN_OUTPUT_COLS[3] = {1, 3, 2}" in header
    values = [
        np.asarray(7, np.int64),
        np.arange(3, dtype=np.float32),
        np.arange(8, dtype=np.float32).reshape(2, 1, 1, 2, 1, 2),
    ]
    decoded = A._decode_readback(_console(values), specs, "fixture-build")
    assert all(A._same(a, b) for a, b in zip(values, decoded, strict=True))


@pytest.mark.parametrize(
    "mutation", ["build_hash", "rank", "shape", "signedness", "checksum", "completion", "bool_bits"]
)
def test_reused_typed_signature_preserves_all_raw_packet_guards(case, mutation):
    from merlin.llvmlower.c_runtime import _out_specs

    values = copy.deepcopy(case.values[0])
    if mutation == "bool_bits":
        values[2].view(np.uint8)[0] = 2
    console = _console(values)
    if mutation == "build_hash":
        console = console.replace(b"build_hash fixture-build", b"build_hash wrong-build")
    elif mutation == "rank":
        console = console.replace(b"memref_rank_mismatch 0", b"memref_rank_mismatch 1")
    elif mutation == "shape":
        console = console.replace(b"out0 1 2", b"out0 2 1")
    elif mutation == "signedness":
        console = console.replace(b"out0 1 2 4 u", b"out0 1 2 4 s")
    elif mutation == "checksum":
        console = console.replace(b"OUT_BIN_END v1 ", b"OUT_BIN_END v1 0", 1)
    elif mutation == "completion":
        console = console.removesuffix(b"DONE\n")
    specs = _out_specs(Path(case.reference["programs"][0]["stage"]) / "model.mlir")
    with pytest.raises(ValueError):
        A._decode_readback(console, specs, "fixture-build")


@pytest.mark.parametrize("output", [1, 2, 3])
def test_wrong_unjudged_cache_bool_or_exact_integer_fails_even_when_judged_final_matches(case, output):
    values = [value.copy() for value in case.values[1]]
    values[output][0] = not values[output][0] if output == 2 else values[output][0] + 1
    _replace_actual(case, 1, values)
    result = A.verify(**case.args)
    assert result["status"] == "fail"
    assert all(row["pass"] for row in result["comparisons"] if row["output"] == 0)
    assert any(not row["pass"] for row in result["comparisons"] if row["output"] == output)


def test_float_policy_is_selected_original_tolerance(case):
    values = [value.copy() for value in case.values[1]]
    values[1][0] += np.float32(2e-6)
    _replace_actual(case, 1, values)
    assert A.verify(**case.args)["status"] == "pass"


def test_numeric_array_names_accept_lexicographic_archive_order_without_changing_binding(tmp_path):
    path = tmp_path / "outputs.npz"
    values = {f"out{i}": np.asarray([i], np.int64) for i in range(12)}
    np.savez(path, **dict(sorted(values.items())))
    result = A.Files().arrays(path, arity=12)
    assert list(result) == [f"out{i}" for i in range(12)]
    assert [int(value[0]) for value in result.values()] == list(range(12))


def test_missing_native_numeric_sidecar_is_not_an_admitted_reference(case):
    sidecar = Path(case.reference["judged_crossjoin"]["sealed_npz"]["path"]).with_name(
        "reference_numeric_contract.json"
    )
    sidecar.unlink()
    with pytest.raises((ValueError, OSError)):
        A.verify(**case.args)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_output",
        "missing_step",
        "wrong_input",
        "wrong_runtime",
        "wrong_candidate",
        "wrong_source",
        "changed_source_pin",
        "wrong_arity",
        "changed_npz_only",
        "nonfinite",
        "wrong_dtype",
    ],
)
def test_missing_or_mismatched_correspondence_never_passes(case, mutation):
    if mutation == "missing_output":
        _npz(case.stepdirs[1] / "outputs.npz", case.values[1][:-1])
    elif mutation == "missing_step":
        (case.stepdirs[1] / "outputs.npz").unlink()
    elif mutation == "wrong_input":
        np.savez(case.stepdirs[1] / "capture/inputs.npz", in0=np.zeros(2, np.float32))
    elif mutation in {"wrong_runtime", "wrong_source"}:
        job = json.loads(case.job_path.read_text())
        if mutation == "wrong_runtime":
            job["selected_identity"]["runtime_pins"]["spike"] = "0" * 64
        else:
            job["selected_identity"]["source_revision"] = "0" * 40
        _write(case.job_path, job)
    elif mutation == "wrong_candidate":
        (case.candidate / "compiler.py").write_text("changed")
    elif mutation == "changed_source_pin":
        case.reference["implementation"]["source_pins"][0]["sha256"] = "0" * 64
        _write(case.reference_path, case.reference)
    elif mutation == "wrong_arity":
        case.reference["programs"][0]["declared_outputs"].pop()
        _write(case.reference_path, case.reference)
    else:
        values = copy.deepcopy(case.values[1])
        if mutation == "nonfinite":
            values[1][0] = np.nan
        elif mutation == "wrong_dtype":
            values[1] = values[1].astype(np.float64)
        else:
            values[1][0] += 10
        _replace_actual(case, 1, values, update_stream=mutation != "changed_npz_only")
    try:
        result = A.verify(**case.args)
    except (OSError, ValueError, KeyError, RuntimeError):
        return
    assert result["status"] != "pass"


def _select_functional_policy(case, atol):
    from merlin.model_check import models as M

    path = case.args["models_path"]
    config = yaml.safe_load(path.read_text())
    config["numerics"] = {
        "version": M.NUMERICS_SCHEMA,
        "functional": {
            "atol": atol,
            "rtol": 0.0,
            "equal_nan": False,
            "finite_required": True,
            "integer_boolean": "exact",
            "outputs": "all_declared_every_step",
        },
    }
    path.write_text(yaml.safe_dump(config))
    case.args["expected_identity"]["models_config_sha256"] = sha256_file(path)
    case.reference["selected_identity"]["models_config_sha256"] = sha256_file(path)
    _write(case.reference_path, case.reference)
    case.args["expected_reference_sha256"] = sha256_file(case.reference_path)
    job = json.loads(case.job_path.read_text())
    job["selected_identity"] = case.args["expected_identity"]
    for name, value in (("atol", atol), ("rtol", 0.0)):
        job["argv"][job["argv"].index("--quantized-reference-" + name) + 1] = str(value)
    _write(case.job_path, job)
    report_path = case.args["report_path"]
    report = json.loads(report_path.read_text())
    report["comparison"]["float_policy"] = A.comparison_policy(path)
    _write(report_path, report)


def test_audit_uses_original_selected_yaml_policy_not_legacy_constant(case):
    _select_functional_policy(case, 0.125)
    values = [value.copy() for value in case.values[1]]
    values[1][0] += np.float32(0.1)
    _replace_actual(case, 1, values)
    result = A.verify(**case.args)
    assert result["status"] == "pass" and result["float_policy"]["atol"] == 0.125
    assert A.FLOAT_POLICY["atol"] == 2e-6  # Compatibility alias cannot control the current audit.
    job = json.loads(case.job_path.read_text())
    job["argv"][job["argv"].index("--quantized-reference-atol") + 1] = "10.0"
    _write(case.job_path, job)
    with pytest.raises(A.AuditError, match="job float policy differs"):
        A.verify(**case.args)


def test_audit_accepts_implicit_selected_yaml_flags_and_enforces_tight_bound(case):
    _select_functional_policy(case, 0.001)
    job = json.loads(case.job_path.read_text())
    for flag in ("--quantized-reference-atol", "--quantized-reference-rtol"):
        index = job["argv"].index(flag)
        del job["argv"][index : index + 2]
    _write(case.job_path, job)
    values = [value.copy() for value in case.values[1]]
    values[1][0] += np.float32(0.01)
    _replace_actual(case, 1, values)
    assert A.verify(**case.args)["status"] == "fail"

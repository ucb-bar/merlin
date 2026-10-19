"""Offline comparison of every declared full-model output, including unjudged intermediates.

This operator consumes a protected original reference supplement, supervisor job
pin and retained full-check files. It executes no model, compiler or simulator.
Raw successful process observations, when retained, join readback to saved NPZs;
deleted transient ELFs remain historical observations, never current byte proofs.
The result is supplemental numerical evidence, not campaign certification or a
new reference selection. Callers must independently admit the operator's inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from merlin.common.digest import sha256_file
from merlin.model_check import bundle as B
from merlin.model_check import chain as CH
from merlin.model_check import full as F
from merlin.model_check import full_build as FB
from merlin.model_check import full_quantized as FQ
from merlin.model_check import models as M

SCHEMA = "merlin.operator.full_declared_output_audit.v1"
REFERENCE_SCHEMA = "merlin.operator.full_declared_output_reference.v1"
FLOAT_POLICY = {"atol": 2e-6, "rtol": 1e-5, "equal_nan": False, "finite_required": True}
DTYPES = {
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "f16": "float16",
    "bf16": "uint16",
    "f32": "float32",
    "f64": "float64",
}


class AuditError(ValueError):
    """Missing correspondence is not a numerical pass."""


def _require(condition, detail):
    if not condition:
        raise AuditError(detail)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON field {key!r}")
        result[key] = value
    return result


class Files:
    """Remember every consumed byte identity and recheck before returning evidence."""

    def __init__(self):
        self.pins = {}

    def pin(self, path, expected=None):
        path = Path(path)
        _require(path.is_absolute() and path.is_file(), f"missing absolute artifact: {path}")
        digest = sha256_file(path)
        _require(expected is None or digest == expected, f"artifact differs from its selected digest: {path}")
        _require(path not in self.pins or self.pins[path] == digest, f"artifact changed while reopening: {path}")
        self.pins[path] = digest
        return {"path": str(path), "sha256": digest}

    def selected(self, pin):
        _require(
            isinstance(pin, dict) and isinstance(pin.get("sha256"), str) and len(pin["sha256"]) == 64,
            "missing selected artifact digest",
        )
        return self.pin(pin["path"], pin["sha256"])

    def json(self, path):
        self.pin(path)
        with Path(path).open() as stream:
            value = json.load(stream, object_pairs_hook=_unique_object)
        _require(isinstance(value, dict), "JSON artifact must be an object")
        self.pin(path)
        return value

    def arrays(self, path, *, arity=None):
        self.pin(path)
        with np.load(path, allow_pickle=False) as archive:
            if arity is not None:
                keys = [f"out{i}" for i in range(arity)]
                _require(
                    len(archive.files) == arity and set(archive.files) == set(keys), "output NPZ arity/names differ"
                )
            else:
                _require(len(set(archive.files)) == len(archive.files), "duplicate NPZ array name")
                keys = archive.files
            result = {key: np.asarray(archive[key]) for key in keys}
        self.pin(path)
        return result

    def unchanged(self):
        for path, digest in tuple(self.pins.items()):
            self.pin(path, digest)


def _argument(argv, name):
    indices = [index for index, value in enumerate(argv) if value == name]
    _require(len(indices) == 1 and indices[0] + 1 < len(argv), f"job has no unique {name}")
    return argv[indices[0] + 1]


def _same(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def _numeric_contract(files, bundle, validation, expected):
    """Native references also have an independently sealed numerical sidecar."""
    from merlin.runtime import linalg_numpy as LN

    path = Path(validation["path"]).with_name("reference_numeric_contract.json")
    name = path.relative_to(bundle.root).as_posix()
    matches = []
    with (bundle.root / "SHA256SUMS").open() as stream:
        for line in stream:
            digest, separator, member = line.rstrip("\n").partition("  ")
            if separator and member == name:
                matches.append(digest)
    _require(len(matches) == 1, "selected numerical sidecar is not uniquely sealed")
    files.pin(path, matches[0])
    receipt = files.json(path)
    evaluator = files.pin(Path(LN.__file__).resolve())
    _require(
        receipt.get("schema") == "merlin.model_check.reference_numeric_contract.v1"
        and receipt.get("outputs_sha256") == validation["sha256"]
        and receipt.get("numeric_contract") == expected
        and expected.get("version") == LN.NUMERIC_CONTRACT_VERSION
        and receipt.get("evaluator_sha256") == evaluator["sha256"],
        "supplement numeric contract differs from the original selected reference",
    )


def _decode_readback(console, specs, build_hash):
    """Ordinary packet/ABI readback checks with one source-bound program signature."""
    from merlin.compile.baremetal_model import console_text_metrics
    from merlin.runtime.out_bin import binary_console_diagnostics, parse_binary_console_details
    from merlin.runtime.whole_model_readback import decode_outputs

    outputs, _numeric, frames = parse_binary_console_details(console)
    metrics = console_text_metrics(binary_console_diagnostics(console).decode("utf-8", errors="replace"))
    _require(
        isinstance(build_hash, str) and bool(build_hash) and metrics.get("build_hash") == build_hash,
        "whole-model console does not identify the linked build",
    )
    _require(metrics.get("memref_rank_mismatch") == "0", "whole-model console lacks a clean memref-rank diagnostic")
    for index, (shape, _dtype) in enumerate(specs):
        # Match render_outputs_header's C producer: the final extent is columns
        # for every rank, and a scalar occupies one word.
        _require(all(type(dim) is int and dim > 0 for dim in shape), "declared readback shape must be static")
        cols, count = shape[-1] if shape else 1, 1
        for dim in shape:
            count *= dim
        frame = frames.get(f"out{index}")
        _require(
            frame is not None and frame.rows == count // cols and frame.cols == cols,
            "whole-model frame shape differs from the declared output",
        )
    values = decode_outputs(outputs, frames, specs)
    for value, (_shape, dtype) in zip(values, specs, strict=True):
        if dtype == "i1":
            _require(bool(np.all((value == 0) | (value == 1))), "Boolean readback has noncanonical wire bits")
    return [
        np.asarray(value).astype(bool) if dtype == "i1" else np.asarray(value)
        for value, (_shape, dtype) in zip(values, specs, strict=True)
    ]


def _execution(files, directory, *, specs, build_hash, runtime, producer_sha, extension_runtime_key):
    observations = []
    for path in sorted((directory / "invocations").glob("*/invocation.json")):
        record = files.json(path)
        if record.get("stage") == "selected_rocc_process":
            observations.append((path, record))
    _require(len(observations) == 1, "step lacks one unambiguous original selected Spike observation")
    path, record = observations[0]
    _require(
        record.get("schema") == "merlin.invocation_record.v1"
        and record.get("kind") == "subprocess"
        and record.get("status") == "completed"
        and type(record.get("returncode")) is int
        and record["returncode"] == 0
        and all(
            record.get(key) is True for key in ("inputs_unchanged", "dependencies_unchanged", "executable_unchanged")
        ),
        "step has no successful unchanged original execution",
    )
    executable = files.selected(record["executable"])
    _require(
        executable["sha256"] == runtime["spike"] and record["argv"][0] == executable["path"],
        "step executed another Spike binary",
    )
    dependencies = [files.selected(pin) for pin in record["dependencies"]]
    _require(
        any(pin["sha256"] == runtime[extension_runtime_key] for pin in dependencies),
        "step has no selected accelerator extension binding",
    )
    for stream in ("stdout", "stderr"):
        files.selected(record[stream])
    recipe_path = directory / "compilation_recipe.json"
    recipe = files.json(recipe_path)
    _require(
        recipe.get("schema") == "merlin.compilation_recipe.v1"
        and recipe.get("status") == "completed"
        and recipe.get("commands")
        and all(row.get("status") == "returned" for row in recipe["commands"]),
        "step has no completed original relink recipe",
    )
    files.selected(recipe["producer"])
    _require(recipe["producer"]["sha256"] == producer_sha, "relink recipe has another source producer")
    original_elf = recipe["executable"]
    _require(
        recipe["commands"][-1].get("output") == original_elf, "relink receipt differs from its observed final command"
    )
    unavailable = []
    for command in recipe["commands"]:
        files.selected(command["executable"])
        for identity in command["inputs"]:
            if Path(identity["path"]).is_file():
                files.selected(identity)
            else:
                unavailable.append(identity["path"])
    _require(
        original_elf["path"] == str(directory / "model.elf")
        and record.get("inputs") == [{"path": original_elf["path"], "sha256": original_elf["sha256"]}],
        "executed ELF differs from the original relink product",
    )
    # These are records of already-deleted transient files. Do not claim their
    # current bytes were reopened, or manufacture replacement ELF receipts.
    if Path(original_elf["path"]).is_file():
        files.selected(original_elf)
    values = _decode_readback(Path(record["stdout"]["path"]).read_bytes(), specs, build_hash)
    return values, {
        "execution": files.pin(path),
        "relink_recipe": files.pin(recipe_path),
        "original_elf_sha256": original_elf["sha256"],
        "transient_elf_current_reopen": "retained" if Path(original_elf["path"]).is_file() else "unavailable_deleted",
        "transient_recipe_inputs_current_unavailable": sorted(set(unavailable)),
        "raw_readback_reopened": True,
    }


def verify(
    *,
    reference_path,
    expected_reference_sha256,
    job_pin_path,
    report_path,
    models_path,
    run_root,
    expected_identity,
    requirement_path,
    extension_runtime_key,
    expected_identity_path=None,
):
    """Reopen one complete input chain against all original declared reference outputs."""
    files = Files()
    files.pin(reference_path, expected_reference_sha256)
    if expected_identity_path is not None:
        _require(files.json(expected_identity_path) == expected_identity, "selected identity file changed")
    requirement = files.pin(requirement_path)
    from merlin.compile import baremetal_model
    from merlin.llvmlower import c_runtime
    from merlin.runtime import out_bin, whole_model_readback

    operator_sources = [
        files.pin(Path(module.__file__).resolve())
        for module in (CH, F, FB, FQ, M, B, baremetal_model, c_runtime, out_bin, whole_model_readback)
    ]
    operator_sources.append(files.pin(Path(__file__).resolve()))
    reference, job, report = (files.json(Path(path)) for path in (reference_path, job_pin_path, report_path))
    _require(
        reference.get("schema") == REFERENCE_SCHEMA
        and reference.get("status") == "complete"
        and reference.get("representation") == "declared_storage",
        "reference supplement is incomplete",
    )
    _require(job.get("selected_identity") == expected_identity, "job selection differs from the admitted experiment")
    for key, value in reference["selected_identity"].items():
        _require(
            key in {"bundle_manifest_sha256", "bundle_sums_sha256", "models_config_sha256"}
            and expected_identity.get(key) == value,
            "reference belongs to another selected bundle/configuration",
        )
    _require(
        set(reference["selected_identity"]) == {"bundle_manifest_sha256", "bundle_sums_sha256", "models_config_sha256"},
        "reference selection is incomplete",
    )
    files.pin(models_path, expected_identity["models_config_sha256"])
    models = M.load(models_path)
    model, input_id = models.model(reference["model"]), reference["input_id"]
    _require(
        input_id.count("/") == 1 and all(part and part not in {".", ".."} for part in input_id.split("/")),
        "invalid selected input identifier",
    )
    argv = job["argv"]
    for key, value in (
        ("--models", str(models_path)),
        ("--model", model.name),
        ("--input", input_id),
        ("--out", str(report_path)),
        ("--mode", "full"),
        ("--engine", "spike"),
        ("--full-comparison", "quantized-reference"),
    ):
        _require(_argument(argv, key) == value, f"job {key} differs from the audited invocation")
    bundle = B.Bundle(Path(_argument(argv, "--bundle")))
    files.pin(bundle.root / "MANIFEST.json", expected_identity["bundle_manifest_sha256"])
    files.pin(bundle.root / "SHA256SUMS", expected_identity["bundle_sums_sha256"])
    _require(
        float(_argument(argv, "--quantized-reference-atol")) == FLOAT_POLICY["atol"]
        and float(_argument(argv, "--quantized-reference-rtol")) == FLOAT_POLICY["rtol"],
        "job float policy differs",
    )
    _require(
        report.get("model") == model.name and report.get("comparison", {}).get("float_policy") == FLOAT_POLICY,
        "full report has another model or numeric comparison",
    )
    submission = job["submission"]
    _require(
        _argument(argv, "--submission") == submission["path"]
        and report.get("submission")
        == {"path": submission["path"], "tree_sha256": submission["model_check_tree_sha256"]},
        "report/command differs from the frozen candidate",
    )
    from merlin.common.tree_hash import hash_tree
    from merlin.model_check.cli import tree_sha256

    _require(
        hash_tree(Path(submission["path"]))["sha256"] == submission["hash_tree_sha256"]
        and tree_sha256(Path(submission["path"])) == submission["model_check_tree_sha256"],
        "frozen candidate changed",
    )
    for pin in reference["implementation"]["source_pins"]:
        files.selected(pin)
    files.selected(reference["input"])
    _require(
        Path(reference["input"]["path"]).resolve() == F.derived_input(bundle, model.name, input_id).resolve(),
        "supplement input differs from the selected model input",
    )
    sealed_values, sealed_validation = FQ.sealed_reference(bundle, model, models_path, input_id)
    _numeric_contract(files, bundle, sealed_validation, reference["implementation"]["numeric_contract"])
    arrays = files.arrays(Path(reference["input"]["path"]))
    programs = reference["programs"]
    _require(
        [row["program"] for row in programs] == [program.name for program in model.programs],
        "reference program roster/order differs from the model",
    )
    report_programs = report.get("programs", [])
    _require(
        len(report_programs) == len(model.programs)
        and {row["program"] for row in report_programs} == {p.name for p in model.programs},
        "compiled report program roster differs",
    )
    report_programs = {row["program"]: row for row in report_programs}
    actual_chain, reference_chain = CH.ChainRun(), CH.ChainRun()
    work = Path(job["work"]) / submission["model_check_tree_sha256"][:16] / model.name
    _require(
        _argument(argv, "--work") == job["work"] and Path(run_root).resolve().parent == (work / "runs").resolve(),
        "audited run root is outside the original job's model work directory",
    )
    comparisons, steps, declarations = [], [], []
    for program, declaration in zip(model.programs, programs, strict=True):
        _require(
            declaration["steps_declared"] == program.steps
            and [row["step"] for row in declaration["steps"]] == list(range(program.steps)),
            "reference does not cover every declared step",
        )
        stage = Path(declaration["stage"])
        for name in ("catalog", "reference_program", "capture_program"):
            files.selected(declaration[name])
        catalog = bundle.catalog(model.name, program.name)
        _require(
            declaration["capture_program"]["path"] == str(stage / "model.mlir")
            and stage.resolve() == catalog.stage.resolve()
            and Path(declaration["reference_program"]["path"]).resolve() == catalog.ref_prepared.resolve()
            and Path(declaration["catalog"]["path"]) == bundle.program_dir(model.name, program.name) / "catalog.json",
            "capture/reference catalog stage differs",
        )
        from merlin.llvmlower.c_runtime import _out_specs

        specs = _out_specs(stage / "model.mlir")
        schema = [
            {"output": index, "dtype": np.dtype(DTYPES[dtype]).str, "shape": shape, "element_type": dtype}
            for index, (shape, dtype) in enumerate(specs)
        ]
        _require(
            schema == declaration["declared_outputs"] and bool(schema), "reference omits or changes declared outputs"
        )
        image = report_programs[program.name]
        build = files.json(work / "programs" / program.name / "build.json")
        _require(
            build.get("status") == "linked"
            and build["key"]["stage"] == str(stage.resolve())
            and build["key"]["target"] == models.target
            and build["key"]["submission_sha256"] == submission["model_check_tree_sha256"]
            and all(build[key] == image["build"][key] for key in ("elf_sha256", "build_hash")),
            "original model build differs from the selected candidate/stage/report",
        )
        declarations.append(
            {
                "program": program.name,
                "capture_program": declaration["capture_program"],
                "steps_declared": program.steps,
                "outputs_per_step": len(schema),
                "outputs_expected": program.steps * len(schema),
            }
        )
        rows = image.get("runs", [])
        _require(
            len(rows) == program.steps
            and [row["step"] for row in rows] == list(range(program.steps))
            and all(
                row.get("status") == "completed" and row.get("engine") == "spike" and row.get("input") == input_id
                for row in rows
            ),
            "actual execution chain is incomplete",
        )
        actual_chain.steps[program.name], reference_chain.steps[program.name] = [], []
        files.pin(stage / "inputs.npz")
        template = files.arrays(stage / "inputs.npz")
        for step, refstep in enumerate(declaration["steps"]):
            directory = Path(run_root) / input_id / f"{program.name}.{step}"
            bound = CH.bind(program, model, arrays, step, actual_chain)
            retained_inputs = files.arrays(directory / "capture/inputs.npz")
            _require(set(bound) == set(template) == set(retained_inputs), "step bound-input arity differs")
            _require(
                all(
                    np.asarray(bound[key]).shape == template[key].shape
                    and _same(np.asarray(bound[key]).astype(template[key].dtype, copy=False), retained_inputs[key])
                    for key in bound
                ),
                "step input binding differs from the actual retained inputs",
            )
            files.selected(refstep["outputs_file"])
            expected = list(files.arrays(Path(refstep["outputs_file"]["path"]), arity=len(schema)).values())
            actual = list(files.arrays(directory / "outputs.npz", arity=len(schema)).values())
            _require(len(refstep["outputs"]) == len(schema), "reference array identities are incomplete")
            for value, declared, identity in zip(expected, schema, refstep["outputs"], strict=True):
                _require(
                    all(identity.get(key) == field for key, field in declared.items())
                    and value.dtype.str == declared["dtype"]
                    and list(value.shape) == declared["shape"]
                    and hashlib.sha256(value.tobytes()).hexdigest() == identity["sha256_of_array_bytes"],
                    "reference output differs from its declared array identity",
                )
            readback, custody = _execution(
                files,
                directory,
                specs=specs,
                build_hash=image["build"]["build_hash"],
                runtime=expected_identity["runtime_pins"],
                producer_sha=build["key"]["source"]["model_check/full_build.py"],
                extension_runtime_key=extension_runtime_key,
            )
            _require(
                len(readback) == len(actual) and all(_same(a, b) for a, b in zip(readback, actual, strict=True)),
                "saved outputs differ from the original successful Spike readback",
            )
            bindings = [{"program": program.name, "step": step, "output": index} for index in range(len(schema))]
            comparisons.extend(FQ.compare_outputs(actual, expected, bindings, FLOAT_POLICY))
            steps.append(
                {
                    "program": program.name,
                    "step": step,
                    "outputs": len(schema),
                    "actual": files.pin(directory / "outputs.npz"),
                    "reference": refstep["outputs_file"],
                    **custody,
                }
            )
            actual_chain.steps[program.name].append(F.carried(actual, specs))
            reference_chain.steps[program.name].append(CH.to_compute(expected, [dtype for _, dtype in specs]))
    crossjoin = reference["judged_crossjoin"]
    files.selected(crossjoin["sealed_npz"])
    _require(
        crossjoin["sealed_npz"]["path"] == sealed_validation["path"]
        and crossjoin["sealed_npz"]["sha256"] == sealed_validation["sha256"],
        "sealed judged reference differs",
    )
    judged = CH.judged_outputs(model, reference_chain)
    sealed = list(files.arrays(Path(crossjoin["sealed_npz"]["path"]), arity=len(judged)).values())
    _require(
        len(judged) == crossjoin["outputs"] == len(sealed_values)
        and all(_same(a, b) for a, b in zip(judged, sealed, strict=True)),
        "supplement differs from original sealed judged outputs",
    )
    files.unchanged()
    _require(
        hash_tree(Path(submission["path"]))["sha256"] == submission["hash_tree_sha256"]
        and tree_sha256(Path(submission["path"])) == submission["model_check_tree_sha256"],
        "candidate changed during audit",
    )
    return {
        "schema": SCHEMA,
        "status": "pass" if all(row["pass"] for row in comparisons) else "fail",
        "scope": "offline supplemental all-declared-output comparison; no new execution or campaign qualification",
        "model": model.name,
        "input_id": input_id,
        "selected_identity": expected_identity,
        "execution_policy": {"engine": "spike", "extension_runtime_key": extension_runtime_key},
        "requirement": {
            "source": requirement,
            "scope": "Every declared output must match the selected quantized reference",
        },
        "submission": submission,
        "reference_manifest": files.pin(reference_path),
        "job_pin": files.pin(job_pin_path),
        "full_report": files.pin(report_path),
        "run_root": str(run_root),
        "float_policy": FLOAT_POLICY,
        "steps_compared": len(steps),
        "steps_expected": sum(program.steps for program in model.programs),
        "n_judged_outputs": len(judged),
        "n_declared_outputs": sum(row["outputs_expected"] for row in declarations),
        "programs": declarations,
        "operator_sources": operator_sources,
        "outputs_compared": len(comparisons),
        "comparisons": comparisons,
        "steps": steps,
        "artifacts": [files.pin(path) for path in files.pins],
        "unchanged": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference",
        "job-pin",
        "report",
        "models",
        "run-root",
        "expected-identity",
        "requirement-source",
        "out",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--extension-runtime-key", required=True)
    parser.add_argument("--reference-sha256", required=True)
    args = parser.parse_args()
    identity = json.loads(args.expected_identity.read_text())
    try:
        result = verify(
            reference_path=args.reference,
            expected_reference_sha256=args.reference_sha256,
            job_pin_path=args.job_pin,
            report_path=args.report,
            models_path=args.models,
            run_root=args.run_root,
            expected_identity=identity,
            requirement_path=args.requirement_source,
            extension_runtime_key=args.extension_runtime_key,
            expected_identity_path=args.expected_identity,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        result = {
            "schema": SCHEMA,
            "status": "error",
            "detail": f"{type(error).__name__}: {error}",
            "scope": "offline supplemental comparison unavailable; not a numerical pass",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

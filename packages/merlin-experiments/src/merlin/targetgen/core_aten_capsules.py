"""Materialize and grade captured single-call capsules with the Core ATen boundary contract.

The batch builder owns result projection, mutation returns and semantic I/O. This adapter
only separates public compiler inputs from private answers and joins the existing capsule
entrypoints to that boundary grader. It does not provide a candidate compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import yaml

from merlin.targetgen.capsule_common import load_capsule
from merlin.targetgen.core_aten_batch import build_core_aten_batch, caller_boundary_layouts, caller_input_storage
from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch
from merlin.targetgen.core_aten_capture import case_capture_name
from merlin.targetgen.core_aten_cover import _provenance_ops
from merlin.targetgen.golden_store import FILES, load_golden, write_golden
from merlin.targetgen.package_runtime import CONTRACT_VERSION

CONTRACT = "full_call_v1"


class CapsuleBuildError(ValueError):
    """All case-local refusals from a bounded parallel build; no corpus is promoted."""

    def __init__(self, failures):
        self.failures = failures
        first = failures[0]
        super().__init__(f"{len(failures)} capsule(s) refused; first: {first['case_id']}: {first['error']}")


class NumericFixtureError(ValueError):
    """Retain private admission decisions without printing reference answers."""

    def __init__(self, message, plan):
        # The stage receipt already binds shared specs/facts. Repeating their
        # complete selection and provider catalogs per refused case is costly
        # at corpus scale, and expected bytes are unnecessary for diagnosis.
        self.plan = {key: value for key, value in plan.items() if key not in {"selection", "provider_catalog"}}
        if isinstance(self.plan.get("integer_reference"), dict):
            self.plan["integer_reference"] = {
                key: value for key, value in self.plan["integer_reference"].items() if key != "expected_output_bytes"
            }
        super().__init__(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, document) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_full_call(capsule: dict) -> bool:
    return (capsule.get("operation", {}).get("attributes", {})).get("core_aten_contract") == CONTRACT


def write_capsules(
    corpus: dict,
    captures: Path,
    public: Path,
    private: Path,
    *,
    label="public",
    tiers=("L2",),
    numeric_flow=None,
    workers=1,
    admission_lanes=False,
    truth_lanes=False,
):
    """Write one capsule per case, using precisely the round-2 result projection.

    ``private`` must be outside the compiler-input tree. No seal or hardware admission is
    implied. Additive bounded case IDs retain the capture tool's content-derived names.
    """
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("capsule workers must be an integer in 1..8")
    if truth_lanes:
        admission_lanes = True
    if admission_lanes and numeric_flow is None:
        raise ValueError("admission-derived lanes require numeric_flow")
    public, private, captures = public.resolve(), private.resolve(), captures.resolve()
    if private.is_relative_to(public) or public.is_relative_to(private):
        raise ValueError("public inputs and private answers require disjoint roots")
    if label not in {"public", "dev", "hidden"} or not tiers:
        raise ValueError("explicit label and nonempty oracle tiers required")
    capture_names = [case_capture_name(case) for case in corpus["cases"]]
    names = [name.replace("-", "_") for name in capture_names]
    if len(names) != len(set(names)):
        raise ValueError("duplicate capsule identifiers")
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.chmod(0o700)
    # Spawn isolates numeric preparation's provider/environment state. Submit only a
    # small window of single-case documents; a full-suite copy per task is quadratic.
    metadata = {key: value for key, value in corpus.items() if key not in {"cases", "selected_cases"}}
    tasks = (
        (
            {**metadata, "cases": [case]},
            captures,
            public,
            private,
            label,
            tiers,
            numeric_flow,
            admission_lanes,
            truth_lanes,
        )
        for case in corpus["cases"]
    )
    if workers == 1:
        rows = [_write_case(task) for task in tasks]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            # Keep futures and pickled documents bounded, preserving declaration order.
            from itertools import islice

            while window := list(islice(tasks, workers * 2)):
                rows.extend(pool.map(_try_write_case, window))
        failures = [row for row in rows if "error" in row]
        if failures:
            raise CapsuleBuildError(failures)
    manifest = {
        "schema": "merlin.core_aten_capsules.v1",
        "label": label,
        "count": len(rows),
        "capsules": rows,
        "provenance": {
            "scope": "packaging retained captures; no execution or hardware verdict",
            "denominator_sha256": corpus.get("denominator_sha256"),
        },
    }
    _json(public / "manifest.json", manifest)
    if label == "hidden":
        (public / "manifest.json").chmod(0o600)
    return manifest


def _try_write_case(task):
    try:
        return _write_case(task)
    except Exception as exc:  # noqa: BLE001 -- a case refusal must not hide other cases
        case = task[0]["cases"][0]
        return {
            "case_id": case.get("case_id", case["overload"]),
            "error": f"{type(exc).__name__}: {exc}",
            **({"numeric_preparation": exc.plan} if isinstance(exc, NumericFixtureError) else {}),
        }


def _write_case(task):
    corpus, captures, public, private, label, tiers, numeric_flow, admission_lanes, *policy = task
    truth_lanes = bool(policy and policy[0])
    case = corpus["cases"][0]
    capture_name = case_capture_name(case)
    name = capture_name.replace("-", "_")
    source = captures / capture_name
    destination, answers = public / name, private / name
    if destination.exists() or answers.exists():
        raise ValueError(f"refusing to overwrite capsule {name}")
    capture = json.loads((source / "capture.json").read_text())
    if capture.get("overload") != case["overload"]:
        raise ValueError(f"capture overload differs from case: {name}")
    if not capture["capture_meta"].get("result_contract"):
        raise ValueError(f"full-call capsule lacks same-conversion boundary: {name}")
    with tempfile.TemporaryDirectory(prefix="capsule-", dir=private) as scratch:
        bundle = Path(scratch)
        report = build_core_aten_batch({**corpus, "cases": [case]}, captures, bundle)
        if report["bundled_count"] != 1 or not (bundle / "semantic_io.json").is_file():
            raise ValueError(f"full-call capsule did not bundle: {name}: {report['cases']}")
        from merlin.targetgen.core_aten_capture import descriptor_specialization_matches

        if case["overload"] not in _provenance_ops(bundle / "model.mlir") and not descriptor_specialization_matches(
            case, capture
        ):
            raise ValueError(f"full-call interface lacks exact overload provenance: {name}")
        admission = None
        if numeric_flow is not None:
            admission = prepare_numeric_fixture(bundle, report, numeric_flow, require_device=not admission_lanes)
        device = admission is not None and _device_fixture(admission)
        commitment = None
        if truth_lanes:
            from merlin.targetgen.functional_truth import expectation, require_placement

            commitment = expectation(case)
            try:
                require_placement(commitment, device=device)
            except ValueError as exc:
                raise NumericFixtureError(str(exc), admission or {}) from exc
        if admission_lanes and not device:
            # Refused numeric preparations cannot alter the original host ABI.
            # Rebuild the raw call, preserving the precise admission decision.
            report = build_core_aten_batch({**corpus, "cases": [case]}, captures, bundle)
            numeric_flow = None
        destination.mkdir(mode=0o700 if label == "hidden" else 0o755)
        answers.mkdir(mode=0o700)
        shutil.copyfile(bundle / "model.mlir", destination / "capsule.interface.mlir")
        for filename in ("inputs.npz", "input_order.json", "weights.safetensors.manifest.json"):
            shutil.copyfile(bundle / filename, destination / filename)
        shutil.copyfile(source / "capsule.pytorch.py", destination / "capsule.pytorch.py")
        # The public ABI contains no expected results, mutation observations or answers.
        record = report["cases"][0]
        # Full-call native results use storage bits, even when the source corpus
        # carries a looser diagnostic comparison. Do not mutate that corpus.
        record["case"] = {
            **record["case"],
            "comparison_parameters": {"rtol": 0, "atol": 0, "storage_bits": True},
        }
        _json(
            destination / "call.json",
            {
                "schema_version": 1,
                "overload": case["overload"],
                "input_abi": record["input_abi"],
                "output_abi": record["output_abi"],
                "output_indices": record["output_indices"],
                "post_indices": record["semantic_boundary"]["post_indices"],
                "comparison": case["comparison"],
                "comparison_parameters": case.get("comparison_parameters", {}),
                "output_count": report["output_count"],
            },
        )
        public_files = {p.name: _sha(p) for p in sorted(destination.iterdir())}
        # Retain original evidence host-side, without copying capture-work trees.
        _json(answers / "capture.json", capture)
        write_golden(
            answers,
            {
                "schema": "merlin.core_aten_capsule.v1",
                "batch_map": report,
                "semantic_io": json.loads((bundle / "semantic_io.json").read_text()),
                "public_files": public_files,
            },
        )
        binding = {filename: _sha(answers / filename) for filename in FILES if (answers / filename).is_file()}
        dtypes = [item["dtype"] for item in record["output_abi"]]
        policy_dtype = next((dtype for dtype in dtypes if dtype.startswith(("f", "complex<"))), dtypes[0])
        if policy_dtype.startswith("complex<"):
            policy_dtype = policy_dtype.removeprefix("complex<").removesuffix(">")
        declaration = {
            "name": name,
            "kind": "model_slice",
            "source_role": "pytorch_model_slice",
            "label": label,
            "source_reference": case["overload"],
            "interface_mlir": "capsule.interface.mlir",
            "operation": {
                "op": "model",
                "attributes": {
                    "core_aten_contract": CONTRACT,
                    "overload": case["overload"],
                },
            },
            "numeric_policy": {
                "compare": "exact_bits",
                "dtype": policy_dtype,
                "rtol": 0.0,
                "atol": 0.0,
            },
            "expected": {"instruction_classes": []},
            "required_oracle_tiers": list(tiers),
            "full_call_golden": binding,
        }
        if numeric_flow is not None:
            declaration["operation"]["attributes"]["numeric_flow"] = numeric_flow["policy"]
            declaration["numeric_policy"].update(
                compare="exact_bits",
                rtol=0.0,
                atol=0.0,
            )
        (destination / "capsule.yaml").write_text(yaml.safe_dump(declaration, sort_keys=False))
        load_capsule(destination)
        for path in answers.iterdir():
            path.chmod(0o600)
        if label == "hidden":
            for path in destination.iterdir():
                path.chmod(0o600)
        return {
            **(
                {
                    "lane_expectation": commitment["lane"] if commitment else "device" if device else "host-guard",
                    "native_admission": admission,
                    **({"functional_truth": commitment} if commitment else {}),
                }
                if admission_lanes
                else {}
            ),
            "id": name,
            "overload": case["overload"],
            "case_id": case.get("case_id", case["overload"]),
            "digests": {**public_files, "capsule.yaml": _sha(destination / "capsule.yaml")},
            "golden_binding": binding,
            "provenance": {
                "capture_sha256": _sha(source / "capture.json"),
                "capture_source_sha256": _sha(source / "capsule.linalg.mlir"),
                "model2mlir_revision": capture.get("model2mlir_revision"),
                "captured_overloads": capture.get("captured_overloads"),
                "capture_abi_version": capture["capture_meta"].get("capture_abi_version"),
                "retained_capture_provenance": capture.get("provenance"),
            },
        }


def _native_fixture(plan: dict) -> bool:
    """The same complete native proof required at preparation and readback."""
    return (
        plan.get("native_admission", {}).get("status") != "host"
        and plan.get("numerical_family") in {"F1", "F2", "F4", "F5"}
        and plan.get("status") == "admitted"
        and plan.get("exact_dtype_sites", 0) > 0
        and (plan.get("host_floor") or {}).get("status") == "required"
        and (plan.get("host_floor") or {}).get("comparison") == "native_eager_bits_every_boundary_result"
        and all(site.get("status") == "admitted" for site in plan.get("sites", []))
        and len(plan.get("sites", [])) == plan.get("exact_dtype_sites")
    )


def _device_fixture(plan: dict) -> bool:
    if plan.get("native_admission", {}).get("status") == "host":
        return False
    admitted = sum(
        plan.get(key, 0) or 0
        for key in ("quantized_contractions", "native_contractions", "quantized_non_contraction_sites")
    )
    return _native_fixture(plan) or (admitted > 0 and (plan.get("integer_reference") or {}).get("status") == "prepared")


def prepare_numeric_fixture(bundle: Path, report: dict, flow: dict, *, require_device=True) -> dict:
    """Freeze a selected declared chain and independent reference, never candidate answers."""
    from merlin.targetgen.core_aten_device import load_execution_provider, selected_facts
    from merlin.targetgen.core_aten_numeric import prepare_spec_int8

    if flow["policy"] != "spec_int8":
        raise ValueError("unsupported full-call numeric fixture policy; explicit conformant preparation required")
    provider = load_execution_provider(flow["target"], Path(flow["execution_provider"]))
    with selected_facts(flow["target"], Path(flow["rtl_facts"])) as facts:
        plan = prepare_spec_int8(
            bundle,
            report,
            target=flow["target"],
            facts=facts,
            software_spec=Path(flow["software_spec"]),
            host_capabilities=Path(flow["host_capabilities"]),
            provider=provider,
            package=Path(flow["device_package"]),
        )
    # A captured W8A8 graph is quantized at preparation (quantized_contractions); a capture that is
    # already i8 x i8 -> i32 is admitted as is (native_contractions); a contraction-free program's
    # int8 sites are admitted as quantized islands (quantized_non_contraction_sites). Each needs the
    # independent exact integer reference.
    if require_device and not _device_fixture(plan):
        raise NumericFixtureError(
            "scored capsule has no admitted contraction with an exact integer reference: "
            + str(
                [
                    (r.get("case_id", r.get("overload")), r.get("routing"))
                    for r in report["cases"]
                    if r.get("status") == "bundled"
                ]
            )
            + "; numeric refusal: "
            + str(
                plan.get("refusal_code")
                or plan.get("refusal_details")
                or [s.get("reasons") for key in ("sites", "non_contraction_sites") for s in plan.get(key, [])]
            ),
            plan,
        )
    report["numeric_preparation"] = plan
    for record in report["cases"]:
        record["numeric_policy"] = flow["policy"]
    return plan


def load_private(capsule: dict, private_root: Path | None = None) -> dict:
    """Verify the answer commitment and every public compiler-input byte before grading."""
    root = Path(capsule["__dir__"]).resolve()
    selected = private_root or os.environ.get("MERLIN_CORE_ATEN_PRIVATE_ROOT")
    answers = (Path(selected) if selected else root.parent.parent / "_private" / root.parent.name) / capsule["name"]
    if answers.is_symlink() or not answers.is_dir():
        raise ValueError("private full-call answer directory is absent or indirect")
    binding = capsule.get("full_call_golden")
    if not isinstance(binding, dict) or "golden.yaml" not in binding or set(binding) - set(FILES):
        raise ValueError("full-call golden binding is absent or malformed")
    for filename, digest in binding.items():
        if (answers / filename).is_symlink() or _sha(answers / filename) != digest:
            raise ValueError("private full-call answer digest changed")
    golden = load_golden(answers)
    if golden.get("schema") != "merlin.core_aten_capsule.v1":
        raise ValueError("private full-call record schema differs")
    for filename, digest in golden["public_files"].items():
        if Path(filename).name != filename or (root / filename).is_symlink() or _sha(root / filename) != digest:
            raise ValueError(f"public full-call input changed: {filename}")
    report = golden["batch_map"]
    if len(report["cases"]) != 1 or report["cases"][0]["overload"] != capsule["operation"]["attributes"]["overload"]:
        raise ValueError("private case binding differs from declaration")
    return golden


def grade_readback(
    capsule: dict,
    output_bytes,
    *,
    output_shapes=None,
    semantic_readback=None,
    execution_error=None,
    private_root=None,
    provenance=None,
    routing=None,
) -> dict:
    """The same numeric and full-call semantic checker used by the round-2 campaign."""
    golden = load_private(capsule, private_root)
    report = golden["batch_map"]
    if routing is not None:
        for record in report["cases"]:
            record["routing"] = routing
    if provenance is not None:
        report["provenance"] = provenance
    if output_bytes is not None and len(output_bytes) != report["output_count"]:
        execution_error = "full-call execution did not emit exactly the declared result count"
    if output_shapes and len(output_shapes) != report["output_count"]:
        execution_error = "full-call execution result shape frame count differs from the declared ABI"
    verdict = grade_core_aten_batch(
        report,
        output_bytes,
        output_shapes=output_shapes or None,
        semantic_readback=semantic_readback,
        execution_error=execution_error,
    )
    row = next(iter(verdict["cases"].values()))
    if capsule.get("operation", {}).get("attributes", {}).get("numeric_flow") and row["status"] != "execution_failed":
        plan = report.get("numeric_preparation", {})
        reference_passed = row.get("integer_reference", {}).get("status") == "pass" or (
            _native_fixture(plan)
            and plan.get("native_admission", {}).get("status") != "host"
            and row.get("host_floor", {}).get("status") == "pass"
        )
        if routing is None or not reference_passed:
            row.update(
                status="mismatch",
                reason="mandatory exact numeric reference or attributed execution evidence did not pass",
            )
            verdict["passed_count"] = 0
            verdict["status_counts"] = {"mismatch": 1}
    # An observed boundary retains its imposed-metadata limitation; numeric-only never passes.
    if row["status"] == "pass" and row.get("semantic_scope") not in {"full", "values+mutation+observed-metadata"}:
        raise ValueError("full-call capsule cannot earn a numeric-only pass")
    return verdict


def runtime_bundle(capsule: dict, destination: Path) -> dict:
    golden = load_private(capsule)
    root = Path(capsule["__dir__"])
    destination.mkdir(parents=True, exist_ok=True)
    source = (root / "capsule.interface.mlir").read_text()
    prepared = caller_input_storage(source, golden["batch_map"]["cases"])
    (destination / "model.mlir").write_text(_legacy_view_storage(capsule, golden, source, prepared))
    for filename in ("inputs.npz", "input_order.json", "weights.safetensors.manifest.json"):
        shutil.copyfile(root / filename, destination / filename)
    # Reconstruct caller layouts also for previously sealed normalized captures.
    # The sealed records remain immutable; only runtime boundary metadata changes.
    semantic_io = caller_boundary_layouts(
        (destination / "model.mlir").read_text(), golden["semantic_io"], golden["batch_map"]["cases"]
    )
    _json(destination / "semantic_io.json", semantic_io)
    plan = golden["batch_map"].get("numeric_preparation")
    if plan is not None:
        _json(destination / "spec-int8-admission.json", plan)
    return golden["batch_map"]


def _legacy_view_storage(capsule: dict, golden: dict, source: str, prepared: str) -> str:
    """Recover a bound same-conversion view contract, never eager answer metadata.

    The copied capture.json is not independently committed. Its boundary may be
    used only when it agrees with the compiler boundary in the bound golden
    document, which also commits the exact public interface bytes. Eager
    expected/post-call observations are not consulted. Missing or unsupported
    contracts leave computed descriptors unknown; inconsistent contracts refuse.
    """
    from xdsl.dialects.builtin import StringAttr

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.result_layout import ATTRIBUTE, read_layout
    from merlin.targetgen.core_aten_batch import _tensor_type

    if read_layout(source) is not None:
        return prepared
    root = Path(capsule["__dir__"]).resolve()
    selected = os.environ.get("MERLIN_CORE_ATEN_PRIVATE_ROOT")
    answers = (Path(selected) if selected else root.parent.parent / "_private" / root.parent.name) / capsule["name"]
    path = answers / "capture.json"
    if not path.exists():
        return prepared
    if path.is_symlink() or not path.is_file():
        raise ValueError("legacy result contract capture is indirect")
    capture = json.loads(path.read_text())
    contract = capture.get("capture_meta", {}).get("result_contract")
    if contract is None:
        return prepared
    record = golden["batch_map"]["cases"][0]
    boundary = golden["semantic_io"]
    metadata = capture["capture_meta"]
    if (
        contract.get("authority") != "same_conversion_exported_program"
        or record["semantic_boundary"].get("authority") != contract["authority"]
        or capture.get("overload") != record["overload"]
        or not metadata.get("ok")
        or metadata.get("opaque") != 0
        or metadata.get("input_abi") != record["input_abi"]
        or metadata.get("output_abi") != record["output_abi"]
        or golden["public_files"].get("capsule.interface.mlir") != hashlib.sha256(source.encode()).hexdigest()
    ):
        raise ValueError("legacy result contract is not bound to the compiled call")
    # Only the old single unmutated user-result projection is recoverable here.
    # Complex component ABIs and mutation/result projections keep their existing
    # lowering path rather than borrowing an inferred storage identity.
    results = contract.get("results", [])
    if len(results) != 1 or results[0].get("role") != "user_output":
        return prepared
    if any(abi["dtype"].startswith("complex<") for abi in record["input_abi"] + record["output_abi"]):
        return prepared
    module = parse_mlir_text(prepared)
    function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    arguments = list(function.body.block.args)
    returned = next(op for op in function.body.block.ops if op.name == "func.return")
    posts = record["semantic_boundary"]["post_indices"]
    users = record["output_indices"]
    if (
        record["input_indices"] != list(range(len(arguments)))
        or len(contract.get("inputs", [])) != len(arguments)
        or len(posts) != len(arguments)
        or len(users) != 1
        or len(returned.operands) != len(boundary["outputs"])
        or boundary["input_indices"] != record["input_indices"]
        or boundary["post_indices"] != posts
    ):
        raise ValueError("legacy result contract projection differs from source ABI")
    keys = ("dtype", "shape", "stride", "storage_offset", "requires_grad")
    result, ordinal = results[0], users[0]
    committed = boundary["outputs"][ordinal]
    aliases = result.get("alias_inputs", [])
    dtypes = {
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
    }
    if (
        any(result.get(key) != committed.get(key) for key in (*keys, "role", "alias_inputs"))
        or str(returned.operands[ordinal].type) != _tensor_type(record["output_abi"][0])
        or result["shape"] != record["output_abi"][0]["shape"]
        or result["dtype"] != dtypes.get(record["output_abi"][0]["dtype"])
        or any(type(i) is not int or not 0 <= i < len(arguments) for i in aliases)
        or len(set(aliases)) != len(aliases)
        or committed.get("alias_outputs", []) != [posts[i] for i in aliases]
    ):
        raise ValueError("legacy result contract differs from the committed compiler boundary")
    if not aliases:
        return prepared  # A fresh computed result must remain a natural descriptor.
    for argument, abi, entry, post in zip(arguments, record["input_abi"], contract["inputs"], posts):
        if (
            str(argument.type) != _tensor_type(abi)
            or entry["shape"] != abi["shape"]
            or entry["dtype"] != dtypes.get(abi["dtype"])
            or returned.operands[post] is not argument
            or any(entry.get(key) != boundary["outputs"][post].get(key) for key in keys)
        ):
            raise ValueError("legacy result contract input is not the bound forwarded argument")
    layout = read_layout(prepared)
    for i in aliases:
        if any(contract["inputs"][i][key] != layout["inputs"][i][key] for key in ("shape", "stride", "storage_offset")):
            raise ValueError("legacy view contract differs from original caller storage")
        # The independently forwarded post-input descriptor must name the same
        # caller storage as the recovered view. Leaving it in a fresh alias class
        # would make the observed view/post relationship false despite good strides.
        layout["results"][posts[i]] = {**layout["inputs"][i], "alias_inputs": [i], "role": "user_output"}
    layout["results"][ordinal] = dict(result)
    module.attributes[ATTRIBUTE] = StringAttr(json.dumps(layout))
    module.verify()
    return str(module) + "\n"


def run_capsule(capsule, package_dir, paths, config, adapters, pkg, contract, timeout, no_oracle):
    """Compile through the standard package ABI; compare executed candidate boundary bytes.

    An adapter may implement ``run_full_call(bundle, llvm_mlir, target, timeout)`` for
    its substrate, returning lossless output_bytes/output_shapes/semantic_readback plus
    provenance. An ordinary command-buffer adapter cannot certify this boundary ABI.
    """
    from merlin.common import provenance
    from merlin.targetgen.capsule_common import run_entrypoints
    from merlin.targetgen.spike_full_call import (
        SPEC_EXECUTION_IDENTITY,
        full_call_execution_identity,
        unqualified_spec_device,
    )
    from merlin.targetgen.tier_policy import required_tiers, tier_depth_order

    ladder = tier_depth_order(required_tiers(capsule))  # an invalid operator floor is not a capsule verdict
    paths.run_path.mkdir(parents=True, exist_ok=True)
    result = {
        "capsule": capsule["name"],
        "kind": capsule["kind"],
        "label": capsule["label"],
        "contract_version": CONTRACT_VERSION,
        "cohort": "guard" if capsule.get("cohort") == "host_guard" else capsule.get("cohort", capsule["label"]),
        "scored": capsule.get("scored", True),
        "lane": "host",
        "executed_instructions": 0,
        "status": "incomplete",
        "tiers": {},
        "numeric": {"status": "skipped"},
        "trace_check": {"status": "skipped", "violations": []},
        "failure": None,
        "spec_qualified_device": unqualified_spec_device("mandatory_tiers_not_qualified"),
        "full_call_execution_identity": {},
        "provenance": provenance.record(
            sources=[Path(__file__), Path(capsule["__dir__"]) / "capsule.yaml"],
            extra={"target": config.target, "executions": {}},
        ),
    }
    qualifications = []
    try:
        from merlin.targetgen.candidate_scope import require_candidate_product_scope

        require_candidate_product_scope(pkg, adapters, capsule=capsule, package_dir=package_dir, contract=contract)
        source_adapters = [a for a in (adapters or {}).values() if getattr(a, "compiles_source_bundle", False)]
        if source_adapters:
            cb, emission = None, ""
        else:
            pkg, cb, emission = run_entrypoints(
                pkg,
                package_dir,
                capsule,
                paths,
                contract=contract,
                timeout=timeout,
                fourth_output_name=config.fourth_output_name,
            )
        if not source_adapters:
            result["candidate_emission"] = {"sha256": hashlib.sha256(emission.encode()).hexdigest()}
        else:
            result["compilation_mode"] = "source_bound_submitted_device_catalog"
        bundle = paths.run_path / ".private_full_call_runtime"
        bundle.mkdir(mode=0o700, exist_ok=True)
        runtime_bundle(capsule, bundle)
        for tier in ladder:
            adapter = (adapters or {}).get(tier)
            if getattr(adapter, "_merlin_deferred", False) and not no_oracle:
                result["tiers"][tier] = {
                    "status": "skipped",
                    "mandatory": True,
                    "not_run_is_not_pass": True,
                    "reason": "deferred: this feedback grade leaves the tier to the certification grades",
                }
                continue
            execute = getattr(adapter, "run_full_call", None)
            if no_oracle or execute is None:
                result["tiers"][tier] = {
                    "status": "unavailable",
                    "mandatory": True,
                    "not_run_is_not_pass": True,
                    "detail": "selected oracle does not implement full-call boundary execution",
                    "reason": "selected oracle does not implement full-call boundary execution",
                }
                result["failure"] = {"plane": "oracle_unavailable", "category": "NOT_RUN_IS_NOT_PASS", "tier": tier}
                break
            kwargs = (
                {"package_dir": package_dir, "capsule": capsule}
                if getattr(adapter, "compiles_source_bundle", False)
                else {}
            )
            run = execute(
                bundle=bundle, llvm_mlir=emission, command_buffer=cb, target=config.target, timeout=timeout, **kwargs
            )
            result.update(lane=run.get("lane", "host"), executed_instructions=run.get("executed_instructions", 0))
            if run.get("execution_evidence") is not None:
                result["execution_evidence"] = run["execution_evidence"]
            if not isinstance(run.get("provenance"), dict):
                raise ValueError("full-call execution lacks provenance")
            result["provenance"]["executions"][tier] = run["provenance"]
            verdict = grade_readback(
                capsule,
                run.get("output_bytes"),
                output_shapes=run.get("output_shapes"),
                semantic_readback=run.get("semantic_readback"),
                execution_error=run.get("execution_error"),
                provenance=run["provenance"],
                routing=run.get("routing"),
            )
            row = next(iter(verdict["cases"].values()))
            # Only the selected trusted oracle's closed producer may issue this
            # receipt. Candidate result dictionaries never supply either field.
            try:
                identity = full_call_execution_identity(capsule, bundle, run)
            except Exception:  # noqa: BLE001 -- unavailable additive credit does not alter functional grading
                identity = {}
            qualifier = getattr(adapter, "qualify_full_call", None)
            qualification = unqualified_spec_device("trusted_qualifier_unavailable")
            if callable(qualifier):
                try:
                    qualification = qualifier(
                        capsule, bundle=bundle, execution=run, verdict=verdict, package_dir=package_dir, timeout=timeout
                    )
                    if (
                        not isinstance(qualification, dict)
                        or set(identity) != set(SPEC_EXECUTION_IDENTITY)
                        or any(
                            qualification.get("bindings", {}).get(key) != identity[key]
                            for key in SPEC_EXECUTION_IDENTITY
                        )
                    ):
                        qualification = unqualified_spec_device("qualification_identity_differs")
                except Exception:  # noqa: BLE001 -- qualification cannot replace ordinary grading
                    qualification = unqualified_spec_device("qualification_evidence_invalid")
            qualifications.append((qualification, identity))
            execution_failed = row["status"] == "execution_failed"
            passed = row["status"] == "pass"
            if capsule.get("lane_expectation") == "device":
                passed = passed and result["lane"] == "device" and result["executed_instructions"] > 0
            elif capsule.get("lane_expectation") in {"host-guard", "no-work"}:
                passed = passed and result["lane"] == "host" and result["executed_instructions"] == 0
            result["tiers"][tier] = {
                "status": "pass" if passed else "fail",
                "mandatory": True,
                "not_run_is_not_pass": False,
                "derived_from_rtl": bool(run.get("derived_from_rtl")),
                "engine": run.get("engine"),
                **({"cycles": run["cycles"]} if run.get("cycles") is not None else {}),
                "semantic_scope": row.get("semantic_scope"),
                "provenance": run["provenance"],
                "spec_qualified_device": qualification,
                "full_call_execution_identity": identity,
                "reason": row["reason"]
                if execution_failed
                else None
                if passed
                else row.get("reason", "compiled full-call boundary or execution lane differs from the call contract"),
            }
            result["numeric"] = {
                "integer_reference": row.get("integer_reference", {}),
                "selected_policy": row.get("numeric_policy", "fp32"),
                "status": "skipped" if execution_failed else "pass" if passed else "fail",
                "semantic_scope": row.get("semantic_scope"),
            }
            if tier in config.rtl_tiers and run.get("derived_from_rtl") is not True:
                result["tiers"][tier]["status"] = "unavailable"
                result["tiers"][tier]["not_run_is_not_pass"] = True
                result["tiers"][tier]["reason"] = "selected oracle did not derive execution from RTL"
                result["failure"] = {"plane": "oracle_unavailable", "category": "NOT_RUN_IS_NOT_PASS", "tier": tier}
                break
            if not passed:
                result["status"] = "fail"
                # Full observations and expected values remain exclusively in the private runtime.
                result["failure"] = {
                    "plane": "compile" if execution_failed else "numeric",
                    "category": "PROTOCOL_VIOLATION" if execution_failed else "NUMERIC_MISMATCH",
                    "tier": tier,
                    "detail": row["reason"]
                    if execution_failed
                    else "compiled full-call boundary differs from the call contract",
                }
                _json(bundle / "private-verdict.json", verdict)
                break
        else:
            result["status"] = "pass"
            if qualifications and all(q.get("status") == "qualified" for q, _ in qualifications):
                result["spec_qualified_device"], result["full_call_execution_identity"] = qualifications[-1]
    except Exception as exc:  # noqa: BLE001 -- every capsule gets a durable refusing verdict
        result["status"] = "fail"
        result["failure"] = {"plane": "compile", "category": "PROTOCOL_VIOLATION", "detail": str(exc)[:1000]}
    if no_oracle:
        result["status"] = "not_gradeable_no_oracle"
    from merlin.targetgen.contract import schemas

    schemas.validate(result, "capsule_result", contract=contract)
    _json(paths.run_path / "capsule_result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--label", choices=("public", "dev", "hidden"), default="public")
    parser.add_argument("--tier", action="append", required=True)
    args = parser.parse_args(argv)
    manifest = write_capsules(
        json.loads(args.corpus.read_text()), args.captures, args.public, args.private, label=args.label, tiers=args.tier
    )
    print(f"Packaged {manifest['count']} full-call capsules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

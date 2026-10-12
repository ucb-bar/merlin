"""Original host custody for a separately emitted raw-IR transformation chain.

The trace supplies checkpoints, never proof authority. A fresh selected common
service checks each pair, and an actual object invocation consumes the same final
bytes. This proves no compiler-internal execution history, numerics or machine
semantics. Saved documents cannot recreate the process-local chain completion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from merlin.common import invocation_record

SCHEMA = "merlin.transformation_trace.v1"
RESULT_SCHEMA = "merlin.emission_chain.v1"
_COMMAND = "emit_transformation_trace"
_LIVE: dict[int, tuple] = {}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _pin(path):
    path = Path(path).absolute()
    if path.resolve() != path or any(owner.is_symlink() for owner in (path, *path.parents)) or not path.is_file():
        raise ValueError("emission proof custody requires plain existing original owners")
    before = path.stat()
    with path.open("rb") as stream:
        identity = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("emission proof member changed during byte observation")
    return {"path": str(path), "sha256": identity}


def _service_selection(service):
    from merlin.targetgen.common_transforms import CommonTransformService

    if type(service) is not CommonTransformService:
        raise TypeError("emission proof selection requires the exact original common service")
    if any(name in vars(service) for name in ("check", "_verify", "_toolchain", "apply")):
        raise ValueError("emission proof cannot select instance callbacks or proof policy")
    CommonTransformService._verify(service)
    if not callable(getattr(service, "check", None)):
        raise ValueError("original common service has no authored transformation checker")
    manifest = _pin(service.bundle_manifest)
    if service.bundle_manifest.read_bytes() != service.bundle_bytes or not service.snapshot_digest:
        raise ValueError("emission proof has no original frozen bundle/snapshot owner")
    return (
        manifest,
        service.snapshot_digest,
        service.profile["sha256"],
        _sha(json.dumps(service.profile, sort_keys=True, allow_nan=False).encode()),
    )


@dataclass(frozen=True)
class EmissionChainSelection:
    """Original host selects this capability; no candidate callback is admitted."""

    service: object
    maximum_trace_bytes: int = 2 * 1024 * 1024
    maximum_checkpoints: int = 32
    maximum_checkpoint_bytes: int = 128 * 1024
    original_selection: tuple = field(init=False, repr=False)

    def __post_init__(self):
        for value in (self.maximum_trace_bytes, self.maximum_checkpoints, self.maximum_checkpoint_bytes):
            if type(value) is not int or value < 1:
                raise ValueError("emission traces require positive bounded original host limits")
        object.__setattr__(self, "original_selection", _service_selection(self.service))

    def verify(self, *, package_root, evidence_root):
        if _service_selection(self.service) != self.original_selection:
            raise ValueError("emission proof original authority changed")
        manifest = self.service.bundle_manifest
        if manifest.is_relative_to(Path(package_root).resolve()) or manifest.is_relative_to(
            Path(evidence_root).resolve()
        ):
            raise ValueError("emission proof authority overlaps compiler-writable or product ownership")


def _entry_names(source):
    from xdsl.context import Context
    from xdsl.dialects import arith, builtin, func, llvm, memref, scf
    from xdsl.parser import Parser

    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, func.Func, llvm.LLVM, arith.Arith, memref.MemRef, scf.Scf):
        context.load_dialect(dialect)
    module = Parser(context, source).parse_module()
    module.verify()
    return tuple(
        operation.sym_name.data
        for operation in module.body.block.ops
        if type(operation) in (func.FuncOp, llvm.FuncOp) and operation.body.blocks
    )


def _checkpoint_chain(raw, selection, original_source, final_artifact, target_entry):
    if len(raw) > selection.maximum_trace_bytes:
        raise ValueError("transformation trace exceeds its original host byte budget")

    def no_duplicates(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("transformation trace contains duplicate JSON keys")
            result[key] = value
        return result

    document = json.loads(raw, object_pairs_hook=no_duplicates)
    if (
        type(document) is not dict
        or set(document) != {"schema", "checkpoints"}
        or document["schema"] != SCHEMA
        or type(document["checkpoints"]) is not list
        or not 2 <= len(document["checkpoints"]) <= selection.maximum_checkpoints
    ):
        raise ValueError("trace must provide a bounded ordered raw-IR chain, with no proof/status assertions")
    checkpoints = []
    for checkpoint in document["checkpoints"]:
        if type(checkpoint) is not dict or set(checkpoint) != {"ir", "entry"}:
            raise ValueError("trace checkpoints contain raw IR and entry only")
        source, entry = checkpoint["ir"], checkpoint["entry"]
        if (
            type(source) is not str
            or not source.strip()
            or len(source.encode()) > selection.maximum_checkpoint_bytes
            or type(entry) is not str
            or not entry
            or len(entry.encode()) > 1024
        ):
            raise ValueError("trace checkpoint exceeds its original raw-IR/entry limits")
        if entry not in _entry_names(source):
            raise ValueError("trace entry is not defined by its complete actual module")
        checkpoints.append((source, entry))
    actual_entries = _entry_names(original_source)
    if len(actual_entries) != 1 or checkpoints[0] != (original_source, actual_entries[0]):
        raise ValueError("trace first checkpoint is not the complete actual original source/entry")
    if checkpoints[-1] != (final_artifact, target_entry):
        raise ValueError("trace last checkpoint is not the complete actual emitted product/entry")
    return tuple(checkpoints)


def _require_roles(document, *, inputs=(), outputs=()):
    for role, expected in (("inputs", inputs), ("outputs", outputs)):
        if any(pin not in document[role] for pin in expected):
            raise ValueError("original invocation does not bind the actual emission source/product roles")


@dataclass(frozen=True)
class EmissionChainResult:
    report_json: str

    def record(self):
        """Descriptive data only; this does not issue acceptance authority."""
        return json.loads(self.report_json)


def check_emission_chain(
    *,
    selection,
    package,
    original_source,
    compiler_input,
    emitted_artifact,
    emission_invocation,
    target_entry,
    evidence_root,
    remaining,
):
    """Observe trace emission and invoke fresh original-host proof checks.

    `emission_invocation` is the original observed lower-interface CALL owned by
    the compile route, not a candidate path or a deserialized receipt.
    """
    from merlin.targetgen import package_runtime

    evidence_root = Path(evidence_root)
    if type(selection) is not EmissionChainSelection:
        raise TypeError("exact original emission-chain selection required")
    selection.verify(package_root=package.directory, evidence_root=evidence_root)
    if type(emission_invocation) is not invocation_record.CallInvocation:
        raise TypeError("emission custody needs its actual original observed host call")
    original = invocation_record.verify(emission_invocation.path)
    source_pin, compiler_input_pin, artifact_pin = _pin(original_source), _pin(compiler_input), _pin(emitted_artifact)
    _require_roles(original, inputs=(source_pin,), outputs=(compiler_input_pin, artifact_pin))
    if compiler_input_pin["sha256"] != source_pin["sha256"]:
        raise ValueError("compiler input staging changed the original source bytes")
    if original.get("callable") != "merlin.targetgen.capsule_common.lower_interface":
        raise ValueError("emission custody requires the ordinary original package lowering call")
    report = {
        "schema": RESULT_SCHEMA,
        "status": "not_declared",
        "source": source_pin,
        "compiler_input": compiler_input_pin,
        "emitted_artifact": artifact_pin,
        "compiler_source_invocation": _pin(emission_invocation.path),
        "edges": [],
        "compiler_internal_pass_execution": "not_established",
        "numerics": "not_established",
        "machine_semantics": "not_established",
        "outside_formal_coverage": "independent_checks_required",
        "original_profile_sha256": selection.original_selection[2],
    }
    if _COMMAND not in (package.manifest.get("commands") or {}):
        report["gap"] = "compiler declares no raw-IR transformation trace"
        return EmissionChainResult(json.dumps(report, sort_keys=True))
    # Trace command receives the same observed original input, in the same
    # protected package executor. It cannot supply invocations or proof results.
    if any(
        Path(path).stat().st_size > selection.maximum_checkpoint_bytes for path in (compiler_input, emitted_artifact)
    ):
        raise ValueError("actual source/product exceed the original checkpoint byte budget")
    source = Path(compiler_input).read_bytes().decode("utf-8")
    artifact = Path(emitted_artifact).read_bytes().decode("utf-8")
    evidence_root.mkdir(parents=True, exist_ok=True)
    trace_path = evidence_root / "transformation_trace.json"
    if trace_path.exists() or trace_path.is_symlink():
        raise ValueError("trace product requires a fresh original host owner")
    dependencies = tuple(Path(row["path"]) for row in original["dependencies"])
    with invocation_record.observe_call(
        evidence_root,
        stage=_COMMAND,
        function=package_runtime.run_entrypoint,
        arguments={"command": _COMMAND},
        inputs=(compiler_input,),
        outputs=(trace_path,),
        dependencies=dependencies,
    ) as observation:
        completed = package_runtime.run_entrypoint(
            package,
            _COMMAND,
            Path(compiler_input),
            trace_path,
            timeout=remaining(),
            invocation_directory=evidence_root,
        )
        if completed.returncode != 0:
            raise ValueError("compiler transformation-trace command failed")
        observation.returned(stdout=completed.stdout, stderr=completed.stderr)
    trace_record = invocation_record.verify(observation.path)
    _require_roles(trace_record, inputs=(compiler_input_pin,), outputs=(_pin(trace_path),))
    with trace_path.open("rb") as stream:
        raw = stream.read(selection.maximum_trace_bytes + 1)
    checkpoints = _checkpoint_chain(raw, selection, source, artifact, target_entry)
    report.update(trace=_pin(trace_path), trace_invocation=_pin(observation.path))
    for index, ((before, entry), (after, following_entry)) in enumerate(zip(checkpoints, checkpoints[1:])):
        if entry != following_entry:
            checked = {"status": "unsupported", "reason": "entry transport is outside fixed mathematical domains"}
        else:
            # The service issues a LIVE original completion before returning its
            # sealed descriptive document. No stored common-check summary enters.
            selection.verify(package_root=package.directory, evidence_root=evidence_root)
            checked = type(selection.service).check(
                selection.service,
                {"schema": "merlin.common_check_request.v1", "before_ir": before, "after_ir": after, "entry": entry},
                work_root=evidence_root / f"edge-{index}",
                timeout_s=max(1, int(remaining())),
            )
        status = checked.get("status")
        if status not in {"verified", "unsupported"}:
            raise ValueError("covered transformation did not obtain a fresh original live proof: " + str(status))
        if status == "verified" and (
            checked.get("source_sha256") != _sha(before.encode())
            or checked.get("target_sha256") != _sha(after.encode())
        ):
            raise ValueError("live transformation proof binds different actual IR")
        report["edges"].append(
            {
                "index": index,
                "source_sha256": _sha(before.encode()),
                "target_sha256": _sha(after.encode()),
                "entry": entry,
                "target_entry": following_entry,
                "status": status,
                "check": checked,
            }
        )
        remaining()
    report["status"] = (
        "verified_relationship" if all(edge["status"] == "verified" for edge in report["edges"]) else "partial"
    )
    report["scope"] = "separate raw-IR chain relationship; actual internal compiler pass execution unproved"
    selection.verify(package_root=package.directory, evidence_root=evidence_root)
    invocation_record.verify(emission_invocation.path)
    invocation_record.verify(observation.path)
    if (
        _pin(original_source) != source_pin
        or _pin(compiler_input) != compiler_input_pin
        or _pin(emitted_artifact) != artifact_pin
    ):
        raise ValueError("original source or emitted product changed during proof checking")
    result = EmissionChainResult(json.dumps(report, sort_keys=True))
    _LIVE[id(result)] = (
        result,
        selection,
        source_pin,
        compiler_input_pin,
        artifact_pin,
        _pin(trace_path),
        evidence_root,
        package.directory,
    )
    return result


def consume_emission_chain(result, *, selection, object_invocation, object_path, consumer_source):
    """Bind fresh proof custody to actual object-consumer inputs and products."""
    if type(result) is not EmissionChainResult:
        raise TypeError("original emission-chain result required")
    report = result.record()
    if report["status"] == "not_declared":
        return report
    actual = _LIVE.get(id(result))
    if actual is None or actual[0] is not result or actual[1] is not selection:
        raise ValueError("saved or fabricated emission-chain data has no original live completion")
    if type(object_invocation) is not invocation_record.CallInvocation:
        raise TypeError("emission proof consumer needs the actual observed object-production call")
    _, _, source_pin, compiler_input_pin, artifact_pin, trace_pin, evidence_root, package_root = actual
    for pin in (source_pin, compiler_input_pin, artifact_pin, trace_pin):
        if _pin(pin["path"]) != pin:
            raise ValueError("emission chain member changed before actual object consumption")
    for name in ("compiler_source_invocation", "trace_invocation"):
        pin = report[name]
        if _pin(pin["path"]) != pin:
            raise ValueError("original emission invocation changed before actual object consumption")
        invocation_record.verify(Path(pin["path"]))
    document = invocation_record.verify(object_invocation.path)
    if document.get("callable") != "merlin.targetgen.contract.compile.llvm_mlir_to_object":
        raise ValueError("emission chain was not consumed by the ordinary LLVM/object producer")
    consumed_pin, object_pin = _pin(consumer_source), _pin(object_path)
    if consumed_pin["sha256"] != artifact_pin["sha256"]:
        raise ValueError("object producer consumed different IR bytes from the proved emitted product")
    _require_roles(document, inputs=(artifact_pin,), outputs=(consumed_pin, object_pin))
    selection.verify(package_root=package_root, evidence_root=evidence_root)
    report.update(
        object_consumer_invocation=_pin(object_invocation.path),
        actual_consumer_source=consumed_pin,
        object_product=object_pin,
        product_custody="observed original emitted IR consumed by object producer",
    )
    _LIVE.pop(id(result))
    return report


def reread_emitted_product(result, *, selection):
    """Require current live custody before handing exact final bytes to LLVM."""
    if type(result) is not EmissionChainResult or type(selection) is not EmissionChainSelection:
        raise TypeError("original emission-chain result and selection required")
    report = result.record()
    if report["status"] != "not_declared":
        live = _LIVE.get(id(result))
        if live is None or live[0] is not result or live[1] is not selection:
            raise ValueError("emission product has no current original live chain completion")
        for pin in live[2:6]:
            if _pin(pin["path"]) != pin:
                raise ValueError("emission chain changed before object production")
        selection.verify(package_root=live[7], evidence_root=live[6])
        if _pin(report["trace_invocation"]["path"]) != report["trace_invocation"]:
            raise ValueError("original trace invocation changed before object production")
        invocation_record.verify(Path(report["trace_invocation"]["path"]))
    if _pin(report["compiler_source_invocation"]["path"]) != report["compiler_source_invocation"]:
        raise ValueError("original emission invocation changed before object production")
    invocation_record.verify(Path(report["compiler_source_invocation"]["path"]))
    pin = report["emitted_artifact"]
    if _pin(pin["path"]) != pin:
        raise ValueError("actual emitted product changed before object production")
    return Path(pin["path"]).read_bytes().decode("utf-8")


def discard_emission_chain(result):
    """Release bounded process-local custody after a refused/interrupted build."""
    live = _LIVE.get(id(result))
    if live is not None and live[0] is result:
        _LIVE.pop(id(result))


def reopen_emission_custody(report, *, selection, package_root, evidence_root):
    """Reopen descriptive transport bytes; never recreate live formal acceptance."""
    if type(selection) is not EmissionChainSelection or type(report) is not dict:
        raise TypeError("original emission-chain selection and transport document required")
    selection.verify(package_root=package_root, evidence_root=evidence_root)
    if (
        report.get("schema") != RESULT_SCHEMA
        or report.get("status") not in {"not_declared", "partial", "verified_relationship"}
        or report.get("original_profile_sha256") != selection.original_selection[2]
        or report.get("numerics") != "not_established"
        or report.get("machine_semantics") != "not_established"
        or report.get("compiler_internal_pass_execution") != "not_established"
    ):
        raise ValueError("emission custody has no matching descriptive scope/original selection")
    for name in ("source", "compiler_input", "emitted_artifact", "compiler_source_invocation"):
        pin = report[name]
        if _pin(pin["path"]) != pin:
            raise ValueError("emission transport original member changed")
    emitted = invocation_record.verify(Path(report["compiler_source_invocation"]["path"]))
    if emitted.get("callable") != "merlin.targetgen.capsule_common.lower_interface":
        raise ValueError("emission transport has no ordinary original source lowering")
    _require_roles(emitted, inputs=(report["source"],), outputs=(report["compiler_input"], report["emitted_artifact"]))
    if report["source"]["sha256"] != report["compiler_input"]["sha256"]:
        raise ValueError("emission transport source staging changed")
    if report["status"] == "not_declared":
        if report.get("edges") != []:
            raise ValueError("undeclared trace cannot provide formal edges")
        return {"status": "reopened", "live_formal_acceptance": "not_issued_by_saved_transport_report"}
    for name in ("trace", "trace_invocation", "object_consumer_invocation", "actual_consumer_source", "object_product"):
        pin = report[name]
        if _pin(pin["path"]) != pin:
            raise ValueError("emission transport trace/consumer member changed")
    trace = invocation_record.verify(Path(report["trace_invocation"]["path"]))
    if trace.get("callable") != "merlin.targetgen.package_runtime.run_entrypoint":
        raise ValueError("emission trace has no original command invocation")
    _require_roles(trace, inputs=(report["compiler_input"],), outputs=(report["trace"],))
    with Path(report["trace"]["path"]).open("rb") as stream:
        raw = stream.read(selection.maximum_trace_bytes + 1)
    checkpoints = _checkpoint_chain(
        raw,
        selection,
        Path(report["source"]["path"]).read_text(),
        Path(report["emitted_artifact"]["path"]).read_text(),
        report["edges"][-1]["target_entry"],
    )
    if len(report["edges"]) != len(checkpoints) - 1:
        raise ValueError("emission trace edge coverage changed")
    for index, (edge, ((before, entry), (after, following_entry))) in enumerate(
        zip(report["edges"], zip(checkpoints, checkpoints[1:]), strict=True)
    ):
        if (
            edge["index"] != index
            or edge["source_sha256"] != _sha(before.encode())
            or edge["target_sha256"] != _sha(after.encode())
            or edge["entry"] != entry
            or edge["target_entry"] != following_entry
            or edge["status"] not in {"verified", "unsupported"}
        ):
            raise ValueError("emission trace actual checkpoint correspondence changed")
    consumer = invocation_record.verify(Path(report["object_consumer_invocation"]["path"]))
    if consumer.get("callable") != "merlin.targetgen.contract.compile.llvm_mlir_to_object":
        raise ValueError("emission transport has no ordinary object consumer")
    _require_roles(
        consumer,
        inputs=(report["emitted_artifact"],),
        outputs=(report["actual_consumer_source"], report["object_product"]),
    )
    if report["actual_consumer_source"]["sha256"] != report["emitted_artifact"]["sha256"]:
        raise ValueError("emission transport consumer source differs")
    return {"status": "reopened", "live_formal_acceptance": "not_issued_by_saved_transport_report"}

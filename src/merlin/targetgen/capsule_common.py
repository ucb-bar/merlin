"""Shared, target-agnostic capsule I/O for the capsule runners.

`capsule_runner` (spike/verilator oracle) and a SIMT target's own capsule runner (cyclotron oracle) had
byte-identical copies of these helpers. They are the single source now; both runners import them (the
oracle-specific `run_capsule`/`run_suite` stay per-runner). Kept in `targetgen` (library), not in the
experiment harness, since the library runners are the consumers.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from merlin.verify.lean_emission_context import observe_emission

if TYPE_CHECKING:
    from aet.core.run_paths import RunPaths

    from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection

from .contract import schemas


def _target_consumer_record(pkg):
    """A package transport promise, never instruction or execution authority."""
    from .package_runtime import CertFailure

    expected = {
        "schema": "merlin.target_ir_consumer.v1",
        "command": "emit_target_bundle",
        "input": "selected_target_mlir",
        "output": "command_buffer_and_target_artifact",
    }
    manifest = pkg.manifest
    declaration = manifest.get("target_ir_consumer")
    command = manifest.get("commands", {}).get("emit_target_bundle")
    argv = command.get("argv") if type(command) is dict else None
    if (
        manifest.get("abi_version", "0.1") != "0.1"
        or type(declaration) is not dict
        or declaration != expected
        or type(command) is not dict
        or set(command) != {"argv"}
        or type(argv) is not list
        or not argv
        or any(type(value) is not str or not value for value in argv)
        or not all(any(placeholder in value for value in argv) for placeholder in ("{input_mlir}", "{output_json}"))
    ):
        raise CertFailure(
            "contract",
            _cat("STRUCTURAL_INVARIANT_VIOLATION"),
            "selected target configuration requires ABI 0.1 and an explicit "
            "merlin.target_ir_consumer.v1 emit_target_bundle consuming "
            "{input_mlir} and producing {output_json} plus the target artifact",
        )
    return json.loads(json.dumps({"declaration": declaration, "command": command, "target": manifest.get("target")}))


def _configuration_selection_record(selection):
    from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection

    if type(selection) is not ConfigurationIRSelection:
        raise ValueError("target configuration requires the original typed selection")
    selection.revalidate()
    return json.loads(json.dumps(selection.record(), sort_keys=True, allow_nan=False))


def _configuration_file(path):
    from merlin.common.digest import sha256_file

    path = Path(path)
    if path.is_symlink() or not path.is_file() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("target configuration requires unchanged direct regular artifact files")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def revalidate_target_configuration(pkg, generated, selection, *, expected_receipt_sha256):
    """Reopen the caller-pinned transport receipt and every consumed artifact.

    This proves invocation/file custody, not arbitrary consumer equivalence or
    instruction semantics. The caller retains the original receipt digest.
    """
    from merlin.common.digest import sha256_file
    from merlin.common.strict_json import loads

    receipt = Path(generated) / "target_configuration.json"
    pin = _configuration_file(receipt)
    if pin["sha256"] != expected_receipt_sha256:
        raise ValueError("target configuration receipt changed after publication")
    document = loads(receipt.read_text())
    if (
        type(document) is not dict
        or document.get("schema") != "merlin.target_configuration_transport.v1"
        or document.get("selection") != _configuration_selection_record(selection)
        or document.get("consumer") != _target_consumer_record(pkg)
        or document.get("owner_sha256") != sha256_file(Path(__file__))
    ):
        raise ValueError("target configuration transport selection or implementation changed")
    for pin in document["files"].values():
        if _configuration_file(pin["path"]) != pin:
            raise ValueError("target configuration source or consumed output changed")
    return document


class _TargetConfigurationInvocation:
    """One original selection and immutable files across public invocations."""

    def __init__(self, pkg, selection):
        self.pkg, self.selection = pkg, selection
        self.consumer = _target_consumer_record(pkg)
        self.record = _configuration_selection_record(selection)
        if self.consumer["target"] != selection.selected_contracts.target:
            raise ValueError("target configuration selected facts and package target disagree")
        self.files = {}
        self.result = None

    def verify(self):
        if self.consumer != _target_consumer_record(self.pkg) or self.record != _configuration_selection_record(
            self.selection
        ):
            raise ValueError("target configuration consumer or original selection changed during emission")
        for pin in self.files.values():
            if _configuration_file(pin["path"]) != pin:
                raise ValueError("target configuration immutable source changed during emission")
        if self.result is not None:
            self.result.revalidate()

    def remember(self, role, path):
        self.files[role] = _configuration_file(path)

    def transform(self, original, selected):
        from merlin.transforms.certificates import ir_text

        self.remember("original_target", original)
        self.verify()
        self.result = self.selection.apply(self.selection.parse(Path(original).read_text()))
        self.verify()
        Path(selected).write_text(ir_text(self.result.module), encoding="utf-8")
        self.remember("selected_target", selected)
        self.verify()

    def publish(self, generated, cb_path, artifact_path):
        from merlin.common.digest import sha256_file

        self.remember("command_buffer", cb_path)
        self.remember("target_artifact", artifact_path)
        self.verify()
        document = {
            "schema": "merlin.target_configuration_transport.v1",
            "selection": self.record,
            "consumer": self.consumer,
            "files": self.files,
            "configuration": self.result.report,
            "owner_sha256": sha256_file(Path(__file__)),
            "scope": "original selection and actual passed-file/output custody; consumer semantic equivalence unproved",
        }
        payload = json.dumps(document, sort_keys=True, allow_nan=False, indent=2) + "\n"
        path = Path(generated) / "target_configuration.json"
        path.write_text(payload, encoding="utf-8")
        digest = hashlib.sha256(payload.encode()).hexdigest()
        revalidate_target_configuration(self.pkg, generated, self.selection, expected_receipt_sha256=digest)


def verify_capture_tool(directory: Path, capsule: dict) -> None:
    """Verify an optional frozen capture package without claiming loader closure.

    Older capsules do not carry this receipt. A capsule that does claim a
    selected installed package must carry both its exact record and wheel.
    """
    selected = capsule.get("capture_tool")
    if selected is None:
        return
    if not isinstance(selected, dict):
        raise ValueError("capsule capture-tool identity is malformed")
    if selected.get("status") == "not_available":
        return
    if selected.get("status") != "verified_selected_package":
        raise ValueError("capsule capture-tool status is unsupported")

    def member(name: object) -> Path:
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("capsule capture-tool member path is unsafe")
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"capsule capture-tool member is absent or indirect: {name}")
        return path

    sidecar = member(selected.get("path"))
    raw = sidecar.read_bytes()
    if hashlib.sha256(raw).hexdigest() != selected.get("sha256"):
        raise ValueError("capsule capture-tool record bytes changed")
    try:
        record = json.loads(raw)
        wheel_identity = record["wheel"]
        module_identity = record["modules"]
        if record.get("schema") != "merlin.capsule_capture_tool.v1" or record.get("status") != selected["status"]:
            raise ValueError("capsule capture-tool record identity changed")
        wheel = member(wheel_identity["path"])
        payload = wheel.read_bytes()
        if len(payload) != wheel_identity["bytes"] or hashlib.sha256(payload).hexdigest() != wheel_identity["sha256"]:
            raise ValueError("capsule capture-tool wheel bytes changed")
        if not isinstance(module_identity, dict) or not module_identity:
            raise ValueError("capsule capture-tool module inventory is absent")
        with zipfile.ZipFile(wheel) as archive:
            modules = sorted(name for name in archive.namelist() if name.startswith("m2m/") and name.endswith(".py"))
            if modules != sorted(module_identity):
                raise ValueError("capsule capture-tool wheel module set changed")
            for name in modules:
                source = archive.read(name)
                identity = module_identity[name]
                if len(source) != identity["bytes"] or hashlib.sha256(source).hexdigest() != identity["sha256"]:
                    raise ValueError(f"capsule capture-tool wheel module changed: {name}")
    except (OSError, KeyError, TypeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"capsule capture-tool cannot be verified: {exc}") from exc


#: Statuses that are NOT a measurement of the submission, and so belong in neither the numerator nor
#: the denominator of a score. Each is a different reason the capsule produced no verdict:
#:   not_graded      the target's contract declares no capability for it -- it can never pass
#:   gated           its own gate deferred it (a whole-model capstone waiting on the op suite)
#:   screened_only   it cleared the cheap screen and the certify budget stopped before the gold tier
#:   budget_exhausted  it started and its wall-clock ceiling ran out before it finished
#:   cert_not_measured its elaborated-RTL tier was ATTEMPTED and produced no verdict (no completion
#:                     witness, no engine, or killed on the clock), so the cheaper tiers that did pass
#:                     are a screen and a screen may never certify -- see `tier_integrity`
#:
#: Counting any of them as a FAILURE is what makes ``all_pass`` unreachable, which disables an agent
#: loop's only early exit and turns every run into a fixed-price purchase of its whole round budget.
#: Counting one as a PASS would be a phantom certification. They are excluded and listed BY NAME.
#: ``infrastructure_fault`` is here for the same reason as the rest and one more: it is not a
#: measurement of the submission, so counting it as a failure would attribute a harness bug to the
#: agent. It is NOT enough on its own, though -- excluding rows from the denominator can leave a
#: 2-of-33 grade reading "2/2, all_pass", so :func:`capsule_grade.grade` also forces ``gradeable``
#: False whenever any row carries it. Never measured AND never reported as success.
NOT_MEASURED_STATUSES = (
    "not_graded",
    "gated",
    "screened_only",
    "budget_exhausted",
    "infrastructure_fault",
    "cert_not_measured",
)


def _flat(nested) -> list:
    """Preserve row-major scalar order across logical ranks and dense readback rows."""
    out: list = []
    for value in nested:
        if isinstance(value, list):
            out.extend(_flat(value))
        else:
            out.append(value)
    return out


def _stderr_excerpt(stderr: str, limit: int = 400) -> str:
    """An excerpt of a failing tool's stderr that keeps BOTH ends.

    ``stderr[-400:]`` is the obvious choice and it is wrong for the commonest case. A one-line parser
    error prefixed by a long absolute path gets its head cut, so the reader is handed ``seError: /scratch/
    .../input.interface.mlir`` -- the exception TYPE gone and the path kept, which is exactly backwards.
    A Python traceback wants its tail; a compiler diagnostic wants its head. Keep both and mark the gap.
    """
    stderr = stderr or ""
    if len(stderr) <= limit:
        return stderr
    head = limit // 2
    tail = limit - head - 20
    return f"{stderr[:head]}\n  [... {len(stderr) - head - tail} chars elided ...]\n{stderr[-tail:]}"


def _cat(name: str):
    """Resolve a FailureCategory by name, tolerant to the enum's membership."""
    from aet.core.failures import FailureCategory

    try:
        return getattr(FailureCategory, name)
    except AttributeError:
        return FailureCategory.RUNNER_CRASH


def validate_interface_tensor_dtypes(cb: dict, interface_mlir: str) -> None:
    """Require the command buffer to preserve every interface tensor's logical dtype.

    ``physical`` describes how bytes of the *same logical tensor* are laid out for readback.  It is
    not permission to narrow an interface result and let the oracle decode fewer bytes.  Without this
    binding check a backend can, for example, replace an ``f32`` result with ``bf16`` storage, attach
    an ignored ``logical_dtype`` hint, and have the independent oracle compare numerically equal BF16
    values instead of verifying the declared 32-bit ABI.

    Dtype aliases are resolved through the shared format registry, so MLIR spellings such as
    ``f8E4M3FN`` remain equivalent to the command-buffer spelling ``fp8_e4m3``.  Plain machine types
    (for example ``i32``/``int32``) are compared by signedness family and width.  Non-``merlin_iface``
    inputs are left to their own frontend contract.
    """
    if "merlin_iface." not in interface_mlir:
        return

    from merlin.common import quant_formats as qf

    from .contract.interface_emit import InterfaceGrammarError, parse_interface_mlir

    try:
        declared = parse_interface_mlir(interface_mlir)
    except InterfaceGrammarError as exc:
        raise schemas.ContractViolation(f"cannot bind command-buffer tensors to the interface: {exc}") from exc

    def identity(token: object) -> tuple[str, object]:
        key = str(token)
        if key.startswith("torch."):
            key = key[len("torch.") :]
        if qf.has(key):
            return "format", qf.get(key).name
        bits = qf.machine_bits(key)
        if bits is None:
            return "opaque", key
        for prefix, family in (
            ("float", "float"),
            ("uint", "uint"),
            ("int", "int"),
            ("f", "float"),
            ("u", "uint"),
            ("i", "int"),
        ):
            if key.startswith(prefix) and key[len(prefix) :].isdigit():
                return family, bits
        return "opaque", key

    emitted = cb.get("tensors") or {}
    problems: list[str] = []
    for name, expected in declared.get("tensors", {}).items():
        actual = emitted.get(name)
        if not isinstance(actual, dict):
            continue
        want, got = expected.get("dtype"), actual.get("dtype")
        if identity(want) != identity(got):
            problems.append(f"tensor {name!r} changes logical dtype from interface {want!r} to command buffer {got!r}")
    if problems:
        raise schemas.ContractViolation("command-buffer/interface ABI mismatch: " + "; ".join(problems))


def load_capsule(capsule_dir: str | Path, *, contract: str | Path | None = None) -> dict:
    """Load + validate a capsule.yaml; stamp its directory for interface-MLIR resolution."""
    d = Path(capsule_dir)
    cap = yaml.safe_load((d / "capsule.yaml").read_text(encoding="utf-8"))
    try:
        schemas.validate_capsule(cap, contract=contract)
    except schemas.ContractViolation as e:
        # Fail closed on a schema-invalid capsule (a corpus bug must surface, never be silently dropped),
        # but name the offending capsule so a discovery-time crash is instantly diagnosable.
        raise schemas.ContractViolation(f"capsule '{d.name}' ({d}): {e}") from e
    verify_capture_tool(d, cap)
    cap["__dir__"] = str(d)
    return cap


def tier_status(entry) -> str | None:
    """The status of one tier record, whichever shape it arrived in.

    An OP capsule records a tier as a dict (``{"status": ..., "cycles": ..., "derived_from_rtl": ...}``);
    a MODEL capsule records it as a bare ``"pass"``/``"fail"`` string. Both aggregators assumed the dict,
    which was survivable only while model capsules were always `gated` and so never reached them. The
    first submission to clear the op gate crashed the grade with ``'str' object has no attribute 'get'``
    -- AFTER every capsule had been simulated, so the run burned its full wall-clock and wrote no score.
    One normalizer, used by every reader, rather than each one re-deciding what a tier looks like."""
    if isinstance(entry, dict):
        return entry.get("status")
    return entry if isinstance(entry, str) else None


def tier_field(entry, field: str):
    """A named field of a tier record, or ``None`` when the record is the bare-string form."""
    return entry.get(field) if isinstance(entry, dict) else None


def oracle_kind(oracle):
    """The oracle's PROVENANCE STRING, whichever shape it arrived in.

    An adapter reports its oracle either as a bare string (``"<target>-spike"``) or as a record
    (``{"kind": ..., "derived_from_rtl": ..., "fidelity": ...}``) — the record form exists so an oracle
    can state whether it is elaborated RTL rather than leaving that to be guessed from its tier name.
    Callers that only want the human-readable identity go through here, so enriching an adapter never
    turns a recorded string field into a dict.
    """
    if isinstance(oracle, dict):
        return oracle.get("kind")
    return oracle


def discover_capsules(root, *, labels: set[str] | None = None, contract: str | Path | None = None) -> list[dict]:
    """Load every capsule under ``root`` (recursively), optionally filtered by label.

    ``root`` is one path OR several. Several, because a target's graded suite is not one directory: the
    capsules are split by KIND into sibling categories (``isa`` / ``layers`` / ``model_slices``), and a
    caller that passes only the primary one silently grades a subset. Passing their common parent instead
    is not the fix — that parent also contains OTHER targets' corpora, which would both leak foreign
    capsules in and (today) fail to load at all.

    Duplicates are dropped by capsule directory, so overlapping roots cost nothing and cannot
    double-count a capsule into the denominator.
    """
    roots = [root] if isinstance(root, (str, Path)) else list(root)
    caps, seen = [], set()
    for r in roots:
        for cy in sorted(Path(r).rglob("capsule.yaml")):
            if cy.parent in seen:
                continue
            seen.add(cy.parent)
            cap = load_capsule(cy.parent, contract=contract)
            if labels is None or cap.get("label") in labels:
                caps.append(cap)
    return caps


def make_run_paths(
    runs_root: str | Path, run_id: str, *, suite: str, target: str, dtype: str, benchmark: str
) -> RunPaths:
    """Build the per-run RunPaths (via RunSpec) and create its directory scaffold.

    ``runs_root`` is RESOLVED to an absolute path first. The grader runs capsules on threads and some of
    them enter a context that chdirs the process into another tree (mlc resolves its arc artifacts
    relative to its own root), so a run path kept relative is resolved against whatever directory the
    process happens to be in when a sibling thread writes. Measured: a suite died on
    ``FileNotFoundError: .../capsule_result.json`` after the capsule had already run, with the directory
    plainly present on disk — it had been created in one cwd and written from another."""
    from aet.core.run_paths import RunPaths
    from aet.core.run_spec import RunSpec

    spec = RunSpec(
        project="merlin",
        suite=suite,
        method=run_id,
        seed=0,
        run_id=run_id,
        project_root=Path(runs_root).resolve(),
        tracking_mode="local",
        target=target,
        dtype=dtype,
        benchmark=benchmark,
    )
    paths = RunPaths.from_spec(spec, run_id)
    for dd in (paths.run_path, paths.logs, paths.artifacts_dir, paths.generated, paths.contracts):
        dd.mkdir(parents=True, exist_ok=True)
    return paths


def run_entrypoints(
    pkg,
    package_dir: str | Path,
    capsule: dict,
    paths,
    *,
    contract: str | Path | None,
    timeout: int,
    fourth_output_name: str,
    additional_forbidden: tuple[str, ...] = (),
):
    """Shared ABI front half: build the package (if needed) and run the 4 contract entrypoints
    (parse -> lower_interface_to_target -> emit_command_buffer -> lower_target_to_llvm), writing the
    standard artifacts and validating the command buffer. Returns ``(pkg, cb, fourth_text)`` where
    ``fourth_text`` is the lower_target_to_llvm stdout (written to ``fourth_output_name`` — the target
    dialect chooses LLVM-dialect MLIR vs a SIMT kernel). Raises CertFailure on any plane failure.
    The oracle tiers (L2+) are the caller's, since they diverge per target.

    The entrypoint walk itself is :func:`lower_interface`, which a whole model's compute groups go
    through too: this function is the capsule route's half of it (build the package, resolve the
    staged interface) and nothing more.
    """
    from .package_runtime import (
        INFRASTRUCTURE_PLANE,
        CertFailure,
        InfraCategory,
        InfraFailure,
        build_package,
        integrity_scan,
        load_package,
    )

    if pkg is None:
        pkg = load_package(package_dir, contract=contract)
        integrity_scan(pkg, **({"additional_forbidden": additional_forbidden} if additional_forbidden else {}))
        build_package(pkg)
    if not pkg.tool.exists():
        raise CertFailure("build", _cat("ELABORATION_ERROR"), f"tool missing: {pkg.tool}")

    iface_rel = capsule.get("interface_mlir", "capsule.interface.mlir")
    cap_dir = Path(capsule["__dir__"]) if "__dir__" in capsule else None
    iface_path = (cap_dir / iface_rel) if cap_dir is not None else Path(iface_rel)
    if not iface_path.is_file():
        # WHOSE fault is a missing interface MLIR? Two different answers, and conflating them cost a run
        # its official verdict. If the capsule's own staged DIRECTORY is gone, the cohort was never
        # materialized -- or was collected out from under this grade, which is the measured case: the
        # grader resolves the cohort symlink to a concrete staging dir once and reads capsules from it for
        # the whole grade, and a sibling materialization rmtree'd that dir 20 s in. Every capsule after
        # that was recorded `schema / structural_invariant_violation`, i.e. "your package is structurally
        # invalid", 31 times, for a submission that had just scored 33/34. That is an INFRASTRUCTURE
        # fault: nothing about the submission was measured, and it must never be spelled like a verdict.
        # Only a capsule dir that IS present, with the interface file missing inside it, is a genuine
        # corpus/schema defect -- and that one stays exactly as it was.
        if cap_dir is not None and not cap_dir.is_dir():
            raise InfraFailure(
                INFRASTRUCTURE_PLANE,
                InfraCategory.COHORT_NOT_MATERIALIZED,
                f"staged capsule input is missing: the cohort was not materialized (or its staging "
                f"directory was removed mid-grade). Expected capsule directory {cap_dir} (for "
                f"{iface_path.name}) does not exist. This is a harness/staging fault, NOT a defect in "
                f"the graded submission, and nothing about the submission was measured for this "
                f"capsule; re-materialize the cohort and re-grade.",
            )
        raise CertFailure(
            "schema", _cat("STRUCTURAL_INVARIANT_VIOLATION"), f"capsule interface MLIR not found: {iface_path}"
        )
    cb, artifact = lower_interface(
        pkg, iface_path, paths.generated, contract=contract, timeout=timeout, artifact_name=fourth_output_name
    )
    return pkg, cb, artifact


@observe_emission
def lower_interface(
    pkg,
    interface: str | Path,
    generated: str | Path,
    *,
    contract: str | Path | None,
    timeout: int,
    invoke=None,
    artifact_name: str = "lowered.llvm.mlir",
    overlap: bool = False,
    memo: dict | None = None,
    target_configuration: ConfigurationIRSelection | None = None,
    shared_pass_requirement: dict | None = None,
) -> tuple[dict, str]:
    """ONE stated ``merlin_iface`` capsule through the package's own entrypoints -> (buffer, artifact).

    THE CAPSULE ROUTE AND THE MODEL ROUTE ARE THE SAME JOB and used to be two implementations of it.
    `llvmlower.whole_program` put each compute group of a captured model to the package by writing
    the interface and calling ONE entrypoint, with no ``parse``, no ``lower_interface_to_target``, no
    contract validation of what came back and -- for every package that does not declare the optional
    bundle, which was all 76 in this tree when this was written -- no target artifact at all. So a
    group could splice a buffer the grader would have refused, and the whole-model artifact a
    performance arm reads was a file of placeholders. A capsule and a model group now reach a backend
    through this function, so the two cannot disagree about what "the package lowered it" means.

    ``invoke`` is the runner (default :func:`oot_runner.run_entrypoint`). An analysis worker runs a
    submission inside a host-created sandbox and must keep doing so; the SEQUENCE is not its business,
    so it injects the invocation rather than reimplementing the walk.

    Which commands produce the buffer and the artifact is
    :func:`oot_runner.analysis_emission_entrypoints` -- the package's own declaration, read once,
    here. It was a second inline feature-detection in the model route.

    ``overlap`` runs ``lower_interface_to_target`` and the buffer command at the same time: both read
    only the parsed interface and neither reads the other's output, so a caller with many groups to
    ask (a whole model) pays the slower of the two rather than their sum. The verdict is unchanged --
    the target plane is still judged first, and a failure there is reported as before.

    ``memo`` (a dict the caller owns for ONE package) records each accepted lowering by the interface's
    bytes: a whole model asks the same interface for many groups (SmolVLA: 27 distinct for 1,551), and
    re-reading the package's recorded replies -- up to 160 MB of target IR a reply -- for every one of
    them was most of a statement. A hit writes the same input and command buffer into ``generated`` and
    returns the recorded buffer (a copy) and artifact; a refusal is never recorded, so it is re-derived.

    An original ``target_configuration`` selection requires the package's explicit
    versioned target-IR consumer. That command receives the actual selected IR,
    returning both command buffer and final artifact; legacy independent emission
    is never substituted. This route disables memoization and overlapping calls.

    Explicit ``shared_pass_requirement`` enforces the selected compiler interface
    contract before invoking the package and across command boundaries. It does
    not interpret arbitrary emitted operations or qualify claimed capabilities.

    Raises :class:`CertFailure` on any plane failure and :class:`BackendDeclined` on a stated decline.
    """
    import contextlib
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from functools import partial

    from .package_runtime import (
        BackendDeclined,
        CertFailure,
        analysis_emission_entrypoints,
        run_entrypoint,
    )
    from .shared_pass_context import selected_shared_pass_requirement

    shared_pass_requirement = selected_shared_pass_requirement(shared_pass_requirement)
    pass_contract = None
    if shared_pass_requirement is not None:
        import copy

        from .shared_pass_contract import validate_shared_pass_contract

        shared_pass_requirement = copy.deepcopy(shared_pass_requirement)
        pass_contract = validate_shared_pass_contract(pkg.manifest, requirement=shared_pass_requirement)
        pass_commands = copy.deepcopy(pkg.manifest["commands"])
        # Each invocation rechecks the selected declarations; a legacy memo hit
        # must not bypass these checks. Declared surfaces still need independent
        # changed-IR consumption checks, not just a successful command invocation.
        overlap, memo = False, None

    selected_target = (
        _TargetConfigurationInvocation(pkg, target_configuration) if target_configuration is not None else None
    )
    # V2 ordinary emission consumes the actual lowered target even when no
    # original host-selected target transform is requested. This is transport,
    # not proof of a declared optimization or consumer semantic equivalence.
    identity_target = (
        pass_contract is not None
        and shared_pass_requirement["schema"] == "merlin.shared_pass_requirement.v2"
        and selected_target is None
    )
    routed_target = selected_target is not None or identity_target
    identity_files = {}

    def verify_identity_target():
        identities = set()
        for pin in identity_files.values():
            if _configuration_file(pin["path"]) != pin:
                raise ValueError("shared pass target input or consumed output changed during emission")
            info = Path(pin["path"]).stat()
            identity = info.st_dev, info.st_ino
            if identity in identities:
                raise ValueError("shared pass target artifacts must not alias")
            identities.add(identity)

    # An exact original runtime may execute this complete body in a fresh sealed
    # child. The OUTER ordinary observer still owns the live source/product call.
    # Explicit runners and alternate target/memo/overlap protocols remain local.
    if (
        invoke is None
        and selected_target is None
        and contract is None
        and memo is None
        and not overlap
        and shared_pass_requirement is None
    ):
        from .package_runtime import active_package_executor

        delegate = getattr(active_package_executor(), "process_lower_interface_v1", None)
        if callable(delegate):
            result = delegate(pkg, interface, generated, timeout=timeout, artifact_name=artifact_name)
            if result is not NotImplemented:
                return result
    if routed_target:
        overlap, memo = False, None
    generated = Path(generated)
    if routed_target:
        reserved = {
            "input.interface.mlir",
            "lowered.target.mlir",
            "selected.target.mlir",
            "command_buffer.json",
            "target_configuration.json",
        }
        if identity_target:
            reserved.add("shared_pass_target.json")
        if Path(artifact_name).name != artifact_name or artifact_name in reserved or artifact_name in ("", ".", ".."):
            raise ValueError("target configuration artifact name overlaps its retained inputs or receipt")
        if any(parent.is_symlink() for parent in (generated, *generated.parents)):
            raise ValueError("target configuration requires a direct private invocation directory")
        if any((generated / name).exists() or (generated / name).is_symlink() for name in (*reserved, artifact_name)):
            raise ValueError("target configuration requires fresh retained invocation artifacts")
    generated.mkdir(parents=True, exist_ok=True)
    invoke = invoke or partial(run_entrypoint, invocation_directory=generated)
    if pass_contract is not None:
        pass_invoke = invoke

        def require_pass_contract():
            if (
                validate_shared_pass_contract(pkg.manifest, requirement=shared_pass_requirement) != pass_contract
                or pkg.manifest.get("commands") != pass_commands
            ):
                raise ValueError("shared pass compiler contract changed during lowering")

        def invoke(*args, **kwargs):
            require_pass_contract()
            result = pass_invoke(*args, **kwargs)
            require_pass_contract()
            return result

    inp = generated / "input.interface.mlir"
    interface_text = Path(interface).read_text(encoding="utf-8")
    inp.write_text(interface_text, encoding="utf-8")
    if routed_target:
        if selected_target is not None:
            selected_target.remember("interface_source", interface)
            selected_target.remember("interface_snapshot", inp)
        else:
            identity_files["interface_source"] = _configuration_file(interface)
            identity_files["interface_snapshot"] = _configuration_file(inp)
        original_invoke = invoke

        def verify_target_input():
            if selected_target is not None:
                selected_target.verify()
            else:
                verify_identity_target()

        def invoke(*args, **kwargs):
            verify_target_input()
            result = original_invoke(*args, **kwargs)
            verify_target_input()
            return result

    memo_key = None
    if memo is not None:
        import copy
        import hashlib

        # Keyed by what decides the answer -- the contract and the interface's bytes -- never by where a
        # caller files the artifact (``artifact_name`` is per group), or no second group would ever hit.
        memo_key = (str(contract), hashlib.sha256(interface_text.encode("utf-8")).hexdigest())
        recorded = memo.get(memo_key)
        if recorded is not None:
            cb_text, cb_recorded, artifact_recorded = recorded
            (generated / "command_buffer.json").write_text(cb_text, encoding="utf-8")
            (generated / artifact_name).write_text(artifact_recorded, encoding="utf-8")
            return copy.deepcopy(cb_recorded), artifact_recorded

    p = invoke(pkg, "parse", inp, timeout=timeout)
    if p.returncode != 0:
        raise CertFailure("parse", _cat("TOOL_CRASH"), f"parse rc={p.returncode}: {_stderr_excerpt(p.stderr)}")

    _bundled = routed_target or analysis_emission_entrypoints(pkg) == ("emit_analysis_bundle",)
    _buffer_cmd = (
        "emit_target_bundle" if routed_target else "emit_analysis_bundle" if _bundled else "emit_command_buffer"
    )
    cb_path = generated / "command_buffer.json"
    early = None
    # Retain the original overlap, but join its worker on every exit before the
    # caller's package/resource scopes can release the invocation workspace.
    with ThreadPoolExecutor(max_workers=1) if overlap else contextlib.nullcontext() as pool:
        if overlap:
            early = pool.submit(copy_context().run, invoke, pkg, _buffer_cmd, inp, cb_path, timeout=timeout)
        p = invoke(pkg, "lower_interface_to_target", inp, timeout=timeout)
    if p.returncode != 0 or not p.stdout.strip():
        if early is not None:
            with contextlib.suppress(Exception):
                early.result()
        raise CertFailure(
            "interface_to_target",
            _cat("ELABORATION_ERROR"),
            f"lower_interface_to_target rc={p.returncode}: {_stderr_excerpt(p.stderr)}",
        )
    if identity_target and (
        (generated / "lowered.target.mlir").exists() or (generated / "lowered.target.mlir").is_symlink()
    ):
        raise ValueError("shared pass target requires fresh host-owned output paths")
    (generated / "lowered.target.mlir").write_text(p.stdout, encoding="utf-8")
    buffer_input = inp
    if selected_target is not None:
        buffer_input = generated / "selected.target.mlir"
        selected_target.transform(generated / "lowered.target.mlir", buffer_input)
    elif identity_target:
        original_target = generated / "lowered.target.mlir"
        identity_files["original_target"] = _configuration_file(original_target)
        buffer_input = generated / "selected.target.mlir"
        if buffer_input.exists() or buffer_input.is_symlink():
            raise ValueError("shared pass target requires fresh host-owned output paths")
        buffer_input.write_bytes(original_target.read_bytes())
        identity_files["selected_target"] = _configuration_file(buffer_input)
        if identity_files["original_target"]["sha256"] != identity_files["selected_target"]["sha256"]:
            raise ValueError("shared pass identity target copy changed bytes")
        verify_identity_target()

    # ONE PROCESS FOR BOTH ARTIFACTS WHEN THE PACKAGE DECLARES IT. A whole model is one capsule per
    # group, so the per-group cost is paid once per layer; asking twice for what one invocation can
    # return doubles the wall time of every round's gate. A package that declares no bundle gets the
    # two-command protocol it always got.
    p = early.result() if early is not None else invoke(pkg, _buffer_cmd, buffer_input, cb_path, timeout=timeout)
    _artifact = p.stdout if _bundled else ""
    if p.returncode != 0:
        raise CertFailure(
            "target_to_command_buffer",
            _cat("STRUCTURAL_INVARIANT_VIOLATION"),
            f"{_buffer_cmd} rc={p.returncode}: {_stderr_excerpt(p.stderr)}",
        )
    if not cb_path.exists():
        # Exit 0 and no output file. Reporting this as "rc=0: <empty stderr>" tells the agent
        # nothing -- worse, rc=0 reads as success, so the message contradicts itself. One package
        # spent twelve rounds and 215 self-checks against exactly that blank string. The cause is
        # nearly always the manifest argv template: it omits {output_json}, so the tool prints the
        # buffer to stdout and the path the runner reads is never written. Say so, and name it.
        _argv = " ".join(pkg.manifest.get("commands", {}).get(_buffer_cmd, {}).get("argv", []) or ["<undeclared>"])
        _hint = (
            f"your manifest argv for {_buffer_cmd} does not reference {{output_json}}, so "
            "nothing is written to the path the runner reads"
            if "{output_json}" not in _argv
            else "the argv does reference {output_json}, so the tool exited before writing it"
        )
        raise CertFailure(
            "target_to_command_buffer",
            _cat("STRUCTURAL_INVARIANT_VIOLATION"),
            f"{_buffer_cmd} exited 0 but wrote no file at the output path it "
            f"was given ({cb_path.name}). Declared argv: {_argv} -- {_hint}. The "
            f"runner reads the FILE, never stdout. stderr: {_stderr_excerpt(p.stderr, 200)!r}",
        )
    if identity_target:
        identity_files["command_buffer"] = _configuration_file(cb_path)
    try:
        cb = json.loads(cb_path.read_text(encoding="utf-8"))
        schemas.validate_command_buffer(cb, contract=contract)
        validate_interface_tensor_dtypes(cb, inp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, schemas.ContractViolation) as e:
        raise CertFailure(
            "command_buffer_schema", _cat("PROTOCOL_VIOLATION"), f"command_buffer.json invalid: {e}"
        ) from e

    # AN EXPLICIT DECLINE, READ BEFORE ANYTHING IS RUN. A backend states here that it does not handle
    # this capsule; the alternative it used to have was to emit a program that writes nothing, which
    # reaches the grader as zeros and is scored as wrong arithmetic. Reading the declaration first means
    # no oracle is paid for a program the backend already said it did not write, and the round feedback
    # can name the shape instead of reporting a numeric mismatch that never happened.
    _declined = cb.get("declined")
    if _declined:
        if not isinstance(_declined, dict) or not str(_declined.get("reason") or "").strip():
            raise CertFailure(
                "command_buffer_schema",
                _cat("PROTOCOL_VIOLATION"),
                "command_buffer declares `declined` without a non-empty `reason`: a "
                "decline has to say WHAT it could not lower, or it is just a silent drop "
                "with extra steps",
            )
        raise BackendDeclined(str(_declined["reason"]), shape=_declined.get("shape"), op=_declined.get("op"))

    # the 4th entrypoint: emit the target's codegen artifact (RoCC LLVM / SIMT kernel / ...). The
    # resolver aliases the legacy name lower_target_to_llvm, so packages using either spelling work.
    # A bundling package already returned it on the same stdout that wrote the buffer.
    if not _bundled:
        p = invoke(pkg, "emit_target_artifact", inp, timeout=timeout)
        if p.returncode != 0 or not p.stdout.strip():
            raise CertFailure(
                "emit_target_artifact",
                _cat("ELABORATION_ERROR"),
                f"emit_target_artifact rc={p.returncode}: {_stderr_excerpt(p.stderr)}",
            )
        _artifact = p.stdout
    elif not _artifact.strip():
        raise CertFailure(
            "emit_target_artifact",
            _cat("ELABORATION_ERROR"),
            "emit_analysis_bundle wrote a command buffer and no target artifact on stdout",
        )
    if identity_target and ((generated / artifact_name).exists() or (generated / artifact_name).is_symlink()):
        raise ValueError("shared pass target requires fresh host-owned output paths")
    (generated / artifact_name).write_text(_artifact, encoding="utf-8")
    if identity_target:
        identity_files["target_artifact"] = _configuration_file(generated / artifact_name)
    from . import artifact_scale as _artifact_scale

    _too_long = _artifact_scale.refusal(_artifact)
    if _too_long:
        raise CertFailure("emit_target_artifact", _cat("STRUCTURAL_INVARIANT_VIOLATION"), _too_long)
    if selected_target is not None:
        selected_target.publish(generated, cb_path, generated / artifact_name)
    if pass_contract is not None:
        require_pass_contract()
    if identity_target:
        verify_identity_target()
        receipt = generated / "shared_pass_target.json"
        if receipt.exists() or receipt.is_symlink():
            raise ValueError("shared pass target requires a fresh host-owned receipt")
        payload = (
            json.dumps(
                {
                    "schema": "merlin.shared_pass_target_transport.v1",
                    "requirement": shared_pass_requirement,
                    "consumer": pass_contract["target_consumer"],
                    "files": identity_files,
                    "owner_sha256": _configuration_file(Path(__file__).resolve())["sha256"],
                    "scope": "byte-identical target selection and invocation custody; semantic equivalence unproved",
                },
                sort_keys=True,
                allow_nan=False,
                indent=2,
            )
            + "\n"
        )
        with receipt.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        verify_identity_target()
    if memo_key is not None:
        import copy

        memo[memo_key] = (cb_path.read_text(encoding="utf-8"), copy.deepcopy(cb), _artifact)
    return cb, _artifact

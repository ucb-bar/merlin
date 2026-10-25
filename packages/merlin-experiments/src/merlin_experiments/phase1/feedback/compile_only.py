"""Original public source-only compilation through the ordinary package box.

The host selects cases, ABI, facts, build tools and the instruction policy.  A
candidate can select public member identifiers only.  Nothing executes the
linked no-data probe and no result here may earn numerical or RTL credit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path

from merlin.common import invocation_record
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import compile_only_execution as CO
from merlin.targetgen import package_runtime as P
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService
from merlin.targetgen.rtl.facts import observed_facts
from merlin_experiments.phase0 import public_compile_sources as SOURCES
from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService
from merlin_experiments.phase2 import campaign as PC
from merlin_experiments.phase2.contracts import exact_tree_record, sha256_file

SCHEMA = "merlin.public_compile_only_feedback.v1"
ACTION = "compile-only"
SCALE_PROBE_FLAG = "--scale-probe"


def _case_row(case):
    return {
        "member": case.member,
        "scale": case.scale,
        "variant": json.loads(case.derivation_json).get("variant", "original"),
        "source_sha256": hashlib.sha256(case.source_ir.encode()).hexdigest(),
        "original_abi_sha256": hashlib.sha256(
            json.dumps(case.original_abi.record(), sort_keys=True).encode()
        ).hexdigest(),
        "compilation_status": "not_attempted",
        "lowering_status": "not_attempted",
        "numeric": "not_attempted",
        "rtl": "not_attempted",
        "static_obligations": {
            name: "UNKNOWN"
            for name in (
                "semantic_coverage",
                "index_bounds",
                "resource_legality",
                "complete_output_coverage",
            )
        },
    }


@contextmanager
def _snapshot_operation(owner):
    """Batch custody reads within one original operation, never its grading lifetime."""
    if type(owner) is not CommonTransformService:
        raise ValueError("compile-only custody requires the exact original profile owner")
    from merlin.common.bundle_snapshot import verified_snapshot_scope
    from merlin.targetgen.tool_profile import (
        COMMON_SUBSTRATE_V3,
        COMMON_SUBSTRATE_V4,
        COMMON_SUBSTRATE_V5,
        COMMON_SUBSTRATE_V6,
        COMMON_SUBSTRATE_V7,
        COMMON_SUBSTRATE_V8,
        COMMON_SUBSTRATE_V9,
        COMMON_SUBSTRATE_V10,
        COMMON_SUBSTRATE_V11,
        COMMON_SUBSTRATE_V12,
    )

    with ExitStack() as checks:
        if owner.profile.get("common_substrate") in {
            COMMON_SUBSTRATE_V3,
            COMMON_SUBSTRATE_V4,
            COMMON_SUBSTRATE_V5,
            COMMON_SUBSTRATE_V6,
            COMMON_SUBSTRATE_V7,
            COMMON_SUBSTRATE_V8,
            COMMON_SUBSTRATE_V9,
            COMMON_SUBSTRATE_V10,
            COMMON_SUBSTRATE_V11,
            COMMON_SUBSTRATE_V12,
        }:
            checks.enter_context(verified_snapshot_scope(owner.workspace, owner.bundle, repo=owner.repo))
        yield checks.close


@contextmanager
def _compile_imports(owner):
    """Issue one original namespace; each command still has independent freshness."""
    with ExitStack() as lifetime:
        with _snapshot_operation(owner):
            lifetime.enter_context(owner.compiler_imports())
        # No verified snapshot scope survives into the compilation body.
        # The original import admission re-verifies again on lifetime exit.
        yield


def _emit_scale_case(candidate, source, output, *, contract, target, timeout_s, ir_budget):
    """The ordinary package pipeline only; no object, linker, tensor or engine."""
    from merlin.common.cpu_pool import shared_compute_pool
    from merlin.llvmlower.device_ir_scalability import DeviceIRBudget, analyze_device_ir
    from merlin.llvmlower.toolchain import mlir_translate
    from merlin.targetgen import capsule_common as CC

    if ir_budget is not None and type(ir_budget) is not DeviceIRBudget:
        raise ValueError("scale probe budget must be the original typed host selection")
    if P.active_package_executor() is None:
        raise ValueError("scale probe lost its original scoped package executor")
    deadline = time.monotonic() + timeout_s

    def remaining_seconds():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("original scale emission deadline expired")
        return remaining

    package = P.load_package(candidate, contract=contract)
    if package.target != target:
        raise ValueError("scale probe package belongs to another original target")
    P.integrity_scan(package)
    P.build_package(package, timeout=remaining_seconds())
    generated = output / "generated"
    generated.mkdir(parents=True, mode=0o700)
    source_pin = CO._pin(source)
    products = tuple(
        generated / name
        for name in (
            "input.interface.mlir",
            "command_buffer.json",
            "lowered.target.mlir",
            "lowered.llvm.mlir",
        )
    )
    def invoke(*args, **kwargs):
        return P.run_entrypoint(
            *args, **{**kwargs, "timeout": remaining_seconds(), "invocation_directory": generated}
        )

    with invocation_record.observe_call(
        output,
        stage="public_scale_source_lowering",
        function=CC.lower_interface,
        arguments={"target": target, "timeout_s": timeout_s},
        inputs=(source,),
        outputs=products,
        dependencies=(Path(__file__),),
    ) as observed:
        _, artifact = CC.lower_interface(
            package, source, generated, contract=contract, timeout=remaining_seconds(), invoke=invoke
        )
        observed.returned(stdout=artifact)
    if CO._pin(source) != source_pin or products[-1].read_text() != artifact:
        raise ValueError("scale probe original source or actual emitted product changed")
    invocation_record.verify(observed.path)
    # Count actual LLVM instructions, not the superficially similar LLVM-dialect
    # MLIR. Translation verifies the emitted IR but never builds or runs an object.
    lowered_pin = CO._pin(products[-1])
    llvm = generated / "lowered.ll"
    command = [str(mlir_translate().resolve()), "--mlir-to-llvmir", str(products[-1]), "-o", str(llvm)]
    with invocation_record.observe(
        output,
        stage="public_scale_llvm_translation",
        argv=command,
        inputs=(products[-1],),
        outputs=(llvm,),
        dependencies=(Path(__file__),),
    ) as translated:
        with shared_compute_pool(timeout_s=remaining_seconds()):
            process = subprocess.run(
                command, capture_output=True, text=True, timeout=remaining_seconds()
            )
        translated.complete(process)
    if process.returncode:
        raise ValueError("original stock translation refused emitted IR")
    invocation_record.verify(translated.path)
    if CO._pin(products[-1]) != lowered_pin or CO._pin(source) != source_pin:
        raise ValueError("scale probe original input or emitted IR changed during translation")
    analysis = analyze_device_ir(llvm)
    decision = ir_budget.evaluate(analysis) if ir_budget is not None else {"mode": "not_selected"}
    return {
        "lowering_status": "resource_refused" if decision.get("status") == "refused" else "emitted",
        "compilation_status": "not_attempted",
        "code_growth": {
            **analysis,
            "emitted_mlir_bytes": products[-1].stat().st_size,
            "emitted_mlir_sha256": lowered_pin["sha256"],
        },
        "resource_budget": decision,
        "emission_invocation_sha256": sha256_file(observed.path),
        "translation_invocation_sha256": sha256_file(translated.path),
        "static_obligations": {
            name: "UNKNOWN"
            for name in (
                "semantic_coverage",
                "index_bounds",
                "resource_legality",
                "complete_output_coverage",
            )
        },
    }


def snapshot_submission(source, destination):
    """One bounded compiler-source copy; never copy tensor data or build trees."""
    from merlin_experiments.phase2.functional_guard_execution import copy_plain_file
    from merlin_experiments.phase2.retained_revisions import source_inventory

    source, destination = Path(source), Path(destination)
    before = source_inventory(source, max_bytes=32 << 20, max_files=8192, exclude_generated=True)
    destination.mkdir(mode=0o700)
    for relative, kind, _mode in before["entries"]:
        selected, copied = source / relative, destination / relative
        if kind == "directory":
            copied.mkdir(parents=True, exist_ok=True)
        else:
            copied.parent.mkdir(parents=True, exist_ok=True)
            copy_plain_file(selected, copied)
            copied.chmod(0o555 if selected.stat().st_mode & 0o111 else 0o444)
    after = source_inventory(source, max_bytes=32 << 20, max_files=8192, exclude_generated=True)
    observed = source_inventory(destination, max_bytes=32 << 20, max_files=8192)
    if before != after or observed != before:
        shutil.rmtree(destination)
        raise ValueError("compile-only compiler changed during its bounded source snapshot")
    for path in [*destination.rglob("*"), destination]:
        if path.is_dir():
            path.chmod(0o555)
    return destination


class _Executor:
    """Reuse the ordinary original package command, isolated input and output."""

    def __init__(self, target, candidate, output, *, common_service, inputs=None, scope=None):
        self.target, self.candidate, self.output, self.inputs = target, candidate, output, inputs
        self.common_service, self.scope = common_service, scope

    def build_package(self, pkg, *, timeout):
        if pkg.directory != self.candidate or any(
            (pkg.manifest.get("build") or {}).get(k) for k in ("configure", "command")
        ):
            raise ValueError("compile-only requires the ordinary checked package without unboxed build steps")

    def run_entrypoint(
        self,
        pkg,
        name,
        input_mlir,
        output_json=None,
        *,
        timeout=600,
        invocation_directory=None,
        write_bytecode=False,
        artifact_profile=None,
    ):
        if pkg.directory != self.candidate or write_bytecode or artifact_profile is not None:
            raise ValueError("compile-only invocation lost its original package/protocol owner")
        if invocation_directory is None or not Path(invocation_directory).is_relative_to(self.output):
            raise ValueError("compile-only invocation escaped its private product owner")
        if type(self) is not _Executor or P.active_package_executor() is not self:
            raise ValueError("compile-only invocation lost its exact original scoped executor")
        with (
            _snapshot_operation(self.common_service) as finish_snapshot,
            tempfile.TemporaryDirectory(prefix="call-", dir=self.output.parent) as directory,
        ):
            workspace = Path(directory)
            incoming = workspace / "input.mlir"
            from merlin_experiments.phase2.functional_guard_execution import copy_plain_file

            copy_plain_file(input_mlir, incoming)
            outgoing = workspace / "output.json" if output_json is not None else None
            policy = PC.package_sandbox_policy(
                self.target, workspace, self.candidate, inputs=self.inputs, scope=self.scope
            )
            # The normal package policy reuses the exact live original profile's
            # public snapshot leaves and destination-derived import roots.
            policy = replace(policy, argv=(*policy.argv, "--ro-bind", str(incoming), str(incoming)))
            argv = P._resolve_argv(pkg, name, incoming, outgoing)
            if P._needs_interpreter(pkg, argv):
                argv = [sys.executable, *argv]
            shell = policy.env_prefix
            from merlin_experiments.phase1.brokers.compiler_call_channel import compiler_call_transport
            from merlin_experiments.phase2 import normal_resource_launch as resources

            with ExitStack() as observations:
                with compiler_call_transport(
                    self.common_service,
                    workspace=workspace,
                    package=pkg.directory,
                    input_path=incoming,
                    command_name=name,
                    timeout_s=timeout,
                    borrow_compute=True,
                ) as transport:
                    shell = transport.env_prefix(shell) + 'exec "$@"'
                    with resources.bounded_normal_scratch(
                        policy.argv, owner=invocation_directory, boundary=resources.resources.active_resource_boundary()
                    ) as prefix:
                        command = [
                            *prefix,
                            "--chdir",
                            str(self.candidate),
                            "bash",
                            "-c",
                            shell,
                            "compile-package",
                            *argv,
                        ]
                        selected_env = dict(os.environ)
                        observed = observations.enter_context(
                            invocation_record.observe(
                                invocation_directory,
                                stage=name,
                                argv=command,
                                inputs=(input_mlir,),
                                dependencies=(Path(__file__),),
                                env=selected_env,
                            )
                        )
                        process_started_ns, process_started = time.time_ns(), time.monotonic()
                        process = subprocess.run(
                            command,
                            capture_output=True,
                            text=True,
                            timeout=transport.remaining_timeout(),
                            env=selected_env,
                        )
                        process_finished_ns, process_wall_s = time.time_ns(), time.monotonic() - process_started
                # The real transport has reaped its proof helper and performed
                # its independent final owner checks. Close full snapshot
                # freshness before publishing either receipt or staged output.
                self.common_service._verify()
                finish_snapshot()
                observed.document["process_timing"] = {
                    "started_ns": process_started_ns,
                    "finished_ns": process_finished_ns,
                    "wall_s": process_wall_s,
                    "scope": "actual candidate process only; transport and custody checks excluded",
                }
                observed.complete(process)
                if outgoing is not None and (outgoing.exists() or outgoing.is_symlink()):
                    copy_plain_file(outgoing, output_json, output=True)
            return process


class _InstructionPolicy:
    def __init__(self, target, roles, facts, facts_path):
        self.target, self.roles, self.facts, self.facts_path = target, roles, facts, facts_path

    def evaluate(self, *, elf, evidence_root):
        from merlin.perf.isa_prohibition import scan_elf
        from merlin.targetgen.elf_lanes import executable_sections

        if self.roles:
            with observed_facts(self.target, self.facts, self.facts_path):
                scan = scan_elf(elf, target=self.target, roles=self.roles)
            accepted = scan.get("status") == "measured" and scan.get("clean") is True
        else:
            # The original policy explicitly declares no prohibitions. Do not
            # convert this into a positive instruction/physical legality claim.
            sections = executable_sections(elf.read_bytes())
            scan = {"status": "none_declared", "roles": [], "executable_sections": len(sections)}
            accepted = bool(sections)
        evidence_root.mkdir(parents=True, mode=0o700)
        report = evidence_root / "policy.json"
        data = {
            "status": "accepted" if accepted else "refused",
            "elf_sha256": sha256_file(elf),
            "original_policy": scan,
            "scope": "original static prohibition only; no execution/effects proof",
        }
        report.write_text(json.dumps(data, sort_keys=True))
        return {**data, "report_path": str(report), "report_sha256": sha256_file(report)}


def _roles(capsules):
    selections = []
    for capsule in capsules:
        arm = ((capsule.get("performance") or {}).get("arms") or {}).get("candidate") or {}
        policy = arm.get("instruction_policy")
        if policy is None:
            raise ValueError("public compilation has no explicit original instruction policy")
        roles = policy.get("prohibited_instruction_roles")
        if not isinstance(roles, list) or any(type(role) is not str for role in roles):
            raise ValueError("public compilation instruction policy is unresolved")
        selections.append(tuple(sorted(roles)))
    if selections and len(set(selections)) != 1:
        raise ValueError("public compilation members disagree on the original instruction policy")
    return selections[0] if selections else ()


def _case_instruction_policy(capsule, *, target):
    """Reuse the original capsule/native selections; absence grants no permissions."""
    field = "performance.arms.candidate.instruction_policy.prohibited_instruction_roles"
    guidance = ("Use the ordinary certifying agent_selfcheck.py request without --compile-only; "
                "the operator must repair unavailable frozen policy metadata.")

    def unavailable(code, missing_field):
        return {"roles": None, "pins": (), "diagnostic": {
            "code": code, "missing_field": missing_field, "action": guidance,
            "permission_inferred": False}}

    policy = ((capsule.get("performance") or {}).get("arms") or {}).get("candidate") or {}
    declared = policy.get("instruction_policy")
    roles, pins = set(), {}
    if declared is not None:
        try:
            roles.update(_roles([capsule]))
        except (ValueError, AttributeError, TypeError):
            return unavailable("invalid_capsule_instruction_policy", field)
    if os.environ.get("MERLIN_TARGET_EXPERIMENT"):
        from merlin.targetgen import native_model_policy

        try:
            selected = native_model_policy._frozen_model_policy(Path(capsule["__dir__"]), target=target)
            roles.update(selected["prohibited_instruction_roles"])
            for name in ("descriptor", "manifest", "capsule_declaration", "instruction_coverage"):
                pin = selected[name]
                path = Path(pin["path"])
                if path.resolve() != path or path.is_symlink() or sha256_file(path) != pin["sha256"]:
                    raise ValueError("selected native policy changed")
                pins[str(path)] = pin["sha256"]
            source = Path(native_model_policy.__file__).resolve()
            pins[str(source)] = sha256_file(source)
        except Exception:  # noqa: BLE001 — original unavailable authority cannot authorize a build
            return unavailable("selected_native_instruction_policy_unavailable",
                               "selected_corpus.instruction_policy")
    elif declared is None:
        return unavailable("missing_original_instruction_policy", field)
    declaration = Path(capsule["__dir__"]) / "capsule.yaml"
    pins[str(declaration)] = sha256_file(declaration)
    return {"roles": tuple(sorted(roles)), "pins": tuple(sorted(pins.items())), "diagnostic": None}


def _growth(actual):
    """Only actual reopened products supply sizes; schedules/claims supply none."""

    def file(name):
        pin = actual["products"].get(name)
        if pin is None:
            return None
        path = Path(pin["path"])
        if path.is_symlink() or not path.is_file() or sha256_file(path) != pin["sha256"]:
            raise ValueError("compile-only growth product changed")
        return path

    llvm, stack = file("generated/lowered.llvm.mlir"), file("build/kernel.stack_frame.json")
    obj = file("build/kernel.o")
    elf = Path(actual["elf"]["path"]) if "path" in actual["elf"] else None
    if elf is not None and sha256_file(elf) != actual["elf"]["sha256"]:
        raise ValueError("compile-only linked growth product changed")
    stack_record = json.loads(stack.read_text()) if stack is not None else {}
    return {
        "emitted_llvm_bytes": llvm.stat().st_size if llvm is not None else None,
        "emitted_llvm_lines": sum(1 for _ in llvm.open()) if llvm is not None else None,
        "object_bytes": obj.stat().st_size if obj is not None else None,
        "linked_elf_bytes": elf.stat().st_size if elf is not None else None,
        "retained_entry_bytes": actual["retained_entry"]["size_bytes"],
        "stack_frame": {
            key: stack_record.get(key)
            for key in (
                "schema",
                "status",
                "entry_symbol",
                "max_static_bytes",
                "frame_bytes",
                "headroom_bytes",
                "report_rows",
                "llvm_ir_sha256",
                "object_sha256",
                "stack_usage_report_sha256",
            )
        }
        if stack is not None
        else None,
        "scope": "actual source/object/link size and selected stack limit; index/device legality unknown",
    }


def observe(
    *,
    candidate,
    target_experiment,
    capsules,
    contract,
    facts_path,
    common_service,
    work_root,
    timeout_s,
    selected="all",
    sandbox_inputs=None,
    scale_probe=False,
    ir_budget=None,
):
    """Run exact original public cases and reopen actual linked products.

    ``common_service`` is the original host's frozen profile owner, never request
    metadata. Source/ABI construction and build callbacks stay in
    this original service. Direct saved success claims are never inputs.
    """
    from merlin.llvmlower.toolchain import readelf
    from merlin.perf import isa_prohibition
    from merlin.targetgen import elf_lanes
    from merlin.targetgen.native_model_execution import _build_service_for

    observation_started = time.monotonic()
    if type(common_service) is not CommonTransformService:
        raise ValueError("compile-only requires the original admitted compiler tool profile")
    with _snapshot_operation(common_service):
        common_service._verify()
        original_profile = common_service.profile
        emission_selection = None
        if original_profile.get("common_substrate") == "merlin.common_compiler_substrate.v3":
            from merlin.verify.lean_emission_chain import EmissionChainSelection

            emission_selection = EmissionChainSelection(common_service)
        if type(timeout_s) is not int or not 1 <= timeout_s <= 600:
            raise ValueError("compile-only aggregate deadline must be 1..600 seconds")
        candidate, contract, facts_path, work_root = map(Path, (candidate, contract, facts_path, work_root))
        if any(path.resolve() != path or path.is_symlink() for path in (candidate, contract, facts_path, work_root)):
            raise ValueError("compile-only requires canonical ordinary original owners")
        if work_root.exists() or any(
            work_root.is_relative_to(path) or path.is_relative_to(work_root)
            for path in (candidate, contract, facts_path.parent)
        ):
            raise ValueError("compile-only products require a fresh separate host owner")
        if type(scale_probe) is not bool:
            raise ValueError("public scale mode is an original explicit selection")
        facts_raw = facts_path.read_bytes()
        facts = json.loads(facts_raw)
        if (facts.get("facts") or {}).get("target") != target_experiment.target:
            raise ValueError("compile-only selected facts identify another target")
        if scale_probe:
            from merlin_experiments.phase0 import public_scale_sources

            edges = {
                row.get("cols")
                for row in (facts.get("facts") or {}).get("arrays", [])
                if type(row.get("cols")) is int and row["cols"] > 0
            }
            cases, source_manifest = public_scale_sources.derive(
                capsules,
                selected=selected,
                index_bits=original_profile.get("host_index_bits"),
                tile_edge=next(iter(edges)) if len(edges) == 1 else None,
            )
        else:
            cases, source_manifest = SOURCES.derive(
                capsules, selected=selected, index_bits=original_profile.get("host_index_bits")
            )
        before = exact_tree_record(candidate)["sha256"]
        work_root.mkdir(parents=True, mode=0o700)
        sources = work_root / "sources"
        sources.mkdir()
        members = {case.member for case in cases}
        policies = {} if scale_probe else {
            str(capsule.get("name") or Path(capsule["__dir__"]).name):
                _case_instruction_policy(capsule, target=target_experiment.target)
            for capsule in capsules
            if str(capsule.get("name") or Path(capsule["__dir__"]).name) in members
        }
        service = _build_service_for(target_experiment.target) if any(
            policy["diagnostic"] is None for policy in policies.values()
        ) else None
        pins = sorted(
            set(
                [
                    Path(__file__),
                    Path(isa_prohibition.__file__),
                    Path(elf_lanes.__file__),
                    facts_path,
                    *(path for case in cases for path, _ in case.provenance),
                ]
            )
        )
    deadline_started = time.monotonic()
    deadline, rows = deadline_started + timeout_s, []
    # Every case of this one request shares its immutable package-policy setup.
    policy_scope = PC.PackagePolicyScope()
    dispatch_timing = {}
    from . import scale_probe_child

    parallel = scale_probe and scale_probe_child.batch_workers() is not None
    if parallel:
        rows = scale_probe_child.emit_cases(
            owner=common_service,
            candidate=candidate,
            target_experiment=target_experiment,
            capsules=capsules,
            cases=cases,
            contract=contract,
            facts_path=facts_path,
            work_root=work_root,
            deadline=deadline,
            ir_budget=ir_budget,
            sandbox_inputs=sandbox_inputs,
            timing=dispatch_timing,
        )
    with nullcontext() if parallel else _compile_imports(common_service):
        for index, case in enumerate(() if parallel else cases):
            case.verify()
            root, source = work_root / f"case-{index:03d}", sources / f"case-{index:03d}.mlir"
            source.write_text(case.source_ir)
            remaining = min(600, math.floor(deadline - time.monotonic()))
            row = _case_row(case)
            admission = None
            if not scale_probe:
                selected_policy = policies[case.member]
                if selected_policy["diagnostic"] is not None:
                    row.update(compilation_status="unavailable", reason="instruction_policy_unavailable",
                               instruction_policy_diagnostic=selected_policy["diagnostic"])
                    rows.append(row)
                    continue
                checker = _InstructionPolicy(target_experiment.target, selected_policy["roles"], facts, facts_path)
                bound = {str(path): sha256_file(path) for path in pins}
                bound.update(selected_policy["pins"])
                admission = LinkedElfAdmissionService(target_experiment.target, checker.evaluate,
                                                     tuple(sorted(bound.items())))
            if remaining < 1:
                row["reason"] = "original aggregate compile budget exhausted"
                rows.append(row)
                continue
            started = time.monotonic()
            try:
                executor = _Executor(
                    target_experiment,
                    candidate,
                    root,
                    common_service=common_service,
                    inputs=sandbox_inputs,
                    scope=policy_scope,
                )
                with (
                    P.scoped_package_executor(executor),
                    observed_facts(target_experiment.target, facts, facts_path),
                ):
                    if scale_probe:
                        row.update(
                            _emit_scale_case(
                                candidate,
                                source,
                                root,
                                contract=contract,
                                target=target_experiment.target,
                                timeout_s=remaining,
                                ir_budget=ir_budget,
                            )
                        )
                        row["wall_seconds"] = time.monotonic() - started
                        case.verify()
                        rows.append(row)
                        continue
                    CO.compile_source_only(
                        package_dir=candidate,
                        source=source,
                        original_abi=case.original_abi,
                        contract_root=contract,
                        target=target_experiment.target,
                        output_root=root,
                        build_service=service,
                        elf_admission=admission,
                        readelf=readelf().resolve(),
                        timeout_s=remaining,
                        emission_verification=emission_selection,
                    )
                    actual = CO.verify_compile_only_report(
                        root / "compile_only_result.json",
                        build_service=service,
                        elf_admission=admission,
                        emission_verification=emission_selection,
                    )
                row.update(
                    compilation_status=actual["compilation_status"],
                    retained_entry=actual["retained_entry"],
                    elf_sha256=actual["elf"]["sha256"],
                    instruction_policy=actual["instruction_policy"]["status"],
                    actual_product_sha256s={name: pin["sha256"] for name, pin in actual["products"].items()},
                    invocation_sha256s=[pin["sha256"] for pin in actual["invocations"]],
                    code_growth=_growth(actual),
                    emission_relationship={
                        key: (actual.get("emission_proof_chain") or {}).get(key)
                        for key in ("status", "scope", "outside_formal_coverage")
                    },
                )
            except Exception as error:  # noqa: BLE001 - failures are incomplete observations, never grades
                row.update(compilation_status="unavailable", reason=type(error).__name__)
                if scale_probe:
                    row["lowering_status"] = "unavailable"
                emitted = root / "generated/lowered.llvm.mlir"
                if scale_probe and emitted.is_file() and not emitted.is_symlink():
                    # A timed-out/refused later stage must not hide already emitted
                    # growth. No lexical LLVM counts are inferred from MLIR text.
                    row["partial_emission"] = {
                        "bytes": emitted.stat().st_size,
                        "sha256": sha256_file(emitted),
                        "scope": "unverified retained MLIR bytes",
                    }
            row["wall_seconds"] = time.monotonic() - started
            case.verify()
            rows.append(row)
    final_checks_started = time.monotonic()
    if before != exact_tree_record(candidate)["sha256"] or facts_path.read_bytes() != facts_raw:
        raise ValueError("compile-only compiler or frozen hardware inputs changed during original compilation")
    common_service._verify()
    result = {
        "schema": SCHEMA,
        "candidate_sha256": before,
        "source_manifest": source_manifest,
        "compiler_tool_profile_sha256": original_profile["sha256"],
        "facts_sha256": hashlib.sha256(facts_raw).hexdigest(),
        "cases": rows,
        "mode": "scale_emission_only" if scale_probe else "object_and_link",
        "all_pass": False,
        "numeric": "not_attempted",
        "rtl": "not_attempted",
        "timing_cycles": None,
        "wall_timing": {
            "initial_source_setup_s": deadline_started - observation_started,
            "selected_compile_budget_s": timeout_s,
            "dispatch": dispatch_timing if parallel else None,
            "final_source_custody_s": time.monotonic() - final_checks_started,
            "total_observation_s": time.monotonic() - observation_started,
            "scope": "all original observation wall stages; compile deadline begins after source setup, "
            "final custody remains mandatory and separately recorded",
        },
        "scope": "original public emission/code-growth only; no object/ABI/effects or execution proof"
        if scale_probe
        else "original public structural compile-only; not correctness, generality or timing acceptance",
    }
    (work_root / "public_compile_only.json").write_text(json.dumps(result, sort_keys=True, indent=2))
    return result


def run(
    *,
    submission,
    target_experiment,
    capsules_root,
    contract,
    facts_path,
    common_service,
    work_root,
    timeout_s,
    selected="all",
    sandbox_inputs=None,
    scale_probe=False,
    ir_budget=None,
):
    # Discover metadata/source only; run_capsule/reference/harness generators are
    # deliberately absent from this path.
    capsules = CR.discover_capsules(str(capsules_root), labels={"public", "dev"}, contract=contract)
    return observe(
        candidate=submission,
        target_experiment=target_experiment,
        capsules=capsules,
        contract=contract,
        facts_path=facts_path,
        common_service=common_service,
        work_root=work_root,
        timeout_s=timeout_s,
        selected=selected,
        sandbox_inputs=sandbox_inputs,
        scale_probe=scale_probe,
        ir_budget=ir_budget,
    )

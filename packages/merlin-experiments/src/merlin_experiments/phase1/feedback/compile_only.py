"""Original public source-only compilation through the ordinary package box.

The host selects cases, ABI, facts, build tools and the instruction policy.  A
candidate can select public member identifiers only.  Nothing executes the
linked no-data probe and no result here may earn numerical or RTL credit.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
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

    def __init__(self, target, candidate, output, *, common_service, inputs=None):
        self.target, self.candidate, self.output, self.inputs = target, candidate, output, inputs
        self.common_service = common_service

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
        with tempfile.TemporaryDirectory(prefix="call-", dir=self.output.parent) as directory:
            workspace = Path(directory)
            incoming = workspace / "input.mlir"
            from merlin_experiments.phase2.functional_guard_execution import copy_plain_file

            copy_plain_file(input_mlir, incoming)
            outgoing = workspace / "output.json" if output_json is not None else None
            policy = PC.package_sandbox_policy(self.target, workspace, self.candidate, inputs=self.inputs)
            # The normal package policy reuses the exact live original profile's
            # public snapshot leaves and destination-derived import roots.
            policy = replace(policy, argv=(*policy.argv, "--ro-bind", str(incoming), str(incoming)))
            argv = P._resolve_argv(pkg, name, incoming, outgoing)
            if P._needs_interpreter(pkg, argv):
                argv = [sys.executable, *argv]
            shell = policy.env_prefix
            from merlin_experiments.phase1.brokers.compiler_call_channel import compiler_call_transport
            from merlin_experiments.phase2 import normal_resource_launch as resources

            with compiler_call_transport(self.common_service, workspace=workspace, package=pkg.directory,
                    input_path=incoming, command_name=name, timeout_s=timeout) as transport:
                shell = transport.env_prefix(shell) + 'exec "$@"'
                with resources.bounded_normal_scratch(
                    policy.argv, owner=invocation_directory, boundary=resources.resources.active_resource_boundary()
                ) as prefix:
                    process = invocation_record.run(
                        [*prefix, "--chdir", str(self.candidate), "bash", "-c", shell, "compile-package", *argv],
                        directory=invocation_directory,
                        stage=name,
                        inputs=(input_mlir,),
                        dependencies=(Path(__file__),),
                        capture_output=True,
                        text=True,
                        timeout=transport.remaining_timeout(),
                    )
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

    if type(common_service) is not CommonTransformService:
        raise ValueError("compile-only requires the original admitted compiler tool profile")
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
    cases, source_manifest = SOURCES.derive(
        capsules, selected=selected, index_bits=original_profile.get("host_index_bits")
    )
    facts_raw = facts_path.read_bytes()
    facts = json.loads(facts_raw)
    if (facts.get("facts") or {}).get("target") != target_experiment.target:
        raise ValueError("compile-only selected facts identify another target")
    before = exact_tree_record(candidate)["sha256"]
    work_root.mkdir(parents=True, mode=0o700)
    sources = work_root / "sources"
    sources.mkdir()
    service = _build_service_for(target_experiment.target)
    checker = _InstructionPolicy(target_experiment.target, _roles(capsules), facts, facts_path)
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
    admission = LinkedElfAdmissionService(
        target_experiment.target, checker.evaluate, tuple((str(path), sha256_file(path)) for path in pins)
    )
    deadline, rows = time.monotonic() + timeout_s, []
    for index, case in enumerate(cases):
        case.verify()
        root, source = work_root / f"case-{index:03d}", sources / f"case-{index:03d}.mlir"
        source.write_text(case.source_ir)
        remaining = min(600, math.floor(deadline - time.monotonic()))
        row = {
            "member": case.member,
            "scale": case.scale,
            "source_sha256": hashlib.sha256(case.source_ir.encode()).hexdigest(),
            "original_abi_sha256": hashlib.sha256(
                json.dumps(case.original_abi.record(), sort_keys=True).encode()
            ).hexdigest(),
            "compilation_status": "not_attempted",
            "numeric": "not_attempted",
            "rtl": "not_attempted",
            "static_obligations": {
                key: "UNKNOWN"
                for key in ("semantic_coverage", "index_bounds", "resource_legality", "complete_output_coverage")
            },
        }
        if remaining < 1:
            row["reason"] = "original aggregate compile budget exhausted"
            rows.append(row)
            continue
        started = time.monotonic()
        try:
            executor = _Executor(
                target_experiment, candidate, root, common_service=common_service, inputs=sandbox_inputs
            )
            with (
                common_service.compiler_imports(),
                P.scoped_package_executor(executor),
                observed_facts(target_experiment.target, facts, facts_path),
            ):
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
        row["wall_seconds"] = time.monotonic() - started
        case.verify()
        rows.append(row)
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
        "all_pass": False,
        "numeric": "not_attempted",
        "rtl": "not_attempted",
        "timing_cycles": None,
        "scope": "original public structural compile-only; not correctness, generality or timing acceptance",
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
    )

"""Original frozen-profile import admission and isolated package execution.

Numerical graders keep their original sources, oracles and verdicts. The
compiler sees one input, its declared output and the same public source grants
as its author. Byte-identical grader copies may move without changing compiler
authority; mutable source, callback-selected policies and foreign owners refuse.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import threading
import time
import weakref
from pathlib import Path

import yaml

from merlin.development.targetgen.execution.host.common_transforms import CommonTransformService
from merlin.development.targetgen.packaging import loading as P
from merlin.evaluation.capsule.operations import setup_timing

# ---- compiler_profile_runtime ------------------------------------------------------------------

_INVOCATION_PUBLICATION = contextvars.ContextVar("original_snapshot_publication", default=None)

_EMISSION_OBSERVATION = contextvars.ContextVar("original_profile_emission_observation", default=None)

_CONCURRENT_EXECUTORS = weakref.WeakSet()


@contextlib.contextmanager
def _verification_compute():
    """Let ready execution use shared CPUs before background custody work.

    No held lease is preempted and no check is skipped. This context ends before
    compiler execution, restoring its original author/grade priority.
    """
    from merlin.common.process import cpu_pool

    with cpu_pool.cpu_priority(cpu_pool.VERIFICATION), cpu_pool.shared_compute_pool():
        yield


def finish_snapshot_invocation():
    """Complete the request's custody boundary before output/receipt publication."""
    close = _INVOCATION_PUBLICATION.get()
    if close is not None:
        close()


def _checkpoint_command_admission():
    """Defer new background commands after a trusted epoch/final checkpoint.

    Ordinary source edits do not request this stop. Never call this from final
    custody or while an owned child is running. Final frozen-compiler checks do
    not carry the background snapshot admission scope and remain eligible.
    """
    from merlin.development.targetgen.grading.scheduling import background_admission_selected

    if not background_admission_selected():
        return
    from merlin.development.backend.campaign import control as joint_control

    reason = joint_control.stop_reason()
    if reason is not None:
        raise P.NotMeasuredFailure(
            "preparation_admission",
            "BACKGROUND_DEFERRED",
            reason,
            status="gated",
            execution_status="deferred_before_command",
            correctness="unknown",
        )


def snapshot_scoped_entrypoint(function):
    """Only exact original execution owners can batch one compiler invocation."""

    from merlin.development.backend.feedback.execution.profile_executor import _Executor

    @functools.wraps(function)
    def call(self, *args, **kwargs):
        from merlin.common.storage.bundle_snapshots import verified_snapshot_scope
        from merlin.development.backend.qualification.components.package_execution import (
            ComponentPackageExecutor,
        )

        if type(self) is _Executor:
            selected = self
        elif type(self) is ComponentPackageExecutor:
            selected = self.original_execution
            if selected is None:
                return function(self, *args, **kwargs)
            if type(selected) is not _Executor or P.active_package_executor() is not self:
                raise ValueError("snapshot batching lost its exact original component executor")
        else:
            raise ValueError("snapshot batching requires an exact original executor")
        if type(selected.owner) is not CommonTransformService:
            raise ValueError("snapshot batching requires its original profile owner")
        _checkpoint_command_admission()
        with (
            setup_timing.selected(
                selected.work_root.parent,
                selected.source_sha256,
                operation=args[1] if len(args) > 1 and isinstance(args[1], str) else function.__name__,
            ),
            contextlib.ExitStack() as stack,
        ):
            owner = selected.owner
            stack.enter_context(
                setup_timing.timed_context(
                    "invocation_snapshot",
                    verified_snapshot_scope(
                        owner.workspace,
                        owner.bundle,
                        repo=owner.repo,
                        entry=selected.snapshot_entry_for_call(),
                        guarded_exit=True,
                    ),
                )
            )
            token = _INVOCATION_PUBLICATION.set(stack.close)
            try:
                return function(self, *args, **kwargs)
            finally:
                _INVOCATION_PUBLICATION.reset(token)

    return call


def selected_service(*, workspace, run_dir, repo):
    """Read only the original archived manifest; historical runs keep no scope."""
    manifest = Path(run_dir) / "input_bundle_manifest.yaml"
    if not manifest.is_file():
        return None
    bundle = yaml.safe_load(manifest.read_bytes())
    if bundle.get("compiler_tool_profile") is None:
        return None
    from merlin.development.backend.feedback.execution.snapshot_evaluator import selected
    from merlin.development.backend.feedback.execution.snapshots import SnapshotInputs

    owner_type = SnapshotInputs if selected() else CommonTransformService
    return owner_type(workspace=Path(workspace), bundle_manifest=manifest, repo=Path(repo))


def functional_service(run_dir):
    """Recover the same original owner from admitted Phase 1 runtime provenance."""
    from merlin.development.backend.optimization.campaign import functional_boundary as PC
    from merlin.development.backend.optimization.contracts import mapping_file

    environment_path = Path(run_dir) / "environment.yaml"
    if not environment_path.is_file():
        archive = Path(run_dir) / "input_bundle_manifest.yaml"
        if archive.is_file() and mapping_file(archive, yaml_file=True).get("compiler_tool_profile") is not None:
            raise ValueError("original compiler profile has no admitted environment")
        return None
    environment = mapping_file(environment_path, yaml_file=True)
    if environment.get("compiler_tool_profile") is None:
        return None
    original = PC.verify_private_input_snapshot(Path(run_dir), environment)
    from merlin.development.backend.feedback.execution.snapshot_evaluator import selected
    from merlin.development.backend.feedback.execution.snapshots import SnapshotInputs

    owner_type = SnapshotInputs if selected() else CommonTransformService
    service = owner_type(
        workspace=original["workspace"], bundle_manifest=original["bundle_path"], repo=original["repo"]
    )
    if service.profile != environment["compiler_tool_profile"]:
        raise ValueError("compiler execution differs from the original Phase 1 tool profile")
    return service


def compiler_profile_binding(service, *, run_dir):
    """Close ordinary retained-job identity over its original Phase 1 owner."""
    if service is None:
        if functional_service(run_dir) is not None:
            raise ValueError("ordinary compiler lost its admitted original profile")
        return None
    from merlin.development.backend.feedback.execution.snapshot_evaluator import selected
    from merlin.development.backend.feedback.execution.snapshots import SnapshotInputs

    if selected():
        if (
            type(service) is not SnapshotInputs
            or service.bundle_manifest != Path(run_dir) / "input_bundle_manifest.yaml"
        ):
            raise ValueError("external compiler selected another snapshot input owner")
        service._verify()
        return service.record()
    if (
        type(service) is not CommonTransformService
        or service.bundle_manifest != Path(run_dir) / "input_bundle_manifest.yaml"
    ):
        raise ValueError("ordinary compiler selected another original profile owner")
    service._verify()
    from merlin.development.backend.optimization.contracts import mapping_file

    if service.profile != mapping_file(Path(run_dir) / "environment.yaml", yaml_file=True)["compiler_tool_profile"]:
        raise ValueError("ordinary compiler changed its original profile")
    return {
        "archive": str(service.bundle_manifest),
        "profile_sha256": service.profile["sha256"],
        "snapshot_sha256": service.snapshot_digest,
    }


@contextlib.contextmanager
def original_compiler_imports(service):
    if service is None:
        yield
        return
    if type(service) is not CommonTransformService:
        raise ValueError("compiler imports require the exact original owner")
    from merlin.verification.lean.emission_custody import original_emission_checks

    with service.compiler_imports(), original_emission_checks(service):
        yield


@contextlib.contextmanager
def selfcheck_execution(bundle_manifest, *, context, submission, work_root):
    """Close setup custody before grading; keep original execution scopes alive."""
    from merlin.common.storage.bundle_snapshots import verified_snapshot_scope
    from merlin.development.backend.feedback.execution import snapshot_evaluator as snapshot_execution
    from merlin.target.registry.descriptor_loading import load_target_experiment

    if bundle_manifest is None:
        if snapshot_execution.selected():
            raise ValueError("snapshot execution requires its selected input bundle")
        yield
        return
    if snapshot_execution.selected():
        from merlin.development.backend.feedback.execution.snapshots import SnapshotInputs

        owner = SnapshotInputs(workspace=Path.cwd(), bundle_manifest=bundle_manifest, repo=context.repo)
        with snapshot_execution.execution(
            owner,
            target_experiment=load_target_experiment(context.descriptor),
            candidate=Path(submission),
            work_root=Path(work_root),
        ):
            yield
        return
    workspace = Path.cwd()
    bundle = yaml.safe_load(Path(bundle_manifest).read_bytes())
    with contextlib.ExitStack() as execution:
        with verified_snapshot_scope(workspace, bundle, repo=context.repo):
            owner = CommonTransformService(workspace=workspace, bundle_manifest=bundle_manifest, repo=context.repo)
            execution.enter_context(
                _original_execution_lifetime(
                    owner,
                    target_experiment=load_target_experiment(context.descriptor),
                    candidate=Path(submission),
                    work_root=Path(work_root),
                )
            )
        # The full final setup read has completed. Actual grading is outside reuse.
        yield


@contextlib.contextmanager
def selfcheck_diagnostics(bundle_manifest, *, context, arguments):
    """Keep historical diagnostics intact; fresh ones copy one bounded compiler."""
    if bundle_manifest is None:
        yield
        return
    from merlin.development.backend.feedback.execution.compile_only import snapshot_submission
    from merlin.development.backend.session.inputs import run_record as run_inputs

    original = arguments.submission
    with tempfile.TemporaryDirectory(prefix="selfcheck-diagnostic-") as temporary:
        source = Path(temporary) / "submission"
        snapshot_submission(Path(original).resolve(), source)
        source.chmod(0o700)  # Declared builds may create an excluded build/ tree.
        manifest = source / "manifest.yaml"
        if manifest.is_file():
            manifest.chmod(0o600)
        run_inputs.strip_build_state(source)
        if manifest.is_file():
            manifest.chmod(0o444)
        arguments.submission = str(source)
        try:
            with selfcheck_execution(
                bundle_manifest, context=context, submission=source, work_root=Path(temporary) / "profile-package-work"
            ):
                yield
        finally:
            arguments.submission = original


def selfcheck_compile_only(a, *, context):
    try:
        import tempfile

        from merlin.development.backend.feedback.execution.compile_only import run
        from merlin.development.targetgen.execution.host.common_transforms import CommonTransformService
        from merlin.target.registry.descriptor_loading import load_target_experiment

        if a.compiler_bundle_manifest is None or a.rtl_facts is None or a.contract is None:
            raise ValueError("compile-only requires the original frozen bundle, facts and contract")
        owner = CommonTransformService(
            workspace=Path.cwd(), bundle_manifest=a.compiler_bundle_manifest, repo=context.repo
        )
        owner._verify()
        from merlin.development.backend.feedback.scalability.resource_gate import selected_budget

        scale_probe = bool(getattr(a, "scale_probe", False))
        ir_budget = selected_budget(owner) if scale_probe else None
        temporary_root = a.compiler_bundle_manifest.parent / "compile-only-work"
        temporary_root.mkdir(mode=0o700, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="probe-", dir=temporary_root) as temporary:
            from merlin.development.backend.feedback.execution.compile_only import snapshot_submission

            copied = snapshot_submission(Path(a.submission).absolute(), Path(temporary) / "compiler")
            result = run(
                submission=copied,
                target_experiment=load_target_experiment(context.descriptor),
                capsules_root=a.capsules_root,
                contract=a.contract,
                facts_path=a.rtl_facts,
                common_service=owner,
                work_root=Path(temporary) / "products",
                timeout_s=min(a.timeout, 600),
                selected=a.capsules,
                scale_probe=scale_probe,
                ir_budget=ir_budget,
                public_artifact_root=Path(getattr(a, "artifact_root", None) or "selfcheck_out/compile-only").absolute(),
            )
        owner._verify()
    except Exception as error:
        result = {
            "schema": "merlin.public_compile_only_feedback.v1",
            "status": "unavailable",
            "reason": type(error).__name__,
            "all_pass": False,
            "numeric": "not_attempted",
            "rtl": "not_attempted",
            "timing_cycles": None,
        }
    txt = json.dumps(result, indent=2, sort_keys=True)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    return 0 if result.get("cases") else 2


@setup_timing.measured("package_policy.public_python_roots")
def _public_python_roots(owner):
    """Derive import roots from pinned public destinations, including split packages."""
    from merlin.development.targetgen.authoring import tool_registry
    from merlin.development.targetgen.sandbox import grants as BW

    sources = list(owner.profile["common_sources"])
    for name in owner.profile["treatment_tools"]:
        if name in {"isa_dialect_generation", "lean_proof_authoring"}:
            sources.extend(
                {"path": path, "sha256": None} for path in tool_registry.spec(name).bundle_paths if path.endswith(".py")
            )
    from merlin.common.storage.bundle_snapshots import guarded_projection

    def derive():
        # Import roots depend only on watched frozen declarations. The mount
        # builder still checks live host locations for every composed policy.
        _, grants = BW._snapshot_grants(owner.workspace, owner.bundle, owner.repo)
        roots = set()
        for row in sources:
            tail = row["path"].removeprefix("merlin/python/")
            matches = [
                (destination, source)
                for declared, destination, source in grants
                if declared == row["path"] or declared.endswith("/" + tail)
            ]
            if len(matches) != 1:
                raise ValueError("original compiler API has no unique pinned public destination")
            destination, source = matches[0]
            if source.is_symlink() or not source.is_file():
                raise ValueError("original compiler API must be an exact frozen source leaf")
            if row["sha256"] is not None and hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("original compiler API bytes differ from its source profile")
            relative = Path(tail)
            if not relative.parts or relative.parts[0] != "merlin" or not str(destination).endswith("/" + tail):
                raise ValueError("original compiler API destination cannot define a Python package root")
            root = destination.parents[len(relative.parts) - 1]
            if ":" in str(root) or "\0" in str(root):
                raise ValueError("original compiler API root cannot form an exact Python path")
            roots.add(root)
        return tuple(sorted(roots))

    return guarded_projection(
        owner.workspace, owner.bundle, owner.repo, ["original.public_python_roots.v1", sources], derive
    )


@setup_timing.measured("package_policy.public_grants")
def _public_package_parts(argv, env_prefix, owner):
    """Compose exact original grants; internal callers must close masks last."""
    from merlin.development.backend.authoring.profiles import resolve, verify_bundle
    from merlin.development.targetgen.sandbox import grants as BW

    author = owner.bundle.get("author_profile")
    if author is not None:
        verify_bundle(resolve(author.get("name")), owner.bundle)
        roots = ()
    else:
        roots = _public_python_roots(owner)
    prefix = [*argv, *BW._bundle_mount_args(owner.workspace, owner.bundle, owner.repo, visibility_prefix=list(argv))]
    root = shlex.quote(":".join(map(str, roots)))
    return prefix, env_prefix + f"export PYTHONPATH={root}:$PYTHONPATH; "


def _original_package_composition(target, inputs, scope):
    """A private raw-parts hook, issued only by the exact live original executor."""
    from merlin.development.backend.feedback.execution.profile_executor import _Executor

    executor = active_original_execution()
    if (
        type(executor) is not _Executor
        or executor.target is not target
        or executor.inputs is not inputs
        or executor.policy_scope is not scope
        or not executor.concurrent_entrypoints_v1()
    ):
        return None
    from merlin.development.targetgen.authoring.profile_admission import (
        _issued_compiler_imports,
        active_compiler_imports,
    )

    owner = active_compiler_imports()
    if owner is None:
        raise ValueError("original package composition lost its import admission")

    def compose(argv, env_prefix, *, target, scope, selected_surfaces):
        from merlin.common.process.cpu_pool import shared_compute_pool
        from merlin.development.targetgen.sandbox.host_surfaces import _final_answer_policy

        expected_surfaces = () if inputs is None else inputs.surfaces
        if (
            active_original_execution() is not executor
            or _issued_compiler_imports() is not owner
            or target is not executor.target
            or scope is not executor.policy_scope
            or tuple(selected_surfaces) != tuple(expected_surfaces)
        ):
            raise ValueError("original package composition changed its owner")
        with shared_compute_pool():
            if not executor.concurrent_entrypoints_v1():
                raise ValueError("original package composition expired")
            argv, env_prefix = _public_package_parts(argv, env_prefix, owner)
            argv, answers = _final_answer_policy(
                argv,
                target,
                owner.workspace,
                owner.bundle,
                repo=owner.repo,
                capsules=None if scope is None else scope.capsules,
                selected_surfaces=selected_surfaces,
            )
            if (
                active_original_execution() is not executor
                or _issued_compiler_imports() is not owner
                or target is not executor.target
                or scope is not executor.policy_scope
            ):
                raise ValueError("original package composition changed its owner")
            if not executor.concurrent_entrypoints_v1():
                raise ValueError("original package composition expired")
            return argv, env_prefix, answers

    return compose


def public_package_policy(policy, *, target, owner, scope=None):
    """Reapply exact original public grants without the original workspace mount.

    ``scope`` (a ``PackagePolicyScope``) shares capsule metadata parsed from
    byte-identical files across one execution scope. Reopen the actual mount
    prefix before omitting a host mask whose input is already hidden.
    """
    from dataclasses import replace

    from merlin.common.process.cpu_pool import shared_compute_pool
    from merlin.development.targetgen.sandbox import grants as BW

    # Alias discovery walks real runtime trees. Schedule this synchronous CPU
    # work with compilation/simulation, and release its slot before any RPC or
    # child wait. The original source and final privacy checks still run fresh;
    # a scope shares only capsule metadata parsed from byte-identical files.
    with shared_compute_pool():
        prefix, env_prefix = _public_package_parts(policy.argv, policy.env_prefix, owner)
        prefix = BW.apply_final_answer_masks(
            prefix,
            target,
            owner.workspace,
            owner.bundle,
            repo=owner.repo,
            capsules=None if scope is None else scope.capsules,
        )
        return replace(policy, argv=tuple(prefix), env_prefix=env_prefix)


def active_public_package_policy(policy, *, target, scope=None):
    """The normal package box consumes only a live original core admission."""
    from merlin.development.targetgen.authoring.profile_admission import active_compiler_imports

    owner = active_compiler_imports()
    return policy if owner is None else public_package_policy(policy, target=target, owner=owner, scope=scope)


def active_profile_service():
    """Recover only the original live scope's service for a normal compiler call."""
    original = active_original_execution()
    if original is not None:
        original.owner._verify()
        return original.owner
    from merlin.development.targetgen.authoring.profile_admission import active_compiler_imports

    admission = active_compiler_imports()
    if admission is None:
        return None
    service = CommonTransformService(
        workspace=admission.workspace, bundle_manifest=admission.bundle_manifest, repo=admission.repo
    )
    if service.profile != admission.profile or service.snapshot_digest != admission.snapshot_sha256:
        raise ValueError("normal compiler call changed its live original profile")
    return service


def active_original_execution():
    """Identify the live original executor without admitting another callback."""
    from merlin.development.backend.feedback.execution.profile_executor import _Executor

    executor = P.active_package_executor()
    if type(executor) is _Executor:
        return executor
    if executor is None:
        return None
    from merlin.development.backend.qualification.components.package_execution import ComponentPackageExecutor

    if type(executor) is ComponentPackageExecutor:
        method = getattr(executor, "original_profile_execution", None)
        original = None if method is None else method()
        if original is not None and type(original) is not _Executor:
            raise ValueError("composed compiler owner did not preserve its exact original execution")
        return original
    return None


@contextlib.contextmanager
def _request_snapshot_admission(service, candidate):
    """All joint evaluator routes share two exact immutable compiler identities."""
    from merlin.development.backend.campaign import control as joint_control
    from merlin.development.backend.feedback.execution import snapshots as snapshot_admission

    if joint_control.selected() is None:
        yield
        return
    raw = os.environ.get("MERLIN_COMPUTE_CPU_SLOTS")
    if not raw:
        raise ValueError("joint snapshot admission requires the original shared compute root")
    root = Path(raw) / "immutable-snapshots"
    key = snapshot_admission.identity(service, candidate)
    from merlin.development.targetgen.grading.scheduling import background_admission_selected

    # Explicit final frozen checks are allowed after editing stops. Ordinary
    # background queue work observes the same trusted stop as command admission.
    stop = joint_control.stop_reason if background_admission_selected() else None
    with snapshot_admission.admitted(root, key, stop_reason=stop):
        if snapshot_admission.identity(service, candidate) != key:
            raise ValueError("compiler snapshot changed while awaiting request admission")
        try:
            yield
        finally:
            if snapshot_admission.identity(service, candidate) != key:
                raise ValueError("compiler snapshot changed during its admitted request")


@contextlib.contextmanager
def original_compiler_execution(service, *, target_experiment, candidate, work_root, sandbox_inputs=None):
    """Batch only setup for an exact owner, never the grading lifetime."""
    from merlin.common.storage.bundle_snapshots import verified_snapshot_scope
    from merlin.development.backend.feedback.execution import snapshot_evaluator as snapshot_execution
    from merlin.development.targetgen.authoring.library_admission import (
        active_compiler_library,
        guarded_compiler_library_verification,
    )

    if snapshot_execution.selected():
        with snapshot_execution.execution(
            service,
            target_experiment=target_experiment,
            candidate=candidate,
            work_root=work_root,
            sandbox_inputs=sandbox_inputs,
        ):
            yield
        return
    if service is None:
        yield
        return
    if type(service) is not CommonTransformService or P.active_package_executor() is not None:
        raise ValueError("original profile execution requires its exact owner and no foreign executor")
    with _request_snapshot_admission(service, candidate), contextlib.ExitStack() as execution:
        # This issued, watched selection lives through all joined workers and
        # executor cleanup. Every library access still checks its exact guard;
        # scope exit independently verifies the reviewed bytes, even on failure.
        execution.enter_context(guarded_compiler_library_verification(active_compiler_library()))
        with verified_snapshot_scope(service.workspace, service.bundle, repo=service.repo):
            execution.enter_context(
                _original_execution_lifetime(
                    service,
                    target_experiment=target_experiment,
                    candidate=candidate,
                    work_root=work_root,
                    sandbox_inputs=sandbox_inputs,
                )
            )
        yield


@contextlib.contextmanager
def _original_execution_lifetime(service, *, target_experiment, candidate, work_root, sandbox_inputs=None):
    """One original host scope for all actual package checks and child mounts."""
    from merlin.development.backend.feedback.execution.profile_executor import _Executor

    if service is None:
        yield
        return
    if type(service) is not CommonTransformService or P.active_package_executor() is not None:
        raise ValueError("original profile execution requires its exact owner and no foreign executor")
    root = Path(work_root)
    if root.resolve() != root or root.is_symlink() or root.is_relative_to(Path(candidate).resolve()):
        raise ValueError("original compiler execution requires an independent canonical private work root")
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scope-", dir=root) as temporary:
        executor = _Executor(service, target_experiment, candidate, Path(temporary), sandbox_inputs=sandbox_inputs)
        with executor.shared_snapshot_entry(), original_compiler_imports(service), P.scoped_package_executor(executor):
            with executor._concurrent_lifetime():
                yield

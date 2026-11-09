"""Fresh count observations through the original one-ELF namespace and guardian."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.cpu_pool import shared_compute_pool
from merlin.perf import _census_process as PROCESS
from merlin.perf.dynamic_census import CensusCleanupUnknown
from merlin.runtime import _native_files as F
from merlin_experiments.phase1 import annotated_spike_evidence as E

from . import _counter_symbol_aliases as ALIASES
from . import campaign as PC
from . import census_launch as BOX
from . import normal_resource_launch as R
from .archive_spike_runtime import ArchiveSpikeRuntime
from .contracts import StageGateError, document_sha256, write_json

_RETAINED = []


class _Stderr:
    def line(self, text):
        pass  # Raw bounded stderr remains in the actual invocation evidence.


#: Bytes of a failed command's stderr tail read back for its stated reason.
_REASON_TAIL_BYTES = 8192


def provider_reason(stage_directory) -> str | None:
    """The last line a failed observation command wrote to stderr (the provider's own refusal).

    A census failure says only that the command did not finish; the reason the provider printed (for
    example a completion it could not attribute) stays in the invocation's raw stderr. Host paths are
    scrubbed; ``None`` when nothing was written or the evidence is unreadable."""
    from merlin.common.path_scrub import scrub_host_paths

    try:
        streams = sorted(Path(stage_directory).glob("invocations/*/stderr.bin"), key=lambda p: p.stat().st_mtime)
        if not streams:
            return None
        with streams[-1].open("rb") as handle:
            handle.seek(max(0, streams[-1].stat().st_size - _REASON_TAIL_BYTES))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    return scrub_host_paths(lines[-1])[:240] if lines else None


def _run(argv, *, prefix, directory, workspace, backing, limits, dependencies, require_done, ownership,
         stdout_protocol="text", extra_inputs=()):
    directory.mkdir(mode=0o700)
    with I.observe(
        directory,
        stage="archive_screen_count" if require_done else "archive_screen_counter_review",
        argv=[*prefix, "--", *argv],
        cwd=workspace,
        env={},
        inputs=(workspace / "program.elf",)
        + ((workspace / "scope.json",) if (workspace / "scope.json").exists() else ()) + tuple(extra_inputs),
        outputs=tuple(Path(v.partition("=")[2]) for v in argv if v.startswith(("--timing-report=", "+signature="))),
        dependencies=dependencies,
    ) as record:
        ownership["closed"] = False
        try:
            with shared_compute_pool(timeout_s=limits.wall_s):
                code = PROCESS.capture(
                    [*prefix, "--", *argv],
                    environment={},
                    cwd=workspace,
                    limits=limits,
                    record=record,
                    census=_Stderr(),
                    writable_roots=(workspace, backing),
                    readonly_files=(workspace / "program.elf", workspace / "scope.json"),
                    require_done=require_done,
                    **({"stdout_protocol": stdout_protocol} if stdout_protocol != "text" else {}),
                )
        except BaseException:
            if getattr(record, "census_started", False) and not getattr(record, "census_cleanup_complete", False):
                raise CensusCleanupUnknown("archive count child cleanup UNKNOWN; retain original job/scratch owner")
            raise
        finally:
            ownership["closed"] = not getattr(record, "census_started", False) or getattr(
                record, "census_cleanup_complete", False
            )
        record.complete_streams(code)
        actual = I.verify(record.path)
    return record.path, actual, (record.directory / "stdout.bin").read_bytes()


def observe(elf, *, runtime, policy, directory, input_sha256, remaining_s, numerical_pass, owned_scope=None, output_export=None,
            native_relocation=None):
    """Observe one already graded original ELF, never an author-selected command.

    Complete original numerical admission precedes this call. Plain and annotated
    consoles must agree byte for byte. All count evidence stays private; the
    caller projects only descriptive counts and empirical intervals.
    """
    from merlin_timing import observations as O

    if numerical_pass is not True:
        return {"status": "UNKNOWN", "reason": "original complete functional screen did not pass"}
    root, scope_owner, ownership = None, None, {"closed": True}
    started = time.monotonic()
    stage = "runtime_validation"
    try:
        if type(runtime) is not ArchiveSpikeRuntime:
            raise StageGateError("archive counts require original reviewed runtime ownership")
        runtime.verify()
        if output_export is not None:
            from .integer_signature_export import IntegerSignatureExport
            if type(output_export) is not IntegerSignatureExport or owned_scope is not None:
                raise StageGateError("integer signature export requires the ordinary unchanged capsule scope")
            output_export.materialize(elf)
        if type(policy) is not PC.PackageSandboxPolicy or policy.coverage_gap:
            raise StageGateError("archive counts require the original complete package namespace")
        stage = "policy_validation"
        PC._verify_frozen_policy(policy)
        limits = runtime.observation_limits(remaining_s)
        limits = replace(limits, max_line_bytes=1 << 20, max_output_bytes=64 << 20)
        limits.verify()
        elf, directory = F.canonical(elf), F.canonical(directory, exists=False)
        if ((owned_scope is None and not elf.is_relative_to(policy.workspace))
                or not directory.is_relative_to(policy.workspace)):
            raise StageGateError("archive count program/output escaped the original numerical job")
        stage = "producer_validation"
        producer = BOX._producer
        if native_relocation is not None:
            from .archive_elf_source import NativeElfRelocation
            if type(native_relocation) is not NativeElfRelocation or owned_scope is not None:
                raise StageGateError("native ELF relocation requires its original kernel producer")
            producer = native_relocation.producer
        if owned_scope is not None:
            from .owned_model_observation import OwnedModelScope, scope_digest
            if type(owned_scope) is not OwnedModelScope:
                raise StageGateError("owned observation requires original synchronous producer contract")
            scope_digest(json.loads(runtime.timing_json))
            producer = owned_scope.producer
        elif "owned_model_scope" in json.loads(runtime.timing_json):
            raise StageGateError("owned runtime requires a verified synchronous producer")
        before = producer(elf, limits)
        stage = "namespace_setup"
        directory.mkdir(mode=0o700)
        root = Path(tempfile.mkdtemp(prefix="archive-count-", dir=directory))
        workspace = root / "workspace"
        workspace.mkdir(mode=0o700)
        copied = workspace / "program.elf"
        F.copy_member(elf, copied, budget=[1, limits.max_elf_bytes])
        prefix = list(BOX._narrow_prefix(policy, workspace))
        # The guardian executes an exact file with execve, not PATH search.
        launcher = shutil.which(prefix[0])
        if launcher is None:
            raise StageGateError("original namespace launcher is unavailable")
        prefix[0] = str(F.canonical(Path(launcher)))
        for path in runtime.execution_files:
            prefix.extend(["--ro-bind", str(path), str(path)])
        # Preserve original runtime lookup environment; add only explicitly
        # reviewed dependency directories rather than granting their contents.
        lookup = ":".join(dict.fromkeys(str(p.parent) for p in (*runtime.libraries, *runtime.dependencies)))
        prefix.extend(["--setenv", "LD_LIBRARY_PATH", lookup, "--ro-bind", str(copied), str(copied)])
        boundary = R.resources.active_resource_boundary()
        if boundary is None:
            prefix, backing = BOX._private_backing(tuple(prefix), root)
        else:
            scope_owner = R.bounded_normal_scratch(tuple(prefix), owner=root, boundary=boundary)
            prefix = tuple(scope_owner.__enter__())
            backing = root
        dependencies = tuple(Path(p) for p, _d in runtime.files) + tuple(Path(module.__file__).resolve()
            for module in (ALIASES, *ALIASES.source_modules())) + (Path(__file__).resolve(),)
        if owned_scope is not None:
            from . import owned_model_observation
            dependencies += (Path(owned_model_observation.__file__).resolve(),)
        if native_relocation is not None:
            from . import archive_elf_source
            dependencies += (Path(archive_elf_source.__file__).resolve(),)
        review_argv = [str(runtime.disassembler), "-d", "--no-show-raw-insn", str(copied)]
        stage = "scope_disassembly"
        review_process, _actual, listing = _run(
            review_argv,
            prefix=prefix,
            directory=directory / "review_process",
            workspace=workspace,
            backing=backing,
            limits=limits,
            dependencies=dependencies,
            require_done=False,
            ownership=ownership,
        )
        stage = "scope_binding"
        selection, model = json.loads(runtime.timing_json), json.loads(runtime.model_json)
        rows = E.parse_disassembly(listing.decode())
        if owned_scope is None:
            window, review_schema = ALIASES.window(
                rows, elf_bytes=copied.read_bytes(), counter_asm=selection["counter_asm"],
                entry_symbol=runtime.entry_symbol, completion_symbol=runtime.completion_symbol)
        else:
            window, review_schema = owned_scope.window(rows, counter_asm=selection["counter_asm"])
        review = {
            "schema": review_schema,
            "elf_sha256": before["sha256"],
            "disassembler": F.pin(runtime.disassembler, max_bytes=256 << 20),
            "disassembly_sha256": F.digest(review_process.parent / "stdout.bin"),
            "counter_asm": selection["counter_asm"],
            "entry_symbol": runtime.entry_symbol,
            "completion_symbol": runtime.completion_symbol,
            **({"native_relocation": native_relocation.record()} if native_relocation is not None else {}),
            **window,
        }
        binding = {
            "schema": "merlin.spike_scope_binding.v1",
            "elf_sha256": before["sha256"],
            "input_sha256": input_sha256,
            "scope_sha256": model["domain"]["scope_sha256"],
            "begin_pc": window["begin_pc"],
            "end_pc": window["end_pc"],
            "occurrence": window["occurrence"],
            "evidence_sha256": document_sha256(review),
        }
        write_json(directory / "counter_review.json", review)
        write_json(workspace / "scope.json", binding)
        prefix = (*prefix, "--ro-bind", str(workspace / "scope.json"), str(workspace / "scope.json"))
        export_record = None
        if output_export is not None:
            _, _, symbols = output_export.materialize(copied)
            supplement = workspace / "output-symbols.elf"
            supplement.write_bytes(symbols)
            prefix = (*prefix, "--ro-bind", str(supplement), str(supplement))
            export_record = dict(plan=output_export.record(), supplement=F.pin(supplement, max_bytes=64 << 10))
        report_path = workspace / "report.json"
        commands = runtime.commands(copied, report_path, workspace / "scope.json")
        consoles, processes = [], []
        for index, argv in enumerate(commands):
            remaining = remaining_s - (time.monotonic() - started)
            if remaining < 1:
                raise StageGateError("archive count observation exhausted original parent deadline")
            current = replace(limits, wall_s=min(limits.wall_s, remaining))
            stage = "functional_execution" if index == 0 else "annotated_execution"
            if output_export is not None:
                signature = workspace / f"output-{index}.signature"
                argv = output_export.commands(argv, elf=copied, supplement=supplement, signature=signature)
            invocation, actual, console = _run(
                argv,
                prefix=prefix,
                directory=directory / ("functional" if index == 0 else "annotated"),
                workspace=workspace,
                backing=backing,
                limits=current,
                dependencies=dependencies,
                require_done=True,
                ownership=ownership,
                **({"stdout_protocol": "out_bin_v1"} if owned_scope is not None else {}),
            )
            processes.append(
                {"path": str(invocation), "sha256": F.digest(invocation), "stdout_sha256": actual["stdout"]["sha256"]}
            )
            if output_export is not None:
                output_export.values(signature)
                processes[-1]["full_output_signature"] = F.pin(signature, max_bytes=256 << 10)
            if owned_scope is not None:
                owned_scope.validate_console(console)
            consoles.append(console)
        stage = "result_verification"
        if consoles[0] != consoles[1]:
            raise StageGateError("annotation changed the original functional console")
        pin = F.pin(report_path, max_bytes=16 << 20)
        report = json.loads(report_path.read_bytes())
        if (
            not O.feature_complete(report)
            or report.get("scope_binding") != binding
            or report.get("model_sha256") != F.digest(runtime.model_path)
            or report.get("loaded_library_sha256") != model["libraries"]
        ):
            raise StageGateError("archive observation lacks complete current counts in its exact selected scope")
        runtime.verify()
        PC._verify_frozen_policy(policy)
        if producer(elf, limits) != before or F.digest(copied) != before["sha256"]:
            raise StageGateError("original count program changed during observation")
        retained = directory / "report.json"
        shutil.copyfile(report_path, retained)
        if F.digest(retained) != pin["sha256"]:
            raise StageGateError("retained archive count report changed during publication")
        return {
            "schema": "merlin.phase2.archive_count_observation.v1",
            "status": "observed",
            "elf_sha256": before["sha256"],
            "input_sha256": input_sha256,
            "scope_binding": binding,
            "runtime_sha256": runtime.sha256,
            "report": {"path": str(retained), "sha256": pin["sha256"]},
            "processes": processes,
            **({"native_relocation": native_relocation.record()} if native_relocation is not None else {}),
            **({"output_export": export_record} if output_export is not None else {}),
            "functional_identical": True,
            "complete_counts": True,
            "feature_counts": report["feature_counts"],
            "wall_seconds": time.monotonic() - started,
            "native_event_timing_qualified": False,
            "authority": "empirical screening input only",
        }
    except (OSError, ValueError, RuntimeError, KeyError, AttributeError, TypeError) as error:
        stated = None
        if isinstance(directory, Path) and stage in ("functional_execution", "annotated_execution"):
            stated = provider_reason(directory / ("functional" if stage == "functional_execution" else "annotated"))
        if isinstance(directory, Path) and directory.exists():
            write_json(
                directory / "failure_private.json",
                {"category": type(error).__name__, "detail": str(error)[:1000], "provider_reason": stated},
            )
        return {
            "status": "UNKNOWN",
            "reason": "original boxed archive count observation unavailable: " + type(error).__name__,
            **({"provider_reason": stated} if stated else {}),
            "failure_stage": stage,
            "failure_category": type(error).__name__,
            "deadline_exhausted": str(error) in {
                "census wall deadline expired",
                "archive count observation exhausted original parent deadline",
            },
            "wall_seconds": max(0.0, time.monotonic() - started),
        }
    finally:
        if root is not None and not ownership["closed"]:
            _RETAINED.append((root, scope_owner))
        if root is not None and ownership["closed"]:
            if scope_owner is not None:
                scope_owner.__exit__(None, None, None)
            # Keep the actually executed ELF beside process receipts. Other
            # fresh scratch is regenerable after verified child cleanup.
            for path in root.iterdir():
                if path.name != "workspace":
                    shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
            for path in (root / "workspace").iterdir():
                if path.name not in {"program.elf", "scope.json", "report.json"}:
                    shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()

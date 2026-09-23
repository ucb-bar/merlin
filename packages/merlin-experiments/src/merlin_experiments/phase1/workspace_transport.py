"""Candidate workspace assembly and fail-closed visibility checks.

Copy mode provides an answer-filtered convenience tree, not OS isolation. Frozen
bwrap assembly retains the existing snapshot authority and live destination names;
only the sandbox binds those names to the verified run-owned bytes.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.target_experiment import load_target_experiment
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.providers.execution import sandbox_command


def _is_answer_file(path: Path) -> bool:
    name = path.name
    return (
        name.startswith("golden.")
        or ".golden." in name
        or name.startswith("expected_command_buffer")
        or name == "expected_instruction_coverage.yaml"
        or name.endswith(".safetensors")
        or name.endswith(".safetensors.manifest.json")
    )


def _destination(ws: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("workspace destination must be a nonempty relative path")
    destination = ws / relative
    if not destination.parent.resolve().is_relative_to(ws.resolve()):
        raise ValueError("workspace destination escapes through a symlink")
    return destination


def _link_filtered(src: Path, dst: Path, *, link_target: Path | None = None) -> None:
    target = src if link_target is None else link_target
    if src.name == "hidden" or _is_answer_file(src):
        return
    if src.is_dir():
        if not any(p.name == "hidden" or _is_answer_file(p) for p in src.rglob("*")):
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(target)
            return
        dst.mkdir(parents=True, exist_ok=True)
        for child in sorted(src.iterdir()):
            _link_filtered(child, dst / child.name, link_target=target / child.name)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(target)


def assemble_workspace(
    bundle: dict, ws: Path, *, context: InvocationContext, _policy_test_live_inputs: bool = False
) -> list[str]:
    """Freeze grants before exposing their established friendly workspace names."""
    if not _policy_test_live_inputs:
        BW.materialize_bundle_inputs(
            ws, bundle, repo=context.repo, descriptor=load_target_experiment(context.descriptor)
        )
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "submission").mkdir(exist_ok=True)
    skipped = []
    for entry in bundle.get("allowed", []):
        src = BW.resolve_grant(entry["path"], context.repo)
        frozen = src if _policy_test_live_inputs else BW.snapshot_input_paths(ws, bundle, [src], repo=context.repo)[0]
        if not frozen.exists():
            skipped.append(entry["path"])
            continue
        dst = _destination(ws, entry.get("as") or Path(entry["path"]).name)
        if dst.exists() or dst.is_symlink():
            dst = _destination(ws, entry["path"].replace("/", "_").rstrip("_"))
        if not dst.exists() and not dst.is_symlink():
            _link_filtered(frozen, dst, link_target=src)
    if skipped:
        print(f"[workspace] {len(skipped)} granted path(s) could not be placed: {sorted(skipped)}", flush=True)
    return [Path(d["path"]).name for d in bundle.get("denied", [])]


def assert_isolation(ws: Path, bundle: dict, *, context: InvocationContext) -> list[str]:
    """Check denied direct workspace aliases; the sandbox owns full mount isolation."""
    violations = []
    for denied in bundle.get("denied", []):
        target = BW.resolve_grant(denied["path"], context.repo).resolve()
        for entry in ws.iterdir():
            if entry.resolve() == target:
                violations.append(f"denied path reachable: {entry.name} -> {denied['path']}")
    return violations


def assemble_copy_workspace(bundle: dict, ws: Path, *, context: InvocationContext) -> dict:
    """Copy contract/tool inputs minus answers and denies; retain answer-free tool links.

    This is the explicitly weaker unsandboxed diagnostic mode: absolute host reads
    remain possible and transcript/integrity checks remain necessary.
    """
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "submission").mkdir(exist_ok=True)
    report = dict(
        copied=[],
        symlinked=[],
        copied_minus=[],
        answer_files_dropped=0,
        tool_subpaths_excluded=[],
        unresolvable_grants=[],
    )
    denied = [BW.resolve_grant(d["path"], context.repo).resolve() for d in bundle.get("denied", [])]
    for entry in bundle.get("allowed", []):
        rel = entry["path"].rstrip("/")
        src = BW.resolve_grant(entry["path"], context.repo)
        if not src.exists():
            report["unresolvable_grants"].append(entry["path"])
            continue
        dst = _destination(ws, entry.get("as") or rel.lstrip("/"))
        if src.resolve() in denied or _is_answer_file(src) or src.name == "hidden":
            report["tool_subpaths_excluded"].append(rel)
            continue
        if dst.exists() or dst.is_symlink():
            continue
        # Preserve explicit public contracts below a blanket target-package deny.
        # Only denies within this grant are subtracted, as in the native transport;
        # this is not cleanroom's stricter surface-exemption policy.
        excluded = [path for path in denied if not src.resolve().is_relative_to(path)]
        # Contract descendants are never aliases back into the answer-bearing live tree.
        is_contract = rel == "merlin/contract" or rel.startswith("merlin/contract/")
        must_copy = (
            is_contract
            or rel.startswith("merlin/")
            or any(
                p.name == "hidden"
                or _is_answer_file(p)
                or any(p.resolve() == denied or denied in p.resolve().parents for denied in excluded)
                for p in src.rglob("*")
            )
            if src.is_dir()
            else is_contract
        )
        if must_copy and src.is_dir():

            def ignore(directory, names):
                skipped = []
                for name in names:
                    path = Path(directory) / name
                    resolved = path.resolve()
                    if (
                        name in {"hidden", "__pycache__"}
                        or _is_answer_file(path)
                        or any(resolved == d or d in resolved.parents for d in excluded)
                    ):
                        skipped.append(name)
                        if _is_answer_file(path):
                            report["answer_files_dropped"] += 1
                        else:
                            report["tool_subpaths_excluded"].append(str(path))
                return skipped

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, ignore=ignore, symlinks=False)
            report["copied" if is_contract else "copied_minus"].append(rel)
        elif is_contract:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            report["copied"].append(rel)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)
            report["symlinked"].append(str(dst.relative_to(ws)))
    return report


def assemble(bundle: dict, ws: Path, sandbox: str, *, context: InvocationContext):
    from merlin_experiments.phase1.session import AssemblyEvidence

    if sandbox == "bwrap":
        names = assemble_workspace(bundle, ws, context=context)
        return AssemblyEvidence(names, assert_isolation(ws, bundle, context=context), None)
    if sandbox != "none":
        raise ValueError(f"unsupported workspace transport: {sandbox}")
    report = assemble_copy_workspace(bundle, ws, context=context)
    return AssemblyEvidence([Path(d["path"]).name for d in bundle.get("denied", [])], [], report)


def probe(ws: Path, bundle: dict, sandbox: str, *, context: InvocationContext) -> dict:
    """Require an executed, complete probe before reporting the agent view clean."""
    descriptor = load_target_experiment(context.descriptor)
    corpus = descriptor.capsule_corpus
    if corpus is None:
        raise RuntimeError("workspace mask probe requires a descriptor-selected corpus")
    roots = {Path(corpus), context.repo / "merlin/contract/capsules"}
    roots.update(descriptor.graded_roots())
    if sandbox == "bwrap":
        # Host-side enumeration uses verified frozen bytes. Shell probes inspect both
        # live destination names and snapshot aliases; none grants candidate access.
        frozen = BW.snapshot_input_paths(ws, bundle, [Path(corpus)], repo=context.repo)
        roots.update(frozen)
        if not any(p.is_file() for root in frozen for p in root.rglob("capsule.interface.mlir")):
            raise RuntimeError("workspace mask probe has no frozen public specification")
        patterns = [
            "golden.*",
            "*.golden.*",
            "expected_command_buffer*",
            "expected_instruction_coverage.yaml",
            "*.safetensors",
            "*.safetensors.manifest.json",
        ]
        expressions = " -o ".join("-name " + shlex.quote(pattern) for pattern in patterns)
        # find -exec preserves spaces/tabs in filenames; no command substitution word splitting.
        emit = shlex.quote('for f do if test -s "$f"; then printf "LEAK:%s\\n" "$f"; fi; done')
        commands = []
        for root in sorted(roots):
            quoted = shlex.quote(str(root))
            commands.append(
                f"if test -e {quoted}; then find {quoted} \\( {expressions} -o "
                f"\\( -path '*/hidden/*' -name capsule.yaml \\) \\) "
                f"-exec sh -c {emit} sh {{}} + || exit 1; fi"
            )
        with tempfile.NamedTemporaryFile(prefix=".mask-control-", dir=ws) as control:
            control.write(b"probe control\n")
            control.flush()
            script = "; ".join([f"test -s {shlex.quote(control.name)} || exit 1", *commands, "printf 'DONE\\n'"])
            result = subprocess.run(
                ["bash", "-c", sandbox_command(script, ws, bundle, context=context)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        lines = result.stdout.splitlines()
        leaked = [line[5:] for line in lines if line.startswith("LEAK:")]
        completed = result.returncode == 0 and lines and lines[-1] == "DONE" and lines.count("DONE") == 1
        completed = completed and all(line == "DONE" or line.startswith("LEAK:") for line in lines)
        status = "LEAK" if leaked else "OK" if completed else "UNPROVEN"
        failure = None
        if status == "UNPROVEN":
            stderr = str(getattr(result, "stderr", "") or "").strip()[-300:] or "(empty)"
            failure = (
                f"masking probe did not complete (rc={result.returncode}); "
                f"the agent view was not established. stderr: {stderr}"
            )
        return {
            "pilot_golden_visible_to_agent": status,
            "n_answer_files_masked": sum(_is_answer_file(p) for root in frozen for p in root.rglob("*")),
            "leaked_answer_files": leaked[:10],
            "probe_returncode": result.returncode,
            "probe_failure": failure,
        }
    if sandbox != "none":
        raise ValueError(f"unsupported workspace transport: {sandbox}")
    goldens, weights, hidden, bad_links = [], [], [], []
    for directory, dirs, files in os.walk(ws):
        for name in [*dirs, *files]:
            path = Path(directory) / name
            if _is_answer_file(path):
                (weights if ".safetensors" in name else goldens).append(str(path))
            if name == "hidden":
                hidden.append(str(path))
            if path.is_symlink() and any(
                path.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(path.resolve())
                for root in roots
            ):
                bad_links.append(str(path))
    specs = list(Path(corpus).rglob("capsule.interface.mlir"))
    if not specs:
        raise RuntimeError("workspace mask probe has no descriptor-selected public specification")
    leak = bool(goldens or weights or hidden or bad_links)
    return {
        "pilot_golden_visible_to_agent": "LEAK" if leak else "OK",
        "goldens_in_workspace": goldens[:10],
        "weights_in_workspace": weights[:10],
        "hidden_in_workspace": hidden[:10],
        "symlinks_into_capsules": bad_links[:10],
        "spec_present": True,
    }

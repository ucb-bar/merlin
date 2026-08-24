"""Shared, target-agnostic capsule I/O for the capsule runners.

`capsule_runner` (gemmini: spike/verilator oracle) and `muon_capsule_runner` (cyclotron oracle) had
byte-identical copies of these helpers. They are the single source now; both runners import them (the
oracle-specific `run_capsule`/`run_suite` stay per-runner). Kept in `targetgen` (library), not in the
experiment harness, since the library runners are the consumers.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from aet.core.run_paths import RunPaths
from aet.core.run_spec import RunSpec

from .contract import schemas


def _flat(nested) -> list:
    out: list = []
    if nested and isinstance(nested[0], list):
        for r in nested:
            out.extend(r)
    else:
        out.extend(nested)
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


def discover_capsules(root, *, labels: set[str] | None = None,
                      contract: str | Path | None = None) -> list[dict]:
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


def make_run_paths(runs_root: str | Path, run_id: str, *, suite: str, target: str,
                   dtype: str, benchmark: str) -> RunPaths:
    """Build the per-run RunPaths (via RunSpec) and create its directory scaffold."""
    spec = RunSpec(project="merlin", suite=suite, method=run_id, seed=0, run_id=run_id,
                   project_root=Path(runs_root), tracking_mode="local", target=target,
                   dtype=dtype, benchmark=benchmark)
    paths = RunPaths.from_spec(spec, run_id)
    for dd in (paths.run_path, paths.logs, paths.artifacts_dir, paths.generated, paths.contracts):
        dd.mkdir(parents=True, exist_ok=True)
    return paths


def run_entrypoints(pkg, package_dir: str | Path, capsule: dict, paths, *,
                    contract: str | Path | None, timeout: int, fourth_output_name: str):
    """Shared ABI front half: build the package (if needed) and run the 4 contract entrypoints
    (parse -> lower_interface_to_target -> emit_command_buffer -> lower_target_to_llvm), writing the
    standard artifacts and validating the command buffer. Returns ``(pkg, cb, fourth_text)`` where
    ``fourth_text`` is the lower_target_to_llvm stdout (written to ``fourth_output_name`` — the target
    dialect chooses LLVM-dialect MLIR vs a SIMT kernel). Raises CertFailure on any plane failure.
    The oracle tiers (L2+) are the caller's, since they diverge per target.
    """
    from .oot_runner import (CertFailure, build_package, integrity_scan, load_package,
                             run_entrypoint)

    if pkg is None:
        pkg = load_package(package_dir, contract=contract)
        integrity_scan(pkg)
        build_package(pkg)
    if not pkg.tool.exists():
        raise CertFailure("build", _cat("ELABORATION_ERROR"), f"tool missing: {pkg.tool}")

    iface_rel = capsule.get("interface_mlir", "capsule.interface.mlir")
    iface_path = Path(capsule["__dir__"]) / iface_rel if "__dir__" in capsule else Path(iface_rel)
    if not iface_path.is_file():
        raise CertFailure("schema", _cat("STRUCTURAL_INVARIANT_VIOLATION"),
                          f"capsule interface MLIR not found: {iface_path}")
    inp = paths.generated / "input.interface.mlir"
    inp.write_text(iface_path.read_text(encoding="utf-8"), encoding="utf-8")

    p = run_entrypoint(pkg, "parse", inp, timeout=timeout)
    if p.returncode != 0:
        raise CertFailure("parse", _cat("TOOL_CRASH"), f"parse rc={p.returncode}: {_stderr_excerpt(p.stderr)}")

    p = run_entrypoint(pkg, "lower_interface_to_target", inp, timeout=timeout)
    if p.returncode != 0 or not p.stdout.strip():
        raise CertFailure("interface_to_target", _cat("ELABORATION_ERROR"),
                          f"lower_interface_to_target rc={p.returncode}: {_stderr_excerpt(p.stderr)}")
    (paths.generated / "lowered.target.mlir").write_text(p.stdout, encoding="utf-8")

    cb_path = paths.generated / "command_buffer.json"
    p = run_entrypoint(pkg, "emit_command_buffer", inp, cb_path, timeout=timeout)
    if p.returncode != 0:
        raise CertFailure("target_to_command_buffer", _cat("STRUCTURAL_INVARIANT_VIOLATION"),
                          f"emit_command_buffer rc={p.returncode}: {_stderr_excerpt(p.stderr)}")
    if not cb_path.exists():
        # Exit 0 and no output file. Reporting this as "rc=0: <empty stderr>" tells the agent
        # nothing -- worse, rc=0 reads as success, so the message contradicts itself. One package
        # spent twelve rounds and 215 self-checks against exactly that blank string. The cause is
        # nearly always the manifest argv template: it omits {output_json}, so the tool prints the
        # buffer to stdout and the path the runner reads is never written. Say so, and name it.
        _argv = " ".join(pkg.manifest.get("commands", {})
                         .get("emit_command_buffer", {}).get("argv", []) or ["<undeclared>"])
        _hint = ("your manifest argv for emit_command_buffer does not reference {output_json}, so "
                 "nothing is written to the path the runner reads"
                 if "{output_json}" not in _argv else
                 "the argv does reference {output_json}, so the tool exited before writing it")
        raise CertFailure("target_to_command_buffer", _cat("STRUCTURAL_INVARIANT_VIOLATION"),
                          f"emit_command_buffer exited 0 but wrote no file at the output path it "
                          f"was given ({cb_path.name}). Declared argv: {_argv} -- {_hint}. The "
                          f"runner reads the FILE, never stdout. stderr: {_stderr_excerpt(p.stderr, 200)!r}")
    try:
        cb = json.loads(cb_path.read_text(encoding="utf-8"))
        schemas.validate_command_buffer(cb, contract=contract)
    except (json.JSONDecodeError, schemas.ContractViolation) as e:
        raise CertFailure("command_buffer_schema", _cat("PROTOCOL_VIOLATION"),
                          f"command_buffer.json invalid: {e}") from e

    # the 4th entrypoint: emit the target's codegen artifact (RoCC LLVM / SIMT kernel / ...). The
    # resolver aliases the legacy name lower_target_to_llvm, so packages using either spelling work.
    p = run_entrypoint(pkg, "emit_target_artifact", inp, timeout=timeout)
    if p.returncode != 0 or not p.stdout.strip():
        raise CertFailure("emit_target_artifact", _cat("ELABORATION_ERROR"),
                          f"emit_target_artifact rc={p.returncode}: {_stderr_excerpt(p.stderr)}")
    (paths.generated / fourth_output_name).write_text(p.stdout, encoding="utf-8")
    return pkg, cb, p.stdout

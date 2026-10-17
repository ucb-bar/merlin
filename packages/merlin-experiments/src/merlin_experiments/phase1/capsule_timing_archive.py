"""Audit and archive host-only Phase 1 capsule simulator evidence."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import tarfile
import tempfile
from pathlib import Path


class TimingArchiveError(ValueError):
    """The selected run has no usable gSIM capsule timing evidence."""


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _kernel_scopes(text):
    matches = []
    for tail in text.split("/* merlin.kernel_call_timing.v1 ")[1:]:
        comment, closed, _ = tail.partition(" */")
        fields = comment.split()
        if (closed and len(fields) == 3 and fields[0].startswith("selection_sha256=")
                and fields[1] == "boundary=entry-through-completion" and fields[2].startswith("record_b64=")):
            digest = fields[0].partition("=")[2]
            if _sha256(digest):
                matches.append((digest, fields[2].partition("=")[2]))
    return matches


def _cycle_value(line):
    fields = line.strip().split()
    if (len(fields) == 3 and fields[:2] == ["METRIC", "cycles"] and fields[2]
            and fields[2][0] != "0" and all(c in "0123456789" for c in fields[2])):
        return int(fields[2])
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compiler_candidate(document, generated, artifacts):
    """Accept original child custody or the ordinary producer's stable snapshot."""
    source = document.get("compiler_source")
    if source is None:
        return (document.get("original_compiler_child") or {}).get("candidate_sha256")
    if (source.get("schema") != "merlin.capsule_compiler_source.v1"
            or source.get("status") != "stable"):
        return None
    candidate = source.get("candidate_sha256")
    if (not isinstance(candidate, str) or not _sha256(candidate)
            or source.get("snapshot") != "compiler-source/" + candidate):
        return None
    snapshot = generated / source["snapshot"]
    from merlin.common.tree_hash import hash_tree
    from merlin_experiments.phase2.retained_revisions import source_inventory

    try:
        source_inventory(snapshot, max_bytes=32 << 20, max_files=8192, exclude_generated=True)
        if hash_tree(snapshot)["sha256"] != candidate:
            return None
    except (OSError, ValueError, RuntimeError):
        return None
    for name, key in (("package_kernel.elf", "elf"), ("harness.c", "harness"),
                      ("input.interface.mlir", "input_interface"), ("command_buffer.json", "command_buffer")):
        actual = artifacts.get(key)
        if actual is None or source.get("artifacts", {}).get(name) != actual["sha256"]:
            return None
    return candidate


def audit(run_dir: Path) -> dict:
    """Count actual capsule results; a pass with no positive cycle marker is unmeasured."""
    root = run_dir.resolve()
    paths = sorted(root.glob("_qa_work/**/runs/*/*/capsule_result.json"))
    rows = []

    def artifact(path: Path) -> dict | None:
        if path.is_symlink():
            raise TimingArchiveError(f"capsule artifact is a symlink: {path}")
        if not path.is_file():
            return None
        return {
            "path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }

    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise TimingArchiveError(f"capsule result is not a regular file: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise TimingArchiveError(f"invalid capsule result: {path}")
        tier = (document.get("tiers") or {}).get("L3") or {}
        if not isinstance(tier, dict):
            raise TimingArchiveError(f"invalid L3 tier: {path}")
        cycles = tier.get("cycles")
        measured = type(cycles) is int and cycles > 0
        command_buffer = (document.get("command_buffer_artifact") or {}).get("command_buffer") or {}
        tensors = command_buffer.get("tensors") or {}
        generated = path.parent / "generated"
        artifacts = {
            name: artifact(location)
            for name, location in {
                "elf": generated / "package_kernel.elf",
                "harness": generated / "harness.c",
                "input_interface": generated / "input.interface.mlir",
                "command_buffer": generated / "command_buffer.json",
                "static_instruction_trace": generated / "instruction_trace.json",
                "numeric_report": generated / "numeric_report.yaml",
                "rtl_console": path.parent / "artifacts" / "rtl_console.log",
                "functional_console": path.parent / "artifacts" / "functional_console.log",
            }.items()
        }
        elf_ref, harness_ref = artifacts["elf"], artifacts["harness"]
        harness = generated / "harness.c"
        scope_matches = _kernel_scopes(harness.read_text(encoding="utf-8")) if harness_ref else []
        selection_sha256 = None
        timing_selection = None
        if len(scope_matches) == 1:
            selected_sha, encoded = scope_matches[0]
            try:
                selected_record = json.loads(base64.b64decode(encoded, validate=True))
            except (ValueError, UnicodeDecodeError):
                selected_record = None
            if (
                type(selected_record) is dict
                and selected_record.get("schema") == "merlin.kernel_call_timing.v1"
                and selected_record.get("selection_sha256") == selected_sha
                and selected_record.get("boundary")
                == "immediately before kernel entry through explicitly selected completion"
                and type(selected_record.get("counter_semantics")) is dict
                and selected_record["counter_semantics"].get("unit") == "cycles"
                and selected_record["counter_semantics"].get("width_bits") == 64
                and selected_record["counter_semantics"].get("wrap") == "modulo"
                and selected_record["counter_semantics"].get("source_pins")
            ):
                selection_sha256 = selected_sha
                timing_selection = selected_record
        rtl_console = path.parent / "artifacts" / "rtl_console.log"
        console_cycles = (
            [
                value
                for line in rtl_console.read_text(encoding="utf-8").splitlines()
                if (value := _cycle_value(line)) is not None
            ]
            if artifacts["rtl_console"]
            else []
        )
        rows.append(
            {
                "capsule": document.get("capsule"),
                "result": path.relative_to(root).as_posix(),
                "shape": {
                    name: {"dtype": spec.get("dtype"), "shape": spec.get("shape")}
                    for name, spec in sorted(tensors.items())
                },
                "command_buffer_sha256": (document.get("command_buffer_artifact") or {}).get("artifact_sha256"),
                "candidate_sha256": _compiler_candidate(document, generated, artifacts),
                "measured_now": not bool(tier.get("carried")),
                "carried": tier.get("carried"),
                "artifacts": artifacts,
                "elf": elf_ref["path"] if elf_ref else None,
                "elf_sha256": elf_ref["sha256"] if elf_ref else None,
                "harness_sha256": harness_ref["sha256"] if harness_ref else None,
                "timing_selection_sha256": selection_sha256,
                "timing_selection": timing_selection,
                "cycle_scope": "entry-through-completion" if selection_sha256 else None,
                "status": tier.get("status"),
                "result_status": document.get("status"),
                "engine": tier.get("engine"),
                "derived_from_rtl": tier.get("derived_from_rtl"),
                "cycle_accurate": tier.get("cycle_accurate"),
                "cycles": cycles,
                "cycles_measured": measured,
                "console_cycle_verified": len(console_cycles) == 1 and console_cycles[0] == cycles,
                "timing": tier.get("timing"),
                "sim_provenance": tier.get("sim_provenance"),
            }
        )
    gsim_pass = [r for r in rows if r["engine"] == "gsim" and r["status"] == "pass" and r["measured_now"]]
    complete = [
        r
        for r in gsim_pass
        if r["cycles_measured"]
        and all(
            isinstance(r.get(name), str) and _sha256(r[name])
            for name in ("elf_sha256", "harness_sha256", "candidate_sha256", "command_buffer_sha256")
        )
        and r["timing_selection_sha256"]
        and r["console_cycle_verified"]
        and all(
            r["artifacts"][name]
            for name in ("harness", "input_interface", "command_buffer", "static_instruction_trace", "numeric_report")
        )
        and isinstance((r["sim_provenance"] or {}).get("sha256"), str)
        and _sha256(r["sim_provenance"]["sha256"])
        and r["result_status"] == "pass"
        and r["derived_from_rtl"] is True
        and r["cycle_accurate"] is True
    ]
    return {
        "schema": "merlin.phase1.capsule_timing_archive.v2",
        "run_dir": str(root),
        "capsule_records": len(rows),
        "gsim_pass_records": len(gsim_pass),
        "gsim_reused_records": sum(r["engine"] == "gsim" and r["status"] == "pass" and not r["measured_now"]
                                   for r in rows),
        "gsim_pass_with_cycles": sum(r["cycles_measured"] for r in gsim_pass),
        "gsim_pass_complete": len(complete),
        "timing_ready": bool(gsim_pass) and len(complete) == len(gsim_pass),
        "timing_target_scope": "selected_gsim_elaboration_only",
        "cycle_scope": "kernel entry through selected completion; excludes output publication",
        "records": rows,
    }


def archive(run_dir: Path, destination: Path, *, require_ready: bool = True) -> dict:
    """Package complete capsule result directories after timing readiness succeeds."""
    index = audit(run_dir)
    if require_ready and not index["timing_ready"]:
        raise TimingArchiveError(
            f"gSIM capsule timing incomplete: {index['gsim_pass_with_cycles']}/"
            f"{index['gsim_pass_records']} passing records have positive cycle markers; "
            f"{index['gsim_pass_complete']}/{index['gsim_pass_records']} have ELF, compiler, command-buffer, "
            "simulator and kernel-scope identities, input/numeric artifacts and one matching console marker"
        )
    if not require_ready:
        index = {**index, "export_scope": "diagnostic_all_results; timing readiness is not implied"}
    if destination.exists():
        raise TimingArchiveError(f"archive already exists: {destination}")
    root = Path(index["run_dir"])
    capsule_dirs = sorted({(root / row["result"]).parent for row in index["records"]})
    files = []
    for directory in capsule_dirs:
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise TimingArchiveError(f"capsule artifact is a symlink: {path}")
            if path.is_file():
                files.append(path)
            elif not path.is_dir():
                raise TimingArchiveError(f"capsule artifact is not a regular file: {path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="capsule-timing-", suffix=".tar", dir=destination.parent, delete=False
    ) as temporary:
        partial = Path(temporary.name)
    try:
        with tarfile.open(partial, "w") as tar:
            payload = (json.dumps(index, sort_keys=True, indent=2) + "\n").encode("utf-8")
            member = tarfile.TarInfo("index.json")
            member.size = len(payload)
            member.mode = 0o644
            tar.addfile(member, io.BytesIO(payload))
            for path in files:
                member = tar.gettarinfo(path, arcname="run/" + path.relative_to(root).as_posix())
                member.uid = member.gid = member.mtime = 0
                member.uname = member.gname = ""
                with path.open("rb") as source:
                    tar.addfile(member, source)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return {
        **index,
        "archive": str(destination),
        "archive_sha256": _file_sha256(destination),
        "artifact_files": len(files),
    }


def finalize_run(run_dir: Path) -> dict:
    """Persist the audit and, only for a complete measured run, its training tar."""
    run_dir = Path(run_dir)
    status_path = run_dir / "capsule_timing_archive_status.json"
    try:
        index = audit(run_dir)
        (run_dir / "capsule_timing_audit.json").write_text(json.dumps(index, sort_keys=True, indent=2) + "\n")
        if index["timing_ready"]:
            destination = run_dir / "capsule_timing.tar"
            if destination.exists():
                with tarfile.open(destination) as existing:
                    member = existing.extractfile("index.json")
                    previous = json.load(member) if member is not None else None
                if previous != index:
                    digest = hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()[:12]
                    destination = run_dir / f"capsule_timing_{digest}.tar"
            if not destination.exists():
                archive(run_dir, destination)
            status = {
                "timing_ready": True,
                "archive": str(destination),
                "archive_sha256": _file_sha256(destination),
                "error": None,
            }
        else:
            status = {
                "timing_ready": False,
                "archive": None,
                "archive_sha256": None,
                "error": (
                    f"{index['gsim_pass_complete']}/{index['gsim_pass_records']} passing gSIM records "
                    "have measured cycles and required identities"
                ),
            }
        if index["timing_ready"]:
            status["evidence_archive"] = status["archive"]
            status["evidence_archive_sha256"] = status["archive_sha256"]
        else:
            # Preserve paid native evidence even if a compiler version failed,
            # an optional promotion omitted another tier, or custody is missing.
            # The distinct diagnostic tar never becomes a training-ready claim.
            digest = hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()[:12]
            evidence = run_dir / f"capsule_evidence_{digest}.tar"
            if not evidence.exists():
                archive(run_dir, evidence, require_ready=False)
            status["evidence_archive"] = str(evidence)
            status["evidence_archive_sha256"] = _file_sha256(evidence)
    except (OSError, ValueError, tarfile.TarError) as exc:
        status = {
            "timing_ready": False,
            "archive": None,
            "archive_sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    status_path.write_text(json.dumps(status, sort_keys=True, indent=2) + "\n")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="write a host-only .tar after positive gSIM cycle validation")
    args = parser.parse_args(argv)
    try:
        result = archive(args.run_dir, args.out) if args.out else audit(args.run_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"capsule timing archive: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["timing_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

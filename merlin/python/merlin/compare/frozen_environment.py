"""Content-addressed execution-environment provenance for CPU-host paper campaigns.

The protocol source hash is necessary but not sufficient for a performance experiment: changing
the compiler binary, Python environment, Codex configuration, external AET/Chia checkout, or K1
board can change the treatment.  This module captures those inputs once and verifies them without
trusting a caller-authored summary.

The companion source tar is deliberately uncompressed and normalized.  Clean tracked external
sources are recoverable from their Git commit; binary diffs and every non-ignored untracked byte are
stored in the tar.  Merlin protocol files are stored in full because the campaign may be frozen from
a dirty working tree.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable

from merlin.common.paths import repo_root


MANIFEST_VERSION = 1


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) +
            "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _setting(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    dotenv = repo_root() / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _run_identity(argv: list[str], *, timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {
            "argv": argv,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # an absent/broken tool is retained, never silently omitted
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": str(exc)}


def _tool(path: str | Path | None, version_args: Iterable[str] = ("--version",)) -> dict[str, Any]:
    requested = str(path or "")
    candidate = Path(requested).expanduser() if requested else None
    if candidate is None or not candidate.is_file():
        return {"requested_path": requested, "present": False}
    resolved = candidate.resolve()
    return {
        "requested_path": requested,
        "resolved_path": str(resolved),
        "present": True,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "version": _run_identity([str(resolved), *version_args]),
    }


def _file_identity(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    if not expanded.is_file():
        return {"path": str(expanded), "present": False}
    resolved = expanded.resolve()
    return {"path": str(expanded), "resolved_path": str(resolved), "present": True,
            "size_bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def _git_snapshot(path: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=30).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True, check=True, timeout=60).stdout
        diff = subprocess.run(
            ["git", "-C", str(path), "diff", "--no-ext-diff", "--binary", "HEAD", "--"],
            capture_output=True, check=True, timeout=60).stdout
        names = subprocess.run(
            ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard", "-z"],
            capture_output=True, check=True, timeout=60).stdout
        untracked: list[dict[str, Any]] = []
        for encoded in sorted(name for name in names.split(b"\0") if name):
            relative = encoded.decode("utf-8", errors="surrogateescape")
            source = path / relative
            if source.is_symlink():
                payload = source.readlink().as_posix().encode("utf-8")
                kind = "symlink"
            elif source.is_file():
                payload = source.read_bytes()
                kind = "file"
            else:
                payload = b""
                kind = "missing_or_non_file"
            untracked.append({"path": relative, "kind": kind, "size_bytes": len(payload),
                              "sha256": hashlib.sha256(payload).hexdigest()})
        return {
            "root": str(path), "git_commit": commit, "dirty": bool(status),
            "status_sha256": hashlib.sha256(status).hexdigest(),
            "diff_sha256": hashlib.sha256(diff).hexdigest(), "untracked": untracked,
        }
    except Exception as exc:
        return {"root": str(path), "git_commit": None, "error": str(exc)}


def _spike_paths() -> dict[str, Path]:
    chipyard = Path(_setting("MERLIN_CHIPYARD") or "/path/to/chipyard")
    root = chipyard / ".conda-env" / "riscv-tools" / "bin"
    gcc = Path(_setting("MERLIN_RISCV_GCC") or root / "riscv64-unknown-elf-gcc")
    return {
        "riscv_gcc": gcc,
        "riscv_objdump": gcc.with_name("riscv64-unknown-elf-objdump"),
        "riscv_objcopy": gcc.with_name("riscv64-unknown-elf-objcopy"),
        "spike": Path(_setting("MERLIN_SPIKE") or root / "spike"),
    }


def _k1_toolchain_root() -> Path | None:
    configured = _setting("MERLIN_K1_TOOLCHAIN")
    if not configured:
        return None
    root = Path(configured)
    candidates = [root, *sorted(root.glob("spacemit-toolchain-linux-glibc-*"))]
    return next((candidate for candidate in candidates
                 if (candidate / "bin" / "clang").is_file()), None)


def capture_local_identity(*, agent: dict[str, Any], telemetry: dict[str, Any]) -> dict[str, Any]:
    """Capture exact executables, dependency state, config identity, and source revisions."""
    root = repo_root().resolve()
    llvm = root / "third_party" / "llvm-install" / "bin"
    k1_root = _k1_toolchain_root()
    spike = _spike_paths()
    controller_python = Path(sys.executable)
    chia_python_raw = Path(str(telemetry.get("chia_python", "")))
    chia_python = chia_python_raw if chia_python_raw.is_absolute() else root / chia_python_raw
    codex_config_root = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    tool_specs: dict[str, tuple[str | Path | None, tuple[str, ...]]] = {
        "controller_python": (controller_python, ("--version",)),
        "chia_python": (chia_python, ("--version",)),
        "codex": (shutil.which("codex"), ("--version",)),
        "bwrap": (shutil.which("bwrap"), ("--version",)),
        "native_clang": (shutil.which("clang"), ("--version",)),
        "native_gcc": (shutil.which("gcc"), ("--version",)),
        "uv": (shutil.which("uv"), ("--version",)),
        "llvm_clang": (llvm / "clang", ("--version",)),
        "mlir_opt": (llvm / "mlir-opt", ("--version",)),
        "riscv_gcc": (spike["riscv_gcc"], ("--version",)),
        "riscv_objdump": (spike["riscv_objdump"], ("--version",)),
        "riscv_objcopy": (spike["riscv_objcopy"], ("--version",)),
        # Spike installations do not consistently implement --version; --help still binds the build.
        "spike": (spike["spike"], ("--help",)),
        "k1_clang": (k1_root / "bin" / "clang" if k1_root else None, ("--version",)),
        "k1_objcopy": (k1_root / "bin" / "llvm-objcopy" if k1_root else None,
                       ("--version",)),
    }
    dependencies = {
        "merlin_pyproject": _file_identity(root / "pyproject.toml"),
        "merlin_uv_lock": _file_identity(root / "uv.lock"),
        "aet_pyproject": _file_identity(Path(str(telemetry.get("aet_source", ""))) /
                                         "pyproject.toml"),
        "aet_uv_lock": _file_identity(Path(str(telemetry.get("aet_source", ""))) / "uv.lock"),
        "chia_pyproject": _file_identity(Path(str(telemetry.get("chia_source", ""))) /
                                          "pyproject.toml"),
        "chia_uv_lock": _file_identity(Path(str(telemetry.get("chia_source", ""))) / "uv.lock"),
        "codex_config": _file_identity(codex_config_root / "config.toml"),
    }
    distribution_program = (
        "import importlib.metadata as m,json; "
        "print(json.dumps(sorted((d.metadata.get('Name',''),d.version) "
        "for d in m.distributions()),separators=(',',':')))"
    )
    installed_distributions = _run_identity(
        [str(chia_python), "-c", distribution_program], timeout=60)
    key = Path(_setting("MERLIN_K1_SSH_KEY"))
    key_identity: dict[str, Any] = {"path": str(key), "present": key.is_file()}
    if key.is_file() and shutil.which("ssh-keygen"):
        public = _run_identity([str(shutil.which("ssh-keygen")), "-y", "-f", str(key)])
        if public["returncode"] == 0:
            # Bind the credential without putting private or public key material in the artifact.
            key_identity["public_key_sha256"] = hashlib.sha256(
                public["stdout"].encode("utf-8")).hexdigest()
    return {
        "tools": {name: _tool(path, args) for name, (path, args) in tool_specs.items()},
        "dependencies": dependencies,
        "chia_installed_distributions": installed_distributions,
        "agent": {
            "driver": agent.get("driver"), "model": agent.get("model"),
            "reasoning_effort": agent.get("reasoning_effort"),
            "orchestrator": agent.get("orchestrator"), "billing": agent.get("billing"),
        },
        "transport": {
            "k1_host": _setting("MERLIN_K1_HOST"),
            "k1_ssh_port": _setting("MERLIN_K1_SSH_PORT"),
            "k1_ssh_key_identity": key_identity,
        },
        "source_revisions": {
            "merlin": _git_snapshot(root),
            "aet": _git_snapshot(Path(str(telemetry.get("aet_source", "")))),
            "chia": _git_snapshot(Path(str(telemetry.get("chia_source", "")))),
        },
    }


def _local_completeness_errors(identity: Any) -> list[str]:
    if not isinstance(identity, dict):
        return ["local identity is absent"]
    errors: list[str] = []
    tools = identity.get("tools")
    if not isinstance(tools, dict):
        errors.append("tool identity set is absent")
    else:
        for name, row in tools.items():
            if (not isinstance(row, dict) or row.get("present") is not True or
                    not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64 or
                    not isinstance(row.get("version"), dict) or
                    row["version"].get("returncode") != 0):
                errors.append(f"required tool is not fully identified: {name}")
    dependencies = identity.get("dependencies")
    if not isinstance(dependencies, dict):
        errors.append("dependency/config identity set is absent")
    else:
        for name, row in dependencies.items():
            if not isinstance(row, dict) or row.get("present") is not True:
                errors.append(f"required dependency/config file is absent: {name}")
    distributions = identity.get("chia_installed_distributions")
    if not isinstance(distributions, dict) or distributions.get("returncode") != 0:
        errors.append("Chia interpreter distribution set could not be captured")
    revisions = identity.get("source_revisions")
    if not isinstance(revisions, dict):
        errors.append("source revision identity set is absent")
    else:
        for name in ("merlin", "aet", "chia"):
            row = revisions.get(name)
            commit = row.get("git_commit") if isinstance(row, dict) else None
            if (not isinstance(commit, str) or len(commit) not in {40, 64} or
                    any(character not in "0123456789abcdef" for character in commit)):
                errors.append(f"source revision is not fully identified: {name}")
    key = (identity.get("transport") or {}).get("k1_ssh_key_identity")
    if not isinstance(key, dict) or key.get("present") is not True or not key.get(
            "public_key_sha256"):
        errors.append("K1 SSH credential identity could not be captured")
    return errors


def _ssh_argv() -> list[str]:
    argv = ["ssh", "-i", _setting("MERLIN_K1_SSH_KEY")]
    port = _setting("MERLIN_K1_SSH_PORT")
    if port:
        argv += ["-p", port]
    return argv + ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                   "-o", "StrictHostKeyChecking=no", _setting("MERLIN_K1_HOST")]


def capture_k1_identity(probe_source: Path) -> dict[str, Any]:
    """Capture stable board identity and independently executed ISA/hart/VLEN facts."""
    from merlin.mining import k1

    probe = k1.run_arch_probe(probe_source)
    probe_values = probe.get("values", {})
    immutable_probe = {
        "source": probe.get("source"), "source_sha256": probe.get("source_sha256"),
        "returncode": probe.get("returncode"),
        # rdtime proves CSR accessibility but is intentionally different on every execution.
        "values": {name: probe_values.get(name) for name in (
            "k1_cpu_probe_version", "online_harts", "vlenb")},
    }
    command = (
        "set -eu; "
        "emit(){ k=$1; p=$2; if test -r \"$p\"; then "
        "printf '%s_sha256=' \"$k\"; sha256sum \"$p\" | cut -d' ' -f1; "
        "printf '%s_b64=' \"$k\"; base64 \"$p\" | tr -d '\\n'; printf '\\n'; "
        "else printf '%s_sha256=absent\\n%s_b64=\\n' \"$k\" \"$k\"; fi; }; "
        "printf 'uname_b64='; uname -a | base64 | tr -d '\\n'; printf '\\n'; "
        "emit os_release /etc/os-release; emit cpuinfo /proc/cpuinfo; "
        "emit device_model /proc/device-tree/model; "
        "emit device_compatible /proc/device-tree/compatible; "
        "emit device_serial /proc/device-tree/serial-number; "
        "emit machine_id /etc/machine-id; emit online_cpus /sys/devices/system/cpu/online"
    )
    proc = subprocess.run(_ssh_argv() + [command], capture_output=True, text=True, timeout=60)
    if proc.returncode:
        raise RuntimeError(f"K1 immutable identity probe failed: {proc.stderr.strip()}")
    raw: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in raw:
            raise RuntimeError(f"malformed K1 immutable identity output: {line!r}")
        raw[key] = value
    required = {
        "uname_b64", "os_release_sha256", "os_release_b64", "cpuinfo_sha256", "cpuinfo_b64",
        "device_model_sha256", "device_model_b64", "device_compatible_sha256",
        "device_compatible_b64", "device_serial_sha256", "device_serial_b64",
        "machine_id_sha256", "machine_id_b64", "online_cpus_sha256", "online_cpus_b64",
    }
    if set(raw) != required:
        raise RuntimeError(f"incomplete K1 immutable identity output: {sorted(set(raw) ^ required)}")
    decoded: dict[str, str] = {}
    for key in ("uname", "os_release", "device_model", "device_compatible", "online_cpus"):
        try:
            decoded[key] = base64.b64decode(raw[f"{key}_b64"], validate=True).decode(
                "utf-8", errors="replace").replace("\x00", "\\0").strip()
        except Exception as exc:
            raise RuntimeError(f"invalid base64 in K1 {key} identity") from exc
    # Serial and machine-id contents are intentionally not exposed; their hashes identify the board.
    hashes = {key.removesuffix("_sha256"): value for key, value in raw.items()
              if key.endswith("_sha256")}
    if any(value == "absent" for key, value in hashes.items()
           if key in {"os_release", "cpuinfo", "device_model", "device_compatible",
                      "machine_id", "online_cpus"}):
        raise RuntimeError(f"K1 immutable identity lacks required files: {hashes}")
    return {
        "transport_host": _setting("MERLIN_K1_HOST"),
        "transport_port": _setting("MERLIN_K1_SSH_PORT"),
        "decoded": decoded,
        "file_sha256": hashes,
        "architecture_probe": immutable_probe,
    }


def _tar_add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(payload))


def _external_bundle_entries(label: str, root: Path) -> list[tuple[str, bytes]]:
    entries: list[tuple[str, bytes]] = []
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, check=True,
        timeout=30).stdout.strip() + b"\n"
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        capture_output=True, check=True, timeout=60).stdout
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--no-ext-diff", "--binary", "HEAD", "--"],
        capture_output=True, check=True, timeout=60).stdout
    names = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True, check=True, timeout=60).stdout
    entries.extend(((f"external/{label}/git_commit", commit),
                    (f"external/{label}/status.porcelain", status),
                    (f"external/{label}/working_tree.diff", diff)))
    for encoded in sorted(name for name in names.split(b"\0") if name):
        relative = encoded.decode("utf-8", errors="surrogateescape")
        source = root / relative
        if source.is_symlink():
            payload = source.readlink().as_posix().encode("utf-8")
        elif source.is_file():
            payload = source.read_bytes()
        else:
            continue
        entries.append((f"external/{label}/untracked/{relative}", payload))
    return entries


def write_source_bundle(output: Path, *, source_paths: dict[str, Path],
                        telemetry: dict[str, Any]) -> dict[str, Any]:
    """Write the reconstructable, deterministic source bundle using exclusive creation."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen source bundle: {output}")
    entries: list[tuple[str, bytes]] = []
    source_index: dict[str, Any] = {}
    for label, source in sorted(source_paths.items()):
        resolved = source.resolve()
        payload = resolved.read_bytes()
        archive_name = f"merlin_protocol/{label}/{resolved.name}"
        entries.append((archive_name, payload))
        source_index[label] = {"path": str(resolved), "archive_path": archive_name,
                               "size_bytes": len(payload),
                               "sha256": hashlib.sha256(payload).hexdigest()}
    for label, key in (("aet", "aet_source"), ("chia", "chia_source")):
        root = Path(str(telemetry.get(key, ""))).resolve()
        entries.extend(_external_bundle_entries(label, root))
    # Preserve the entire dirty/untracked Merlin overlay too. Explicit protocol files above make the
    # bundle directly auditable; commit + binary diff + untracked bytes make the full checkout
    # reconstructable even when a transitive import was not named in the contract table.
    entries.extend(_external_bundle_entries("merlin", repo_root().resolve()))
    index = {"version": 1, "sources": source_index,
             "entries": [{"path": name, "size_bytes": len(payload),
                           "sha256": hashlib.sha256(payload).hexdigest()}
                          for name, payload in sorted(entries)]}
    entries.append(("INDEX.json", _canonical_json(index)))
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp",
                                                   dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, payload in sorted(entries):
                _tar_add_bytes(archive, name, payload)
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite frozen source bundle: {output}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(output.resolve()), "sha256": sha256_file(output),
            "source_index": source_index, "entry_count": len(entries)}


def capture_frozen_environment(output: Path, *, source_paths: dict[str, Path],
                               agent: dict[str, Any], telemetry: dict[str, Any],
                               probe_source: Path, include_live_board: bool) -> dict[str, Any]:
    """Create one immutable manifest plus its sibling source bundle."""
    output = output.resolve()
    bundle = output.with_name(f"{output.stem}.sources.tar")
    local_before = capture_local_identity(agent=agent, telemetry=telemetry)
    source_bundle = write_source_bundle(bundle, source_paths=source_paths, telemetry=telemetry)
    try:
        local_after = capture_local_identity(agent=agent, telemetry=telemetry)
        if local_after != local_before:
            raise RuntimeError(
                "local tool/config/dependency/source identity changed while creating source bundle")
        source_bundle["path"] = bundle.name
        manifest = {
            "version": MANIFEST_VERSION,
            "capture_complete": include_live_board,
            "local": local_after,
            "k1": capture_k1_identity(probe_source) if include_live_board else None,
            "source_bundle": source_bundle,
        }
        if include_live_board:
            completeness = _local_completeness_errors(manifest["local"])
            if completeness:
                raise RuntimeError(
                    "cannot freeze an incomplete local environment: " + "; ".join(completeness))
        payload = _canonical_json(manifest)
        try:
            descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite frozen environment manifest: {output}") from exc
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return {"manifest": manifest, "path": str(output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_bundle_path": str(bundle),
                "source_bundle_sha256": source_bundle["sha256"]}
    except Exception:
        bundle.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise


def validate_frozen_environment(path: Path, *, expected_sha256: str,
                                source_paths: dict[str, Path], agent: dict[str, Any],
                                telemetry: dict[str, Any], probe_source: Path,
                                check_local: bool, check_board: bool) -> dict[str, Any]:
    """Verify artifact integrity and, when requested, exact live local/K1 identity."""
    errors: list[str] = []
    evidence: dict[str, Any] = {"manifest_path": str(path.resolve())}
    if not path.is_file():
        return {"ready": False, "errors": [f"frozen environment manifest is absent: {path}"],
                "evidence": evidence}
    actual_sha = sha256_file(path)
    evidence["manifest_sha256"] = actual_sha
    if actual_sha != expected_sha256:
        errors.append("frozen environment manifest digest differs from experiment.yaml")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ready": False, "errors": [*errors, f"environment manifest is invalid: {exc}"],
                "evidence": evidence}
    if not isinstance(manifest, dict) or manifest.get("version") != MANIFEST_VERSION:
        errors.append("environment manifest is not the supported version")
        return {"ready": False, "errors": errors, "evidence": evidence}
    if manifest.get("capture_complete") is True:
        errors.extend(_local_completeness_errors(manifest.get("local")))
    bundle = manifest.get("source_bundle")
    if not isinstance(bundle, dict):
        errors.append("environment manifest has no source bundle identity")
    else:
        bundle_path = Path(str(bundle.get("path", "")))
        if not bundle_path.is_absolute():
            bundle_path = path.parent / bundle_path
        if not bundle_path.is_file():
            errors.append("frozen source bundle is absent")
        elif sha256_file(bundle_path) != bundle.get("sha256"):
            errors.append("frozen source bundle digest differs from environment manifest")
        source_index = bundle.get("source_index")
        if not isinstance(source_index, dict) or set(source_index) != set(source_paths):
            errors.append("frozen source bundle does not contain the exact protocol source set")
        else:
            for label, source in sorted(source_paths.items()):
                row = source_index.get(label, {})
                if (not source.is_file() or row.get("path") != str(source.resolve()) or
                        row.get("sha256") != sha256_file(source)):
                    errors.append(f"protocol source differs from frozen source bundle: {label}")
    if check_local:
        if manifest.get("capture_complete") is not True:
            errors.append("environment manifest was captured without a live K1 identity")
        current_local = capture_local_identity(agent=agent, telemetry=telemetry)
        evidence["local_identity_matches"] = current_local == manifest.get("local")
        if current_local != manifest.get("local"):
            errors.append("local tool/config/dependency/source identity differs from frozen environment")
    if check_board:
        if manifest.get("capture_complete") is not True or not isinstance(manifest.get("k1"), dict):
            errors.append("environment manifest has no frozen live K1 identity")
        else:
            current_k1 = capture_k1_identity(probe_source)
            evidence["k1_identity_matches"] = current_k1 == manifest.get("k1")
            if current_k1 != manifest.get("k1"):
                errors.append("K1 device/kernel/OS/ISA identity differs from frozen environment")
    evidence["capture_complete"] = manifest.get("capture_complete")
    evidence["source_bundle_sha256"] = bundle.get("sha256") if isinstance(bundle, dict) else None
    return {"ready": not errors, "errors": errors, "evidence": evidence}

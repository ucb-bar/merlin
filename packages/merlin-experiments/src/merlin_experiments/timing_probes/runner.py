"""Build and measure selected probes on the selected cycle-accurate engine, cached by hardware identity.

The engine command, build recipe and kernel-call timing window are the shared ones capsules use:
``merlin.perf.layer_bench`` builds with the target backend's own ``harness_build_recipe`` and runs
the backend's pinned, revalidated gSIM command; ``merlin.runtime.kernel_call_timing`` supplies the
reviewed counter window (``kernel_entry`` through ``kernel_complete``) and its source marker.

A measurement is reused only when its key is unchanged: the elaborated FIRRTL and emulator bytes,
the operator toolchain (compiler, flags, runtime sources, link script), the timing selection and the
probe's exact program source. Labels are written in the timing archive layout (``index.json`` plus
``run/<path>``) with the capsule archive's row fields, so calibration can merge them with Phase 1
capsule labels. Each probe runs ``repeats`` times; differing cycles fail its determinism check.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .backends import backend_path
from .space import TimingSpace, canonical_sha

ARCHIVE = "merlin.timing_probe_archive.v1"
RESULT = "merlin.timing_probe_result.v1"
_RTL_ASSERTION = "Assertion failed"


class ProbeRunError(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def private_dir(path: Path) -> Path:
    """Create ``path`` and every missing ancestor owner-only (0700); tighten ``path`` if it exists."""
    path = Path(path).resolve()
    for directory in [*reversed(path.parents), path]:
        if not directory.exists():
            directory.mkdir(mode=0o700)
            os.chmod(directory, 0o700)
    os.chmod(path, 0o700)
    return path


def hardware_identity(target: str) -> dict:
    """What a cycle label is a fact about: elaborated design, engine bytes and operator toolchain."""
    from merlin.runtime.backends import base
    from merlin.targetgen import gsim_emulator

    backend = base.get_backend(target)
    selected = getattr(backend, "execution_engine_identity", None)
    if selected is not None:
        # The capsule L3 path: the contract-selected engine pin, receipt and command template.
        chosen = selected("gsim")
        engine = {
            "available": True,
            "refused": False,
            "path": chosen["binary"]["path"],
            "binary_sha256": chosen["binary"]["sha256"],
            "receipt": chosen.get("receipt"),
            "argv_template": list(backend.execution_argv_template("gsim")),
            "startup_transport": chosen.get("startup_transport"),
            "selection_sha256": chosen.get("selection_sha256"),
        }
    else:
        engine = gsim_emulator.citation(target, env_var=getattr(backend, "GSIM_EMU_ENV", None))
    receipt = engine.get("receipt") or {}
    recipe = base.harness_build_recipe(target)
    toolchain = {
        "compiler": {"path": str(recipe.compiler), "sha256": _sha_file(recipe.compiler)},
        "cflags": list(recipe.cflags),
        "ldflags": list(recipe.ldflags),
        "link_script_sha256": _sha_file(recipe.link_script),
        "load_address": recipe.load_address,
        "support_sources": {Path(p).name: _sha_file(p) for p in recipe.support_sources},
        "header_dependencies": {Path(p).name: _sha_file(p) for p in recipe.header_dependencies},
    }
    core = {
        "target": target,
        "engine": "gsim",
        "firrtl_sha256": receipt.get("firrtl_sha256"),
        "emulator_sha256": engine.get("binary_sha256"),
        "toolchain_sha256": canonical_sha(toolchain),
        "engine_command_sha256": canonical_sha({k: engine.get(k) for k in ("argv_template", "startup_transport")}),
    }
    if not engine.get("available") or engine.get("refused") or not all(core.values()):
        raise ProbeRunError(f"engine or design identity is incomplete: {engine.get('reason')} {core}")
    return {**core, "identity_sha256": canonical_sha(core), "engine_citation": engine, "toolchain": toolchain}


def _console(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    metrics = []
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) == 3 and fields[:2] == ["METRIC", "cycles"] and fields[2].isdigit() and fields[2][0] != "0":
            metrics.append(int(fields[2]))
    return {
        "metrics": metrics,
        "done": any(line.strip() == "DONE" for line in text.splitlines()),
        "assertion": _RTL_ASSERTION in text,
    }


def _measure(probe_dir: Path, source: str, *, target: str, repeats: int, max_cycles: int, timeout_s: float) -> dict:
    """Build with the backend's harness recipe; run on its selected gSIM command (the capsule L3 path)."""
    from merlin.perf.layer_bench.build import build_program
    from merlin.runtime.backends import base

    backend = base.get_backend(target)
    probe_dir = private_dir(probe_dir)
    (probe_dir / "probe.c").write_text(source, encoding="utf-8")
    started = time.monotonic()
    built = build_program(
        [probe_dir / "probe.c"],
        probe_dir / "build",
        target=target,
        max_loaded_bytes=None,
        elf_name="probe.elf",
        support_first=True,
    )
    build_s = time.monotonic() - started
    elf = probe_dir / "probe.elf"
    shutil.copyfile(built.elf, elf)
    limit = getattr(backend, "execution_timeout_limit", None)
    deadline = min(timeout_s, limit("gsim")) if limit else timeout_s
    runs = []
    for replicate in range(repeats):
        console = probe_dir / f"rtl_console_r{replicate}.log"
        began, failure = time.monotonic(), None
        try:
            text = backend.run_elf(elf, simulator="gsim", timeout=deadline, private_max_cycles=max_cycles)
        except Exception as exc:  # noqa: BLE001 -- a refused or failed run is recorded, never a label
            text, failure = "", f"{type(exc).__name__}: {str(exc)[-600:]}"
        wall = time.monotonic() - began
        console.write_text(text if isinstance(text, str) else text.decode(errors="replace"), encoding="utf-8")
        parsed = _console(console)
        refused = parsed["assertion"] or (
            failure is not None and ("declared failure" in failure or _RTL_ASSERTION in failure)
        )
        ok = failure is None and parsed["done"] and len(parsed["metrics"]) == 1 and not refused
        runs.append(
            {
                "replicate": replicate,
                "status": "pass" if ok else ("refused_by_rtl" if refused else "fail"),
                "cycles": parsed["metrics"][0] if len(parsed["metrics"]) == 1 else None,
                "console": console.name,
                "console_sha256": _sha_file(console),
                "sim_wall_s": round(wall, 3),
                "failure": failure,
            }
        )
    cycles = {run["cycles"] for run in runs}
    return {
        "elf_sha256": _sha_file(elf),
        "loaded_bytes": built.loaded_bytes,
        "build_s": round(build_s, 3),
        "runs": runs,
        "deterministic": len(runs) >= 2 and len(cycles) == 1 and None not in cycles,
        "status": "pass" if all(r["status"] == "pass" for r in runs) else runs[-1]["status"],
    }


def _spike(elf: Path, *, target: str, timeout_s: float, out: Path) -> dict:
    """A functional-model pass on the same ELF: a pairing record for the timing model, never a label."""
    from merlin.runtime.backends import base

    backend = base.get_backend(target)
    try:
        text = backend.run_elf(elf, simulator="spike", timeout=int(timeout_s))
    except Exception as exc:  # noqa: BLE001 -- recorded, never fatal to the gSIM label
        return {"status": "fail", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"}
    out.write_text(text if isinstance(text, str) else text.decode(errors="replace"), encoding="utf-8")
    parsed = _console(out)
    identify = getattr(backend, "execution_engine_identity", None) or getattr(backend, "simulator_identity")
    return {
        "status": "pass" if parsed["done"] else "fail",
        "metric_cycles": parsed["metrics"],
        "console": out.name,
        "console_sha256": _sha_file(out),
        "simulator": identify("spike"),
        "kind": "functional",
    }


def run(
    selection: dict,
    *,
    space: TimingSpace,
    backend,
    contract: dict,
    target: str,
    out_dir: Path,
    repeats: int = 2,
    jobs: int = 1,
    max_cycles: int = 200_000_000,
    timeout_s: float = 3600.0,
    spike: bool = False,
    expect_identity: str | None = None,
) -> dict:
    from merlin.runtime.kernel_call_timing import from_contract

    timing = from_contract(contract)
    if timing is None:
        raise ProbeRunError("the selected contract has no reviewed kernel-call timing selection")
    if (
        selection.get("target") != target
        or selection.get("identity", {}).get("timing_space_sha256") != space.identity["timing_space_sha256"]
    ):
        raise ProbeRunError("the selection belongs to a different target or timing space")
    identity = hardware_identity(target)
    if expect_identity and identity["identity_sha256"] != expect_identity:
        raise ProbeRunError(f"hardware identity {identity['identity_sha256']} differs from {expect_identity}")
    out_dir = private_dir(out_dir)
    cache = private_dir(out_dir / "cache" / identity["identity_sha256"][:16])
    backend_identity = {**backend.identity(), "backend_path_sha256": _sha_file(backend_path(space))}

    def one(probe: dict) -> dict:
        request = probe["request"]
        realized = backend.realize(request, space)
        commands = [[c.name, c.rs1, c.rs2] for c in realized.commands]
        if hashlib.sha256(json.dumps(commands).encode()).hexdigest() != probe["commands_sha256"]:
            raise ProbeRunError(f"{probe['id']}: the backend no longer realizes the selected command stream")
        source = backend.render(realized, timing, probe_id=probe["id"])
        source_sha = hashlib.sha256(source.encode()).hexdigest()
        key = canonical_sha(
            {
                "hardware": identity["identity_sha256"],
                "source_sha256": source_sha,
                "timing_selection_sha256": timing.selection_sha256,
                "repeats": repeats,
            }
        )
        entry = cache / key[:24]
        result_path = entry / "probe_result.json"
        if result_path.is_file():
            stored = json.loads(result_path.read_text())
            if stored.get("key") == key and _sha_file(entry / "probe.elf") == stored["measurement"]["elf_sha256"]:
                return {**stored, "cache_hit": True}
        private_dir(entry)
        (entry / "probe_commands.json").write_text(
            json.dumps(
                {
                    "probe": probe["id"],
                    "request": request,
                    "commands": [dict(zip(("name", "rs1", "rs2"), c)) for c in commands],
                }
            )
        )
        measurement = _measure(
            entry, source, target=target, repeats=repeats, max_cycles=max_cycles, timeout_s=timeout_s
        )
        record = {
            "schema": RESULT,
            "key": key,
            "probe": probe["id"],
            "request": request,
            "description": realized.description,
            "covers": [c["requirement"] for c in probe.get("covers") or ()],
            "source_sha256": source_sha,
            "measurement": measurement,
            "spike": _spike(
                entry / "probe.elf", target=target, timeout_s=min(timeout_s, 1800), out=entry / "spike_console.log"
            )
            if spike
            else {"status": "not_requested"},
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        result_path.write_text(json.dumps(record, indent=1, sort_keys=True))
        return {**record, "cache_hit": False}

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = list(pool.map(one, selection["probes"]))
    archive = write_archive(
        out_dir,
        cache,
        results,
        identity=identity,
        timing=timing,
        space=space,
        backend_identity=backend_identity,
        selection=selection,
    )
    return {
        "schema": "merlin.timing_probe_run.v1",
        "target": target,
        "hardware_identity": {k: v for k, v in identity.items() if k != "engine_citation"},
        "backend": backend_identity,
        "timing_selection_sha256": timing.selection_sha256,
        "wall_s": round(time.monotonic() - started, 3),
        "probes": [
            {
                "probe": r["probe"],
                "status": r["measurement"]["status"],
                "deterministic": r["measurement"]["deterministic"],
                "cycles": [run["cycles"] for run in r["measurement"]["runs"]],
                "cache_hit": r["cache_hit"],
                "sim_wall_s": [run["sim_wall_s"] for run in r["measurement"]["runs"]],
                "build_s": r["measurement"]["build_s"],
                "covers": r["covers"],
                "description": r["description"],
                "spike": r["spike"].get("status"),
            }
            for r in results
        ],
        "archive": archive,
    }


def _artifact(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": _sha_file(path), "bytes": path.stat().st_size}


def write_archive(
    out_dir: Path,
    cache: Path,
    results: list[dict],
    *,
    identity: dict,
    timing,
    space: TimingSpace,
    backend_identity: dict,
    selection: dict,
) -> dict:
    """``timing_probe_labels.tar``: one row per replicate, in the capsule archive's row vocabulary."""
    rows, files = [], []
    for result in results:
        entry = cache / result["key"][:24]
        measurement = result["measurement"]
        refs = {
            name: _artifact(cache, entry / file)
            for name, file in (("elf", "probe.elf"), ("harness", "probe.c"), ("probe_commands", "probe_commands.json"))
        }
        files += [entry / "probe.elf", entry / "probe.c", entry / "probe_commands.json", entry / "probe_result.json"]
        for run in measurement["runs"]:
            console = entry / run["console"]
            files.append(console)
            rows.append(
                {
                    "record_kind": "operator_timing_probe",
                    "probe": result["probe"],
                    "capsule": result["probe"],
                    "probe_request": result["request"],
                    "replicate": run["replicate"],
                    "result": (entry / "probe_result.json").relative_to(cache).as_posix(),
                    "status": "pass" if run["status"] == "pass" else run["status"],
                    "result_status": measurement["status"],
                    "engine": "gsim",
                    "measured_now": True,
                    "cycles": run["cycles"],
                    "cycles_measured": type(run["cycles"]) is int and run["cycles"] > 0,
                    "console_cycle_verified": run["status"] == "pass",
                    "cycle_scope": "entry-through-completion",
                    "timing_selection_sha256": timing.selection_sha256,
                    "timing_selection": timing.record(),
                    "elf": refs["elf"]["path"],
                    "elf_sha256": measurement["elf_sha256"],
                    "harness_sha256": refs["harness"]["sha256"],
                    "candidate_sha256": None,
                    "command_buffer_sha256": None,
                    "probe_source_sha256": result["source_sha256"],
                    "deterministic": measurement["deterministic"],
                    "artifacts": {**refs, "rtl_console": _artifact(cache, console)},
                    "sim_provenance": {
                        "engine": "gsim",
                        "binary": identity["engine_citation"].get("path"),
                        "sha256": identity["emulator_sha256"],
                        "firrtl_sha256": identity["firrtl_sha256"],
                    },
                    "hardware_identity_sha256": identity["identity_sha256"],
                    "toolchain_sha256": identity["toolchain_sha256"],
                    "derived_from_rtl": True,
                    "cycle_accurate": True,
                    "timing": {"sim_wall_s": run["sim_wall_s"], "build_s": measurement["build_s"]},
                }
            )
    passing = [r for r in rows if r["status"] == "pass"]
    complete = [r for r in passing if r["cycles_measured"] and r["deterministic"]]
    index = {
        "schema": ARCHIVE,
        "target": space.target,
        "record_kind": "operator_timing_probe",
        "timing_target_scope": "selected_gsim_elaboration_only",
        "cycle_scope": "kernel entry through selected completion; excludes output publication",
        "hardware_identity": {k: v for k, v in identity.items() if k != "engine_citation"},
        "timing_space_sha256": space.identity["timing_space_sha256"],
        "backend": backend_identity,
        "selection_sha256": canonical_sha(selection),
        "records": rows,
        "probe_records": len(rows),
        "gsim_pass_records": len(passing),
        "gsim_pass_complete": len(complete),
        "timing_ready": bool(rows) and len(complete) == len(rows),
        "visibility": "operator-private calibration labels; never granted to an author or graded",
    }
    destination = out_dir / "timing_probe_labels.tar"
    with tempfile.NamedTemporaryFile(prefix="probe-labels-", suffix=".tar", dir=out_dir, delete=False) as handle:
        partial = Path(handle.name)
    try:
        with tarfile.open(partial, "w") as tar:
            payload = (json.dumps(index, sort_keys=True, indent=1) + "\n").encode()
            member = tarfile.TarInfo("index.json")
            member.size, member.mode = len(payload), 0o600
            tar.addfile(member, io.BytesIO(payload))
            for path in sorted(set(files)):
                info = tar.gettarinfo(path, arcname="run/" + path.relative_to(cache).as_posix())
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode = 0o600
                with path.open("rb") as source:
                    tar.addfile(info, source)
        os.chmod(partial, 0o600)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "sha256": _sha_file(destination),
        "records": len(rows),
        "complete": len(complete),
        "timing_ready": index["timing_ready"],
    }

#!/usr/bin/env python3
"""Run a chip example's images on FireSim: plan one queue job per leg, submit it, grade what came back.

    firesim_flow.py legs <bundle> --board B --package P --matrix-unit U --matrix-config C
                    --matrix-support-target T [--harts N] [--timeout S] [--submit]
        Build the DEVICE leg (contractions routed to the matrix unit) and its CONTROL twin (identical
        lowering, nothing routed) through merlin-compile. Both exact images are gated on spike before
        they are copied to a queue-readable artifact directory and planned for FireSim.

    firesim_flow.py submit <job-dir> [--submit] [--timeout S] [--priority N]
        Print the queue submission for every leg of <job-dir>/firesim_job.json. With --submit, run the
        legs one at a time through the queue and grade them.

    firesim_flow.py collect <job-dir> [--job LEG=ID ...] [--uartlog LEG=PATH ...]
        Grade finished runs: which bitstream actually ran, the image's build hash, cycles, and the output
        against the bundle's W8A8 golden. Writes a measurement under
        out/artifacts/measurements/firesim_<bitstream>/<model>/.

A JOB DIRECTORY holds `firesim_job.json` (the legs, their ELFs and build hashes, the board and its
bitstream), `firesim/workload/<workload>.json` (the FireSim workload definition), and, per leg,
`firesim/<leg>/` with the queue evidence. `chip_flow.py op` writes one for a FireSim board.

Every host fact is configuration, read from the environment or `.env`, never assumed:
    MERLIN_CHIPYARD                 the chipyard whose sims/firesim deploys the bitstream
    MERLIN_EXT_FIRESIM_QUEUE        the shared queue's state root (jobs/<id>/ lives there)
    MERLIN_FIRESIM_QUEUE_CLIENT     the raw queue client; default $MERLIN_EXT_FIRESIM_QUEUE/bin/firesim-queue
    MERLIN_FIRESIM_WORKLOAD         the FireSim workload name the ELF is staged into (default merlin-zephyr)
    MERLIN_FIRESIM_CLIENT_DROP_ENV  variables the client must not forward to a daemon run by another account
    MERLIN_FIRESIM_CLIENT_PATH_PREFIX  directories put first on the client's PATH for a daemon run by
                                    another account: the queue's cwd launchers (docs/guides/firesim.md)
The board names its bitstream (`bitstream:` in the board catalog); nothing here names a chip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
#: How this script is named in the commands it prints: runnable from the repository root.
SELF = "examples/lib/firesim_flow.py"

MANIFEST = "firesim_job.json"
SCHEMA = "merlin_example_firesim_job_v1"
RESULT_SCHEMA = "merlin_example_firesim_result_v1"
DEFAULT_WORKLOAD = "merlin-zephyr"
#: Legs in submission order. The device leg first: it is the one that can fail in a new way.
LEGS = ("device", "control")
#: The queue accepts exactly these priorities.
PRIORITIES = (0, 5, 10)


# --------------------------------------------------------------------------------------------- host


def _env(key: str) -> str:
    from merlin.common.paths import env

    return env(key) or ""


def host_config() -> dict[str, Any]:
    """Where this host's FireSim and queue live. Every entry may be absent; callers say what is missing."""
    chipyard = Path(_env("MERLIN_CHIPYARD")) if _env("MERLIN_CHIPYARD") else None
    deploy = chipyard / "sims" / "firesim" / "deploy" if chipyard else None
    workload = _env("MERLIN_FIRESIM_WORKLOAD") or DEFAULT_WORKLOAD
    workload_json = deploy / "workloads" / f"{workload}.json" if deploy else None
    installed = workload_json is not None and workload_json.is_file()
    # The boot binary name is the workload's own (`common_bootbinary`): the queue stages the ELF under
    # that name, and FireSim boots whatever the workload names, so the two must agree.
    bootbinary = f"{workload}.elf"
    if installed:
        try:
            bootbinary = json.loads(workload_json.read_text())["common_bootbinary"] or bootbinary
        except (OSError, ValueError, KeyError):
            pass
    queue_root = Path(_env("MERLIN_EXT_FIRESIM_QUEUE")) if _env("MERLIN_EXT_FIRESIM_QUEUE") else None
    client = _env("MERLIN_FIRESIM_QUEUE_CLIENT")
    if not client and queue_root is not None:
        client = str(queue_root / "bin" / "firesim-queue")
    drop = [n for n in _env("MERLIN_FIRESIM_CLIENT_DROP_ENV").replace(",", " ").split() if n]
    path_prefix = [Path(p) for p in _env("MERLIN_FIRESIM_CLIENT_PATH_PREFIX").split(os.pathsep) if p]
    return {
        "chipyard": chipyard,
        "deploy": deploy,
        "workload": workload,
        "workload_installed": installed,
        "bootbinary": bootbinary,
        "queue_root": queue_root,
        "client": Path(client) if client else None,
        "client_drop_env": drop,
        "client_path_prefix": path_prefix,
    }


def _owner_uid(path: Path) -> int:
    return Path(path).stat().st_uid


def staging_refusal(deploy: Path, workload: str, bootbinary: str, queue_root: Path) -> str | None:
    """Why the queue could not stage this workload's boot binary, or None.

    The daemon stages with ``shutil.copy2``: it writes the bytes and then sets the destination's times and
    mode, which only the file's OWNER may do. A destination that exists and belongs to anyone but the
    daemon's user (the owner of ``daemon.pid``) is therefore overwritten and then fails the job with
    ``Operation not permitted`` -- group write does not help. Refused here, before the queue touches it.
    """
    pid = Path(queue_root) / "daemon.pid"
    dst = Path(deploy) / "workloads" / workload / bootbinary
    if not pid.is_file() or not dst.exists():
        return None
    daemon_uid, owner = _owner_uid(pid), _owner_uid(dst)
    if owner == daemon_uid:
        return None
    return (
        f"{dst} exists and is owned by uid {owner}, not by the queue daemon's uid {daemon_uid}: the daemon would "
        f"overwrite it and then fail staging with 'Operation not permitted' (copy2 sets times/mode, which only "
        f"the owner may do). Use a workload whose staged directory the daemon creates itself "
        f"(MERLIN_FIRESIM_WORKLOAD), or have the file's owner remove it."
    )


def _board(name: str):
    from merlin.runtime.boards import board

    brd = board(name)
    if not brd.bitstream:
        raise SystemExit(
            f"board {name!r} declares no `bitstream`, so it is not a FireSim board: pick one whose catalog "
            f"entry names the bitstream its images run on"
        )
    return brd


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _capture_sha256(bundle: Path) -> str:
    """Bind both image builds to every named byte of the captured model and reference."""
    h = hashlib.sha256()
    for path in sorted(bundle.rglob("*")):
        if not path.is_file():
            continue
        h.update(str(path.relative_to(bundle)).encode())
        h.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- plan


def plan(
    job_dir: Path,
    *,
    board: str,
    name: str,
    bundle: Path,
    legs: dict[str, dict[str, Any]],
    harts: int = 1,
    timeout_s: int = 3600,
    capture_sha256: str | None = None,
    host: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``<job_dir>/firesim_job.json`` and the workload definition; return the manifest.

    ``legs`` maps a leg name to at least ``{"elf": path}``, plus ``build_hash`` when the build reported
    one. The ELF is hashed here, so a later rebuild in place is detectable at collect time.
    """
    brd = _board(board)
    host = host or host_config()
    job_dir = Path(job_dir).resolve()
    rows = {}
    for leg in LEGS:
        if leg not in legs:
            continue
        elf = Path(legs[leg]["elf"]).resolve()
        if not elf.is_file():
            raise SystemExit(f"{leg} leg: no ELF at {elf}")
        rows[leg] = {**legs[leg], "elf": str(elf), "elf_sha256": _sha256(elf)}
    if not rows:
        raise SystemExit("nothing to plan: no device or control leg was given")
    manifest = {
        "schema": SCHEMA,
        "name": name,
        "bundle": str(Path(bundle).resolve()),
        "capture_sha256": capture_sha256,
        "board": brd.name,
        "hw_config": brd.bitstream,
        "vlen": brd.vlen,
        "harts": harts,
        "workload": host["workload"],
        "bootbinary": host["bootbinary"],
        "timeout_s": int(timeout_s),
        "legs": rows,
        "jobs": {},
    }
    write_workload(job_dir, manifest)
    _save(job_dir, manifest)
    return manifest


def write_workload(job_dir: Path, manifest: dict[str, Any]) -> Path:
    """The FireSim workload definition the ELF boots under, for a host that does not have it yet."""
    path = Path(job_dir) / "firesim" / "workload" / f"{manifest['workload']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "benchmark_name": manifest["workload"],
        "common_bootbinary": manifest["bootbinary"],
        "common_rootfs": None,
        "common_simulation_outputs": ["uartlog"],
    }
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return path


def _save(job_dir: Path, manifest: dict[str, Any]) -> None:
    Path(job_dir).mkdir(parents=True, exist_ok=True)
    (Path(job_dir) / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load(job_dir: Path) -> dict[str, Any]:
    path = Path(job_dir) / MANIFEST
    if not path.is_file():
        raise SystemExit(f"{job_dir} is not a FireSim job directory (no {MANIFEST})")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: schema {manifest.get('schema')!r}, this reads {SCHEMA}")
    return manifest


def hwdb_entry(manifest: dict[str, Any], host: dict[str, Any]) -> Path | None:
    """The one-entry hwdb file for the bitstream, when the host has one. The queue then flashes a
    snapshot of THAT file, so an edit to the shared hwdb between submit and dispatch cannot change
    what runs."""
    if host["deploy"] is None:
        return None
    path = host["deploy"] / "built-hwdb-entries" / manifest["hw_config"]
    return path if path.is_file() else None


def submission_argv(
    manifest: dict[str, Any],
    leg: str,
    host: dict[str, Any],
    *,
    priority: int = 5,
    project: str = "merlin-examples",
    timeout_s: int | None = None,
) -> list[str]:
    """The exact queue submission for one leg -- always naming the bitstream (`--hw-config`).

    Without `--hw-config` the queue inherits the shared runtime template's default design, and a job
    silently runs on whatever bitstream someone else last selected: observed here, a batch of jobs
    meant for one design ran on another with plausible-looking output. Same argument order as
    `merlin.perf.firesim_checkpoint.QueueSubmission.argv`, which performs the real submission.
    """
    if priority not in PRIORITIES:
        raise SystemExit(f"queue priority is one of {PRIORITIES}, got {priority}")
    argv = [
        str(host["client"] or "firesim-queue"),
        "runworkload-full",
        "--chipyard",
        str(host["chipyard"] or "$MERLIN_CHIPYARD"),
        "--workload",
        manifest["workload"],
        "--bootbinary",
        manifest["bootbinary"],
        "--stage-from",
        manifest["legs"][leg]["elf"],
        "--hw-config",
        manifest["hw_config"],
    ]
    entry = hwdb_entry(manifest, host)
    if entry is not None:
        argv += ["--hwdb-config-artifact", str(entry)]
    argv += ["--priority", str(priority), "--project", project, "--timeout", str(timeout_s or manifest["timeout_s"])]
    return argv


def describe(job_dir: Path, manifest: dict[str, Any], host: dict[str, Any], **kw) -> str:
    """What to run, printed: the queue submissions, and the same thing for a FireSim with no queue."""
    job_dir = Path(job_dir)
    wl = job_dir / "firesim" / "workload" / f"{manifest['workload']}.json"
    lines = [
        f"FireSim jobs for {manifest['name']} on board {manifest['board']} "
        f"(bitstream {manifest['hw_config']}, {manifest['harts']} hart(s)):"
    ]
    for leg in manifest["legs"]:
        row = manifest["legs"][leg]
        lines.append(f"  {leg:8} {row['elf']}  build_hash={row.get('build_hash') or '?'}")
    lines.append("")
    if host["chipyard"] is None:
        lines.append("  MERLIN_CHIPYARD is unset: set it to the chipyard whose sims/firesim deploys this bitstream.")
    elif not host["workload_installed"]:
        lines.append(f"  workload {manifest['workload']!r} is not installed in {host['deploy']}/workloads; once:")
        lines.append(
            f"    cp {wl} {host['deploy']}/workloads/ && mkdir -p {host['deploy']}/workloads/{manifest['workload']}"
        )
    if host["client"] is not None:
        drop = " ".join(f"-u {n}" for n in host["client_drop_env"])
        path = os.pathsep.join(str(p) for p in host["client_path_prefix"])
        prefix = ("env " + drop + (f" PATH={path}:$PATH" if path else "") + " ") if (drop or path) else ""
        lines.append("  Through the shared queue (one job per leg; each waits for its turn on the FPGA):")
        for leg in manifest["legs"]:
            lines.append("    " + prefix + shlex.join(submission_argv(manifest, leg, host, **kw)))
        lines.append(f"  or run them all and grade: {SELF} submit {job_dir} --submit")
    else:
        lines.append("  No queue configured (MERLIN_EXT_FIRESIM_QUEUE unset). On a FireSim host of your own:")
        lines.append(f"    1. install {wl} into $FIRESIM/deploy/workloads/")
        lines.append(f"    2. config_runtime.yaml: target_config.default_hw_config: {manifest['hw_config']}")
        lines.append(f"                            workload.workload_name: {manifest['workload']}.json")
        for leg in manifest["legs"]:
            lines.append(
                f"    3. ({leg}) cp {manifest['legs'][leg]['elf']} "
                f"$FIRESIM/deploy/workloads/{manifest['workload']}/{manifest['bootbinary']}"
            )
            lines.append("       firesim kill; firesim infrasetup && firesim runworkload; firesim kill")
    lines.append(
        f"  then grade: {SELF} collect {job_dir} "
        + " ".join(f"--job {leg}=<id>" for leg in manifest["legs"])
        + "   (or --uartlog LEG=<results-workload/.../uartlog>)"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- submit


def submit(job_dir: Path, *, priority: int = 5, project: str = "merlin-examples", timeout_s: int | None = None) -> dict:
    """Run every leg through the queue, one at a time, and keep each job's evidence under
    ``<job_dir>/firesim/<leg>/``. Uses the checkpoint harness's submission, which hashes the staged ELF
    before the queue is asked to run it and copies the client log, daemon log and UART out of the queue's
    own tree."""
    from merlin.perf import firesim_checkpoint as fc

    job_dir = Path(job_dir).resolve()
    manifest = load(job_dir)
    for leg, row in manifest["legs"].items():
        elf = Path(row["elf"])
        if not elf.is_file() or _sha256(elf) != row.get("elf_sha256"):
            raise SystemExit(f"{leg} leg: planned ELF is missing or changed; rebuild and gate it before submission")
    host = host_config()
    missing = [k for k in ("chipyard", "queue_root", "client") if host[k] is None]
    if missing:
        raise SystemExit(f"cannot submit: {', '.join(missing)} not configured (see this file's docstring)")
    if not host["workload_installed"]:
        raise SystemExit(
            f"workload {manifest['workload']!r} is not installed in {host['deploy']}/workloads -- "
            f"install {job_dir}/firesim/workload/{manifest['workload']}.json first"
        )
    refused = staging_refusal(host["deploy"], manifest["workload"], manifest["bootbinary"], host["queue_root"])
    if refused:
        raise SystemExit(f"cannot submit: {refused}")
    queue_host = fc.QueueHost(
        queue_executable=host["client"], queue_state_root=host["queue_root"], chipyard=host["chipyard"]
    )
    env = (
        fc.client_environment(drop=host["client_drop_env"], path_prefix=host["client_path_prefix"])
        if (host["client_drop_env"] or host["client_path_prefix"])
        else None
    )
    for leg, row in manifest["legs"].items():
        sub = fc.QueueSubmission(
            host=queue_host,
            workload=manifest["workload"],
            bootbinary=manifest["bootbinary"],
            elf=Path(row["elf"]),
            hw_config=manifest["hw_config"],
            hwdb_config_artifact=hwdb_entry(manifest, host),
            timeout_s=int(timeout_s or manifest["timeout_s"]),
            priority=priority,
            project=project,
        )
        print("$ " + shlex.join(sub.argv()), flush=True)
        evidence = job_dir / "firesim" / leg
        try:
            ev = fc.submit(sub, evidence, env=env)
        except fc.JobIncomplete as exc:
            manifest["jobs"][leg] = exc.job_id
            _save(job_dir, manifest)
            raise SystemExit(f"{leg} leg: {exc}") from exc
        manifest["jobs"][leg] = ev.job_id
        _save(job_dir, manifest)
        print(f"  {leg}: queue job {ev.job_id} DONE in {ev.wall_s:.0f}s; uart {ev.uart_log}", flush=True)
    return manifest


# --------------------------------------------------------------------------------------------- collect


def results_dir_for_job(job_id: int, workload: str, host: dict[str, Any]) -> Path | None:
    """FireSim's per-job results directory, ``results-workload/<ts>-<workload>-q<id>/<workload>0/``.

    A job run by a daemon of another account writes into its private deploy overlay; one run by the
    submitter's own daemon writes into the chipyard's deploy tree. Both are looked at.
    """
    roots = []
    if host.get("queue_root") is not None:
        roots.append(host["queue_root"] / "jobs" / str(job_id) / "deploy_overlay" / "results-workload")
    if host.get("deploy") is not None:
        roots.append(host["deploy"] / "results-workload")
    for root in roots:
        if not root.is_dir():
            continue
        runs = sorted(root.glob(f"*-{workload}-q{job_id}"), key=lambda p: p.stat().st_mtime, reverse=True)
        for run in runs:
            leaf = run / f"{workload}0"
            if (leaf / "uartlog").is_file():
                return leaf
    return None


def read_hw_summary(results_dir: Path | None) -> dict[str, str]:
    """``HW_CFG_SUMMARY``: FireSim's own statement of which hardware config and bitstream tar it ran."""
    out: dict[str, str] = {}
    if results_dir is None:
        return out
    path = Path(results_dir) / "HW_CFG_SUMMARY"
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def read_sim_footer(uart: str) -> dict[str, Any]:
    """The FireSim driver's own footer: did the target pass, how long, at what effective clock."""
    out: dict[str, Any] = {}
    for line in uart.splitlines():
        text = line.strip()
        if text.startswith("*** PASSED ***") or text.startswith("*** FAILED ***"):
            out["sim_passed"] = text.startswith("*** PASSED ***")
            words = text.split()
            if "after" in words:
                try:
                    out["target_cycles_total"] = int(words[words.index("after") + 1])
                except (ValueError, IndexError):
                    pass
        key, sep, value = text.partition(":")
        if not sep:
            continue
        first = value.split()[0] if value.split() else ""
        try:
            if key == "Wallclock Time Elapsed":
                out["fpga_wall_s"] = float(first)
            elif key == "Effective Target Frequency":
                out["effective_mhz"] = float(first)
            elif key == "Host Frequency":
                out["host_mhz"] = float(first)
        except ValueError:
            pass
    return out


def _bitstream_artifact(tar_uri: str | None) -> dict[str, Any]:
    """Which declared artifact (merlin/contract/hardware_pins.yaml) is the bitstream tar that ran.

    Matched by the RESOLVED PATH FireSim reported, never by name: a hwdb key can be re-pointed at a new
    tar, and the bytes are what the registry pins. A tar the registry does not declare is said so."""
    from merlin.common import provenance

    if not tar_uri:
        return {"matched": False, "why": "FireSim reported no bitstream tar (no HW_CFG_SUMMARY beside the uartlog)"}
    path = Path(tar_uri.partition("file://")[2] or tar_uri)
    try:
        arts = provenance.load_artifacts()
    except Exception as exc:  # noqa: BLE001
        return {"matched": False, "tar": str(path), "why": f"pin registry unreadable: {exc}"}
    for name, art in arts.items():
        where = art.resolve()
        if where is not None and Path(where) == path:
            check = provenance.verify_artifact(name)
            return {
                "matched": True,
                "name": name,
                "ok": check.ok,
                "digest": check.digest,
                "gaps": list(check.gaps),
                "built_from": list(getattr(art, "built_from", ()) or ()),
            }
    return {"matched": False, "tar": str(path), "why": "this bitstream tar is not a declared artifact"}


def _references(bundle: Path) -> dict:
    import numpy as np

    refs = {}
    if (bundle / "golden.npy").is_file():
        refs["fp32"] = np.load(bundle / "golden.npy")
    if (bundle / "golden_w8a8.npy").is_file():
        refs["w8a8"] = np.load(bundle / "golden_w8a8.npy")
    return refs


def grade_leg(
    uart: str,
    *,
    row: dict[str, Any],
    refs: dict,
    hw_config: str,
    hw_summary: dict[str, str],
) -> dict[str, Any]:
    """One leg's verdict. Every check that could not be made is a named problem, never a pass."""
    from merlin.runtime.backends import output_identity
    from merlin.runtime.backends import zephyr_model as zm

    out: dict[str, Any] = {"problems": [], **read_sim_footer(uart)}
    observed_hw = hw_summary.get("RuntimeHWConfig")
    out["hw_config"] = observed_hw
    if observed_hw is None:
        out["problems"].append("hardware config unverified: FireSim's HW_CFG_SUMMARY was not found")
    elif observed_hw != hw_config:
        out["problems"].append(f"ran on {observed_hw!r}, not the board's bitstream {hw_config!r}")
    out["bitstream_tar"] = hw_summary.get("Bitstream tar")
    try:
        res = zm._parse_console(uart, 0)
    except zm.ZephyrModelError:
        stages = [ln for ln in uart.splitlines() if ln.startswith(("STAGE ", "ALIVE ", "FAIL "))]
        out["status"] = "no_output"
        out["problems"].append("the console has no OUT/DONE: the image did not finish")
        out["last_progress"] = stages[-3:]
        return out
    metrics = res.get("metrics", {})
    out["cycles"] = metrics.get("cycles")
    observed_hash = metrics.get("build_hash")
    out["build_hash"] = observed_hash
    expected = row.get("build_hash")
    if expected and str(observed_hash) != str(expected):
        out["problems"].append(f"console build_hash {observed_hash!r} is not this leg's image ({expected!r})")
    if "w8a8" not in refs:
        out["problems"].append("no golden_w8a8.npy beside the bundle: the W8A8 grade cannot be made")
    if refs:
        gate = zm._gate(res["prefix"], refs)
        out["grade"] = {
            k: gate[k]
            for k in sorted(gate)
            if k.endswith(("_cos", "_rel", "_max_rel", "_argmax"))
            or k in ("ok", "tier_ok", "tiers", "n_compared", "n_reference", "comparison_complete")
        }
        if not gate.get("ok") or gate.get("tier_ok") != "w8a8":
            out["problems"].append(f"output fails the gate (tier_ok={gate.get('tier_ok')})")
    digest = res.get("out_hash")
    expected_identity = row.get("spike_output_identity")
    if expected_identity is not None:
        expected_hash = expected_identity.get("out_hash")
        if not expected_hash:
            out["problems"].append("spike gate has no whole-output HASH")
        elif digest != expected_hash:
            out["problems"].append(
                f"whole-output HASH differs from the spike-gated image: {digest!r} != {expected_hash!r}"
            )
        out["out_hash_matches_spike"] = bool(expected_hash) and digest == expected_hash
        observed_identity = output_identity.output_identity(res)
        if observed_identity.get("prefix_sha256") != expected_identity.get("prefix_sha256"):
            out["problems"].append("printed output prefix differs from the spike-gated image")
        out["prefix_matches_spike"] = observed_identity.get("prefix_sha256") == expected_identity.get("prefix_sha256")
    if digest and "w8a8" in refs and int(digest["elems"]) == int(refs["w8a8"].size):
        # The OUT line is a prefix of a large tensor; the hash covers all of it.
        out["out_hash_matches_w8a8"] = zm.out_hash_fnv1a32(refs["w8a8"]) == digest["value"]
    out["status"] = "verified" if not out["problems"] else "graded_with_problems"
    return out


def collect(
    job_dir: Path,
    *,
    jobs: dict[str, int] | None = None,
    uartlogs: dict[str, Path] | None = None,
    measurement: bool = True,
) -> dict[str, Any]:
    """Grade every leg that has a finished run and write the result. Returns the result document."""
    job_dir = Path(job_dir).resolve()
    manifest = load(job_dir)
    host = host_config()
    jobs = {**{k: int(v) for k, v in manifest.get("jobs", {}).items()}, **(jobs or {})}
    uartlogs = dict(uartlogs or {})
    bundle = Path(manifest["bundle"])
    expected_capture = manifest.get("capture_sha256")
    if expected_capture and _capture_sha256(bundle) != expected_capture:
        raise SystemExit("capture bundle changed since the paired FireSim images were built")
    for row in manifest["legs"].values():
        for name, sidecar in (row.get("routing_sidecars") or {}).items():
            path = Path(sidecar["path"])
            if not path.is_file() or _sha256(path) != sidecar["sha256"]:
                raise SystemExit(f"routing sidecar changed since the FireSim image was built: {name}")
    refs = _references(bundle)
    legs: dict[str, Any] = {}
    for leg, row in manifest["legs"].items():
        uart_path, results_dir = None, None
        if leg in uartlogs:
            uart_path = Path(uartlogs[leg])
            results_dir = uart_path.parent
        elif leg in jobs:
            results_dir = results_dir_for_job(jobs[leg], manifest["workload"], host)
            evidence = job_dir / "firesim" / leg / "uartlog"
            uart_path = evidence if evidence.is_file() else (results_dir / "uartlog" if results_dir else None)
        if uart_path is None or not uart_path.is_file():
            legs[leg] = {"status": "not_run", "problems": [f"no uartlog for the {leg} leg (pass --job or --uartlog)"]}
            continue
        uart = uart_path.read_text(errors="replace")
        graded = grade_leg(
            uart, row=row, refs=refs, hw_config=manifest["hw_config"], hw_summary=read_hw_summary(results_dir)
        )
        # The ELF must still be the one that was planned: a rebuild in place since then would make the
        # recorded hash describe bytes that never ran.
        if Path(row["elf"]).is_file() and _sha256(Path(row["elf"])) != row.get("elf_sha256"):
            graded["problems"].append("the leg's ELF changed on disk since it was planned")
            graded["status"] = "graded_with_problems"
        legs[leg] = {
            "elf": row["elf"],
            "elf_sha256": row.get("elf_sha256"),
            "queue_job": jobs.get(leg),
            "uartlog": str(uart_path),
            "unit_instruction_counts": row.get("unit_instruction_counts"),
            **graded,
        }
    result = _result(manifest, legs)
    result_path = job_dir / "firesim" / "results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    if result["all_verified"] and manifest["hw_config"] == "alveo_u250_firesim_kodiak_opu_1core":
        sidecars = manifest["legs"].get("device", {}).get("routing_sidecars") or {}
        if "opu_signatures.json" not in sidecars:
            # Older single-operator plans lack the immutable routing numerator. Preserve their
            # valid correctness grade and state why no utilization can be calculated.
            result["utilization_unavailable"] = "the plan has no SHA-checked OPU signatures sidecar"
        else:
            # Calculate from the saved, fully graded pair and SHA-checked routing sidecar. Keep the
            # capacity and occupancy semantics in one place: opu_utilization.py owns both.
            cmd = [PY, str(Path(__file__).with_name("opu_utilization.py")), str(job_dir)]
            measured = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            if measured.returncode:
                raise RuntimeError(f"OPU utilization calculation failed: {measured.stderr[-1000:]}")
            utilization = json.loads(measured.stdout)
            (job_dir / "firesim" / "utilization.json").write_text(json.dumps(utilization, indent=2) + "\n")
            result["utilization"] = utilization
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    if measurement and any(leg.get("cycles") for leg in legs.values()):
        result["measurement"] = str(_write_measurement(manifest, result))
    return result


def _result(manifest: dict[str, Any], legs: dict[str, Any]) -> dict[str, Any]:
    from merlin.common import provenance

    tars = {leg.get("bitstream_tar") for leg in legs.values() if leg.get("bitstream_tar")}
    artifact = _bitstream_artifact(next(iter(tars)) if len(tars) == 1 else None)
    if len(tars) > 1:
        artifact = {"matched": False, "why": f"the legs ran on different bitstream tars: {sorted(tars)}"}
    pins = {}
    for pin in artifact.get("built_from", ()):
        try:
            pins[pin] = provenance.verify(pin)
        except Exception:  # noqa: BLE001 -- an unverifiable pin is recorded as absent, not raised
            continue
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "model": manifest["name"],
        "bundle": manifest["bundle"],
        "board": manifest["board"],
        "bitstream": manifest["hw_config"],
        "harts": manifest["harts"],
        **legs,
        "provenance": provenance.record(pins=pins, extra={"bitstream_artifact": artifact}),
    }
    dev, ctl = legs.get("device", {}), legs.get("control", {})
    if dev.get("cycles") and ctl.get("cycles"):
        result["speedup_control_over_device"] = ctl["cycles"] / dev["cycles"]
    result["all_verified"] = bool(legs) and all(leg.get("status") == "verified" for leg in legs.values())
    return result


def _write_measurement(manifest: dict[str, Any], result: dict[str, Any]) -> Path:
    from merlin.common.artifacts import new_measurement

    # substrate = firesim_<bitstream>, the measurement layout's own convention, so this lands beside
    # every other number measured on the same design.
    mdir = new_measurement(f"firesim_{manifest['hw_config']}", manifest["name"], "device_vs_control")
    (mdir.path / "perf_results.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return mdir.path


def summary(result: dict[str, Any]) -> str:
    lines = [f"== {result['model']} on {result['bitstream']} ({result['harts']} hart(s))"]
    for leg in LEGS:
        row = result.get(leg)
        if not row:
            continue
        grade = row.get("grade") or {}
        lines.append(
            f"  {leg:8} {row.get('status'):21} cycles={row.get('cycles')}  w8a8_cos={grade.get('w8a8_cos', 'n/a')}"
            f"  w8a8_max_rel={grade.get('w8a8_max_rel', 'n/a')}  hw={row.get('hw_config')}"
            f"  wall={row.get('fpga_wall_s')}s @ {row.get('effective_mhz')} MHz"
        )
        for problem in row.get("problems", ()):
            lines.append(f"           problem: {problem}")
    if result.get("speedup_control_over_device"):
        lines.append(f"  device speedup over control: {result['speedup_control_over_device']:.3f}x")
    if utilization := result.get("utilization"):
        lines.append(
            "  OPU utilization: "
            f"routed MAC share={utilization['routed_mac_share']:.1%} "
            f"static tile fill={utilization['static_tile_fill']:.1%} "
            f"whole-inference effective peak={utilization['whole_inference_effective_peak_fraction']:.4%} "
            "array-active occupancy=unmeasured"
        )
    elif reason := result.get("utilization_unavailable"):
        lines.append(f"  OPU utilization: unavailable ({reason})")
    art = result["provenance"].get("bitstream_artifact", {})
    lines.append(
        f"  bitstream artifact: {art.get('name', '?')} ok={art.get('ok')}"
        if art.get("matched")
        else f"  bitstream artifact: UNPINNED ({art.get('why')})"
    )
    if result.get("measurement"):
        lines.append(f"  measurement: {result['measurement']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- legs


def build_legs(args) -> Path:
    """Spike-gate exact merlin-compile images, then plan their FireSim jobs."""
    from merlin.common.artifacts import recaptures_dir
    from merlin.common.paths import artifacts_dir
    from merlin.runtime.backends import output_identity

    brd = _board(args.board)
    bundle = recaptures_dir() / args.bundle
    if not (bundle / "model.mlir").is_file():
        raise SystemExit(f"{bundle} is not a capture bundle; capture it first (run.sh capture <model>)")
    if not (bundle / "golden_w8a8.npy").is_file():
        raise SystemExit(f"{bundle}: FireSim requires the model's W8A8 reference before building")
    capture_path = bundle.resolve()
    capture_sha256 = _capture_sha256(capture_path)
    model = args.bundle.removesuffix("_int8_full")
    job_base = artifacts_dir() / "probes" / brd.name / f"{model}_firesim_h{args.harts}"
    job_base.mkdir(parents=True, exist_ok=True)
    old_plan = job_base / MANIFEST
    if old_plan.is_file():
        archive = job_base / f"firesim_job.previous_{time.time_ns()}.json"
        old_plan.rename(archive)
        print(f"[firesim] archived prior plan at {archive}; it cannot be submitted as this build", flush=True)
    # A queued FireSim job stages its ELF later, when the daemon gets the FPGA. Never overwrite an ELF
    # a previous queue request may still reference, even if this build fails its spike gate.
    job_dir = job_base / f"run_{time.time_ns()}"
    package = Path(args.package)
    if not package.is_absolute():
        package = REPO / "out" / "artifacts" / "targets" / package
    legs = {}
    gate = getattr(args, "gate", "twin")
    if gate not in ("twin", "independent-rvv"):
        raise SystemExit(f"unknown FireSim gate: {gate}")
    for leg in LEGS:
        out = job_dir / leg
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            PY,
            "-m",
            "merlin.compile_cli",
            "--workload",
            model,
            "--dtype",
            "int8",
            "--target",
            "rvv",
            "--board",
            brd.name,
            "--package",
            str(package),
            "--harts",
            str(args.harts),
            "--run",
            "spike",
            "--verify",
            "--json",
            "--no-capture",
            "--timeout",
            str(args.spike_timeout),
        ]
        if leg == "device":
            cmd += [
                "--matrix-unit",
                args.matrix_unit,
                "--matrix-config",
                args.matrix_config,
                "--matrix-support-target",
                args.matrix_support_target,
            ]
            if args.matrix_kernel:
                cmd += ["--matrix-kernel", args.matrix_kernel]
            if gate == "independent-rvv":
                cmd += ["--matrix-gate", "shipped"]
        print("$ " + shlex.join(str(c) for c in cmd), flush=True)
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        marker = "\n{\n"
        try:
            result = json.loads(proc.stdout[proc.stdout.rindex(marker) + 1 :])
        except ValueError as exc:
            raise SystemExit(
                f"{leg} leg: merlin-compile returned no result (exit {proc.returncode}): "
                f"{(proc.stderr or proc.stdout)[-1500:]}"
            ) from exc
        matrix = result.get("matrix") or {}
        shipped = matrix.get("shipped_on_simulator") or {}
        if proc.returncode or result.get("status") != "verified" or result.get("verify", {}).get("tier_ok") != "w8a8":
            raise SystemExit(f"{leg} leg failed its spike W8A8 gate: {result.get('status')}; {proc.stderr[-1000:]}")
        if leg == "device":
            device_gate_ok = (
                shipped.get("status") == "verified"
                and shipped.get("verify", {}).get("tier_ok") == "w8a8"
                and matrix.get("unit_instruction_counts")
            )
            if gate == "twin":
                device_gate_ok = device_gate_ok and shipped.get("bit_exact_vs_twin") is True
            if not device_gate_ok:
                raise SystemExit(
                    "device leg: shipped OPU image did not pass its spike model and "
                    + ("bit-exact twin gate" if gate == "twin" else "W8A8 gate")
                )
        identity = shipped.get("output_identity") if leg == "device" else result.get("output_identity")
        if not identity or not identity.get("out_hash") or not identity.get("prefix_sha256"):
            raise SystemExit(f"{leg} leg: spike gate has no complete output identity")
        source = Path(result["binary"])
        elf = out / "zephyr.elf"
        shutil.copy2(source, elf)
        if _sha256(source) != _sha256(elf):
            raise SystemExit(f"{leg} leg: staged ELF differs from the spike-gated image")
        routing_sidecars = None
        if leg == "device":
            routing_source = Path(matrix["routing_report"])
            signature_source = routing_source.parent / "opu_signatures.json"
            if not routing_source.is_file() or not signature_source.is_file():
                raise SystemExit("device leg: exact image has no routing report or OPU signatures")
            routing_dir = out / "routing"
            routing_dir.mkdir()
            routing_sidecars = {}
            for source_sidecar in (routing_source, signature_source):
                staged = routing_dir / source_sidecar.name
                shutil.copy2(source_sidecar, staged)
                if _sha256(source_sidecar) != _sha256(staged):
                    raise SystemExit(f"device leg: staged {source_sidecar.name} differs from the image's sidecar")
                routing_sidecars[source_sidecar.name] = {"path": str(staged), "sha256": _sha256(staged)}
        counts = _unit_counts(elf, args.matrix_support_target, args.matrix_unit)
        if leg == "device" and not counts:
            raise SystemExit("device leg: staged ELF has no matrix-unit instructions")
        if leg == "control" and counts:
            raise SystemExit(f"control leg: staged ELF contains matrix-unit instructions: {counts}")
        row = {
            "leg": leg,
            "elf": str(elf),
            "build_hash": result["build_hash"],
            "board": brd.name,
            "harts": args.harts,
            "vlen": brd.vlen,
            "unit_instruction_counts": counts,
            "routing_sidecars": routing_sidecars,
            "spike_output_identity": identity,
            "spike_gate": {
                "method": gate if leg == "device" else "w8a8",
                "status": result["status"],
                "verify": result["verify"],
                "shipped_on_simulator": shipped if leg == "device" else None,
            },
        }
        (out / "leg.json").write_text(json.dumps(row, indent=2) + "\n")
        legs[leg] = row
    if bundle.resolve() != capture_path or _capture_sha256(capture_path) != capture_sha256:
        raise SystemExit("capture bundle changed while building the paired FireSim images")
    if gate == "independent-rvv":
        same = output_identity.same_output(
            legs["device"]["spike_output_identity"], legs["control"]["spike_output_identity"]
        )
        if same is not True:
            raise SystemExit(f"device/control exact-board Spike outputs do not match bit for bit: {same}")
        legs["device"]["spike_gate"]["independent_rvv_bit_exact"] = True
        (job_dir / "device" / "leg.json").write_text(json.dumps(legs["device"], indent=2) + "\n")
    plan(
        job_dir,
        board=brd.name,
        name=model,
        bundle=bundle,
        legs=legs,
        harts=args.harts,
        timeout_s=args.timeout,
        capture_sha256=capture_sha256,
    )
    return job_dir


def _unit_counts(elf: Path, support_target: str, unit: str) -> dict[str, int]:
    """Audit the staged image against instruction encodings derived from its selected provider."""
    from merlin.kernels.decode import opu as opu_audit
    from merlin.targetgen.plugins import load_declared

    provider = load_declared(support_target, "matrix_lowering")
    encodings = provider.derive_encodings(provider.load_contract(unit)).encodings
    return {k: int(v) for k, v in sorted((opu_audit.audit_object(elf, encodings).counts or {}).items()) if v}


# --------------------------------------------------------------------------------------------- cli


def _pairs(values: list[str] | None, kind) -> dict:
    out = {}
    for item in values or ():
        leg, sep, value = item.partition("=")
        if not sep or leg not in LEGS:
            raise SystemExit(f"expected LEG=VALUE with LEG one of {LEGS}, got {item!r}")
        out[leg] = kind(value)
    return out


def _after_plan(job_dir: Path, args) -> int:
    manifest = load(job_dir)
    print(describe(job_dir, manifest, host_config(), priority=args.priority, timeout_s=args.timeout), flush=True)
    if not args.submit:
        print("\n  (not submitted: pass --submit to run these on the FPGA)")
        return 0
    submit(job_dir, priority=args.priority, timeout_s=args.timeout)
    result = collect(job_dir)
    print(summary(result))
    return 0 if result["all_verified"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _queue_flags(p):
        p.add_argument("--submit", action="store_true", help="actually run on the FPGA (default: print only)")
        p.add_argument("--priority", type=int, default=5, choices=PRIORITIES)
        p.add_argument("--timeout", type=int, default=None, help="RUNNING-phase limit per leg, seconds")

    lg = sub.add_parser("legs", help="build + audit device/control legs of a whole model, then plan")
    lg.add_argument("bundle", help="capture bundle name under out/artifacts/recaptures/ (e.g. spectformer_int8_full)")
    lg.add_argument("--board", required=True)
    lg.add_argument("--package", required=True, help="RVV package under out/artifacts/targets/")
    lg.add_argument("--harts", type=int, default=1)
    lg.add_argument("--matrix-unit", required=True)
    lg.add_argument("--matrix-config", required=True)
    lg.add_argument("--matrix-support-target", required=True)
    lg.add_argument("--matrix-kernel", help="the device leg's matrix kernel (registered name or C file)")
    lg.add_argument("--spike-timeout", type=int, default=14400, help="spike gate limit per image, seconds")
    lg.add_argument(
        "--gate",
        choices=("twin", "independent-rvv"),
        default="twin",
        help="matrix correctness gate: scalar twin (default), or exact-board OPU Spike vs independent RVV",
    )
    _queue_flags(lg)
    sb = sub.add_parser("submit", help="print (or with --submit, run) the queue jobs of a job directory")
    sb.add_argument("job_dir", type=Path)
    _queue_flags(sb)
    co = sub.add_parser("collect", help="grade finished runs of a job directory")
    co.add_argument("job_dir", type=Path)
    co.add_argument("--job", action="append", metavar="LEG=ID", help="queue job id of a leg")
    co.add_argument(
        "--uartlog", action="append", metavar="LEG=PATH", help="a leg's uartlog, for a run outside the queue"
    )
    co.add_argument("--no-measurement", action="store_true", help="do not write out/artifacts/measurements/")
    args = ap.parse_args(argv)

    if args.cmd == "legs":
        args.timeout = args.timeout or 3600
        return _after_plan(build_legs(args), args)
    if args.cmd == "submit":
        return _after_plan(args.job_dir.resolve(), args)
    result = collect(
        args.job_dir,
        jobs=_pairs(args.job, int),
        uartlogs=_pairs(args.uartlog, Path),
        measurement=not args.no_measurement,
    )
    print(summary(result))
    return 0 if result["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

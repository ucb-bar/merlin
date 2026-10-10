"""Target-bound, operator-selected Phase 1 oracle timing input.

This record is produced only after a real cert-tier observation. Its path is a
resource owned by the selected target, never an alias through shared scripts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

TIMING_SCHEMA = "merlin.oracle-timing.v2"


def _digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def observed_seconds(record: dict) -> float:
    """Elapsed host wall time for the cert grade, not simulated cycles or kernel cost."""
    if record.get("schema") == TIMING_SCHEMA:
        seconds = record.get("per_capsule_s")
    elif any(key in record for key in ("schema", "engine", "per_capsule_s")):
        raise ValueError("oracle timing record has an unsupported observation format")
    else:
        seconds = record.get("verilator_per_capsule_s")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError("oracle timing record has no positive finite observation")
    try:
        finite = math.isfinite(seconds)
    except OverflowError:
        finite = False
    if not finite or seconds <= 0:
        raise ValueError("oracle timing record has no positive finite observation")
    return float(seconds)


def _path_digest(path: Path) -> str:
    return hashlib.sha256(str(path.absolute()).encode("utf-8")).hexdigest()


def barrier_timing_identity(tier: dict) -> dict | None:
    """Project only the actual passed tier's engine/digests into agent-visible feedback.

    No requested engine substitutes for absent provenance. Private paths and
    receipt/tool details remain in the existing host-owned result.
    """
    from merlin.targetgen.rtl_engine_policy import ENGINE_PRIORITY

    if not isinstance(tier, dict) or tier.get("status") != "pass":
        return None
    engine, provenance = tier.get("engine"), tier.get("sim_provenance")
    if engine not in ENGINE_PRIORITY or not isinstance(provenance, dict):
        return None
    binary, digest = provenance.get("binary"), provenance.get("sha256")
    if provenance.get("engine") != engine or not isinstance(binary, str) or not Path(binary).is_absolute():
        return None
    if not _digest(digest):
        return None
    return {"engine": engine, "simulator_sha256": digest, "simulator_path_sha256": _path_digest(Path(binary))}


def _binding_identity(binding: dict) -> dict:
    return {
        "engine": binding["engine"],
        "simulator_sha256": binding["simulator_sha256"],
        "simulator_path_sha256": _path_digest(Path(binding["simulator_path"])),
    }


def barrier_measurements(tier: dict, *, cert_tier: bool) -> dict:
    """Public observations from the actual barrier, with no requested-engine substitute."""
    if not isinstance(tier, dict):
        return {}
    record = {}
    cycles = tier.get("cycles")
    if isinstance(cycles, int) and not isinstance(cycles, bool) and cycles > 0:
        record["barrier_cycles"] = cycles
    identity = barrier_timing_identity(tier) if cert_tier else None
    if identity is not None:
        record["barrier_timing_identity"] = identity
    return record


def _same(left, right) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _selected_engine(target: str) -> str:
    from merlin.targetgen.oracle_policy import select_chipyard_engine

    try:
        engine = select_chipyard_engine(target).get("engine")
    except Exception as exc:  # unavailable metadata cannot validate an observation
        raise ValueError("oracle timing has no selected engine") from exc
    if not isinstance(engine, str) or not engine:
        raise ValueError("oracle timing has no selected engine")
    return engine


def _ordinary_file(path: Path) -> Path:
    path = path.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise ValueError("oracle timing selection is absent or contains a symlink")
    return path


def timing_path(resources: Path, target: str) -> Path:
    if not target or not all(char.isascii() and (char.isalnum() or char in "_-") for char in target):
        raise ValueError("timing target must be a nonempty path-safe identity")
    return Path(resources) / f".oracle_timing.{target}.json"


def requires_chipyard_timing(descriptor: Path | None) -> bool:
    """Require byte-bound timing for Chipyard or an unclassifiable descriptor (fail closed)."""
    if descriptor is None:
        return True
    from merlin.targetgen.target_experiment import load_target_experiment

    try:
        return load_target_experiment(descriptor).sim_via == "chipyard"
    except (OSError, ValueError):
        return True


def read_timing(path: Path, *, target: str) -> dict:
    """Read a measured record, refusing aliases and foreign/invalid observations."""
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"oracle timing path contains a symlink: {path}")
    if not path.is_file():
        raise ValueError(f"oracle timing record is absent: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"oracle timing record is unreadable: {path}") from exc
    if not isinstance(record, dict) or record.get("target") != target:
        raise ValueError(f"oracle timing record is not bound to target {target!r}: {path}")
    if not isinstance(record.get("config"), str) or not record["config"].strip():
        raise ValueError(f"oracle timing record has no simulator config: {path}")
    if not isinstance(record.get("measured_by"), str) or not record["measured_by"].strip():
        raise ValueError(f"oracle timing record has no measurement producer: {path}")
    digest = record.get("simulator_sha256")
    if not _digest(digest):
        raise ValueError(f"oracle timing record has no simulator byte identity: {path}")
    observed_seconds(record)
    return record


def selected_engine_binding(*, descriptor: Path, target: str) -> dict:
    """Resolve the ordinary selected engine; never substitute another timing engine.

    Strict gSIM receipts validate files and saved producer transcripts, not
    emitter-build, clock, counter, runtime or physical correspondence.
    """
    from merlin.common.digest import sha256_file
    from merlin.common.paths import ext_path
    from merlin.targetgen.target_experiment import (
        declared_vs_resolved_contract,
        load_capability_manifest,
        load_target_experiment,
    )

    if descriptor is None:
        raise ValueError("oracle timing has no selected target descriptor")
    try:
        selected = load_target_experiment(descriptor)
        if selected.target != target or selected.sim_via != "chipyard":
            raise ValueError("selected descriptor does not declare this target's Chipyard simulator")
        _, contract_path, agreement = declared_vs_resolved_contract(selected)
        if agreement != "agree" or contract_path is None:
            raise ValueError(f"selected target contract is not agreed and resolvable: {agreement}")
        config = (load_capability_manifest(target, contract_path=contract_path).contract.get("runtime") or {}).get(
            "rtl_sim_config"
        )
        if not isinstance(config, str) or not config.strip():
            raise ValueError("oracle timing has no selected simulator config")
        engine = _selected_engine(target)
        required = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip().lower()
        if required and engine != required:
            raise ValueError("oracle timing engine differs from the required engine")
        binding = {"target": target, "config": config, "engine": engine}
        if engine == "verilator":
            simulator = _ordinary_file(
                ext_path("chipyard") / "sims" / "verilator" / f"simulator-chipyard.harness-{config}"
            )
        elif engine == "gsim":
            from merlin.compile.model_execution_inputs import selected_firrtl
            from merlin.runtime.backends import base as backends
            from merlin.targetgen import gsim_emulator

            facts_path = os.environ.get("MERLIN_RTL_FACTS", "").strip()
            if not facts_path:
                raise ValueError("oracle timing has no explicitly selected RTL facts")
            facts = selected_firrtl(_ordinary_file(Path(facts_path)), target=target, config=config)
            backend = backends.get_backend(target)
            env_var = getattr(backend, "GSIM_EMU_ENV", None)
            exact, _reason = gsim_emulator.selected_firrtl_status(target, env_var=env_var, backend=backend)
            if not exact:
                raise ValueError("oracle timing cannot verify a strict gSIM receipt for the selected FIRRTL")
            model = gsim_emulator.resolve(target, env_var=env_var, backend=backend)
            receipt = model.receipt or {}
            if (
                not model.ok
                or model.flavour != "binary"
                or model.receipt_status != "bound"
                or receipt.get("schema_version") != gsim_emulator.STRICT_RECEIPT_SCHEMA
                or receipt.get("firrtl_sha256") != facts["firrtl_sha256"]
            ):
                raise ValueError("oracle timing has no strict gSIM receipt for the selected FIRRTL")
            simulator = _ordinary_file(model.path)
            getter = getattr(backend, f"{engine}_path", None)
            if not callable(getter) or _ordinary_file(Path(getter())) != simulator:
                raise ValueError("oracle timing backend selects different simulator bytes")
            if sha256_file(simulator) != model.digest:
                raise ValueError("oracle timing simulator changed during selection")
            # A frozen session may copy identical facts into its own snapshot.
            # Reopen that selected file and FIRRTL, retaining byte/config identity
            # without requiring the operator's original facts pathname.
            binding.update(
                rtl_facts={key: facts[key] for key in ("sha256", "firrtl_sha256", "target", "config")},
                receipt=receipt,
            )
        else:
            raise ValueError("oracle timing has no byte-bound support for the selected engine")
        binding.update(simulator_path=str(simulator), simulator_sha256=sha256_file(simulator))
    except (OSError, KeyError, TypeError, RuntimeError):
        raise ValueError("cannot verify selected simulator for oracle timing") from None
    return binding


def read_verified_timing(path: Path, *, descriptor: Path, target: str) -> dict:
    """Match timing to the current selected engine, config, binary and gSIM receipt/facts."""
    record = read_timing(path, target=target)
    # Legacy records state Verilator by their original field name. Refuse them
    # before resolving a different engine's artifacts, even if that engine is absent.
    engine = _selected_engine(target)
    recorded_engine = record.get("engine") if record.get("schema") == TIMING_SCHEMA else "verilator"
    if recorded_engine != engine:
        raise ValueError("oracle timing engine differs from the selected engine")
    binding = selected_engine_binding(descriptor=descriptor, target=target)
    if record["config"] != binding["config"]:
        raise ValueError("oracle timing config differs from the selected target contract")
    if record["simulator_sha256"] != binding["simulator_sha256"]:
        raise ValueError("measured simulator bytes changed")
    if record.get("schema") == TIMING_SCHEMA:
        if not _same(record.get("engine_binding"), binding) or not _same(
            record.get("barrier_timing_identity"), _binding_identity(binding)
        ):
            raise ValueError("oracle timing observation differs from the selected engine binding")
    return record


def write_observed_timing(
    path: Path,
    *,
    descriptor: Path,
    target: str,
    before: dict,
    elapsed_s: float,
    report: dict,
    measured_capsule: str,
    measured_by: str,
    reference: dict | None = None,
) -> dict:
    """Publish the existing readiness producer's actual complete one-capsule L3 observation.

    The caller brackets its real grade with a monotonic clock. This record sizes
    host timeouts only; it grants no numerical, hardware or performance authority.
    """
    observed_seconds({"schema": TIMING_SCHEMA, "per_capsule_s": elapsed_s})
    current = selected_engine_binding(descriptor=descriptor, target=target)
    if not _same(before, current):
        raise ValueError("selected timing engine changed during the observation")
    rows = report.get("per_capsule") if isinstance(report, dict) else None
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or report.get("all_pass") is not True
        or type(report.get("n_capsules")) is not int
        or report["n_capsules"] != 1
        or report.get("sim") != current["engine"]
        or type(report.get("_readiness_returncode")) is not int
        or report["_readiness_returncode"] != 0
    ):
        raise ValueError("oracle timing has no complete passed selected-engine observation")
    row = rows[0]
    if (
        not isinstance(row, dict)
        or row.get("capsule") != measured_capsule
        or row.get("pass") is not True
        or row.get("barrier_tier") != "L3"
        or row.get("barrier_status") != "pass"
        or not _same(row.get("barrier_timing_identity"), _binding_identity(current))
    ):
        raise ValueError("oracle timing has no actual passed barrier matching the selected engine")
    record = {
        "schema": TIMING_SCHEMA,
        "target": target,
        "config": current["config"],
        "engine": current["engine"],
        "per_capsule_s": elapsed_s,
        "simulator_sha256": current["simulator_sha256"],
        "engine_binding": current,
        "barrier_timing_identity": row["barrier_timing_identity"],
        "measured_capsule": measured_capsule,
        "measured_by": measured_by,
        "measurement_scope": "complete one-capsule cert grade host wall time",
    }
    if reference is not None:
        record["reference"] = reference
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("oracle timing output contains a symlink")
    path.write_text(json.dumps(record, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return read_verified_timing(path, descriptor=descriptor, target=target)

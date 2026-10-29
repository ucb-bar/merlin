"""Original frozen model policy and selected functional-engine file identities.

These data readers execute no program and issue no numerical or timing role.
The native evaluator retains its historical wrappers and selection refusal.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


def _policy_bytes(path: Path):
    """Pin the bounded ordinary bytes actually parsed, including ancestor custody."""
    from merlin.compile.analysis.artifact_reader import identity, open_ordinary

    from .native_model_execution import NativeModelExecutionError

    fd = open_ordinary(path)
    try:
        before = os.fstat(fd)
        if not 0 < before.st_size <= 4 << 20:
            raise NativeModelExecutionError("selected instruction policy exceeds its byte bound")
        raw = os.pread(fd, before.st_size, 0)
        if len(raw) != before.st_size or identity(before) != identity(os.fstat(fd)):
            raise NativeModelExecutionError("selected instruction policy changed while reading")
        return raw, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    finally:
        os.close(fd)


def _frozen_corpus_instruction_policy(*, target: str) -> dict[str, Any]:
    """Read only the original corpus's explicit global static prohibitions.

    Model coverage sidecars and required numerical tiers remain the separate
    model reader's responsibility. No absent or unresolved policy grants roles.
    """
    import yaml

    from merlin.targetgen.target_experiment import load_target_experiment

    from .native_model_execution import NativeModelExecutionError

    descriptor_name = os.environ.get("MERLIN_TARGET_EXPERIMENT")
    if not descriptor_name:
        raise NativeModelExecutionError("selected reviewed target experiment is absent")
    descriptor = Path(descriptor_name).absolute()
    if descriptor.resolve(strict=True) != descriptor:
        raise NativeModelExecutionError("selected experiment is not an ordinary file")
    _, descriptor_pin = _policy_bytes(descriptor)
    experiment = load_target_experiment(descriptor)
    if experiment.target != target or experiment.descriptor_sha256 != descriptor_pin["sha256"]:
        raise NativeModelExecutionError("selected experiment names a different target")
    manifest = experiment.capsule_corpus.parent / "MANIFEST.yaml"
    raw, manifest_pin = _policy_bytes(manifest)
    corpus = yaml.safe_load(raw)
    instruction = corpus.get("instruction_policy") if isinstance(corpus, dict) else None
    roles = instruction.get("prohibited_instruction_roles") if isinstance(instruction, dict) else None
    if (
        not isinstance(instruction, dict)
        or instruction.get("status") != "resolved"
        or not isinstance(roles, list)
        or any(not isinstance(role, str) or not role for role in roles)
    ):
        raise NativeModelExecutionError("frozen prohibited-role policy is unresolved")
    if _policy_bytes(descriptor)[1] != descriptor_pin:
        raise NativeModelExecutionError("selected experiment changed while reading policy")
    return {"descriptor": descriptor_pin, "manifest": manifest_pin, "prohibited_instruction_roles": roles}


def _frozen_model_policy(capsule_dir: Path, *, target: str, source_bound_groups: bool = False) -> dict[str, Any]:
    """Join the global policy with unchanged model coverage and tier obligations."""
    import yaml

    from .native_model_execution import NativeModelExecutionError, _digest

    selected = _frozen_corpus_instruction_policy(target=target)
    declaration_path = capsule_dir / "capsule.yaml"
    declaration = yaml.safe_load(declaration_path.read_text(encoding="utf-8"))
    coverage_path = capsule_dir / "expected_instruction_coverage.yaml"
    # Direct-IR model slices carry their reviewed instruction requirements in
    # capsule.yaml; unlike captured whole models, their native intake never
    # writes a duplicate coverage sidecar. Reopen that original declaration,
    # rather than manufacturing a file or removing any instruction obligation.
    declaration_coverage = (
        source_bound_groups is True
        and isinstance(declaration, dict)
        and declaration.get("kind") == "model_slice"
        and not coverage_path.exists()
        and not coverage_path.is_symlink()
    )
    if declaration_coverage:
        coverage_path = declaration_path
        coverage = declaration.get("expected")
    else:
        coverage = yaml.safe_load(coverage_path.read_text(encoding="utf-8"))
    if not all(isinstance(doc, dict) for doc in (declaration, coverage)):
        raise NativeModelExecutionError("frozen model instruction policy is malformed")
    expected = (declaration.get("expected") or {}).get("instruction_classes")
    from merlin.targetgen.public_program_timing import is_public_program

    public_zero_device_demand = (
        is_public_program(declaration)
        and (declaration.get("semantic") or {}).get("must_accelerate") is False
        and expected == []
    )
    if (
        not isinstance(expected, list)
        or (not expected and not public_zero_device_demand and not source_bound_groups)
        or any(not isinstance(name, str) or not name for name in expected)
        or coverage.get("instruction_classes") != expected
    ):
        raise NativeModelExecutionError("frozen instruction coverage or prohibited-role policy is unresolved")
    if source_bound_groups and (
        type(source_bound_groups) is not bool or declaration.get("kind") not in {"model", "model_slice"}
    ):
        raise NativeModelExecutionError("source-bound group instruction policy requires an original model capsule")
    required = declaration.get("required_oracle_tiers")
    if (
        not isinstance(required, list)
        or not required
        or len(required) != len(set(required))
        or any(tier not in {"L0", "L1", "L2", "L3"} for tier in required)
    ):
        raise NativeModelExecutionError("frozen required oracle tiers are malformed")
    return {
        **selected,
        "capsule_declaration": _digest(declaration_path),
        "instruction_coverage": _digest(coverage_path),
        "required_tiers": required,
        "expected_instruction_classes": expected,
        **({"instruction_coverage_origin": "capsule_declaration.expected"} if declaration_coverage else {}),
        **({"public_zero_device_demand": True} if public_zero_device_demand else {}),
        **({"instruction_class_scope": "global reviewed requirements plus independently mandatory "
                                     "source-bound group work"}
           if source_bound_groups else {}),
    }


def functional_engine_bytes(backend, *, target: str):
    """Reopen the original selected functional binary/extension; run nothing."""
    from .native_model_execution import NativeModelExecutionError, _digest

    path_fn = getattr(backend, "spike_path", None)
    extension_fn = getattr(backend, "spike_extension", None)
    if (
        not callable(path_fn)
        or not callable(extension_fn)
        or not callable(getattr(backend, "run_elf", None))
        or not backend.available("spike")
    ):
        raise NativeModelExecutionError("selected backend has no available inspectable Spike ELF runner")
    binary = Path(path_fn()).resolve(strict=True)
    if not binary.is_file() or binary.is_symlink():
        raise NativeModelExecutionError("selected Spike executable is absent or indirect")
    flags, library_dir = extension_fn()
    if not isinstance(flags, (tuple, list)) or not isinstance(library_dir, Path):
        raise NativeModelExecutionError("selected Spike extension invocation is malformed")
    names = [part.split("=", 1)[1] for part in flags if part.startswith("--extension=")]
    explicit = [part.split("=", 1)[1] for part in flags if part.startswith("--extlib=")]
    if len(names) != 1 or len(explicit) > 1:
        raise NativeModelExecutionError("selected Spike extension identity is ambiguous")
    if explicit:
        extension = Path(explicit[0]).resolve(strict=True)
    else:
        if not library_dir.is_dir() or library_dir.is_symlink():
            raise NativeModelExecutionError("selected Spike extension library directory is absent")
        matches = list(library_dir.glob(f"lib{names[0]}.so"))
        if len(matches) != 1:
            raise NativeModelExecutionError("selected default Spike extension bytes are not unique")
        extension = matches[0].resolve(strict=True)
    if not extension.is_file() or extension.is_symlink():
        raise NativeModelExecutionError("selected Spike extension is absent or indirect")
    citation = {
        "engine": "spike",
        "target": target,
        "binary": _digest(binary),
        "extension": _digest(extension),
        "flags": list(flags),
        "library_dir": str(library_dir),
    }

    def revalidate() -> None:
        if (
            Path(path_fn()).resolve(strict=True) != binary
            or _digest(binary) != citation["binary"]
            or tuple(extension_fn()[0]) != tuple(flags)
            or _digest(extension) != citation["extension"]
        ):
            raise NativeModelExecutionError("selected L2 engine or extension changed during execution")

    return backend, citation, revalidate


def audit_rtl_tier(native, completed_dispatch):
    """The original full RTL tier verifier, never called by numerical screening."""
    from collections.abc import Mapping

    from .native_model_execution import NativeModelExecutionError, _digest, _pinned_native_file

    try:
        rtl = (native.get("tiers") or {}).get("L3") or {}
        elf = _pinned_native_file(native, "elf")
        console = _pinned_native_file(native, "console")
        provenance = native.get("simulator_provenance") or {}
        if (
            not isinstance(completed_dispatch, Mapping)
            or completed_dispatch.get("status") != "verified"
            or completed_dispatch.get("numeric_status") not in {"pass", "fail"}
            or rtl.get("status") != completed_dispatch["numeric_status"]
            or completed_dispatch.get("linked_elf") != _digest(elf)
            or completed_dispatch.get("console") != _digest(console)
            or rtl.get("elf") != _digest(elf)
            or rtl.get("console") != _digest(console)
            or rtl.get("engine") != native.get("simulator")
            or rtl.get("engine_citation") != provenance.get("citation")
            or rtl.get("selection") != provenance.get("selection")
            or rtl.get("numeric") != native.get("numeric")
            or (native.get("numeric") or {}).get("status") != completed_dispatch["numeric_status"]
        ):
            raise NativeModelExecutionError("selected RTL tier lacks independently verified candidate completion")
        result = {
            "status": completed_dispatch["numeric_status"],
            "engine": rtl.get("engine"),
            "elf": rtl["elf"],
            "console": rtl["console"],
            "completed_dispatch": completed_dispatch,
        }
        # Fidelity comes from the same selected engine report that the completed
        # dispatch auditor revalidated, never from the L3 label. No selected
        # fidelity means no RTL or cycle-accuracy claim in the score.
        fidelity = (provenance.get("selection") or {}).get("fidelity")
        if isinstance(fidelity, str) and fidelity:
            from merlin.targetgen.rtl_engine_policy import ELABORATED_RTL

            result["fidelity"] = fidelity
            result["derived_from_rtl"] = fidelity == ELABORATED_RTL
    except Exception as exc:  # noqa: BLE001 -- no verified RTL completion is not an L3 pass
        result = {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}
    return result


def finish_numerical_execution(record):
    """Publish only the actually completed same-ELF functional receipt."""
    functional = record.get("tiers", {}).get("L2") or {}
    if functional.get("status") not in ("pass", "fail"):
        return record
    record.update(
        console=functional["console"],
        numeric=functional["numeric"],
        status="numeric_match_diagnostic" if functional["status"] == "pass" else "numeric_mismatch_diagnostic",
        scope="same candidate Spike numerical screen; timing UNKNOWN; required deeper tiers pending",
        simulator_provenance={"citation": functional["engine_citation"]},
    )
    return record


def attach_numerical_screen(result, capsule, check):
    """Expose observed numerical tiers without promoting an incomplete grade."""
    from merlin.targetgen.public_program_timing import numerical_screen_requested

    if not numerical_screen_requested(result, capsule):
        return
    candidate = check.get("candidate_native_model_check") or {}
    tiers = (candidate.get("candidate_required_tiers") or {}).get("tiers") or {}
    native = result.get("candidate_native_execution") or {}
    clean = (
        native.get("simulator") == "spike"
        and native.get("oracle_tier_ceiling") == "L2"
        and set(candidate.get("violations") or []) <= {"candidate_required_tiers_unverified"}
        and all((tiers.get(tier) or {}).get("status") == "pass" for tier in ("L0", "L1", "L2"))
    )
    result["candidate_numerical_screen"] = {
        "status": "pass" if clean else "unverified",
        "ceiling": "L2",
        "timing_cycles": None,
        "scope": "numerics only; full grade remains incomplete",
    }
    result["tiers"] = {
        tier: {
            "tier": tier,
            "status": (
                (tiers.get(tier) or {}).get("status", "unverified")
                if clean or (tiers.get(tier) or {}).get("status") == "fail"
                else "unverified"
            ),
            "derived_from_rtl": False,
            "cycle_accurate": False,
            "candidate_evidence": tiers.get(tier),
        }
        for tier in ("L0", "L1", "L2")
    }
    result["tiers"]["L3"] = {
        "tier": "L3",
        "status": "pending",
        "derived_from_rtl": False,
        "cycle_accurate": False,
        "scope": "outside explicit L2 numerical ceiling",
    }
    if clean:
        result["numeric"] = native["numeric"]
    if check.get("status") == "pass":
        check["status"] = "incomplete"
        check.setdefault("violations", []).append("candidate_required_tiers_unverified")


def observed_execution(backend, elf, *, simulator, citation, directory, timeout, capture_kwargs):
    """Observe the original Python dispatch; never invent a subprocess record."""
    from merlin.common import invocation_record as I

    deps = []
    for role in ("binary", "extension"):
        pin = citation.get(role)
        if isinstance(pin, dict) and pin.get("path"):
            deps.append(Path(pin["path"]))
    if citation.get("path"):
        deps.append(Path(citation["path"]))
    for key in ("receipt_path", "firrtl_path"):
        value = (citation.get("receipt") or {}).get(key)
        if value:
            deps.append(Path(value))
    with I.observe_call(
        directory,
        stage="execution",
        function=backend.run_elf,
        arguments={"simulator": simulator, "timeout": timeout, **capture_kwargs},
        inputs=(Path(elf),),
        dependencies=tuple(deps),
    ) as observation:
        console = backend.run_elf(elf, simulator=simulator, timeout=timeout, **capture_kwargs)
        observation.returned(stdout=console)
    return console, {"path": str(observation.path), "sha256": hashlib.sha256(observation.path.read_bytes()).hexdigest()}

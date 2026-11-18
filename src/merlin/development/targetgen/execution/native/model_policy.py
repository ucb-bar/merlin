"""Original frozen model policy and selected functional-engine file identities.

These data readers execute no program and issue no numerical or timing role.
The native evaluator retains its historical wrappers and selection refusal.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# ---- native_model_policy -----------------------------------------------------------------------


def _policy_bytes(path: Path):
    """Pin the bounded ordinary bytes actually parsed, including ancestor custody."""
    from merlin.analysis.observation.elf_artifacts import identity, open_ordinary
    from merlin.development.targetgen.execution.native.native_model_execution import NativeModelExecutionError

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

    from merlin.development.targetgen.execution.native.native_model_execution import NativeModelExecutionError
    from merlin.target.registry.descriptor_loading import load_target_experiment

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

    from merlin.development.targetgen.execution.native.native_model_execution import (
        NativeModelExecutionError,
        _digest,
    )

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
    from merlin.development.targetgen.execution.host.public_program_timing import is_public_program

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
        **(
            {
                "instruction_class_scope": "global reviewed requirements plus independently mandatory "
                "source-bound group work"
            }
            if source_bound_groups
            else {}
        ),
    }


def functional_engine_bytes(backend, *, target: str):
    """Reopen the original selected functional binary/extension; run nothing."""
    from merlin.development.targetgen.execution.native.native_model_execution import (
        NativeModelExecutionError,
        _digest,
    )

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

    from merlin.development.targetgen.execution.native.native_model_execution import (
        NativeModelExecutionError,
        _digest,
        _pinned_native_file,
    )

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
            from merlin.target.simulators.engine_policy import ELABORATED_RTL

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
    from merlin.development.targetgen.execution.host.public_program_timing import numerical_screen_requested

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
    from merlin.common.process import invocation_records as I

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


# ---- native_instruction_audit ------------------------------------------------------------------
# Original exact-ELF ISA audit shared by explicitly selected execution ABIs.
#
# This supplies no source arithmetic, placement, numerical or timing verdict.


def audit_native_instruction_tier(native, *, target, module, policy, allow_empty_instructions=False):
    from merlin.development.targetgen.execution.native.native_model_execution import (
        NativeModelExecutionError,
        _digest,
        _pinned_native_file,
    )

    try:
        elf = _pinned_native_file(native, "elf")
        from merlin.analysis.whole_model.grading.isa_prohibition import scan_elf
        from merlin.common.target_data import interface as facts_interface
        from merlin.development.targetgen.grading import elf_lanes
        from merlin.runtime.execution.model_inputs import selected_firrtl
        from merlin.target.isa.rocc.decode import decode_module
        from merlin.target.rtl.facts import observed_facts

        selected = native.get("rtl_facts") or {}
        if (
            not isinstance(selected, Mapping)
            or not isinstance(selected.get("path"), str)
            or not isinstance(selected.get("config"), str)
            or selected_firrtl(selected["path"], target=target, config=selected["config"]) != selected
        ):
            raise NativeModelExecutionError("L1 lacks selected byte-bound RTL ISA facts")
        facts_path = Path(selected["path"])
        facts = json.loads(facts_path.read_text(encoding="utf-8"))
        table = facts_interface(facts.get("facts"), "funct_decode_table") or {}
        opcode = table.get("custom_opcode")
        names = {int(key) for key in (table.get("names") or {})}
        if type(opcode) is not int or not names:
            raise NativeModelExecutionError("selected ISA has no complete custom command table")
        blob = elf.read_bytes()
        sections = elf_lanes.executable_sections(blob)
        if not sections:
            raise NativeModelExecutionError("linked candidate ELF has no executable sections")
        custom_count = 0
        for _name, offset, size, address in sections:
            for _at, word in elf_lanes.instruction_words(blob[offset : offset + size], address):
                if word & 0x7F == opcode:
                    custom_count += 1
                    if word >> 25 not in names:
                        raise NativeModelExecutionError("linked candidate ELF has an unknown custom selector")
        with observed_facts(target, facts, path=facts_path):
            decoded = decode_module(module, target=target)
            instructions = decoded.get("instructions")
            if (
                not isinstance(instructions, list)
                or (not instructions and allow_empty_instructions is not True)
                or any(row.get("class") == "UNKNOWN" for row in instructions)
            ):
                raise NativeModelExecutionError("candidate LLVM has missing/unknown selected ISA decode")
            classes = {str(row.get("class")) for row in instructions}
            required = set(policy["expected_instruction_classes"])
            if not required.issubset(classes):
                raise NativeModelExecutionError(
                    f"candidate LLVM omits frozen required instruction classes {sorted(required - classes)}"
                )
            if policy["prohibited_instruction_roles"]:
                prohibition = scan_elf(elf, target=target, roles=policy["prohibited_instruction_roles"])
                if prohibition.get("status") != "measured" or prohibition.get("clean") is not True:
                    raise NativeModelExecutionError("linked candidate ELF has prohibited or unmeasured ISA roles")
            else:
                prohibition = {
                    "status": "not_applicable",
                    "clean": True,
                    "reason": "frozen reviewed policy explicitly prohibits no roles",
                }
        return {
            "status": "pass",
            "scope": "selected ISA decode, frozen class coverage and whole-ELF prohibited roles",
            "observed_classes": sorted(classes),
            "required_classes": sorted(required),
            "prohibition": prohibition,
            "whole_elf_known_custom_commands": custom_count,
            "rtl_facts": selected,
            "elf": _digest(elf),
        }
    except Exception as exc:  # noqa: BLE001 -- unknown ISA legality is not a pass
        return {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}


# ---- native_numerical_audit --------------------------------------------------------------------
# Original same-ELF numerical re-audit shared by explicit execution ABIs.
#
# The caller selects storage-name binding; this never changes numerical policy.


def audit_native_numerical_tier(
    emission, native, *, target, readback_policy, output_values=None, source_intake_protocol=None
):
    from merlin.development.targetgen.execution.native.native_model_execution import (
        NativeModelExecutionError,
        _digest,
        _functional_engine,
        _pinned_native_file,
        _read_pinned_payload,
        checked_source_output_values,
    )
    from merlin.target.contract.harness.readback_policy import read_console

    if output_values is None:
        output_values = checked_source_output_values
    try:
        import yaml

        from merlin.development.targetgen.grading.golden import compare
        from merlin.development.targetgen.grading.golden_store import load_golden
        from merlin.runtime.backends import registry as backends
        from merlin.runtime.engine.command_buffer import declared_output_dtypes

        functional = (native.get("tiers") or {}).get("L2") or {}
        if functional.get("status") not in {"pass", "fail"} or functional.get("engine") != "spike":
            raise NativeModelExecutionError("same candidate ELF has no completed functional L2 execution")
        backend, citation, revalidate = _functional_engine(target)
        if functional.get("engine_citation") != citation:
            raise NativeModelExecutionError("L2 engine citation differs from currently selected bytes")
        revalidate()
        elf = _pinned_native_file(native, "elf")
        if functional.get("elf") != _digest(elf):
            raise NativeModelExecutionError("L2 ran a different ELF")
        console = read_console(_pinned_native_file(functional, "console"), policy=readback_policy)
        observed, _metrics = backend.parse_output(console)
        cb = json.loads(_read_pinned_payload(emission, "command_buffer").decode("utf-8"))
        if readback_policy is not None:
            from merlin.target.contract.harness.readback_policy import require_full_value_roster

            require_full_value_roster(cb, console, observed, policy=readback_policy)
        observed = backends.decode_float_readback(observed, declared_output_dtypes(cb))
        capsule_dir = _pinned_native_file(native["source"], "capsule_declaration").parent
        _pinned_native_file(native["source"], "golden")
        if (capsule_dir / "golden.npz").is_file() != ("golden_arrays" in native["source"]):
            raise NativeModelExecutionError("L2 golden archive presence differs from source pins")
        if "golden_arrays" in native["source"]:
            _pinned_native_file(native["source"], "golden_arrays")
        declaration = yaml.safe_load((capsule_dir / "capsule.yaml").read_text(encoding="utf-8"))
        if source_intake_protocol is not None:
            from merlin.development.targetgen.execution.native import (
                native_pointer_source_binding as source_intake,
            )

            fresh = source_intake.preflight_large_intake(
                cb,
                capsule_dir,
                _pinned_native_file(native["source"], "interface").parent,
                intake_protocol=source_intake_protocol,
            )
            if native.get("source_intake") != fresh:
                raise NativeModelExecutionError("L2 source intake differs from its explicit original host selection")
        golden = load_golden(capsule_dir, **({"compact_arrays": True} if source_intake_protocol is not None else {}))
        observed = output_values(cb, native, golden, observed)
        if set(observed) != set(golden["outputs"]):
            raise NativeModelExecutionError("L2 console has no complete declared full output")
        compare_outputs = (
            source_intake.compare_complete_integer_outputs if source_intake_protocol is not None else compare
        )
        numeric = compare_outputs(
            golden["outputs"],
            observed,
            declaration["numeric_policy"],
            golden_source=str(golden.get("golden_source") or "independent_capsule"),
        )
        if (
            numeric.get("status") not in {"pass", "fail"}
            or numeric != functional.get("numeric")
            or functional.get("status") != numeric["status"]
        ):
            raise NativeModelExecutionError("L2 numeric receipt differs from independently parsed full output")
        return {
            "status": numeric["status"],
            "engine_citation": citation,
            "elf": _digest(elf),
            "console": _digest(_pinned_native_file(functional, "console")),
            "numeric": numeric,
        }
    except Exception as exc:  # noqa: BLE001 -- unbound engine/console is unavailable, not a tier pass
        return {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}


# ---- frozen_source_eligibility -----------------------------------------------------------------
# The trusted eligibility denominator of a frozen model source, from that source alone.
#
# Core compiler code (sibling source routing) and the optional native model executor both reopen this
# census; it lives here so the core never imports the optional executor. ``error`` selects the
# exception type a caller's refusals already use.


def _digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def census(capsule: Mapping[str, Any], *, target: str, error: type[Exception] = ValueError) -> dict[str, Any]:
    """Build only the trusted eligibility denominator from frozen model source.

    No host-dispatch compile, outline, tile ledger, or candidate placement is
    consulted. The router is used to enumerate demands; its *decisions* are
    not native-execution evidence. This can run even when the legacy model
    grade fails before producing a coverage certificate.
    """
    root_value = capsule.get("__dir__")
    interface_name = capsule.get("interface_mlir") or capsule.get("linalg_mlir")
    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    fmt = attrs.get("dtype")
    if (
        not isinstance(root_value, str)
        or not isinstance(interface_name, str)
        or not interface_name
        or Path(interface_name).is_absolute()
        or ".." in Path(interface_name).parts
        or not isinstance(fmt, str)
        or not fmt
    ):
        raise error("frozen model has no exact local interface and routing dtype")
    root = Path(root_value)
    if not root.is_dir() or root.is_symlink():
        raise error("frozen model capsule directory is absent or indirect")
    source = root / interface_name
    if source.is_symlink() or not source.is_file():
        raise error("frozen model interface is absent or indirect")
    raw = source.read_bytes()
    import yaml

    from merlin.capture.demands import model_op_demands_checked
    from merlin.development.targetgen.conformance.coverage_certificate import for_target
    from merlin.target.capabilities.routing import route_plan
    from merlin.target.registry import identity as target_registry

    source_text = raw.decode("utf-8")
    selected_contract = target_registry.resolve(target).capability_contract_path
    if not selected_contract.is_absolute() or selected_contract.is_symlink() or not selected_contract.is_file():
        raise error("selected target capability contract is absent or indirect")
    contract_pin = _digest(selected_contract)
    contract = yaml.safe_load(selected_contract.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or contract.get("name") != target:
        raise error("selected target capability contract names a different target")
    with target_registry.observed_contract(target, contract, source_path=selected_contract):
        demands = model_op_demands_checked(source_text, fmt)
        certificate = for_target(route_plan(demands, target), target, linalg_mlir=source_text, execution=None)
    if _digest(source)["sha256"] != hashlib.sha256(raw).hexdigest():
        raise error("frozen source changed during independent eligibility census")
    if _digest(selected_contract) != contract_pin:
        raise error("selected capability contract changed during eligibility census")
    if (
        certificate.get("target") != target
        or certificate.get("source_mlir_sha256") != hashlib.sha256(raw).hexdigest()
        or certificate.get("n_regions") != len(demands)
    ):
        raise error("independent source denominator lacks exact identity")
    certificate["source_inventory_scope"] = "frozen_source_only_no_legacy_execution"
    certificate["eligibility_capability_contract"] = contract_pin
    return certificate

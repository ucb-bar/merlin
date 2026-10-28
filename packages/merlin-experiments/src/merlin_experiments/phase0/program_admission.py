"""Screen a WRITTEN capsule from its own program, operation by operation.

The entry a capsule was written from names an op and a dtype; it carries no rank, layout, tail,
broadcast, alias or composition observation, so a screen of the entry left every such constraint
unresolved and verified generation refused the whole corpus. Those are properties of the written
program. This screen reads them from it and judges every operation with the same function that
judges a captured application's operations (``operation_accounting.admit_operation_row``):

* a Linalg/source program (PyTorch-sourced capsules and whole models) is inventoried exactly as a
  captured application is (``application_demand_inventory`` + ``build_operation_accounting``);
* an accelerator-interface program is read through ``interface_observations.command_rows``.

Each compute operation must have exactly one lane whose admission is ``admitted`` AND reviewed; that
is its placement. Support operations (packing, residency, readout bookkeeping, shape plumbing) are not
placed here, as in the Phase 0 readiness gate. A host-only probe must place everything on the host.

A whole-program capsule built from a declared iteration capture is additionally held to the
per-operation admission inventory saved for that capture at derivation (``whole_program_inventory``):
the written program must place the same operations on the same lanes.
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "merlin.phase0.whole_program_admission.v1"
_NON_COMPUTE = {"structural", "nested_component", "support_lowering_required"}


def _lanes(entry: dict) -> list[str]:
    return [
        lane
        for lane, key in (("accelerator", "accelerator_admission"), ("host", "host_admission"))
        if (entry.get(key) or {}).get("status") == "admitted" and (entry.get(key) or {}).get("reviewed") is True
    ]


def _decision(entry: dict, *, host_only: bool) -> dict:
    row = entry.get("observed_signature") or {}
    lanes = _lanes(entry)
    refused = all(
        (entry.get(key) or {}).get("status") == "unsupported" for key in ("accelerator_admission", "host_admission")
    )
    if host_only and lanes == ["accelerator"]:
        status, reason = "unsupported", "a host-only probe's program contains accelerator work"
    elif len(lanes) == 1:
        status, reason = "admitted", f"reviewed {lanes[0]} admission"
    elif refused:
        reasons = [(entry.get(key) or {}).get("reason") for key in ("accelerator_admission", "host_admission")]
        detail = [
            decision.get("reason") for decision in entry.get("software_admissions") or [] if decision.get("reason")
        ]
        status, reason = "unsupported", "; ".join(str(item) for item in [*detail, *reasons] if item)
    elif len(lanes) == 2:
        status, reason = "unknown", "both lanes are admitted; placement is ambiguous"
    else:
        status, reason = (
            "unknown",
            "; ".join(
                str(item)
                for item in [
                    *(
                        decision.get("reason")
                        for decision in entry.get("software_admissions") or []
                        if decision.get("status") != "admitted"
                    ),
                    (entry.get("hardware_admission") or {}).get("reason"),
                    (entry.get("host_admission") or {}).get("reason"),
                ]
                if item
            ),
        )
    return {
        **({"source_op_indices": row["source_op_indices"]} if "source_op_indices" in row else {}),
        "operation": entry.get("operation"),
        "mlir_operation": row.get("mlir_operation"),
        "frontend_op": row.get("frontend_op"),
        "count": entry.get("count", 1),
        "classification": entry.get("classification"),
        "placement": lanes[0] if status == "admitted" else None,
        "status": status,
        "reason": reason,
    }


def accelerator_placement_refusal(screen: dict | None) -> str | None:
    """Why a capsule with this written-program screen may not be published, or ``None`` when it may.

    A capsule grades the author's accelerator compiler; host placement is compiled by Merlin's own
    generic host codegen. A program with no operation placed on the accelerator therefore grades none
    of the author's work. Fail closed: a screen without a per-lane placement count is refused too,
    because an absent count is not evidence of accelerator work.
    """
    placements = screen.get("placements") if isinstance(screen, dict) else None
    if not isinstance(placements, dict):
        return "the capsule carries no frozen per-operation placement record"
    count = placements.get("accelerator", 0)
    if type(count) is not int or count < 0:
        return "the capsule's frozen accelerator placement count is malformed"
    if count == 0:
        lanes = ", ".join(f"{lane}={placements[lane]}" for lane in sorted(placements)) or "no placed operation"
        return (
            f"no operation of the written program is placed on the accelerator ({lanes}); host placement "
            "is Merlin-supplied and grades none of the author's work"
        )
    return None


def entry_refusal_is_final(entry: dict, decision: dict) -> bool:
    """Whether a screen of an ENTRY (before its capsule is written) may refuse it outright.

    Only a definite refusal is final before writing. Even that is deferred for a host-only probe:
    its entry cannot say which of its operations are compute and which are support plumbing (a
    transpose probe has no compute operation at all), so the written program's screen decides it.
    """
    probe = entry.get("generalization") or {}
    host_probe = probe.get("must_accelerate") is False and probe.get("eligible") is False
    # A MUST-REFUSE member is refused by the declarations on purpose: that refusal is what it grades,
    # so it is never moved out of the graded cohort for it (merlin.targetgen.expected_refusal).
    if entry.get("outcome") == "refuse":
        return False
    return decision.get("status") == "unsupported" and not host_probe


def summarize(entries: list[dict], *, host_only: bool = False, scope: str) -> dict:
    """Fold per-operation admissions into one screen with the shape ``screen_entry`` returns."""
    decisions = [
        _decision(entry, host_only=host_only) for entry in entries if entry.get("classification") not in _NON_COMPUTE
    ]
    refused = [row for row in decisions if row["status"] == "unsupported"]
    unknown = [row for row in decisions if row["status"] == "unknown"]
    status = "unsupported" if refused else "unknown" if unknown else "admitted"
    return {
        "status": status,
        "constraints_status": "refused" if refused else "unknown" if unknown else "matched",
        "reason": "; ".join(f"{row['operation']}: {row['reason']}" for row in (refused or unknown))
        or "every compute operation of the written program has exactly one reviewed lane",
        "decisions": decisions,
        "placements": dict(
            sorted(collections.Counter(row["placement"] for row in decisions if row["placement"]).items())
        ),
        "scope": scope,
    }


def account_program(program: Path, *, name: str, target: str, evidence, source_context=None) -> list[dict]:
    """Per-operation admission entries of one Linalg/source program, as for a captured application."""
    from merlin.targetgen.application_inventory import application_demand_inventory
    from merlin.targetgen.operation_accounting import build_operation_accounting

    inventory = application_demand_inventory(
        {name: program},
        target,
        detailed=True,
        capability_contract=evidence.contract,
        include_graph=True,
    )
    accounting = build_operation_accounting(
        inventory,
        evidence.software_spec or None,
        capability_contract=evidence.contract,
        host_capabilities=getattr(evidence, "host_capabilities", None),
    )
    _join_source_admissions(
        accounting, {name: program}, evidence=evidence, source_context=source_context, target=target
    )
    # Bind original top-level operations to the same parsed source inventory
    # used for admission. Nested scalar bodies are covered by their owning op.
    from merlin.common import mlir_query as mq
    from merlin.frontends.capture_normalization import normalize_capture_mlir

    normalized, _ = normalize_capture_mlir(program.read_text())
    module = mq.parse(normalized)
    functions = [op for op in module.body.block.ops if mq.op_name(op) == "func.func"]
    if len(functions) == 1 and len(functions[0].body.blocks) == 1:
        top = {
            op: i for i, op in enumerate(op for op in functions[0].body.block.ops if mq.op_name(op) != "func.return")
        }
        walked = tuple(mq.walk(module))
        for entry in accounting["applications"][name]["signatures"]:
            row = entry["observed_signature"]
            ordinals = row.get("ordinals", [])
            if ordinals and all(walked[i] in top for i in ordinals):
                row["source_op_indices"] = [top[walked[i]] for i in ordinals]
    return accounting["applications"][name]["signatures"]


def _join_source_admissions(
    accounting: dict, captures: dict[str, Path], *, evidence, source_context=None, target=None
) -> None:
    """Re-screen exact inventoried source bodies; serialized signatures are insufficient."""
    from merlin.common import mlir_query as mq
    from merlin.frontends.capture_normalization import normalize_capture_mlir
    from merlin.targetgen.operation_accounting import admit_operation_row

    if set(captures) != set(accounting["applications"]):
        raise ValueError("source admission captures must cover the exact application roster")
    if source_context is not None:
        from .connected_source_context import require_context

        require_context(source_context, captures)
    for name, application in accounting["applications"].items():
        raw = Path(captures[name]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != application["capture_sha256"]:
            raise ValueError("source admission capture differs from inventoried bytes")
        normalized, _ = normalize_capture_mlir(raw.decode("utf-8"))
        operations = tuple(mq.walk(mq.parse(normalized)))
        bound_context = source_context.for_program(captures[name]) if source_context is not None else None
        control_context = (
            source_context.control_admission_context(captures[name], target=target)
            if source_context is not None
            else None
        )
        index_context = source_context.index_admission_context(captures[name]) if source_context is not None else None
        for entry in application["signatures"]:
            row = entry["observed_signature"]
            control = control_context is not None and row.get("mlir_operation") in {
                "arith.index_cast",
                "arith.addi",
                "arith.cmpi",
                "cf.assert",
            }
            if entry["host_admission"]["status"] != "unknown" and not control:
                continue
            entry.update(
                admit_operation_row(
                    row,
                    software_spec=evidence.software_spec or None,
                    capability_contract=evidence.contract,
                    host_capabilities=getattr(evidence, "host_capabilities", None),
                    source_operations=tuple(operations[ordinal] for ordinal in row["ordinals"]),
                    source_context=(
                        control_context
                        if control
                        else index_context
                        if index_context is not None and row.get("frontend_op") == "aten.index.Tensor"
                        else bound_context
                    ),
                )
            )
    if source_context is not None:
        require_context(source_context, captures)


def account_interface(program: Path, *, target: str, evidence, numeric_screens: list[dict] | None = None) -> list[dict]:
    """Per-command admission entries of one accelerator-interface program."""
    return account_interface_text(
        program.read_text(encoding="utf-8"), target=target, evidence=evidence, numeric_screens=numeric_screens
    )


def account_interface_text(
    mlir: str, *, target: str, evidence, numeric_screens: list[dict] | None = None
) -> list[dict]:
    """Screen builder-emitted interface bytes before a capsule or golden is written."""
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir
    from merlin.targetgen.interface_observations import command_rows
    from merlin.targetgen.operation_accounting import admit_operation_row

    parsed = parse_interface_mlir(mlir, include_results=True)
    if parsed.get("target") != target:
        raise ValueError("written interface program names a different target")
    if numeric_screens is None:
        numeric_screens = _operand_sum_numeric_screens_text(mlir, evidence)
    by_command = {screen["command_index"]: screen for screen in numeric_screens}
    entries = []
    for row in command_rows(parsed):
        # A conditional SW declaration must see the numerical witness computed
        # from this exact written command and the selected fact-derived model.
        if row.get("command_opcode") == "RESIDUAL_ADD":
            ordinal = (row.get("ordinals") or [None])[0]
            if ordinal in by_command:
                row = {**row, "numeric_screen": by_command[ordinal]}
        entries.append(
            {
                "operation": row["operation"],
                "count": 1,
                "observed_signature": row,
                **admit_operation_row(
                    row,
                    software_spec=evidence.software_spec or None,
                    capability_contract=evidence.contract,
                    host_capabilities=getattr(evidence, "host_capabilities", None),
                ),
            }
        )
    return entries


def _operand_sum_numeric_screens(program: Path, evidence) -> list[dict]:
    return _operand_sum_numeric_screens_text(program.read_text(encoding="utf-8"), evidence)


def _operand_sum_model(evidence) -> tuple[list[dict], dict] | None:
    """The selected i8 operand-sum facets and readout numerics; ``None`` without a fused operand-sum row."""
    spec = getattr(evidence, "software_spec", None) or {}
    if not any(
        (row.get("derived_from_facts") or {}).get("form") == "fused_operand_sum" for row in spec.get("operations") or ()
    ):
        return None
    facets = [
        facet.to_dict() if hasattr(facet, "to_dict") else facet for facet in getattr(evidence, "readout_facets", ())
    ]
    selected = [facet for facet in facets if (facet.get("operand_sum") or {}).get("operand_dtype") in {"i8", "int8"}]
    return selected, (spec.get("numerical_semantics") or {}).get("readout") or {}


def per_operand_readout_witness(
    *, lhs_scale: float, rhs_scale: float, readout_scale: float, relu: bool, evidence
) -> dict | None:
    """The per-operand operand-sum witness INCLUDING the readout scale, for a fused i8 operand sum.

    An interface command states no readout scale for its operand sum, so its per-operand screen
    with an ``acc_scale`` stage stays unknown (``_operand_sum_numeric_screens_text``). A source
    program whose every parameter is a proven constant -- both operand scales, the one readout gain
    and the relu -- supplies that separate readout-scale witness: the exhaustive audit run with that
    gain. ``None`` when the selected spec has no fused operand-sum row (no screen is made then).
    """
    model = _operand_sum_model(evidence)
    if model is None:
        return None
    selected, readout = model
    if (
        len(selected) != 1
        or readout.get("acc_scale_rounding") != "half_even"
        or readout.get("narrowing") != "saturate_to_declared_dtype"
    ):
        result = {"status": "unknown", "reason": "selected i8 operand-sum/readout numerics are incomplete or ambiguous"}
    else:
        from merlin.targetgen.operand_sum_numeric import audit_i8_per_operand_sum

        result = audit_i8_per_operand_sum(
            lhs_scale=lhs_scale,
            rhs_scale=rhs_scale,
            readout_scale=readout_scale,
            relu=relu,
            facet=selected[0],
            bound_lsb=0,
        )
    return {"form": "fused_operand_sum", **result}


def single_rounding_readout_witness(
    *, lhs_scale: float, rhs_scale: float, readout_scale: float, relu: bool, evidence
) -> dict | None:
    """The single-rounding operand-sum witness INCLUDING the readout gain, for a fused i8 operand sum.

    The same gate as ``per_operand_readout_witness`` (a fused operand-sum row, exactly one selected
    i8 facet, half-even/saturating readout numerics), but the exhaustive audit is the single-rounding
    one (``audit_i8_operand_sum``) run with the source program's readout gain, so the witness carries
    that gain under the contract the single-rounding realization binds. ``None`` when the selected
    spec has no fused operand-sum row.
    """
    model = _operand_sum_model(evidence)
    if model is None:
        return None
    selected, readout = model
    if (
        len(selected) != 1
        or readout.get("acc_scale_rounding") != "half_even"
        or readout.get("narrowing") != "saturate_to_declared_dtype"
    ):
        result = {"status": "unknown", "reason": "selected i8 operand-sum/readout numerics are incomplete or ambiguous"}
    else:
        from merlin.targetgen.operand_sum_numeric import audit_i8_operand_sum

        result = audit_i8_operand_sum(
            lhs_scale=lhs_scale,
            rhs_scale=rhs_scale,
            bound_lsb=0,
            relu=relu,
            facet=selected[0],
            readout_scale=readout_scale,
        )
    return {"form": "fused_operand_sum", **result}


def _operand_sum_numeric_screens_text(mlir: str, evidence) -> list[dict]:
    """Bound only a selected, fact-derived sum/readout composition.

    This is a deterministic software-model refusal screen, not a replacement
    for the executable capsule's target oracle. Other fused forms are untouched.
    """
    model = _operand_sum_model(evidence)
    if model is None:
        return []
    selected, readout = model
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir
    from merlin.targetgen.operand_sum_numeric import audit_i8_operand_sum, audit_i8_per_operand_sum

    commands = parse_interface_mlir(mlir).get("commands") or ()
    screens = []
    for index, command in enumerate(commands):
        attrs = command.get("attributes") or {}
        if command.get("opcode") != "RESIDUAL_ADD":
            continue
        stages = set(attrs.get("epilogue") or ())
        if stages - {"relu", "acc_scale"}:
            result = {"status": "unknown", "reason": "operand-sum numeric model does not cover these epilogue stages"}
        elif (
            len(selected) != 1
            or attrs.get("output_dtype") not in {"i8", "int8"}
            or readout.get("acc_scale_rounding") != "half_even"
            or readout.get("narrowing") != "saturate_to_declared_dtype"
        ):
            result = {
                "status": "unknown",
                "reason": "selected i8 operand-sum/readout numerics are incomplete or ambiguous",
            }
        elif attrs.get("operand_sum_rounding", "single_rounding") not in {"single_rounding", "per_operand"}:
            result = {"status": "unknown", "reason": "undeclared operand-sum rounding"}
        elif attrs.get("operand_sum_rounding") == "per_operand" and "acc_scale" in stages:
            result = {
                "status": "unknown",
                "reason": "per-operand interface screen requires a separate readout-scale witness",
            }
        elif "acc_scale" in stages:
            # An interface RESIDUAL_ADD cannot state a readout gain (its ABI epilogue is relu only):
            # the single-rounding audit used to run here with that gain IGNORED. A readout gain is
            # screened only by a source witness that carries it (single_rounding_readout_witness).
            result = {
                "status": "unknown",
                "reason": "single-rounding interface screen requires a separate readout-scale witness",
            }
        else:
            audit = (
                audit_i8_per_operand_sum if attrs.get("operand_sum_rounding") == "per_operand" else audit_i8_operand_sum
            )
            result = audit(
                lhs_scale=attrs.get("lhs_scale"),
                rhs_scale=attrs.get("rhs_scale"),
                bound_lsb=attrs.get("bound_lsb"),
                relu="relu" in stages,
                facet=selected[0],
            )
        screens.append(
            {
                "command_index": index,
                "form": "fused_operand_sum" if attrs.get("epilogue") else "standalone_operand_sum",
                **result,
            }
        )
    return screens


def screen_written(capsule: dict, directory: Path, *, target: str, evidence) -> dict | None:
    """The written program's screen, or ``None`` when the capsule contains no program to read."""
    from .coverage_commitment import _selected_program

    program = _selected_program(capsule, Path(directory))
    if program is None and capsule.get("kind") == "model":
        selected = capsule.get("interface_mlir")
        candidate = Path(directory) / selected if isinstance(selected, str) and selected else None
        program = candidate if candidate is not None and candidate.is_file() and not candidate.is_symlink() else None
    if program is None:
        return None
    semantic = capsule.get("semantic") or capsule.get("generalization") or {}
    runtime_scale_name = None
    host_only = semantic.get("must_accelerate") is False and semantic.get("eligible") is False
    if "linalg_mlir" in capsule or capsule.get("kind") == "model":
        entries = account_program(program, name=str(capsule.get("name")), target=target, evidence=evidence)
        scope = "written source program: per-operation admission, as for a captured application"
        if (capsule.get("operation") or {}).get("op") == "composition_program":
            # A chain member proven to be its contraction's epilogue is judged as that fused stage.
            from .source_fused_stages import observe as observe_fused_stages

            attributes = (capsule.get("operation") or {}).get("attributes") or {}
            case = attributes.get("case") or {}
            inputs = [row for row in capsule.get("inputs") or [] if isinstance(row, dict)]
            names = [row.get("name") for row in inputs]
            scale_name = case.get("runtime_scale_input")
            runtime_scale = names.index(scale_name) if scale_name and names.count(scale_name) == 1 else None
            # The fact-derived extended-K bound the generator resolved for this composition.
            within = ((semantic.get("composition_facts") or {}).get("k_extension_within")) or {}
            entries = observe_fused_stages(
                program,
                entries,
                evidence=evidence,
                readout_strategy=(attributes.get("quantization_scheme") or {}).get("readout_strategy"),
                dataflow=case.get("dataflow"),
                max_k=within.get("max_k") if type(within.get("max_k")) is int else None,
                # The Linalg cannot state a fence; the composition declares it (an unfenced twin
                # has byte-identical Linalg), so only a declared fence lets a pool be observed.
                pooling_fenced=case.get("scratchpad_source_pooling_fenced") is True,
                runtime_scale=runtime_scale,
                inputs=inputs,
            )
            if any(
                ((entry.get("observed_signature") or {}).get("composed_observation") or {}).get("scale_source")
                == "runtime_input"
                for entry in entries
            ):
                runtime_scale_name = scale_name
        numeric_screens = []
    else:
        numeric_screens = _operand_sum_numeric_screens(program, evidence)
        entries = account_interface(program, target=target, evidence=evidence, numeric_screens=numeric_screens)
        scope = "written interface program: per-command admission from its own declared tensors"
    screen = summarize(entries, host_only=host_only, scope=scope)
    if numeric_screens:
        screen["numeric_screens"] = numeric_screens
        failures = [row for row in numeric_screens if row["status"] != "within_bound"]
        if failures:
            screen["status"] = "unsupported" if any(row["status"] == "exceeds_bound" for row in failures) else "unknown"
            screen["constraints_status"] = "refused" if screen["status"] == "unsupported" else "unknown"
            screen["reason"] = "operand-sum numerical screen: " + "; ".join(
                f"command {row['command_index']}: {row.get('reason') or row['status']}" for row in failures
            )
    screen["program_sha256"] = hashlib.sha256(program.read_bytes()).hexdigest()
    materialized = capsule.get("materialized_capture") or {}
    if materialized:
        screen = _check_saved_inventory(screen, materialized, evidence)
    from .numeric_domains import screen as screen_numeric_domain

    domain = screen_numeric_domain(capsule, directory, (evidence.software_spec or {}).get("numerical_semantics") or {})
    if runtime_scale_name is not None:
        # An observed runtime scale holds the capsule's own inputs to the reviewed domain it was
        # observed under, so the reference is never computed outside it.
        from .numeric_domains import screen_runtime_scale
        from .source_fused_stages import runtime_scale_domain

        scale_domain = screen_runtime_scale(
            capsule, directory, runtime_scale_name, runtime_scale_domain(evidence.software_spec or {})
        )
        screen["runtime_scale_domain"] = scale_domain
        if scale_domain["status"] != "within_domain" and (domain is None or domain["status"] == "within_domain"):
            domain = scale_domain
    if domain is not None:
        screen["numeric_domain"] = domain
        if domain["status"] in {"unsupported", "unknown"}:
            screen.update(
                status=domain["status"],
                constraints_status="refused" if domain["status"] == "unsupported" else "unknown",
                reason="selected numerical domain: " + domain["reason"],
            )
    return screen


def _check_saved_inventory(screen: dict, materialized: dict, evidence) -> dict:
    """Hold a capsule built from a declared capture to that capture's derivation-time inventory."""
    saved = getattr(evidence, "whole_program_admission", None) or {}
    applications = saved.get("applications") or {}
    match = [row for row in applications.values() if row.get("capture_sha256") == materialized.get("capture_sha256")]
    if len(match) != 1:
        return {
            **screen,
            "status": "unknown",
            "constraints_status": "unknown",
            "reason": "no derivation-time per-operation admission inventory names this capture",
        }
    expected = match[0]
    if expected.get("status") != "admitted":
        return {**screen, "status": expected.get("status", "unknown"), "reason": expected.get("reason")}
    if expected.get("placements") != screen.get("placements"):
        return {
            **screen,
            "status": "unsupported",
            "constraints_status": "refused",
            "reason": "the written program places operations differently from the derivation-time inventory",
            "expected_placements": expected.get("placements"),
        }
    return {**screen, "saved_inventory": {"capture_sha256": expected["capture_sha256"], "status": "matched"}}


def derive_inventory(inventory: dict, *, target: str, evidence, captures: dict[str, Path] | None = None) -> dict:
    """The per-operation admission inventory of each declared capture, written at derivation.

    ``inventory`` is the derivation's own detailed application inventory of the declared captures,
    judged once more with the selected declarations. Selected captures additionally
    establish parsed source-body requirements against the exact inventoried bytes.
    """
    from merlin.targetgen.operation_accounting import build_operation_accounting

    accounting = build_operation_accounting(
        inventory,
        evidence.software_spec or None,
        capability_contract=evidence.contract,
        host_capabilities=getattr(evidence, "host_capabilities", None),
    )
    if (
        captures is None
        and inventory.get("applications")
        and all(
            isinstance(row.get("capture_source_path"), str) and row["capture_source_path"]
            for row in inventory["applications"].values()
        )
    ):
        captures = {name: Path(row["capture_source_path"]) for name, row in inventory["applications"].items()}
    if captures is not None:
        _join_source_admissions(accounting, captures, evidence=evidence)
    applications = {}
    for label, application in sorted(accounting["applications"].items()):
        screen = summarize(application["signatures"], scope="declared iteration capture at derivation")
        applications[label] = {
            "capture_sha256": application["capture_sha256"],
            "status": screen["status"],
            "reason": screen["reason"],
            "placements": screen["placements"],
            "operations": screen["decisions"],
        }
    document = {"schema": SCHEMA, "target": target, "applications": applications}
    document["sha256"] = operations_digest(document)
    return document


def operations_digest(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(document.get("applications") or {}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

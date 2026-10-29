"""Project recovered CPU operands through the original selected accelerator ISA.

The normalized operand observer supplies field semantics. Static facts are
scoped to instruction sites; they do not establish complete math, execution
counts, allocation safety, overlap, or timing.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

from merlin.compile.analysis.riscv_machine import decode_cpu, finite_control_counts, register_facts
from merlin.compile.analysis.rocc_commands import fields_of
from merlin.kernels.instruction_comparison import OperandSummary
from merlin.kernels.kernel_evidence import EvidenceFact, cca_projection


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _resolved(value):
    if isinstance(value, dict):
        return all(_resolved(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_resolved(item) for item in value)
    return value is not None


class _MechanismObservations:
    """Bounded witnessed operand repetition and lexical issue boundaries.

    Equality here is equality of recovered fields, not an alias, useful-work,
    retention, execution-count or optimization-legality proof.
    """

    def __init__(self, roles, *, max_groups=128, max_witnesses=8):
        self.roles, self.max_groups, self.max_witnesses = roles, max_groups, max_witnesses
        self.groups, self.omitted, self.unresolved_sites, self.unmapped_role_sites = {}, 0, 0, 0
        self.boundaries, self.boundary_count = [], 0
        self.boundary_sequence = hashlib.sha256()
        self.phase, self.run, self.runs, self.run_count = None, None, [], 0

    def _close_phase(self, boundary=None):
        if self.phase is None:
            return
        phase = {**self.phase, "end_boundary": boundary}
        self.boundary_count += 1
        self.boundary_sequence.update(json.dumps(phase, sort_keys=True).encode() + b"\n")
        if len(self.boundaries) < self.max_witnesses:
            self.boundaries.append(phase)
        self.phase = None

    def _close_run(self):
        if self.run is not None:
            self.run_count += 1
            if len(self.runs) < self.max_witnesses:
                self.runs.append(self.run)
            self.run = None

    def fence(self, index, pc):
        self._close_phase(dict(instruction_index=index, pc=pc, kind="cpu_fence"))
        self._close_run()

    def observe(self, index, record):
        instruction, fields = record.get("instruction", "UNKNOWN"), record.get("fields", {})
        roles = self.roles.get(instruction, ())
        self.unmapped_role_sites += not bool(roles)
        resolved = bool(fields) and _resolved(fields) and record.get("status") != "unknown"
        self.unresolved_sites += not resolved
        encoded = json.dumps(dict(instruction=instruction, fields=fields), sort_keys=True, allow_nan=False).encode()
        identity = hashlib.sha256(encoded).hexdigest()
        witness = dict(instruction_index=index, pc=record["pc"])
        if identity in self.groups:
            group = self.groups[identity]
            group["sites"] += 1
            group["last_witness"] = witness
        elif len(self.groups) < self.max_groups:
            self.groups[identity] = dict(instruction=instruction, roles=list(roles), sites=1,
                                        recovered_fields_sha256=identity, fully_recovered=resolved,
                                        fields={key: value for key, value in sorted(fields.items())[:8]},
                                        omitted_fields=max(0, len(fields) - 8),
                                        first_witness=witness, last_witness=witness)
        else:
            self.omitted += 1
        if self.phase is None:
            self.phase = dict(first_witness=witness, last_witness=witness, command_sites=0, role_sites={})
        self.phase["last_witness"] = witness
        self.phase["command_sites"] += 1
        for role in roles:
            self.phase["role_sites"][role] = self.phase["role_sites"].get(role, 0) + 1
        if self.run is not None and self.run["instruction"] != instruction:
            self._close_run()
        if self.run is None:
            self.run = dict(instruction=instruction, roles=list(roles), sites=0,
                            first_witness=witness, last_witness=witness)
        self.run["sites"] += 1
        self.run["last_witness"] = witness
        if "sync" in roles:
            self._close_phase(dict(**witness, kind="selected_sync_role"))
            self._close_run()

    def report(self, operands):
        self._close_phase()
        self._close_run()
        repeated = sorted((group for group in self.groups.values() if group["sites"] > 1),
                          key=lambda group: (-group["sites"], group["recovered_fields_sha256"]))
        invariants = []
        profiles = operands["field_profiles"]
        for field in operands["field_fingerprints"]:
            if (field["distinct_values"] == 1 and field["sites"] > 1
                    and not field["missing_sites"] and not field["unresolved_sites"]):
                label = field["instruction"] + "." + field["field"]
                invariants.append(dict(instruction=field["instruction"], field=field["field"], sites=field["sites"],
                                       value_sha256=field["value_set_sha256"],
                                       value_profile=profiles.get(label),
                                       roles=list(self.roles.get(field["instruction"], ()))))
        transfers = [dict(mechanism="repeated_transfer_operand_fields", status="observed_static_repetition",
                          evidence=group, optimization_applicability="unknown",
                          missing=["canonical_source_and_destination_accesses", "intervening_write_and_alias_effects",
                                   "inherited_configuration_and_operand_bindings",
                                   "capacity_and_lifetime_legality", "dynamic_execution_multiplicity"])
                     for group in repeated if group["fully_recovered"] and
                     set(group["roles"]) & {"operand_load", "weight_load", "dma"}]
        return dict(
            schema="merlin.machine_mechanism_observations.v1", origin="selected_isa_recovered_operands",
            scope="listed_instruction_sites", ordering="lexical_listing_not_dynamic_issue_trace",
            status="partial" if self.omitted or self.unresolved_sites or self.unmapped_role_sites else "observed",
            repeated_operand_groups=repeated[:self.max_witnesses],
            omitted_repeated_operand_groups=max(0, len(repeated) - self.max_witnesses),
            repetition_census="unknown_group_budget_exhausted" if self.omitted else "complete_recovered_field_tuples",
            omitted_group_sites=self.omitted, unresolved_operand_sites=self.unresolved_sites,
            unmapped_instruction_role_sites=self.unmapped_role_sites,
            invariant_fields=invariants[:self.max_witnesses],
            omitted_invariant_fields=max(0, len(invariants) - self.max_witnesses),
            invariant_field_census=operands["field_fingerprint_status"],
            instruction_runs=self.runs, instruction_run_count=self.run_count,
            fence_delimited_groups=self.boundaries, fence_delimited_group_count=self.boundary_count,
            fence_delimited_groups_sha256=self.boundary_sequence.hexdigest(),
            candidates=transfers[:self.max_witnesses],
            physical_vs_useful_reduction=dict(status="unknown", missing=["selected_field_extent_and_unit_bindings",
                                                                         "complete_upstream_reduction_domain"]),
            unknowns=["same_fields_do_not_establish_same_memory_region_or_redundant_work",
                      "hardware_completion_overlap_and_timing_not_established",
                      "retention_and_batching_legality_require_storage_effects"],
        )


def machine_projection(rows, *, model, evidence_id, decoder, instruction_offset=0):
    """Return witnessed fields and CCA axes; unsupported semantics stay absent."""
    semantics, target = decoder.get("semantics"), decoder.get("target")
    if semantics is None:
        return None, dict(
            status="unknown",
            code="selected_operand_decoder_unavailable",
            next_action="Bind the original target contract and operand protocol, then reanalyze.",
        )
    isa = semantics.isa_constants(target)
    states, analysis = register_facts(rows)
    commands, unresolved = [], Counter()
    command_count, next_command_offset, decoded_sites = 0, None, 0
    operand_summary = OperandSummary()
    structure = decoder.get("structure") or {}
    instruction_roles = structure.get("instruction_roles") or {}
    mechanisms = _MechanismObservations(instruction_roles)
    classes, raw_classes, fences = Counter(), Counter(), 0

    def command(index, record):
        nonlocal command_count, next_command_offset
        command_count += 1
        operand_summary.observe(record.get("instruction", "UNKNOWN"), record.get("fields", {}), record.get("operands"))
        mechanisms.observe(index, record)
        if instruction_offset <= index < instruction_offset + 8:
            commands.append(dict(instruction_index=index, **record))
        elif index >= instruction_offset + 8 and next_command_offset is None:
            next_command_offset = index

    for index, row in enumerate(rows):
        if not row.hexcode:
            continue
        cpu = decode_cpu(row)
        if cpu.kind == "fence":
            fences += 1
            mechanisms.fence(index, row.addr)
        if len(row.hexcode) != 8:
            continue
        fields = fields_of(int(row.hexcode, 16))
        if fields["opcode"] != isa["CUSTOM_OPCODE"]:
            continue
        raw_classes[isa["FUNCT_CLASS"].get(fields["funct"], "UNKNOWN")] += 1
        state = states.get(row.addr)
        if state is None:
            unresolved["unreachable_or_unresolved_control"] += 1
            # A relocation elsewhere can invalidate register/control recovery
            # without altering this instruction's selected identity bits. Never
            # apply this independence to a site that itself has relocations.
            klass = isa["FUNCT_CLASS"].get(fields["funct"], "UNKNOWN") if not row.relocations else "UNKNOWN"
            if klass != "UNKNOWN":
                classes[klass] += 1
            command(index, dict(pc=row.addr, word=row.hexcode, instruction=klass, fields={}, status="unknown",
                                reason="unresolved_control_or_operands", identity_scope="encoded_static_site"))
            continue
        operands = [state[fields[name]].record() for name in ("rs1", "rs2")]
        klass, decoded = semantics.decode_instruction(fields["funct"], *operands, isa)
        decoded = _finite(decoded)
        classes[klass] += 1
        decoded_sites += 1
        for name, value in decoded.items():
            if value is None:
                unresolved[klass + "." + name] += 1
        command(
            index,
            dict(
                pc=row.addr,
                word=row.hexcode,
                instruction=klass,
                fields=decoded,
                operands=dict(zip(("rs1", "rs2"), operands)),
            ),
        )
    counted = finite_control_counts(rows, custom_opcode=isa["CUSTOM_OPCODE"])
    facts = []
    witness = (evidence_id + "#selected_machine_operands",)

    def fact(axis, value, reason):
        if value is not None:
            facts.append(EvidenceFact(axis, value, "inferred", witness, reason))

    if counted["status"] == "proved" and not raw_classes["UNKNOWN"]:
        fact("dispatch.n_dispatches", counted["commands"], "complete bounded scalar-control proof")
        fact("dispatch.n_fences", counted["fences"], "complete bounded scalar-control proof")

    observed_roles = Counter()
    for name, count in classes.items():
        for role in instruction_roles.get(name, ()):
            observed_roles[role] += count
    has_compute = bool(observed_roles["accumulate"])
    if has_compute:
        kinds = {unit.get("kind") for unit in (structure.get("compute_units") or []) if "matmul" in unit.get("ops", [])}
        fact(
            "compute.contraction_form",
            "systolic" if kinds == {"systolic"} else None,
            "selected compute instruction fields and original declared compute unit",
        )
        fact(
            "compute.accumulator_dtype",
            (structure.get("output_epilogue") or {}).get("accumulator_dtype"),
            "selected compute instruction and original output datapath type",
        )
    # Absence is only meaningful when the selected ISA supplies a complete
    # role roster. Class-name spellings do not imply hardware-loop semantics.
    if raw_classes and all(instruction_roles.get(name) for name in raw_classes):
        fact(
            "dispatch.loop_offloaded",
            any("loop_descriptor" in instruction_roles[name] for name in raw_classes),
            "complete role roster from the selected ISA definition",
        )
    cca = cca_projection(model, op="accelerator_region", source="machine", backend=(target,), facts=tuple(facts))
    for row in cca.provenance["axis_evidence"].values():
        row["scope"] = "static_instruction_sites"
    for axis in ("dispatch.n_dispatches", "dispatch.n_fences"):
        if axis in cca.provenance["axis_evidence"]:
            cca.provenance["axis_evidence"][axis]["scope"] = "selected_span_entry_to_return"
    operands = operand_summary.report()
    report = dict(
        status="partial" if unresolved or analysis.get("gaps") else "analyzed",
        count_basis="listed_instruction_sites",
        register_analysis=analysis,
        counted_control={
            key: value for key, value in counted.items() if key not in ("instruction_executions", "command_executions")
        },
        decoded_sites=decoded_sites,
        classified_static_sites=sum(classes.values()),
        instruction_classes=dict(classes),
        instruction_roles=instruction_roles,
        role_sites=dict(observed_roles),
        unclassified_instruction_classes=sorted(name for name in raw_classes if not instruction_roles.get(name)),
        missing_declared_instruction_roles=structure.get("missing_declared_instruction_roles", {}),
        unresolved_fields=dict(sorted(unresolved.items())),
        command_witnesses=commands,
        command_page=dict(instruction_offset=instruction_offset, next_instruction_offset=next_command_offset),
        mechanism_observations=mechanisms.report(operands),
        **{
            key: value
            for key, value in operands.items()
            if key not in ("count_basis", "unresolved_fields")
        },
        omitted_command_witnesses=command_count - len(commands),
        fence_sites=fences,
        semantic_region_status="unknown",
        missing=["complete_semantic_region", "dynamic_loop_multiplicity", "hardware_completion_and_timing"],
        decoder_inputs=decoder.get("decoder_inputs", []),
        role_inputs=decoder.get("role_inputs", []),
        next_action="Join the upstream semantic region and complete counted/captured command dataflow; "
        "retain field witnesses at the listed instruction PCs.",
    )
    return cca, report

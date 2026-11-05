"""Project recovered CPU operands through the original selected accelerator ISA.

The normalized operand observer supplies field semantics. Static facts are
scoped to instruction sites; they do not establish complete math, execution
counts, allocation safety, overlap, or timing.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections import Counter

from merlin.compile.analysis.riscv_machine import BRANCHES, CALLS, decode_cpu, finite_control_counts, register_facts
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


#: Hardware/ISA field kinds a target may declare for decoded fields. ``address``
#: is accepted but never projected: placement is not region semantics.
FIELD_KINDS = ("stride", "extent", "scale", "activation", "output_dtype", "address")
_MAX_FIELD_VALUES = 32


def declared_field_kinds(structure):
    """``{instruction.field: kind}`` from the selected target's declaration, or ``{}``."""
    declared = (structure or {}).get("operand_field_kinds") or {}
    if type(declared) is not dict:
        raise ValueError("operand field kinds must map decoded field labels to declared kinds")
    kinds = {}
    for label, kind in declared.items():
        instruction, dot, field = str(label).partition(".")
        if not instruction or dot != "." or not field or kind not in FIELD_KINDS:
            raise ValueError("operand field kind declaration has a malformed label or unknown kind")
        kinds[str(label)] = kind
    return kinds


def _field_kind_facts(kinds, fingerprints, values, witness):
    """Value sets per declared kind; an incompletely recovered field stays an explicit unknown."""
    grouped = {}
    for row in fingerprints:
        label = row["instruction"] + "." + row["field"]
        kind = kinds.get(label)
        if kind is None or kind == "address":
            continue
        observed = values.get(label)
        complete = (not row["missing_sites"] and not row["unresolved_sites"] and observed is not None
                    and len(observed) == row["distinct_values"])
        ordered = sorted((json.loads(v) for v in observed or ()), key=lambda v: (type(v).__name__, v))
        grouped.setdefault(kind, []).append((label, tuple(ordered) if complete else None))
    facts = []
    # A kind the selected target does not declare for any observed field stays
    # an explicit unknown; decoded values never acquire meaning from names.
    for kind in FIELD_KINDS:
        if kind != "address" and kind not in grouped:
            facts.append(EvidenceFact("operand_fields." + kind, None, "unknown", (),
                                      "no selected target field kind declaration binds an observed field"))
    for kind, rows in sorted(grouped.items()):
        unknown = sorted(label for label, value in rows if value is None)
        reason = "selected target field kind declaration; recovered values at listed static sites"
        if unknown:
            reason += "; unrecovered fields: " + ", ".join(unknown[:8])
        if len(unknown) == len(rows):
            # Declared but never recovered: the axis stays unknown, not a known set of nulls.
            facts.append(EvidenceFact("operand_fields." + kind, None, "unknown", (), reason))
            continue
        facts.append(EvidenceFact("operand_fields." + kind, tuple(sorted(rows)), "observed", witness, reason))
    return facts


def lifter_call(function, rows, **options):
    """Call a machine lifter with only the options its signature accepts.

    Entry states and call summaries are optional lifter inputs; an older lifter
    without them is called unchanged and its absent outputs stay unknown.
    """
    accepted = inspect.signature(function).parameters
    return function(rows, **{key: value for key, value in options.items() if key in accepted and value is not None})


def _block_members(rows):
    """``{leader_pc: [pc, ...]}`` under the same leader rule as the register lifter."""
    decoded = {row.addr: decode_cpu(row) for row in rows}
    order = sorted(decoded)
    if not order:
        return {}
    leaders = {rows[0].addr}
    for pc, d in decoded.items():
        following = pc + d.size
        if d.target in decoded and d.kind in BRANCHES | {"jump"}:
            leaders.add(d.target)
        if d.kind in BRANCHES | CALLS | {"jump", "return", "indirect"} and following in decoded:
            leaders.add(following)
    for left, right in zip(order, order[1:]):
        if left + decoded[left].size != right:
            leaders.add(right)
    members, current = {}, None
    for pc in order:
        if pc in leaders or current is None:
            current = pc
        members.setdefault(current, []).append(pc)
    return members


def _loop_pcs(loop, members):
    pcs = set()
    for block in loop.get("blocks") or ():
        if isinstance(block, dict):
            if isinstance(block.get("pcs"), (list, tuple)):
                pcs.update(block["pcs"])
                continue
            block = block.get("start", block.get("leader"))
        pcs.update(members.get(block, ()))
    return pcs


def loop_nest(analysis, rows, *, sites, roles, evidence_id, counted=None):
    """Lifter-proved loop nest with accelerator roles placed at each level.

    ``sites`` maps command PCs to decoded classes. Only loops and trip counts the
    lifter reports are projected; an absent analysis stays an explicit unknown.
    """
    analysis = analysis or {}
    loops = analysis.get("loops")
    if not isinstance(loops, list):
        reason = ("register lifter reports no loop analysis" if analysis.get("status") == "analyzed"
                  else "register lifter did not analyze the span: " + str(analysis.get("reason", "unknown")))
        return dict(value=None, status="unknown", reason=reason, evidence=[])
    members = _block_members(rows)
    calls = {row.addr for row in rows if decode_cpu(row).kind in CALLS}
    placed, levels = {}, []
    spans = [(loop, _loop_pcs(loop, members)) for loop in loops]
    for loop, pcs in spans:
        for pc in pcs & set(sites):
            depth = loop.get("nest_depth") or 0
            if pc not in placed or depth > placed[pc][0]:
                placed[pc] = (depth, loop.get("header_pc"))
    for loop, pcs in sorted(spans, key=lambda item: (item[0].get("nest_depth") or 0, item[0].get("header_pc") or 0)):
        header = loop.get("header_pc")
        own = [pc for pc, (_, owner) in placed.items() if owner == header]
        classes, role_sites = Counter(sites[pc] for pc in own), Counter()
        for klass, count in classes.items():
            for role in roles.get(klass, ()):
                role_sites[role] += count
        levels.append(dict(
            header_pc=header,
            nest_depth=loop.get("nest_depth"),
            parent_header_pc=loop.get("parent_header_pc"),
            latch_pcs=list(loop.get("latch_pcs") or ()),
            trip_count=loop.get("trip_count") or dict(status="unknown", reason="lifter reports no trip count"),
            induction=[dict(reg=row.get("reg"), init=row.get("init"), step=row.get("step"), status=row.get("status"))
                       for row in loop.get("induction") or ()],
            command_sites=len(own),
            commands_in_body=len(pcs & set(sites)),
            instruction_classes=dict(sorted(classes.items())),
            role_sites=dict(sorted(role_sites.items())),
            call_sites_in_body=len(pcs & calls),
            evidence=[f"{evidence_id}#register_analysis.loops[header_pc={header}]"],
        ))
    proved = [level for level in levels if level["trip_count"].get("status") == "proved"]
    dynamic = (counted or {}).get("dynamic_command_count")
    in_loops = set().union(*(pcs for _, pcs in spans)) if spans else set()
    value = dict(
        depth=max((level["nest_depth"] or 0 for level in levels), default=0),
        loops=len(levels),
        proved_trip_counts=len(proved),
        levels=levels,
        top_level_command_sites=len(set(sites) - set(placed)),
        calls_in_loop=len(calls & in_loops),
        dynamic_command_count=dynamic if isinstance(dynamic, dict) else dict(
            status="unknown", reason="lifter reports no dynamic command count"),
    )
    status = "proved" if len(proved) == len(levels) else "partial" if proved else "observed"
    return dict(value=value, status=status, evidence=[f"{evidence_id}#register_analysis.loops"],
                scope=dict(basis="lifter_loop_analysis", ordering="static_cfg"),
                limits=["trip counts hold only under the analyzed entry state; hardware timing is unknown"])


def projection_summary(projection):
    """Compact per-context evidence: control, loops and operand-field resolution."""
    projection = projection or {}
    nest = projection.get("loop_nest") or {}
    counted = projection.get("counted_control") or {}
    return dict(
        status=projection.get("status", "unknown"),
        counted_control=dict(status=counted.get("status"), commands=counted.get("commands"),
                             reason=counted.get("reason"),
                             dynamic_command_count=counted.get("dynamic_command_count")),
        field_resolution=projection.get("field_resolution"),
        loop_nest=dict(status=nest.get("status", "unknown"), reason=nest.get("reason"),
                       depth=(nest.get("value") or {}).get("depth"),
                       proved_trip_counts=(nest.get("value") or {}).get("proved_trip_counts"),
                       trip_counts=[level["trip_count"] for level in (nest.get("value") or {}).get("levels", ())]),
        operand_sequence_sha256=projection.get("operand_sequence_sha256"),
    )


def machine_projection(rows, *, model, evidence_id, decoder, instruction_offset=0, entry_state=None,
                       call_summaries=None):
    """Return witnessed fields and CCA axes; unsupported semantics stay absent.

    ``entry_state`` and ``call_summaries`` are interprocedural lifter inputs
    (caller-agreed argument values; callee return summaries), used when the
    selected lifter accepts them.
    """
    semantics, target = decoder.get("semantics"), decoder.get("target")
    if semantics is None:
        return None, dict(
            status="unknown",
            code="selected_operand_decoder_unavailable",
            next_action="Bind the original target contract and operand protocol, then reanalyze.",
        )
    isa = semantics.isa_constants(target)
    states, analysis = lifter_call(register_facts, rows, entry_state=entry_state, call_summaries=call_summaries)
    field_slots = resolved_slots = 0
    site_classes = {}
    commands, unresolved = [], Counter()
    command_count, next_command_offset, decoded_sites = 0, None, 0
    operand_summary = OperandSummary()
    structure = decoder.get("structure") or {}
    instruction_roles = structure.get("instruction_roles") or {}
    mechanisms = _MechanismObservations(instruction_roles)
    classes, raw_classes, fences = Counter(), Counter(), 0
    field_kinds = declared_field_kinds(structure)
    field_values = {}

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
        site_classes[row.addr] = klass
        decoded_sites += 1
        for name, value in decoded.items():
            label = klass + "." + name
            field_slots += 1
            resolved_slots += value is not None
            if value is None:
                unresolved[label] += 1
            elif label in field_kinds and field_values.get(label, ()) is not None:
                seen = field_values.setdefault(label, set())
                seen.add(json.dumps(value, sort_keys=True, allow_nan=False))
                if len(seen) > _MAX_FIELD_VALUES:
                    field_values[label] = None
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
    counted = lifter_call(finite_control_counts, rows, custom_opcode=isa["CUSTOM_OPCODE"], entry_state=entry_state,
                          call_summaries=call_summaries)
    nest = loop_nest(analysis, rows, sites=site_classes, roles=instruction_roles, evidence_id=evidence_id,
                     counted=counted)
    facts = []
    witness = (evidence_id + "#selected_machine_operands",)

    def fact(axis, value, reason):
        if value is not None:
            facts.append(EvidenceFact(axis, value, "inferred", witness, reason))

    dynamic = counted.get("dynamic_command_count") or {}
    if counted["status"] == "proved" and not raw_classes["UNKNOWN"]:
        fact("dispatch.n_dispatches", counted["commands"], "complete bounded scalar-control proof")
        fact("dispatch.n_fences", counted["fences"], "complete bounded scalar-control proof")
    elif dynamic.get("status") == "proved" and type(dynamic.get("value")) is int and not raw_classes["UNKNOWN"]:
        fact("dispatch.n_dispatches", dynamic["value"], "lifter-proved loop trip counts over listed command sites")
    if nest["status"] != "unknown":
        fact("envelope.calls_in_loop", nest["value"]["calls_in_loop"], "call sites inside lifter-identified loops")
    # Static listing counts hold whether or not control is proved; they are
    # site counts, never executions.
    fact("dispatch.static_command_sites", sum(raw_classes.values()), "listed selected-opcode instruction sites")
    fact("dispatch.static_fence_sites", fences, "listed CPU fence instruction sites")

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
    # Static share of classified sites whose selected role is configuration.
    # Refinements are counted under their decoded class; unknown sites refuse.
    classified = sum(classes.values())
    if classified and not raw_classes["UNKNOWN"] and all(instruction_roles.get(name) for name in classes):
        config_sites = sum(count for name, count in classes.items() if "config" in instruction_roles[name])
        fact("dispatch.config_fraction", config_sites / classified,
             "selected role roster over classified listed instruction sites, not issued commands")
    operands = operand_summary.report()
    facts.extend(_field_kind_facts(field_kinds, operands["field_fingerprints"], field_values, witness))
    cca = cca_projection(model, op="accelerator_region", source="machine", backend=(target,), facts=tuple(facts))
    for row in cca.provenance["axis_evidence"].values():
        row["scope"] = "static_instruction_sites"
    for axis in ("dispatch.n_dispatches", "dispatch.n_fences"):
        if axis in cca.provenance["axis_evidence"]:
            cca.provenance["axis_evidence"][axis]["scope"] = "selected_span_entry_to_return"
    if "envelope.calls_in_loop" in cca.provenance["axis_evidence"]:
        cca.provenance["axis_evidence"]["envelope.calls_in_loop"]["scope"] = "lifter_identified_loops"
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
        operand_layout_roster=structure.get("operand_layout_roster")
        or dict(status="unknown", reason="selected_layout_roster_not_supplied"),
        operand_field_kinds=dict(sorted(field_kinds.items())),
        field_resolution=dict(field_slots=field_slots, resolved_field_slots=resolved_slots,
                              unresolved_command_sites=unresolved["unreachable_or_unresolved_control"],
                              fraction=round(resolved_slots / field_slots, 4) if field_slots else None),
        loop_nest=nest,
        # Recorded only when the lifter consumed it; otherwise arguments stayed unconstrained.
        entry_state=dict(entry_state) if entry_state and "entry_state" in inspect.signature(
            register_facts).parameters else None,
        call_summary_targets=sorted(call_summaries) if call_summaries else [],
        semantic_region_status="unknown",
        missing=["complete_semantic_region", "dynamic_loop_multiplicity", "hardware_completion_and_timing"],
        decoder_inputs=decoder.get("decoder_inputs", []),
        role_inputs=decoder.get("role_inputs", []),
        next_action="Join the upstream semantic region and complete counted/captured command dataflow; "
        "retain field witnesses at the listed instruction PCs.",
    )
    from merlin.kernels.instruction_observations import from_projection

    cca.instructions = from_projection(report, evidence_id=evidence_id, unknowns=cca.provenance.get("unknowns", ()))
    return cca, report

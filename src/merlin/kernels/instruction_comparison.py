"""Bounded comparisons of actual instruction sites and selected operand facts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict

from merlin.compile.analysis.riscv_machine import decode_cpu


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _has_unknown(value):
    if isinstance(value, dict):
        return any(_has_unknown(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_unknown(item) for item in value)
    return value is None


class _Population:
    """Exact digest counts within a shared bound; exhaustion never means equality."""

    def __init__(self, budget):
        self.budget, self.counts, self.complete = budget, Counter(), True

    def observe(self, encoded):
        if not self.complete:
            return
        key = hashlib.sha256(encoded.encode()).hexdigest()
        if key not in self.counts:
            if self.budget[0] == 0:
                self.budget[0] += len(self.counts)
                self.counts.clear()
                self.complete = False
                return
            self.budget[0] -= 1
        self.counts[key] += 1

    def report(self):
        return dict(
            population_status="complete" if self.complete else "unknown_budget_exhausted",
            population_sha256=_digest(sorted(self.counts.items())) if self.complete else None,
            value_set_sha256=_digest(sorted(self.counts)) if self.complete else None,
            distinct_values=len(self.counts) if self.complete else None,
        )


class OperandSummary:
    """Keep order and all recovered values while bounding human-readable profiles.

    The selected ISA observer supplies the field names and values. This helper
    assigns no meaning to class names, addresses, panel sizes or execution counts.
    """

    def __init__(self, *, max_fields=128, max_population_entries=4096):
        if any(type(value) is not int or value < 1 for value in (max_fields, max_population_entries)):
            raise ValueError("operand summaries require finite positive bounds")
        self.sequence, self.classes = hashlib.sha256(), hashlib.sha256()
        self.budget = [max_population_entries]
        self.multiset, self.fields = _Population(self.budget), {}
        self.max_fields, self.max_population_entries = max_fields, max_population_entries
        self.class_sites, self.omitted_fields, self.omitted_class_sites = Counter(), 0, 0
        self.omitted, self.unresolved = Counter(), Counter()
        self.count = 0

    def observe(self, instruction, fields, operands=None):
        value = dict(instruction=instruction, fields=fields, operands=operands)
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.sequence.update(encoded.encode() + b"\n")
        self.classes.update(json.dumps(instruction).encode() + b"\n")
        self.multiset.observe(encoded)
        self.count += 1
        if instruction not in self.class_sites and len(self.class_sites) == self.max_fields:
            self.omitted_fields += len(fields)
            self.omitted_class_sites += 1
            return
        self.class_sites[instruction] += 1
        for name, field in fields.items():
            identity = instruction, name
            if identity not in self.fields:
                if len(self.fields) == self.max_fields:
                    self.omitted_fields += 1
                    continue
                self.fields[identity] = dict(sequence=hashlib.sha256(), population=_Population(self.budget),
                                             sites=0, unresolved=0, profiles=Counter())
            summary = self.fields[identity]
            summary["sites"] += 1
            token = json.dumps(field, sort_keys=True, separators=(",", ":"), allow_nan=False)
            summary["sequence"].update(json.dumps([self.class_sites[instruction], field], sort_keys=True,
                                                separators=(",", ":"), allow_nan=False).encode() + b"\n")
            summary["population"].observe(token)
            key = instruction + "." + name
            if _has_unknown(field):
                self.unresolved[key] += 1
                summary["unresolved"] += 1
            if field is not None and type(field) in (int, bool, str, float):
                # JSON keys retain distinctions such as True, 1 and 1.0.
                profile = summary["profiles"]
                if len(profile) < 16 or token in profile:
                    profile[token] += 1
                else:
                    self.omitted[key] += 1

    def report(self):
        population = self.multiset.report()
        return dict(
            count_basis="listed_instruction_sites",
            commands=self.count,
            operand_fingerprint_schema="merlin.recovered_operands.v2",
            operand_sequence_sha256=self.sequence.hexdigest(),
            instruction_class_sequence_sha256=self.classes.hexdigest(),
            operand_multiset_sha256=population["population_sha256"],
            operand_population_status=population["population_status"],
            field_fingerprints=[
                dict(instruction=instruction, field=name, sites=summary["sites"],
                     missing_sites=self.class_sites[instruction] - summary["sites"],
                     unresolved_sites=summary["unresolved"],
                     sequence_sha256=summary["sequence"].hexdigest(), **summary["population"].report())
                for (instruction, name), summary in sorted(self.fields.items())
            ],
            field_fingerprint_status="complete" if not (self.omitted_fields or self.omitted_class_sites)
            else "unknown_field_budget_exhausted",
            omitted_field_sites=self.omitted_fields,
            omitted_instruction_class_sites=self.omitted_class_sites,
            summary_limits=dict(fields=self.max_fields, population_entries=self.max_population_entries),
            field_profiles={
                instruction + "." + name: [
                    dict(value=json.loads(value), sites=count)
                    for value, count in sorted(summary["profiles"].items())[:4]
                ]
                for (instruction, name), summary in sorted(self.fields.items()) if summary["profiles"]
            },
            omitted_profile_sites=dict(self.omitted),
            unresolved_fields=dict(self.unresolved),
        )


def compare_operand_summaries(left, right):
    """Distinguish a changed operand population from changed order with equal work."""
    a, b = left.get("field_profiles", {}), right.get("field_profiles", {})
    differences = [
        dict(
            field=key,
            reference=a.get(key),
            candidate=b.get(key),
            scope="listed_instruction_sites",
            completeness="bounded_profiles_only",
        )
        for key in sorted(a.keys() | b.keys())
        if a.get(key) != b.get(key)
    ]
    fingerprints = [side.get("field_fingerprints") for side in (left, right)]
    same_schema = left.get("operand_fingerprint_schema") == right.get("operand_fingerprint_schema")
    if all(value is not None for value in fingerprints) and same_schema:
        a_fields, b_fields = ({(row["instruction"], row["field"]): row for row in rows} for rows in fingerprints)
        differences = []
        for instruction, name in sorted(a_fields.keys() | b_fields.keys()):
            key = instruction, name
            a_field, b_field = a_fields.get(key), b_fields.get(key)
            reasons = []
            if a_field is None or b_field is None:
                reasons.append("field_missing_or_unobserved")
            else:
                for field, reason in (("sites", "site_count_changed"), ("missing_sites", "missingness_changed"),
                                      ("unresolved_sites", "unresolved_sites_changed")):
                    if a_field[field] != b_field[field]:
                        reasons.append(reason)
                if a_field["value_set_sha256"] and b_field["value_set_sha256"]:
                    if a_field["value_set_sha256"] != b_field["value_set_sha256"]:
                        reasons.append("recovered_values_changed")
                if a_field["population_sha256"] and b_field["population_sha256"]:
                    if a_field["population_sha256"] != b_field["population_sha256"]:
                        reasons.append("recovered_population_changed")
                elif a_field["sequence_sha256"] != b_field["sequence_sha256"]:
                    reasons.append("recovered_sequence_changed_population_unknown")
            if reasons:
                label = instruction + "." + name
                differences.append(dict(field=label, instruction=instruction, operand_field=name,
                                        reference=a.get(label), candidate=b.get(label),
                                        reference_fingerprint=a_field, candidate_fingerprint=b_field,
                                        scope="listed_instruction_sites", completeness="complete_field_sequences",
                                        reasons=reasons))
        # Counts can change on every field at once. Keep recovered value
        # changes visible before filling the bounded reply with those counts.
        differences.sort(key=lambda row: ("recovered_values_changed" not in row["reasons"],
                                           row["field"]))

    def same(key):
        return left[key] == right[key] if same_schema and left.get(key) and right.get(key) else None

    ordered, population = same("operand_sequence_sha256"), same("operand_multiset_sha256")
    return dict(
        status="compared_recovered_operands" if ordered is not None else "unknown",
        same_recovered_operands=ordered,
        same_recovered_operand_population=population,
        same_instruction_class_order=same("instruction_class_sequence_sha256"),
        order_change="reordered_same_recovered_operands"
        if population is True and ordered is False
        else "none_observed"
        if ordered is True
        else "different_or_unresolved_operand_population",
        operand_field_differences=differences[:8],
        omitted_field_differences=max(0, len(differences) - 8),
        field_localization_status="unknown_fingerprint_schema_mismatch" if not same_schema else
        "complete_observed_field_roster" if all(side.get("field_fingerprint_status") == "complete"
                                                for side in (left, right)) else "bounded_or_unavailable",
        semantic_alignment="unknown",
        execution_multiplicity="unknown",
    )


def instruction_summary(rows, *, decoder=None, offset=0, limit=8):
    """Retain order, registers and immediates; normalize only internal PC labels.

    A same-PC branch can have a different displacement after compression or
    relocation. Normalizing its destination to a local ordinal preserves the
    edge without erasing operand strides, addresses or other immediate choices.
    """
    decoder = decoder or {}
    semantics, target = decoder.get("semantics"), decoder.get("target")
    isa = semantics.isa_constants(target) if semantics else {}
    ordinal = {row.addr: index for index, row in enumerate(rows)}
    tokens, histogram, page = [], Counter(), []
    for index, row in enumerate(rows):
        cpu = decode_cpu(row)
        token = asdict(cpu)
        token.pop("size")
        if cpu.target is not None:
            token["target"] = (
                dict(local_instruction=ordinal[cpu.target]) if cpu.target in ordinal else dict(external_pc=cpu.target)
            )
        if len(row.hexcode) == 8 and int(row.hexcode, 16) & 127 == isa.get("CUSTOM_OPCODE"):
            from merlin.compile.analysis.rocc_commands import fields_of

            fields = fields_of(int(row.hexcode, 16))
            token = dict(kind="endpoint", instruction=isa["FUNCT_CLASS"].get(fields["funct"], "UNKNOWN"), fields=fields)
        elif cpu.kind in ("unknown", "opaque", "fp", "fp_memory"):
            token["word"] = row.hexcode
        if row.relocations:
            token["relocations"] = row.relocations
        tokens.append(token)
        histogram[token["kind"]] += 1
        if offset <= index < offset + limit:
            page.append(
                dict(
                    index=index,
                    pc=row.addr,
                    word=row.hexcode,
                    mnemonic=row.mnemonic,
                    operands=row.operands,
                    decoded=token,
                )
            )
    return dict(
        schema="merlin.instruction_sites.v1",
        count_basis="listed_instruction_sites",
        ordered_structure_sha256=_digest(tokens),
        machine_words_sha256=_digest([r.hexcode for r in rows]),
        instructions=len(rows),
        cpu_kinds=dict(histogram),
        offset=offset,
        page=page,
        next_offset=offset + len(page) if offset + len(page) < len(rows) else None,
        normalization="internal branch targets to instruction ordinals; registers and other immediates retained",
    )


def compare_instruction_summaries(left, right):
    """Report structural differences without promoting static sites into work."""
    a, b = left.get("instruction_view"), right.get("instruction_view")
    if not a or not b:
        return dict(status="unknown", reason="machine_instruction_view_unavailable")
    pa, pb = left.get("mechanism_projection") or {}, right.get("mechanism_projection") or {}
    roles_a, roles_b = pa.get("role_sites", {}), pb.get("role_sites", {})
    operands = compare_operand_summaries(pa, pb)
    return dict(
        status="compared_static_structure",
        count_basis="listed_instruction_sites",
        same_words=a["machine_words_sha256"] == b["machine_words_sha256"],
        same_ordered_structure=a["ordered_structure_sha256"] == b["ordered_structure_sha256"],
        instruction_delta=b["instructions"] - a["instructions"],
        role_sites={
            key: dict(reference=roles_a.get(key, 0), candidate=roles_b.get(key, 0))
            for key in sorted(roles_a.keys() | roles_b.keys())
        },
        **{key: value for key, value in operands.items() if key != "status"},
        next_action="Align complete semantic regions before interpreting instruction differences as opportunities.",
    )

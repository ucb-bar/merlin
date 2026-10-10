"""Restricted data-bound instruction legality checks, without lowering strategies.

Version one checks trace decoding and membership in a complete hardware funct
table. Memory bounds and temporal protocol rules are unavailable, never inferred
from instruction names, schedules, tensor shapes or performance observations.
"""

from __future__ import annotations

import copy
import json

from merlin.targetgen.rtl_checks import Check, CheckReport


class BoundChecks:
    def __init__(self, semantics, rules):
        if type(rules) is not list or not rules or any(type(rule) is not str for rule in rules):
            raise ValueError("hardware checks require a unique explicit rule list")
        if len(set(rules)) != len(rules) or any(rule not in {"decode_clean", "legal_funct"} for rule in rules):
            raise ValueError("unsupported hardware check; strategy rules cannot become legality checks")
        self._semantics = semantics
        self._rules = tuple(rules)

    def load_default_facts(self, target):
        self._semantics.isa_constants(target)
        return copy.deepcopy(self._semantics._facts)

    def project_facts(self, facts_rec):
        if facts_rec != self._semantics._facts:
            raise ValueError("hardware checks require the bound original facts snapshot")
        return copy.deepcopy(facts_rec)

    def screen(self, trace, capsule=None, rtl_facts=None, *, target, command_buffer=None):
        facts = self.load_default_facts(target) if rtl_facts is None else self.project_facts(rtl_facts)
        instructions = trace.get("instructions") if type(trace) is dict else None
        if type(instructions) is not list or any(type(item) is not dict for item in instructions):
            raise ValueError("hardware checks require the complete decoded instruction roster")
        report = CheckReport((capsule or {}).get("name"), None, facts)
        observed = [item for item in instructions if item.get("class") != "FENCE" or item.get("funct") is not None]
        for rule in self._rules:
            if rule == "legal_funct" and not self._semantics.complete_decode:
                report.checks.append(Check(rule, "T0", "error", "skipped", "hardware funct table is incomplete"))
                continue
            bad = []
            for index, item in enumerate(observed):
                code = item.get("funct")
                if rule == "legal_funct":
                    valid = type(code) is int and code in self._semantics.legal_codes
                else:
                    expected = self._semantics._isa["FUNCT_CLASS"].get(code) if type(code) is int else None
                    valid = expected is not None and item.get("class") == expected
                if not valid:
                    bad.append(index)
            passed = bool(observed) and not bad
            report.checks.append(
                Check(
                    rule,
                    "T0",
                    "error",
                    "pass" if passed else "fail",
                    "complete instruction roster checked",
                    expected="nonempty conforming instructions",
                    got={"count": len(observed), "invalid": bad},
                )
            )
        return report

    def compile_trace_checks(self, facts_rec, capsule, prefix):
        self.project_facts(facts_rec)
        if type(prefix) is not str or not prefix.isidentifier():
            raise ValueError("trace check prefix must be an identifier")
        return "\n".join(f"// {prefix}: RTL_CHECK {rule} pass" for rule in self._rules)

    def render_trace(self, trace, facts_rec):
        report = self.screen(trace, rtl_facts=facts_rec, target=self._semantics._target)
        lines = [f"RTL_CHECK {check.id} {check.status}" for check in report.checks]
        lines.extend(json.dumps(item, sort_keys=True) for item in trace["instructions"])
        return "\n".join(lines) + "\n"

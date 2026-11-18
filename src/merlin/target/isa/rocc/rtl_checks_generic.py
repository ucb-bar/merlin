"""Generic RoCC hardware-legality checks, driven by RTL facts and a target's declared protocol.

A complete ``rocc_semantics.rtl_checks`` capability (see :func:`merlin.target.isa.rocc.rtl_checks.selected_checks`)
written once for every RoCC accelerator. It screens a decoded instruction trace
(:mod:`merlin.target.isa.rocc.decode`) for HARDWARE LEGALITY only. Every rule's bound is derivable from
the target's RTL facts or its contract encodings; none encodes a lowering strategy, a schedule, a tile
count, or how a particular compiler chooses to fill an instruction's fields. A program that issues only
legal commands passes, however it is scheduled.

Rule kinds (target-independent code) and what grounds them:

``decode_clean``
    Every instruction decodes to a class of the target's funct->class encoding (``encoding.semantic_class``
    over the RTL decoder's funct field). An undecodable command is one the decoder would not dispatch.
``funct_legal``
    Every emitted funct is in the RTL decoder's legal set (facts ``funct_decode_table.legal_funct``).
``local_address_bounds``
    Every local (scratchpad / accumulator) address an instruction carries, plus the rows it transfers,
    stays inside the memory its encoded space selector chooses. Capacities and row masks come from the
    CIRCT memory facts; the selector is the contract's local-address flag. Payload bits outside a
    store's row field alias a lower row and are reported. A declared all-ones "retain" sentinel is not
    an address.
``precedes``
    A role that consumes state another role must have set (the contract names the pair, e.g. a compute
    that flips preloaded weights needs a preload first) is preceded by it.
``config_before_use``
    A role that reads a configuration register is preceded by the command that writes it.
``bracket``
    The command stream opens and closes with the ordering role (a fence): RoCC commands complete
    asynchronously to the host, so results are only visible to it after a fence.
``output_store_coverage``
    Every declared output cell is written by some store and no store writes past the declared extent.
    Store footprints are computed from the encoded fields with the RTL's own store semantics (rows x
    cols at the configured DRAM pitch; a native-pooling store walks its encoded pooled positions).

The protocol — which classes play which ROLE, which decoded field carries which MEANING, which fields
are local addresses, how a role is spelled, and which rules apply in which order — is the contract's
``rtl_checks`` block. A target whose contract declares none has no protocol: :func:`screen` refuses
(``RtlChecksUnavailable``) rather than borrowing another target's roles.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from merlin.common.target_data import interface as _facts_interface
from merlin.target.isa.rocc.rtl_checks import Check, CheckReport, RtlChecksUnavailable
from merlin.target.rtl.facts import load_facts

# ---- rtl_checks_generic ------------------------------------------------------------------------


def _declared_op(capsule: dict | None) -> str | None:
    if not capsule:
        return None
    op = (capsule.get("operation") or {}).get("op")
    return op.lower() if isinstance(op, str) else None


def _declared_output_shape(capsule: dict | None) -> tuple[int, int] | None:
    """Best-effort (M, N) of the declared output, GENERAL (declared shapes only, no golden).

    For a plain matmul the output is (M, N) with lhs (M,K) and weight (K,N). We read the declared
    input roles. Returns None when the shape cannot be derived generally (the caller has no declared extent
    and must not guess).
    """
    if not capsule:
        return None
    inputs = capsule.get("inputs") or []
    by_role: dict[str, list] = {}
    for t in inputs:
        by_role.setdefault(t.get("role"), []).append(t.get("shape"))
    lhs = by_role.get("input")
    w = by_role.get("weight")
    if not lhs or not w:
        return None
    lhs_shape, w_shape = lhs[0], w[0]
    if not (isinstance(lhs_shape, list) and isinstance(w_shape, list) and len(lhs_shape) == 2 and len(w_shape) == 2):
        return None
    M = lhs_shape[0]
    N = w_shape[1]
    return (int(M), int(N))


SCHEMA = "rtl_checks/v0"  # advisory artifact tag — deliberately NOT a frozen src/merlin/resources/contracts schema


RENDER_SCHEMA = "rtl-trace-render/v1"


PROTOCOL_KEY = "rtl_checks"


_DTYPE_BYTES = {
    "i8": 1,
    "int8": 1,
    "u8": 1,
    "i16": 2,
    "bf16": 2,
    "fp16": 2,
    "f16": 2,
    "i32": 4,
    "fp32": 4,
    "f32": 4,
}


@dataclass(frozen=True)
class Protocol:
    """A target's declared RoCC protocol (the contract's ``rtl_checks`` block), as data."""

    target: str | None
    class_roles: dict[str, frozenset[str]]
    role_labels: dict[str, str]
    fields: dict[str, str]
    rules: tuple[dict, ...]
    local_addresses: tuple[dict, ...] = ()
    pooling: dict | None = None
    vocabulary: tuple[str, ...] | None = None  # encoding.semantic_class labels (None: not declared)
    custom_slot: int | None = None
    local_flags: dict[str, int] = field(default_factory=dict)

    def classes(self, role: str) -> frozenset[str]:
        return frozenset(c for c, roles in self.class_roles.items() if role in roles)

    def label(self, role: str) -> str:
        if role in self.role_labels:
            return self.role_labels[role]
        names = sorted(self.classes(role))
        return "/".join(names) if names else role

    def f(self, meaning: str) -> str:
        return self.fields.get(meaning, meaning)


def protocol_from_contract(contract: dict | None, target: str | None = None) -> Protocol | None:
    """Build a :class:`Protocol` from a target contract, or ``None`` when it declares none."""
    from merlin.target.isa.rocc.legality_checks import RULE_KINDS

    block = (contract or {}).get(PROTOCOL_KEY)
    if not isinstance(block, dict):
        return None
    roles = {
        str(cls): frozenset(str(t) for t in ([tags] if isinstance(tags, str) else (tags or ())))
        for cls, tags in (block.get("class_roles") or {}).items()
    }
    rules = []
    for spec in block.get("rules") or ():
        if not isinstance(spec, dict) or not spec.get("id") or not spec.get("rule"):
            raise RtlChecksUnavailable(f"{target}: an rtl_checks rule needs an id and a rule kind")
        if spec["rule"] not in RULE_KINDS:
            raise RtlChecksUnavailable(f"{target}: rtl_checks rule {spec['id']!r} names unknown kind {spec['rule']!r}")
        rules.append(dict(spec))
    local = []
    for spec in block.get("local_addresses") or ():
        if not isinstance(spec, dict) or not spec.get("role") or not spec.get("address"):
            raise RtlChecksUnavailable(f"{target}: an rtl_checks local_addresses entry needs a role and an address")
        local.append(dict(spec))
    encoding = (contract or {}).get("encoding") or {}
    sem = encoding.get("semantic_class")
    slot = encoding.get("rocc_custom_slot")
    try:
        from merlin.target.isa.rocc import semantics as _sem

        flags = _sem.local_address_flags(_sem.operand_roles(contract), encoding.get("addr_len"))
    except Exception:  # noqa: BLE001 — undeclared/malformed flags leave the accumulator selector UNKNOWN
        flags = {}
    pooling = block.get("native_pooling")
    return Protocol(
        target=target,
        class_roles=roles,
        role_labels={str(k): str(v) for k, v in (block.get("role_labels") or {}).items()},
        fields={str(k): str(v) for k, v in (block.get("fields") or {}).items()},
        rules=tuple(rules),
        local_addresses=tuple(local),
        pooling=dict(pooling) if isinstance(pooling, dict) else None,
        vocabulary=tuple(str(v) for v in sem.values()) if isinstance(sem, dict) and sem else None,
        custom_slot=slot if type(slot) is int else None,
        local_flags=flags,
    )


def protocol_for(target: str) -> Protocol:
    """The protocol ``target``'s contract declares (read fresh: never cached by target name)."""
    from merlin.target.registry import experiment_descriptor as target_experiment

    proto = protocol_from_contract(target_experiment.load_capability_manifest(target).contract, target)
    if proto is None:
        raise RtlChecksUnavailable(f"{target}: the target contract declares no `{PROTOCOL_KEY}` protocol")
    return proto


def _facts_target(facts_rec: dict | None) -> str | None:
    """The target a facts RECORD answers for (a served alias names the asker first)."""
    if not isinstance(facts_rec, dict):
        return None
    served = facts_rec.get("served_for")
    if isinstance(served, dict) and isinstance(served.get("target"), str) and served["target"]:
        return served["target"]
    body = facts_rec.get("facts", facts_rec)
    t = body.get("target") if isinstance(body, dict) else None
    return t if isinstance(t, str) and t else None


def _resolve(protocol: Protocol | None, rtl_facts: dict | None) -> Protocol | None:
    """The explicit protocol, else the one the flat facts' ``target`` declares, else ``None``."""
    if protocol is not None:
        return protocol
    target = (rtl_facts or {}).get("target")
    if isinstance(target, str) and target:
        try:
            return protocol_for(target)
        except Exception:  # noqa: BLE001 — no protocol: protocol-dependent findings stay UNKNOWN
            return None
    return None


def _hint(spec: dict | None, default: str) -> str:
    return (spec or {}).get("fix_hint") or default


def _empty_facts(target: str | None) -> dict[str, Any]:
    return {
        "mesh": None,
        "scratchpad_bytes": None,
        "scratchpad_rows": None,
        "scratchpad_row_mask": None,
        "accumulator_rows": None,
        "accumulator_select_bit": None,
        "accumulator_row_mask": None,
        "local_address_data_width": None,
        "local_address_data_mask": None,
        "local_address_sentinel": None,
        "max_pool_supported": None,
        "legal_funct": None,
        "custom_opcode": None,
        "funct3": None,
        "from": "UNKNOWN (RTL facts not derivable)",
        "target": target,
    }


def _row_mask(rows: object) -> int | None:
    return (1 << max(1, (rows - 1).bit_length())) - 1 if isinstance(rows, int) and rows > 0 else None


def _pool_supported(body: dict, proto: Protocol | None) -> bool | None:
    feature = ((proto.pooling if proto else None) or {}).get("feature")
    if not feature:
        return None
    block = _facts_interface(body, "elaborated_rtl_features") or {}
    value = (block.get("features") or {}).get(feature)
    return value if isinstance(value, bool) and block.get("status") == "derived" else None


def load_default_facts(target: str) -> dict[str, Any]:
    """The RTL-derived facts ``target``'s legality rules read, flattened from its facts record.

    FAIL-CLOSED: anything not derivable stays ``None`` and the rules that need it SKIP. The flat dict
    records ``target`` so target-less helpers resolve the same protocol."""
    facts = _empty_facts(target)
    try:
        proto = protocol_for(target)
    except Exception:  # noqa: BLE001 — base facts remain useful without a protocol
        proto = None
    try:
        rec = load_facts(target)
    except Exception:  # noqa: BLE001
        return facts
    body = rec.get("facts", rec)
    mesh = next((a for a in body.get("arrays", []) if a.get("name") == "mesh"), None)
    if mesh:
        facts["mesh"] = [mesh["rows"], mesh["cols"]]
    try:
        from merlin.target.capabilities.address_space import (
            accumulator_store,
            derive_address_space,
            operand_store,
        )

        space = derive_address_space(target, facts=rec)
        spad, acc = operand_store(space).store, accumulator_store(space).store
        if spad is not None:
            facts.update(scratchpad_bytes=spad.nbytes, scratchpad_rows=spad.total_rows)
            facts["scratchpad_row_mask"] = _row_mask(spad.total_rows)
        if acc is not None:
            facts["accumulator_rows"] = acc.total_rows
            facts["accumulator_row_mask"] = _row_mask(acc.total_rows)
        if space.separate_accumulator_space is True:
            select = (proto.local_flags if proto else {}).get("accumulator_select")
            facts["accumulator_select_bit"] = select if isinstance(select, int) else None
        masks = [m for m in (facts["scratchpad_row_mask"], facts["accumulator_row_mask"]) if isinstance(m, int)]
        if masks and (space.separate_accumulator_space is not True or len(masks) == 2):
            # The local-address data field is as wide as the largest local store's row address.
            facts["local_address_data_mask"] = max(masks)
            facts["local_address_data_width"] = max(masks).bit_length()
    except Exception:  # noqa: BLE001 — only these optional fields stay UNKNOWN
        pass
    try:
        from merlin.target.isa.rocc import semantics

        facts["local_address_sentinel"] = semantics.isa_constants(target).get("RETAIN_SENTINEL")
    except Exception:  # noqa: BLE001
        pass
    table = _facts_interface(body, "funct_decode_table")
    if table:
        facts.update(legal_funct=table.get("legal_funct"), custom_opcode=table.get("custom_opcode"))
        facts["funct3"] = table.get("funct3")
    facts["max_pool_supported"] = _pool_supported(body, proto)
    facts["from"] = f"circt_introspect facts.json ({rec.get('generator', {}).get('version', '?')})"
    return facts


def project_facts(facts_rec: dict) -> dict:
    """The flat facts a facts RECORD grounds without re-resolving the target's own record."""
    body = facts_rec.get("facts", facts_rec)
    mesh = next((a for a in body.get("arrays", []) if a.get("name") == "mesh"), {})
    out: dict[str, Any] = {}
    if mesh:
        out["mesh"] = [mesh["rows"], mesh["cols"]]
    who = _facts_target(facts_rec)
    proto = None
    if who:
        out["target"] = who
        try:
            proto = protocol_for(who)
        except Exception:  # noqa: BLE001
            proto = None
    supported = _pool_supported(body, proto)
    if supported is not None:
        out["max_pool_supported"] = supported
    table = _facts_interface(body, "funct_decode_table")
    if table and table.get("legal_funct"):
        out["legal_funct"] = table.get("legal_funct")
    return out


def _classes(trace: dict) -> list[str]:
    return [i.get("class") for i in trace.get("instructions", [])]


def _first(classes: list[str], names) -> int:
    return next((k for k, c in enumerate(classes) if c in names), -1)


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _check_decode_clean(trace: dict, proto: Protocol, spec: dict | None = None) -> Check:
    cid = (spec or {}).get("id", "T0.decode_clean")
    instrs = trace.get("instructions", [])
    if not instrs:
        return Check(
            cid,
            "T0",
            "error",
            "fail",
            "the trace decodes to no RoCC instruction at all",
            expected=">0 instructions",
            got=0,
            fix_hint=_hint(spec, "the emitted program issues no command on the target's custom opcode"),
        )
    unknown = [i.get("index") for i in instrs if i.get("class") == "UNKNOWN"]
    if unknown:
        slot = f"custom-{proto.custom_slot}" if proto.custom_slot is not None else "custom-opcode"
        return Check(
            cid,
            "T0",
            "error",
            "fail",
            f"{len(unknown)} instruction(s) do not decode to any class of the target's encoding",
            expected=0,
            got=len(unknown),
            evidence={"instruction_indices": unknown[:8]},
            fix_hint=_hint(spec, f"emit only {slot} commands whose funct the target's encoding declares"),
        )
    return Check(
        cid, "T0", "error", "pass", f"all {len(instrs)} instructions decode to declared classes", expected=0, got=0
    )


def _check_decode_funct_legal(trace: dict, rtl_facts: dict, spec: dict | None = None) -> Check:
    cid = (spec or {}).get("id", "T0.decode_funct_legal")
    legal_list = rtl_facts.get("legal_funct")
    if not legal_list:
        return Check(cid, "T0", "error", "skipped", "the RTL decoder's legal funct set is UNKNOWN")
    legal = set(legal_list)
    bad = [i for i in trace.get("instructions", []) if _is_int(i.get("funct")) and i["funct"] not in legal]
    if bad:
        return Check(
            cid,
            "T0",
            "error",
            "fail",
            f"{len(bad)} instruction(s) use funct {sorted({i['funct'] for i in bad})}, which the "
            "RTL decoder does not accept",
            expected=0,
            got=len(bad),
            evidence={"instruction_indices": [i.get("index") for i in bad][:8], "legal_funct": sorted(legal)},
            fix_hint=_hint(spec, "use only functs in the RTL decoder's legal set"),
        )
    return Check(cid, "T0", "error", "pass", "every emitted funct is in the RTL decoder's legal set", expected=0, got=0)


def _active_store_pooled(dec: dict, proto: Protocol) -> bool:
    pooling = proto.pooling or {}
    enable = pooling.get("enable_field")
    return bool(enable) and _is_int(dec.get(enable)) and dec[enable] > 0


def _check_local_address_bounds(
    trace: dict, rtl_facts: dict, proto: Protocol | None = None, spec: dict | None = None
) -> Check:
    """Every local address (and transfer extent) stays inside the memory its space selector picks."""
    cid = (spec or {}).get("id", "T0.local_address_bounds")
    proto = _resolve(proto, rtl_facts)
    if proto is None:
        return Check(cid, "T0", "error", "skipped", "the target declares no rtl_checks protocol")
    spad_rows = rtl_facts.get("scratchpad_rows")
    if not _is_int(spad_rows):
        return Check(cid, "T0", "error", "skipped", "the scratchpad row capacity is UNKNOWN (no RTL memory facts)")
    spad_mask = rtl_facts.get("scratchpad_row_mask")
    spad_mask = spad_mask if _is_int(spad_mask) else _row_mask(spad_rows)
    acc_rows = rtl_facts.get("accumulator_rows")
    acc_mask = rtl_facts.get("accumulator_row_mask")
    acc_mask = acc_mask if _is_int(acc_mask) else _row_mask(acc_rows)
    select = rtl_facts.get("accumulator_select_bit")
    select_known = _is_int(select) and select > 0 and select & (select - 1) == 0
    data_mask = rtl_facts.get("local_address_data_mask")
    if not _is_int(data_mask):
        masks = [m for m in (spad_mask, acc_mask) if _is_int(m)]
        data_mask = max(masks) if masks else None
    sentinel = rtl_facts.get("local_address_sentinel")

    by_role: dict[str, list[dict]] = {}
    for entry in proto.local_addresses:
        by_role.setdefault(str(entry["role"]), []).append(entry)
    store_cfg = (proto.pooling or {}).get("config_role", "config_store")
    pooled = False
    spad, acc = [], []  # (index, first row, exclusive end)
    unresolved_rows, invalid_rows, unresolved_space, noncanonical_spad, noncanonical_acc = [], [], [], [], []
    n_addr = 0
    for inst in trace.get("instructions", []):
        cls, dec, idx = inst.get("class"), inst.get("decoded") or {}, inst.get("index")
        if cls in proto.classes(store_cfg):
            pooled = _active_store_pooled(dec, proto)
        for role, entries in by_role.items():
            if cls not in proto.classes(role):
                continue
            for entry in entries:
                addr = dec.get(entry["address"])
                if not _is_int(addr) or (_is_int(sentinel) and addr == sentinel):
                    continue
                n_addr += 1
                rows = 1
                if entry.get("rows") and not (role == "movement_out" and pooled):
                    rows = dec.get(entry["rows"])
                    if not _is_int(rows):
                        unresolved_rows.append(idx)
                        continue
                    if rows <= 0:
                        invalid_rows.append(idx)
                        continue
                if select_known and addr & select:
                    if not (_is_int(acc_rows) and _is_int(acc_mask)):
                        unresolved_space.append(idx)
                        continue
                    row = addr & acc_mask
                    if _is_int(data_mask) and (addr & data_mask) & ~acc_mask:
                        noncanonical_acc.append(idx)
                    acc.append((idx, row, row + rows))
                else:
                    if _is_int(acc_rows) and not select_known and (not _is_int(data_mask) or addr & ~data_mask):
                        unresolved_space.append(idx)  # a high bit might select the accumulator
                        continue
                    row = addr & spad_mask
                    if _is_int(data_mask) and (addr & data_mask) & ~spad_mask:
                        noncanonical_spad.append(idx)
                    spad.append((idx, row, row + rows))
    if not n_addr:
        return Check(cid, "T0", "error", "skipped", "no instruction carries a decodable local address")
    over_spad = [t for t in spad if t[2] > spad_rows]
    over_acc = [t for t in acc if _is_int(acc_rows) and t[2] > acc_rows]
    evidence = {
        "scratchpad_rows_capacity": spad_rows,
        "scratchpad_row_mask": spad_mask,
        "scratchpad_max_row": max((t[1] for t in spad), default=None),
        "scratchpad_max_row_exclusive": max((t[2] for t in spad), default=None),
        "accumulator_rows_capacity": acc_rows,
        "accumulator_row_mask": acc_mask,
        "accumulator_select_bit": select,
        "accumulator_max_row": max((t[1] for t in acc), default=None),
        "accumulator_max_row_exclusive": max((t[2] for t in acc), default=None),
        "local_address_data_mask": data_mask,
        "noncanonical_scratchpad_instruction_indices": noncanonical_spad[:8],
        "noncanonical_accumulator_instruction_indices": noncanonical_acc[:8],
        "unresolved_address_space_instruction_indices": unresolved_space[:8],
        "invalid_row_count_instruction_indices": invalid_rows[:8],
        "unresolved_row_count_instruction_indices": unresolved_rows[:8],
    }
    problems = []
    if over_spad:
        problems.append(f"scratchpad access reaches row {max(t[2] for t in over_spad)} of {spad_rows}")
    if over_acc:
        problems.append(f"accumulator access reaches row {max(t[2] for t in over_acc)} of {acc_rows}")
    if noncanonical_spad or noncanonical_acc:
        problems.append(
            f"{len(noncanonical_spad) + len(noncanonical_acc)} address(es) set row bits beyond "
            "their memory's depth, which the RTL drops (the access aliases a lower row)"
        )
    if invalid_rows:
        problems.append(f"{len(invalid_rows)} transfer(s) encode a non-positive row count")
    if problems:
        bad = [t[0] for t in over_spad + over_acc] + noncanonical_spad + noncanonical_acc + invalid_rows
        return Check(
            cid,
            "T0",
            "error",
            "fail",
            "local address out of bounds: " + "; ".join(problems),
            expected={"scratchpad_rows": spad_rows, "accumulator_rows": acc_rows},
            got={
                "scratchpad_max_row_exclusive": evidence["scratchpad_max_row_exclusive"],
                "accumulator_max_row_exclusive": evidence["accumulator_max_row_exclusive"],
            },
            evidence={**evidence, "instruction_indices": bad[:8]},
            fix_hint=_hint(spec, "keep every local access inside the RTL-derived memory depth"),
        )
    if unresolved_space or unresolved_rows:
        return Check(
            cid,
            "T0",
            "error",
            "skipped",
            "some local accesses could not be bounded (address space or row count UNKNOWN)",
            evidence={**evidence, "instruction_indices": (unresolved_space + unresolved_rows)[:8]},
        )
    return Check(
        cid, "T0", "error", "pass", f"all {n_addr} local address(es) are inside their memory", evidence=evidence
    )


def _check_precedes(trace: dict, proto: Protocol, spec: dict) -> Check:
    """Every ``then``-role command is preceded by a ``first``-role command."""
    cid = spec["id"]
    first_role, then_role = str(spec.get("first")), str(spec.get("then"))
    first, then = proto.label(first_role), proto.label(then_role)
    cls = _classes(trace)
    k_then, k_first = _first(cls, proto.classes(then_role)), _first(cls, proto.classes(first_role))
    if k_then == -1:
        return Check(cid, "T0", "error", "pass", f"no {then} is issued")
    if k_first == -1 or k_first > k_then:
        where = "none is issued" if k_first == -1 else f"the first is at {k_first}"
        return Check(
            cid,
            "T0",
            "error",
            "fail",
            f"{then} at {k_then} is not preceded by a {first} ({where})",
            expected=f"{first} before {then}",
            got=f"{then} at {k_then}",
            evidence={"instruction_indices": [k_then]},
            fix_hint=_hint(spec, f"issue {first} before {then}"),
        )
    return Check(cid, "T0", "error", "pass", f"{first} at {k_first} precedes the first {then} at {k_then}")

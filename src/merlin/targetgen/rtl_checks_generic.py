"""Generic RoCC hardware-legality checks, driven by RTL facts and a target's declared protocol.

A complete ``rocc_semantics.rtl_checks`` capability (see :func:`merlin.targetgen.rtl_checks.selected_checks`)
written once for every RoCC accelerator. It screens a decoded instruction trace
(:mod:`merlin.targetgen.rocc.decode`) for HARDWARE LEGALITY only. Every rule's bound is derivable from
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

from dataclasses import dataclass, field
from typing import Any

from merlin.common.facts_view import interface as _facts_interface
from merlin.targetgen.rtl.facts import load_facts
from merlin.targetgen.rtl_checks import (
    Check,
    CheckReport,
    RtlChecksUnavailable,
    _declared_op,
    _declared_output_shape,
)

SCHEMA = "rtl_checks/v0"  # advisory artifact tag — deliberately NOT a frozen merlin/contract schema
RENDER_SCHEMA = "rtl-trace-render/v1"

#: Contract key holding a target's protocol.
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


# ======================================================================================== protocol
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
        from merlin.targetgen.rocc import semantics as _sem

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
    from merlin.targetgen import target_experiment

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


# ======================================================================================= RTL facts
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
        from merlin.targetgen.address_space import accumulator_store, derive_address_space, operand_store

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
        from merlin.targetgen.rocc import semantics

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


# =================================================================================== trace helpers
def _classes(trace: dict) -> list[str]:
    return [i.get("class") for i in trace.get("instructions", [])]


def _first(classes: list[str], names) -> int:
    return next((k for k, c in enumerate(classes) if c in names), -1)


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# ======================================================================================= rules
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


def _check_config_before_use(trace: dict, proto: Protocol, spec: dict) -> Check:
    """Each use role is preceded by the role that writes the configuration it reads."""
    cid = spec["id"]
    cls = _classes(trace)
    problems, hints = [], []
    for pair in spec.get("pairs") or ():
        use, cfg = proto.label(str(pair.get("use"))), proto.label(str(pair.get("config")))
        hints.append(f"{cfg} before the first {use}")
        k_use, k_cfg = (
            _first(cls, proto.classes(str(pair.get("use")))),
            _first(cls, proto.classes(str(pair.get("config")))),
        )
        if k_use != -1 and (k_cfg == -1 or k_cfg > k_use):
            problems.append(f"{use} at {k_use} reads configuration no {cfg} has written yet")
    if problems:
        return Check(
            cid,
            "T0",
            "error",
            "fail",
            "; ".join(problems),
            expected="configuration before use",
            got="; ".join(problems),
            fix_hint=_hint(spec, "issue " + " and ".join(hints)),
        )
    return Check(cid, "T0", "error", "pass", "every configuration is written before it is read")


def _check_bracket(trace: dict, proto: Protocol, spec: dict) -> Check:
    """The stream opens and closes with the ordering role."""
    cid = spec["id"]
    role = str(spec.get("role"))
    name, names, cls = proto.label(role), proto.classes(role), _classes(trace)
    if not cls:
        return Check(cid, "T0", "error", "skipped", "empty trace")
    missing = [w for w, c in (("open", cls[0]), ("close", cls[-1])) if c not in names]
    if missing:
        return Check(
            cid,
            "T0",
            "error",
            "fail",
            f"the stream does not {' and '.join(missing)} with a {name} (first={cls[0]}, last={cls[-1]})",
            expected=f"{name} … {name}",
            got=f"{cls[0]} … {cls[-1]}",
            fix_hint=_hint(spec, f"issue a {name} before the first and after the last command"),
        )
    return Check(cid, "T0", "error", "pass", f"the stream opens and closes with a {name}")


# ================================================================================ store coverage
def _stores(trace: dict, rtl_facts: dict, proto: Protocol) -> list[dict]:
    """Every store with the DRAM pitch and footprint its active configuration gives it.

    Footprint rows follow the RTL store semantics: the encoded row count, or — when the active store
    configuration enables native pooling and the RTL supports it — the encoded pooled positions."""
    pooling = proto.pooling or {}
    cfg_classes = proto.classes(pooling.get("config_role", "config_store"))
    pitch_key, dram_key = proto.f("store_pitch"), proto.f("dram")
    rows_key, cols_key = proto.f("rows"), proto.f("cols")
    supported = rtl_facts.get("max_pool_supported")
    pitch, config, out = None, {}, []
    for inst in trace.get("instructions", []):
        dec = inst.get("decoded") or {}
        if inst.get("class") in cfg_classes:
            v = dec.get(pitch_key)
            pitch = v if _is_int(v) and v > 0 else None
            config = dict(dec)
            continue
        if inst.get("class") not in proto.classes("movement_out"):
            continue
        dram = dec.get(dram_key) if isinstance(dec.get(dram_key), dict) else {}
        rows, why = dec.get(rows_key), ""
        if _active_store_pooled(config, proto) and supported is not False:
            positions = [config.get(f) for f in pooling.get("output_positions") or ()]
            if supported is None:
                rows, why = None, "the store enables native pooling but the RTL pooling capability is UNKNOWN"
            elif positions and all(_is_int(p) for p in positions):
                rows = 1
                for p in positions:
                    rows *= p
            else:
                rows, why = None, "the store enables native pooling but its pooled positions are not decoded"
        out.append(
            {
                "index": inst.get("index"),
                "kind": dram.get("kind"),
                "arg_index": dram.get("arg_index"),
                "offset": dram.get("offset"),
                "rows": rows if _is_int(rows) else None,
                "cols": dec.get(cols_key),
                "stride": pitch,
                "unknown": why or "a store carries no decodable offset/rows/cols/pitch",
            }
        )
    return out


def _covered_cells(tiles: list[tuple[int, int, int, int]], R: int, C: int) -> tuple[int, list[int], list[int]]:
    """Cells of an R x C extent the tiles (row0, col0, rows, cols) cover, plus untouched rows/cols."""
    bounds = sorted({0, R} | {b for r0, _c, nr, _n in tiles for b in (r0, r0 + nr) if 0 <= b <= R})
    covered = 0
    for lo, hi in zip(bounds, bounds[1:]):
        spans = sorted((max(c0, 0), min(c0 + nc, C)) for r0, c0, nr, nc in tiles if r0 <= lo and r0 + nr >= hi)
        reach = width = 0
        for a, b in spans:
            if b > max(a, reach):
                width += b - max(a, reach)
                reach = b
        covered += (hi - lo) * width
    rows_hit = {r for r0, _c, nr, _n in tiles for r in range(max(r0, 0), min(r0 + nr, R))}
    cols_hit = {c for _r, c0, _n, nc in tiles for c in range(max(c0, 0), min(c0 + nc, C))}
    return covered, [r for r in range(R) if r not in rows_hit][:16], [c for c in range(C) if c not in cols_hit][:16]


def _store_coverage(trace: dict, outputs: list[dict], rtl_facts: dict, proto: Protocol) -> dict[str, dict]:
    stores = _stores(trace, rtl_facts, proto)
    report: dict[str, dict] = {}
    for o in outputs:
        mine = [s for s in stores if s["kind"] == "argbase" and s["arg_index"] == o["arg_index"]]
        rec: dict = {
            "arg_index": o["arg_index"],
            "extent": [o["rows"], o["cols"]],
            "n_stores": len(mine),
            "arg_indices_stored_to": sorted({s["arg_index"] for s in stores if s["arg_index"] is not None}),
        }
        if not mine:
            blind = [s["index"] for s in stores if s["kind"] not in ("argbase", "const")]
            rec["status"] = "unknown" if blind else "absent"
            if blind:
                rec["unknown_reason"] = "some stores carry no decodable DRAM address"
            report[o["name"]] = rec
            continue
        undecodable = [s for s in mine if not all(_is_int(s[k]) for k in ("offset", "rows", "cols", "stride"))]
        if undecodable:
            rec.update(
                status="unknown",
                unknown_reason="; ".join(sorted({s["unknown"] for s in undecodable})),
                undecodable_instruction_indices=[s["index"] for s in undecodable][:8],
            )
            report[o["name"]] = rec
            continue
        R, C, eb = o["rows"], o["cols"], o["elem_bytes"]
        tiles = [(s["offset"] // s["stride"], (s["offset"] % s["stride"]) // eb, s["rows"], s["cols"]) for s in mine]
        past = [
            s["index"] for s, (r0, c0, nr, nc) in zip(mine, tiles) if r0 < 0 or c0 < 0 or r0 + nr > R or c0 + nc > C
        ]
        covered, miss_rows, miss_cols = _covered_cells(tiles, R, C)
        rec.update(
            status="covered" if covered == R * C and not past else "uncovered",
            covered_cells=covered,
            declared_cells=R * C,
            stores_past_extent=past[:8],
            uncovered_rows=miss_rows,
            uncovered_cols=miss_cols,
        )
        report[o["name"]] = rec
    return report


def _check_output_store_coverage(
    trace: dict,
    capsule: dict | None,
    rtl_facts: dict,
    command_buffer: dict | None = None,
    proto: Protocol | None = None,
    spec: dict | None = None,
) -> Check:
    """Every declared output cell is stored, and no store writes past the declared extent."""
    cid = (spec or {}).get("id", "T0.output_store_coverage")
    proto = _resolve(proto, rtl_facts)
    if proto is None:
        return Check(cid, "T0", "warn", "skipped", "the target declares no rtl_checks protocol")
    outs, why = declared_outputs(capsule, command_buffer)
    if not outs:
        return Check(cid, "T0", "warn", "skipped", f"declared output extent not derivable: {why}")
    cov = _store_coverage(trace, outs, rtl_facts, proto)
    unknown = {n: r for n, r in cov.items() if r["status"] == "unknown"}
    if unknown:
        return Check(
            cid,
            "T0",
            "warn",
            "skipped",
            "store coverage is UNKNOWN for " + "; ".join(f"{n} ({r['unknown_reason']})" for n, r in unknown.items()),
            evidence=unknown,
        )
    bad = {n: r for n, r in cov.items() if r["status"] != "covered"}
    if bad:
        parts = []
        for n, r in bad.items():
            if r["status"] == "absent":
                parts.append(
                    f"{n} (kernel argument #{r['arg_index']}) is never stored; stores address "
                    f"argument(s) {r['arg_indices_stored_to']}"
                )
            else:
                parts.append(
                    f"{n} [{r['extent'][0]}x{r['extent'][1]}]: {r['covered_cells']} of "
                    f"{r['declared_cells']} cells stored, {len(r['stores_past_extent'])} store(s) past the extent"
                )
        return Check(
            cid,
            "T0",
            "warn",
            "fail",
            "declared output(s) not exactly covered: " + "; ".join(parts),
            expected="every declared cell stored, none past the extent",
            got=f"{len(bad)} of {len(outs)} output(s) not covered",
            evidence=bad,
            fix_hint=_hint(
                spec,
                "store every declared output cell through that output's kernel argument, within its declared extent",
            ),
        )
    return Check(
        cid, "T0", "warn", "pass", f"all {len(outs)} declared output(s) are exactly covered by stores", evidence=cov
    )


# ================================================================================ declared outputs
def _attrs(capsule: dict | None) -> dict:
    return ((capsule or {}).get("operation") or {}).get("attributes") or {}


def _input_shape(capsule: dict | None, name: str | None, role: str | None = None):
    """A declared input's shape, by NAME first and by ROLE as the fallback."""
    ins = (capsule or {}).get("inputs") or []
    hit = next((t for t in ins if name is not None and t.get("name") == name), None)
    hit = hit or next((t for t in ins if role is not None and t.get("role") == role), None)
    return hit.get("shape") if hit else None


def _pooled_rows(rows: int, entry: dict) -> tuple[int | None, str]:
    """Rows after a declared pooling stage, through the runtime's own pooled-extent definition."""
    pid, ps, pst = entry.get("pool_in_dims"), entry.get("pool_size"), entry.get("pool_stride")
    if not (pid and ps and pst):
        return None, "a pooling stage is declared without pool_in_dims/pool_size/pool_stride"
    from merlin.runtime.tensor import pool_out_dims

    H, W = int(pid[0]), int(pid[1])
    if H * W <= 0 or rows % (H * W):
        return None, f"declared pool_in_dims [{H}, {W}] does not divide the {rows} pre-pool rows"
    Ho, Wo = pool_out_dims(H, W, ps, pst, entry.get("pool_padding") or [0, 0, 0, 0])
    return (rows // (H * W)) * Ho * Wo, ""


def declared_outputs(capsule: dict | None, command_buffer: dict | None = None) -> tuple[list[dict], str]:
    """Every DECLARED output's committed extent — ``([{name, rows, cols, elem_bytes, arg_index}], "")``
    or ``([], reason)`` — from the capsule's declaration only. ``arg_index`` is resolved from the
    logical kernel ABI for ``command_buffer``, falling back to "declared inputs in order, then the
    outputs" when no buffer resolves."""
    a = _attrs(capsule)
    op = _declared_op(capsule)
    ins = (capsule or {}).get("inputs")
    if not isinstance(ins, list):
        return [], "capsule declares no input list"
    entries: list[tuple] = []
    if op == "resident_reuse":
        w = _input_shape(capsule, a.get("weight"), "weight")
        mms = a.get("matmuls")
        if not (isinstance(w, list) and len(w) == 2 and isinstance(mms, list) and mms):
            return [], "declared residency capsule has no [K, N] weight or no matmul list"
        for mm in mms:
            lhs = _input_shape(capsule, mm.get("lhs"))
            if not (isinstance(lhs, list) and len(lhs) == 2) or not mm.get("out"):
                return [], "a declared matmul has no 2-D lhs shape or no output name"
            entries.append((mm["out"], int(lhs[0]), int(w[1]), mm))
    elif op in ("matmul", "matmul_resident"):
        shape = _declared_output_shape(capsule)
        if shape is None or not a.get("out"):
            return [], "the declared (M, N) and output name are not derivable"
        entries.append((a["out"], shape[0], shape[1], a))
    elif op in ("conv2d", "conv"):
        ifm, w = _input_shape(capsule, a.get("ifm"), "input"), _input_shape(capsule, a.get("weight"), "weight")
        if not (isinstance(ifm, list) and len(ifm) == 4 and isinstance(w, list) and len(w) == 2 and a.get("out")):
            return [], "declared conv is not [N, H, W, Ci] x [Kh*Kw*Ci, Co] with a named output"
        try:
            from merlin.runtime.commandbuffer import conv_out_dims

            Ho, Wo = conv_out_dims(
                int(ifm[1]),
                int(ifm[2]),
                int(a["kh"]),
                int(a["kw"]),
                a.get("stride", [1, 1]),
                a.get("padding", [0, 0, 0, 0]),
                a.get("dilation", [1, 1]),
            )
        except (KeyError, TypeError, ValueError) as e:
            return [], f"declared conv geometry is incomplete ({type(e).__name__})"
        entries.append((a["out"], int(ifm[0]) * Ho * Wo, int(w[1]), a))
    elif op == "movement":
        src = _input_shape(capsule, a.get("src"), "input")
        if not (isinstance(src, list) and len(src) == 2) or not a.get("out"):
            return [], "declared movement has no 2-D source shape or no output name"
        entries.append((a["out"], int(src[0]), int(src[1]), a))
    else:
        return [], f"declared op {op or '<absent>'!r} has no derived output extent"
    outs = []
    for name, rows, cols, entry in entries:
        dt = entry.get("output_dtype") or ((capsule or {}).get("numeric_policy") or {}).get("dtype")
        eb = _DTYPE_BYTES.get(str(dt).lower()) if dt else None
        if eb is None:
            return [], f"output {name!r} declares no dtype, so its DRAM byte extent is not derivable"
        if "maxpool" in (entry.get("epilogue") or []):
            rows, why = _pooled_rows(rows, entry)
            if rows is None:
                return [], f"output {name!r}: {why}"
        outs.append({"name": str(name), "rows": int(rows), "cols": int(cols), "elem_bytes": int(eb)})
    order, _shape, _why = resolve_kernel_arg_order(command_buffer)
    positions = {name: k for k, name in enumerate(order)}
    for j, rec in enumerate(outs):
        rec["arg_index"] = positions.get(rec["name"], len(ins) + j)
    return outs, ""


# ================================================== kernel-argument ABI, resolved from the logical ABI
def resolve_kernel_arg_order(command_buffer: dict | None) -> tuple[list[str], str, str]:
    """``(argument names in harness call order, ABI shape name, reason)`` for a command buffer.

    The order is the one the runner-owned harness calls the kernel with: ``logical_kernel_abi`` in the
    OOT backend contract (:func:`merlin.targetgen.contract.harness_render.kernel_arg_order`). A buffer
    that ABI cannot describe, or one tensor in two slots, is UNKNOWN (``([], "", reason)``)."""
    if not command_buffer:
        return [], "", "no command buffer was passed, so which tensor each kernel argument carries is unknown"
    if not isinstance(command_buffer.get("commands"), list) or not command_buffer["commands"]:
        return [], "", "the command buffer declares no commands"
    if not isinstance(command_buffer.get("tensors"), dict) or not command_buffer["tensors"]:
        return [], "", "the command buffer declares no tensors"
    from merlin.targetgen.contract import harness_render as HR

    try:
        names = HR.kernel_arg_order(command_buffer)
    except (HR.HarnessRenderError, OSError, KeyError, TypeError, ValueError) as e:
        return [], "", f"the logical kernel ABI does not resolve against this command buffer ({e})"
    abi = command_buffer.get("kernel_abi")
    shape = "whole_program" if isinstance(abi, dict) and abi.get("kind") == "whole_program" else "logical"
    if len(set(names)) != len(names):
        return [], "", f"the {shape} argument order {names} puts one tensor in two slots"
    return names, shape, ""


# ===================================================================================== rule engine
@dataclass
class _Ctx:
    trace: dict
    capsule: dict | None
    facts: dict
    command_buffer: dict | None
    proto: Protocol


_RULES: dict[str, Any] = {
    "decode_clean": lambda c, s: _check_decode_clean(c.trace, c.proto, s),
    "funct_legal": lambda c, s: _check_decode_funct_legal(c.trace, c.facts, s),
    "local_address_bounds": lambda c, s: _check_local_address_bounds(c.trace, c.facts, c.proto, s),
    "precedes": lambda c, s: _check_precedes(c.trace, c.proto, s),
    "config_before_use": lambda c, s: _check_config_before_use(c.trace, c.proto, s),
    "bracket": lambda c, s: _check_bracket(c.trace, c.proto, s),
    "output_store_coverage": lambda c, s: _check_output_store_coverage(
        c.trace, c.capsule, c.facts, c.command_buffer, c.proto, s
    ),
}

#: The rule kinds this engine evaluates (a protocol naming any other kind is refused).
RULE_KINDS = tuple(_RULES)


def screen(
    trace: dict,
    capsule: dict | None = None,
    rtl_facts: dict | None = None,
    *,
    target: str,
    command_buffer: dict | None = None,
) -> CheckReport:
    """Run ``target``'s declared legality rules over a decoded trace; return an advisory report.

    ``target`` is REQUIRED and selects both the base RTL facts and the protocol; ``rtl_facts``
    optionally overrides fact keys; ``capsule`` and ``command_buffer`` supply the declared outputs."""
    proto = protocol_for(target)
    facts = load_default_facts(target)
    if rtl_facts:
        facts.update(rtl_facts)
    rep = CheckReport(capsule=(capsule or {}).get("name"), source_trace=trace.get("source"), rtl_facts=facts)
    ctx = _Ctx(trace, capsule, facts, command_buffer, proto)
    rep.checks.extend(_RULES[spec["rule"]](ctx, spec) for spec in proto.rules)
    return rep


# ============================================================================ FileCheck assertions
def _facts_abi(facts: dict) -> tuple[str, str] | None:
    """The RoCC (custom opcode, funct3) the RTL facts derive, or ``None`` — never a default."""
    for i in facts.get("interfaces") or []:
        if i.get("name") == "funct_decode_table" and i.get("custom_opcode") is not None:
            return (f"0x{i['custom_opcode']:x}", f"0x{i['funct3']:x}")
    return None


def _legal_funct(facts_rec: dict) -> set[int]:
    table = _facts_interface(facts_rec.get("facts", facts_rec), "funct_decode_table") or {}
    return set(table.get("legal_funct") or [])


def compile_trace_checks(facts_rec: dict, capsule: dict, prefix: str = "TRACE") -> str | None:
    """FileCheck lines over :func:`render_trace`: only the facts-grounded legality assertions (the
    derived RoCC ABI, zero RTL-illegal functs, zero undecodable commands) — never a count or a
    presence requirement that would encode a lowering."""
    op = _declared_op(capsule)
    if op is None:
        return None
    lines = [f"// RTL-derived legality checks (op={op}) — generated, do not edit"]
    abi = _facts_abi(facts_rec.get("facts", facts_rec))
    if abi is not None:
        lines.append(f"// {prefix}-DAG: ABI custom={abi[0]} funct3={abi[1]}")
    if _legal_funct(facts_rec):
        lines.append(f"// {prefix}-DAG: ILLEGAL_FUNCT_COUNT 0{{{{$}}}}")
    lines.append(f"// {prefix}-DAG: UNKNOWN_COUNT 0{{{{$}}}}")
    return "\n".join(lines) + "\n"


def render_trace(trace: dict, facts_rec: dict) -> str:
    """Canonical text the TRACE FileCheck lines are matched against."""
    instrs = trace.get("instructions", [])
    legal = _legal_funct(facts_rec)
    illegal = str(sum(1 for i in instrs if _is_int(i.get("funct")) and i["funct"] not in legal)) if legal else "-"
    abi = trace.get("abi") or {}
    lines = [
        f"# {RENDER_SCHEMA}",
        f"ABI custom={abi.get('custom_opcode', '-')} funct3={abi.get('funct3', '-')}",
        f"INSTRUCTION_COUNT {len(instrs)}",
        f"ILLEGAL_FUNCT_COUNT {illegal}",
        f"UNKNOWN_COUNT {sum(1 for i in instrs if i.get('class') == 'UNKNOWN')}",
    ]
    for i in instrs:
        f = i.get("funct")
        lines.append(f"INSTR {i.get('index')} {i.get('class')} funct={f if f is not None else '-'}")
    return "\n".join(lines) + "\n"

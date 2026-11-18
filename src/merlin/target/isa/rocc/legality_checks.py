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
from merlin.target.isa.rocc.legality_protocol import (
    Protocol,
    _check_decode_clean,
    _check_decode_funct_legal,
    _check_local_address_bounds,
    _check_precedes,
)
from merlin.target.isa.rocc.rtl_checks import Check, CheckReport, RtlChecksUnavailable
from merlin.target.rtl.facts import load_facts

# ---- rtl_checks_generic ------------------------------------------------------------------------


def _check_config_before_use(trace: dict, proto: Protocol, spec: dict) -> Check:
    """Each use role is preceded by the role that writes the configuration it reads."""
    from merlin.target.isa.rocc.legality_protocol import Protocol, _classes, _first, _hint

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
    from merlin.target.isa.rocc.legality_protocol import Protocol, _classes, _hint

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


def _stores(trace: dict, rtl_facts: dict, proto: Protocol) -> list[dict]:
    """Every store with the DRAM pitch and footprint its active configuration gives it.

    Footprint rows follow the RTL store semantics: the encoded row count, or — when the active store
    configuration enables native pooling and the RTL supports it — the encoded pooled positions."""
    from merlin.target.isa.rocc.legality_protocol import Protocol, _active_store_pooled, _is_int

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
    from merlin.target.isa.rocc.legality_protocol import Protocol, _is_int

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
    from merlin.target.isa.rocc.legality_protocol import Protocol, _hint, _resolve

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
    from merlin.runtime.engine.command_buffer import pool_out_dims

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
    from merlin.target.isa.rocc.legality_protocol import _DTYPE_BYTES, _declared_op, _declared_output_shape

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
            from merlin.runtime.engine.command_buffer import conv_out_dims

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


def resolve_kernel_arg_order(command_buffer: dict | None) -> tuple[list[str], str, str]:
    """``(argument names in harness call order, ABI shape name, reason)`` for a command buffer.

    The order is the one the runner-owned harness calls the kernel with: ``logical_kernel_abi`` in the
    OOT backend contract (:func:`merlin.target.contract.harness.harness_render.kernel_arg_order`). A buffer
    that ABI cannot describe, or one tensor in two slots, is UNKNOWN (``([], "", reason)``)."""
    if not command_buffer:
        return [], "", "no command buffer was passed, so which tensor each kernel argument carries is unknown"
    if not isinstance(command_buffer.get("commands"), list) or not command_buffer["commands"]:
        return [], "", "the command buffer declares no commands"
    if not isinstance(command_buffer.get("tensors"), dict) or not command_buffer["tensors"]:
        return [], "", "the command buffer declares no tensors"
    from merlin.target.contract.harness import harness_render as HR

    try:
        names = HR.kernel_arg_order(command_buffer)
    except (HR.HarnessRenderError, OSError, KeyError, TypeError, ValueError) as e:
        return [], "", f"the logical kernel ABI does not resolve against this command buffer ({e})"
    abi = command_buffer.get("kernel_abi")
    shape = "whole_program" if isinstance(abi, dict) and abi.get("kind") == "whole_program" else "logical"
    if len(set(names)) != len(names):
        return [], "", f"the {shape} argument order {names} puts one tensor in two slots"
    return names, shape, ""


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
    from merlin.target.isa.rocc.legality_protocol import load_default_facts, protocol_for

    proto = protocol_for(target)
    facts = load_default_facts(target)
    if rtl_facts:
        facts.update(rtl_facts)
    rep = CheckReport(capsule=(capsule or {}).get("name"), source_trace=trace.get("source"), rtl_facts=facts)
    ctx = _Ctx(trace, capsule, facts, command_buffer, proto)
    rep.checks.extend(_RULES[spec["rule"]](ctx, spec) for spec in proto.rules)
    return rep


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
    from merlin.target.isa.rocc.legality_protocol import _declared_op

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
    from merlin.target.isa.rocc.legality_protocol import RENDER_SCHEMA, _is_int

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

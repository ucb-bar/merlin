"""Explicit integer-observable post-round clamp legalization.

This requires caller-owned standard roundevenf value semantics and unobserved
errno/FENV. No build route selects it. No source tree is modified.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from merlin.llvmlower import late_quant_rne as K

from .post_round_clamp_guard import CalleePolicy, inspect_module


@dataclass(frozen=True)
class Contract:
    roundeven_value_semantics: str
    supplier_path: str
    supplier_sha256: str
    supplier_source_path: str
    supplier_source_sha256: str
    supplier_recipe_path: str
    supplier_recipe_sha256: str
    fenv_observed: bool
    errno_observed: bool
    value_only: bool
    nan_terminal: str
    host_isa: str
    callee_policy: CalleePolicy

    def validate(self):
        if self.roundeven_value_semantics != "merlin_ieee_words_binary32_nearest_ties_even_v1":
            raise ValueError("unknown roundevenf value semantics")
        if (
            any(type(v) is not bool for v in [self.fenv_observed, self.errno_observed, self.value_only])
            or self.fenv_observed
            or self.errno_observed
            or not self.value_only
        ):
            raise ValueError("value-only unobserved errno/FENV contract required")
        if self.nan_terminal != "llvm_fptosi_poison" or self.host_isa not in {
            "rv64gc",
            "portable",
        }:
            raise ValueError("unknown sourceNaN/hostISA contract")
        p = Path(self.supplier_path)
        if not p.is_absolute() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != self.supplier_sha256:
            raise ValueError("selected roundevenf supplier unavailable/changed")
        source = Path(self.supplier_source_path)
        recipe = Path(self.supplier_recipe_path)
        approved = "8a53be598e3f5f2fd4c269e07894b5f3fa23ae7fdf00acc9c56b987f83f4303f"
        if (
            self.supplier_source_sha256 != approved
            or not source.is_absolute()
            or hashlib.sha256(source.read_bytes()).hexdigest() != approved
        ):
            raise ValueError("unreviewed/changed roundevenf source semantics")
        if not recipe.is_absolute() or hashlib.sha256(recipe.read_bytes()).hexdigest() != self.supplier_recipe_sha256:
            raise ValueError("selected supplier compilation recipe changed")
        rows = [r for r in json.loads(recipe.read_bytes()).get("commands", []) if r.get("requested_output") == str(p)]
        if (
            len(rows) != 1
            or rows[0].get("output", {}).get("sha256") != self.supplier_sha256
            or not any(r.get("path") == str(source) and r.get("sha256") == approved for r in rows[0].get("inputs", []))
            or any(a.startswith("-DMERLIN_EXTERNAL_ROUNDEVEN_F32") for a in rows[0]["argv"])
        ):
            raise ValueError("actual helper object/source/compilation is not selected bitword implementation")


def _same(a, b):
    return a.startswith("%") and b.startswith("%") and K._identity(a) == K._identity(b)


def _lookup(defs, x):
    return defs.get(K._identity(x)) if x.startswith("%") else None


def _clamp(op, defs):
    if op is None or op.opcode != "select" or op.dtype != "float":
        return None
    condition, bound, value = op.operands
    cmp = _lookup(defs, condition)
    if (
        cmp is None
        or cmp.opcode != "fcmp"
        or cmp.dtype != "float"
        or cmp.predicate not in {"olt", "ogt"}
        or not _same(cmp.operands[0], value)
    ):
        return None
    a, b = K._number(bound), K._number(cmp.operands[1])
    if a is None or b != a or not math.isfinite(a) or not a.is_integer():
        return None
    return {
        "input": value,
        "bound": int(a),
        "kind": "lower" if cmp.predicate == "olt" else "upper",
        "select": op,
        "compare": cmp,
    }


def rewrite(text, contract):
    contract.validate()
    guard = inspect_module(text, contract.callee_policy)
    receipt = {
        "schema": "merlin.post_round_clamp_integer_rne.v1",
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "routes": [],
        "contract": {
            **contract.__dict__,
            "callee_policy": contract.callee_policy.__dict__,
        },
        "callee_guard": guard,
        "scope": (
            "Caller-selected value-only context. Final source-owned provider closure "
            "requires a separate actual post-link audit; this emission grants no "
            "execution or performance qualification."
        ),
    }
    tokens = K._tokens(text)
    if any(
        t.text == "strictfp"
        or (t.text.startswith("@") and K._identity(t.text).startswith("llvm.experimental.constrained."))
        for t in tokens
    ):
        receipt["refusal"] = "strict/constrainedFP"
        return text, receipt
    if any(t.text in {"fast", "nnan", "ninf", "nsz", "arcp", "contract", "afn", "reassoc"} for t in tokens):
        receipt["refusal"] = "unsupported FPflags"
        return text, receipt
    forbidden = {
        "fetestexcept",
        "feraiseexcept",
        "fegetexceptflag",
        "fesetexceptflag",
        "fegetenv",
        "fesetenv",
        "feholdexcept",
        "feupdateenv",
        "fegetround",
        "fesetround",
        "feclearexcept",
        "__errno_location",
        "_errno",
        "llvm.read_register.i32",
        "llvm.read_register.i64",
    }
    if any(t.text.startswith("@") and K._identity(t.text) in forbidden for t in tokens):
        receipt["refusal"] = "observable FPenvironment/errno route"
        return text, receipt
    # A local definition is not the explicitly selected standard library function.
    for i, t in enumerate(tokens):
        if t.text == "define":
            target = next((v.text for v in tokens[i + 1 :] if v.text.startswith("@")), None)
            if target and K._identity(target) == "roundevenf":
                receipt["refusal"] = "local roundevenf definition"
                return text, receipt
    functions = K._functions(tokens)
    used = {K._identity(t.text) for t in tokens if t.text.startswith("%")}
    edits = []
    serial = 0
    claimed = set()
    for fn in functions:
        defs = {}
        counts = {}
        for i, t in enumerate(fn):
            if t.text.startswith("%"):
                lhs = i + 1 < len(fn) and fn[i + 1].text == "="
                if not lhs:
                    counts[K._identity(t.text)] = counts.get(K._identity(t.text), 0) + 1
                if lhs:
                    op = K._instruction(fn, i)
                    if op:
                        defs[K._identity(op.result)] = op
        for terminal in sorted(defs.values(), key=lambda x: x.start):
            dtype = terminal.output_type
            if (
                terminal.opcode != "fptosi"
                or terminal.dtype != "float"
                or not dtype.startswith("i")
                or not dtype[1:].isdigit()
                or not 1 <= int(dtype[1:]) <= 24
            ):
                continue
            second = _clamp(_lookup(defs, terminal.operands[0]), defs)
            if not second:
                continue
            first = _clamp(_lookup(defs, second["input"]), defs)
            if not first or {first["kind"], second["kind"]} != {"lower", "upper"}:
                continue
            bounds = {c["kind"]: c["bound"] for c in [first, second]}
            lo, hi = bounds["lower"], bounds["upper"]
            bits = int(dtype[1:])
            if not -(1 << (bits - 1)) <= lo < hi <= (1 << (bits - 1)) - 1:
                continue
            rounded = _lookup(defs, first["input"])
            if (
                rounded is None
                or rounded.opcode != "call"
                or rounded.dtype != "float"
                or rounded.callee != "roundevenf"
                or len(rounded.operands) != 1
            ):
                continue
            # Every floating/control use must be within this exact chain. Float stores/returns/other consumers refuse.
            wanted = {
                rounded.result: 2,
                first["compare"].result: 1,
                first["select"].result: 2,
                second["compare"].result: 1,
                second["select"].result: 1,
            }
            if any(counts.get(K._identity(v), 0) != n for v, n in wanted.items()):
                continue
            if rounded.start in claimed:
                continue
            if (
                not rounded.start
                < first["compare"].start
                < first["select"].start
                < second["compare"].start
                < second["select"].start
                < terminal.start
            ):
                continue
            # Single basic-block chain: reject labels/terminators between call and terminal.
            chain_tokens = [t.text for t in fn if rounded.start <= t.start < terminal.end]
            if any(
                x
                in {
                    "br",
                    "switch",
                    "invoke",
                    "callbr",
                    "phi",
                    "landingpad",
                    "catchpad",
                    "ret",
                }
                for x in chain_tokens
            ):
                continue
            claimed.add(rounded.start)
            while any(
                f"merlin.post_rne.{serial}.{suffix}" in used for suffix in ["lowcmp", "low", "highcmp", "high", "word"]
            ):
                serial += 1
            prefix = f"merlin.post_rne.{serial}"
            serial += 1
            names = [f"%{prefix}.{s}" for s in ["lowcmp", "low", "highcmp", "high", "word"]]
            if any(K._identity(n) in used for n in names):
                raise ValueError("temporary collision")
            used.update(K._identity(n) for n in names)
            lc, lv, hc, hv, word = names
            raw = rounded.operands[0]
            indent = text[text.rfind("\n", 0, rounded.start) + 1 : rounded.start]
            if indent.strip():
                continue
            low, high = f"{float(lo):.6e}", f"{float(hi):.6e}"
            lines = [
                f"{lc} = fcmp olt float {raw}, {low}",
                f"{lv} = select i1 {lc}, float {low}, float {raw}",
                f"{hc} = fcmp ogt float {lv}, {high}",
                f"{hv} = select i1 {hc}, float {high}, float {lv}",
            ]
            if contract.host_isa == "rv64gc":
                lines.extend(
                    [
                        f'{word} = call i32 asm "fcvt.w.s $0, $1, rne", "=r,f"(float {hv})',
                        f"{rounded.result} = sitofp i32 {word} to float",
                    ]
                )
            else:
                lines.append(f"{rounded.result} = call float @llvm.roundeven.f32(float {hv})")
            replacement = ("\n" + indent).join(lines)
            edits.append((rounded.start, rounded.end, replacement))
            receipt["routes"].append(
                {
                    "rounded_result": rounded.result,
                    "terminal_result": terminal.result,
                    "raw_input": raw,
                    "output_type": dtype,
                    "bounds": [lo, hi],
                    "source_nan": "terminal LLVMfptosipoison refined only; floatNaNusesrefused",
                    "signed_zero": "only integerterminalobservable; sourcefloatchainusesclosed",
                    "source_round_call_span": [rounded.start, rounded.end],
                    "source_form": "ordered_post_round_clamp_integer_terminal",
                }
            )
    changed = text
    for start, end, replacement in reversed(sorted(edits)):
        changed = changed[:start] + replacement + changed[end:]
    if (
        edits
        and contract.host_isa == "portable"
        and not any(t.text.startswith("@") and K._identity(t.text) == "llvm.roundeven.f32" for t in tokens)
    ):
        changed += "\ndeclare float @llvm.roundeven.f32(float)\n"
    receipt["rewritten_sha256"] = hashlib.sha256(changed.encode()).hexdigest()
    receipt["rewrites"] = len(edits)
    receipt["preservation"] = (
        "Only eligible roundcall replaced; "
        "producerarith/originalSSAids/postclamps/terminalconversions retainbytes. "
        "No externalfloatingconsumer accepted."
    )
    return changed, receipt

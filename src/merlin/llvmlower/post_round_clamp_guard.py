"""Closed-module guard for an explicitly source-selected value-only ABI policy.

The provider owns the no-observer premise; artifact pins are custody, not a
semantic theorem. Unknown calls, aliases and input inline asm are refused.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from merlin.llvmlower import late_quant_rne as K

INTRINSICS = frozenset(
    {
        "llvm.memcpy.p0.p0.i64",
        "llvm.maximum.f32",
        "llvm.minimum.f32",
        "llvm.roundeven.f32",
    }
)


def _pin(row):
    if (
        type(row) is not dict
        or set(row) != {"path", "sha256", "bytes"}
        or type(row["path"]) is not str
        or type(row["sha256"]) is not str
        or type(row["bytes"]) is not int
    ):
        raise ValueError("typed full artifact pin required")
    p = Path(row["path"])
    if (
        not p.is_absolute()
        or p.stat().st_size != row["bytes"]
        or hashlib.sha256(p.read_bytes()).hexdigest() != row["sha256"]
    ):
        raise ValueError("callee policy artifact unavailable or changed")


@dataclass(frozen=True)
class CalleePolicy:
    path: str
    sha256: str

    def validate(self):
        p = Path(self.path)
        if not p.is_absolute() or hashlib.sha256(p.read_bytes()).hexdigest() != self.sha256:
            raise ValueError("selected callee policy changed")
        j = json.loads(p.read_bytes())
        if (
            j.get("schema") != "merlin.host_value_only_callee_policy.v1"
            or j.get("caller_premise") != "closed_freestanding_value_only_no_fenv_errno_observation"
            or j.get("approval_scope") != "explicit_trusted_provider_policy_not_inferred"
        ):
            raise ValueError("explicit closed caller observer policy required")
        if j.get("fenv_observed") is not False or j.get("errno_observed") is not False:
            raise ValueError("observer policy not value-only")
        if (
            type(j.get("external_callees")) is not list
            or not j["external_callees"]
            or len(set(j["external_callees"])) != len(j["external_callees"])
            or any(type(s) is not str or not s for s in j["external_callees"])
        ):
            raise ValueError("explicit unique callee roster required")
        if any(s.startswith("llvm.") for s in j["external_callees"]):
            raise ValueError("LLVM intrinsics have a separate exact allowlist")
        rows = j.get("bindings")
        if (
            type(rows) is not list
            or {r.get("symbol") for r in rows} != set(j["external_callees"])
            or len(rows) != len(j["external_callees"])
        ):
            raise ValueError("one actual source/object binding per external callee required")
        for r in rows:
            if (
                type(r) is not dict
                or set(r)
                != {
                    "symbol",
                    "actual_symbol",
                    "source_pins",
                    "object_pins",
                    "effect_evidence_pins",
                }
                or type(r["actual_symbol"]) is not str
            ):
                raise ValueError("unknown callee binding format")
            for key in ["source_pins", "object_pins", "effect_evidence_pins"]:
                if type(r[key]) is not list or not r[key]:
                    raise ValueError("source/object/provider evidence closure required")
                for pin in r[key]:
                    _pin(pin)
        return j


def inspect_module(text, policy):
    """Require plain direct calls and one ordinary external roundeven declaration."""
    if type(policy) is not CalleePolicy:
        raise TypeError("typed callee policy required")
    approved = policy.validate()
    tokens = K._tokens(text)
    ts = [t.text for t in tokens]
    if any(t in {"asm", "alias", "ifunc", "blockaddress", "callbr", "invoke", "indirectbr"} for t in ts):
        raise ValueError("input inlineasm/alias/ifunc/exotic transfer refused")
    defined = {}
    declared = {}
    decl_indices = set()
    call_indices = set()
    calls = []
    for i, t in enumerate(tokens):
        if t.text not in {"define", "declare"}:
            continue
        k = i + 1
        while k < len(tokens) and not tokens[k].text.startswith("@"):
            if tokens[k].text in {"define", "declare"}:
                raise ValueError("unknown function header")
            k += 1
        if k + 1 >= len(tokens) or tokens[k + 1].text != "(":
            raise ValueError("unknown function header")
        name = K._identity(tokens[k].text)
        target = defined if t.text == "define" else declared
        if name in target or name in (declared if t.text == "define" else defined):
            raise ValueError("duplicate function binding")
        target[name] = (i, k)
        decl_indices.add(k)
    if "roundevenf" in defined or "roundevenf" not in declared:
        raise ValueError("roundevenf requires unique external declaration")
    i, k = declared["roundevenf"]
    if ts[i:k] != ["declare", "float"] or ts[k : k + 4] != [
        tokens[k].text,
        "(",
        "float",
        ")",
    ]:
        raise ValueError("roundevenf external signature must be float(float)")
    allowed = set(approved["external_callees"]) | INTRINSICS
    if set(declared) - allowed:
        raise ValueError("unapproved external declaration: " + ",".join(sorted(set(declared) - allowed)))
    for i, t in enumerate(tokens):
        if t.text != "call":
            continue
        # Only ordinary direct scalar/void calls; function pointer/bitcast forms refuse.
        k = i + 1
        if k >= len(tokens):
            raise ValueError("missing direct call type")
        if ts[k] in {"{", "[", "<"}:
            pairs = {"{": "}", "[": "]", "<": ">"}
            stack = []
            while k < len(tokens):
                if ts[k] in pairs:
                    stack.append(pairs[ts[k]])
                elif ts[k] in {"}", "]", ">"}:
                    if not stack or ts[k] != stack.pop():
                        raise ValueError("unbalanced direct call result type")
                    if not stack:
                        k += 1
                        break
                k += 1
            if stack:
                raise ValueError("unclosed direct call result type")
        elif (
            ts[k] in {"void", "float", "double", "half", "bfloat", "ptr"}
            or ts[k].startswith("%")
            or ts[k].startswith("i")
            and ts[k][1:].isdigit()
        ):
            k += 1
        else:
            raise ValueError("unsupported direct call prefix/type")
        if k + 1 >= len(tokens) or not ts[k].startswith("@") or ts[k + 1] != "(":
            raise ValueError("indirect or nonordinary call refused")
        name = K._identity(ts[k])
        call_indices.add(k)
        calls.append(name)
        if name not in defined and name not in declared:
            raise ValueError("undeclared callee refused")
        if name not in defined and name not in allowed:
            raise ValueError("unapproved actual callee")
    callable_names = set(defined) | set(declared)
    for i, t in enumerate(tokens):
        if t.text.startswith("@") and K._identity(t.text) in callable_names and i not in decl_indices | call_indices:
            raise ValueError("function address/rebinding/global reference refused")
    return {
        "schema": "merlin.host_closed_callee_observation.v1",
        "policy": {"path": policy.path, "sha256": policy.sha256},
        "defined_functions": sorted(defined),
        "declared_functions": sorted(declared),
        "called_symbols": sorted(set(calls)),
        "input_inline_asm": False,
        "indirect_calls": False,
        "roundeven_binding": "unique ordinary external float(float)",
        "scope": (
            "Explicit trusted source-selected value-only caller premise; "
            "source/object/proof pins do not independently establish whole-program "
            "safety."
        ),
    }

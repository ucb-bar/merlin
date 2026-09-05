#!/usr/bin/env python3
"""Gate: the kernel ABI's argument order has ONE definition, and no harness has drifted from it.

The definition is ``kernel_abi.arg_order_by_command_shape`` in
``merlin/contract/mlir_oot_backend_contract.yaml`` — a row per COMMAND SHAPE, each carrying a token
list that expands to the pointer arguments in order. This gate resolves those tokens against a probe
command buffer of every declared shape, then renders EVERY registered backend's real runner-owned
harness for the same buffer, parses the actual entry call out of the emitted C, and fails if the two
disagree.

WHY. ``arg_order`` was a single prose string saying weight-first, and it was true of exactly one of the
three shapes a harness renderer dispatches to. The whole-op renderer had grown a second rule — the
interface's DECLARATION order — and the pure-movement renderer a third — ``(src, dst)`` — with neither
written anywhere. Twenty-eight shipped capsules declare their weight first, so declaration order and
weight-first coincide and nothing noticed; the four that declare the activation first
(``IFM, W, Y0``) were compiled weight-first against a harness passing activation-first. The kernel then
gathered activations out of the weight buffer (244 bytes past its end) and MVIN'd the weight with the
activation's pitch. Four functional failures, all of them one undocumented rule.

WHAT IT PROVES, AND WHAT IT DOES NOT. It proves the *document* and the *emitted harness* agree, per
shape, for a buffer of that shape. It does not claim the orders are alike — they are not, and the
contract now says so. It also checks the shape ROSTER: the opcode set a renderer treats as a whole op
is read off the backend and compared with the contract's, so a fourth command shape cannot arrive
undocumented.

FAIL CLOSED. A shape whose probe cannot be rendered, a backend whose whole-op opcode set cannot be
read, a contract row with an unknown token, an emitted call that cannot be parsed, or zero renderers
found — each is a REFUSAL that exits nonzero and says so. None of them is ever reported as success.

Parsing is STRUCTURAL (yaml/json loads, ``str.partition``/``split``) — no regex, per the repo rule.
Nothing here names a target: the roster comes from the backend registry and the probe buffers are built
from the contract's own token vocabulary.

Usage::

    python build_tools/scripts/check_kernel_abi_arg_order.py            # check (exit 1 on drift)
    python build_tools/scripts/check_kernel_abi_arg_order.py --verbose  # print every order it resolved
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT / "merlin" / "python"))

_CONTRACT = _ROOT / "merlin" / "contract" / "mlir_oot_backend_contract.yaml"

#: The external tensor roles that become pointer arguments on the whole-op shape. Named here because
#: the token's own definition in the contract names them, and the two are compared below.
_EXTERNAL_ROLES = ("input", "weight", "bias", "output")

#: The attribute a backend module exposes for the opcode set its whole-op harness claims. Read, never
#: assumed: a backend that does not expose it cannot be certified and the gate says so.
_WHOLE_OP_ATTR = "_NATIVE_INTERFACE_OPS"


# --------------------------------------------------------------------------------------------------
# probe command buffers — one per declared shape, built from the contract's token vocabulary alone
# --------------------------------------------------------------------------------------------------
def _probe_movement() -> dict:
    return {"abi_version": "0.1",
            "tensors": {"S": {"shape": [4, 4], "dtype": "i8", "role": "input"},
                        "D": {"shape": [4, 4], "dtype": "i32", "role": "output"}},
            "commands": [{"opcode": "MOVEMENT", "operands": {"src": "S", "dst": "D"},
                          "attributes": {"output_dtype": "i32"}}]}


def _probe_native_whole_op(opcode: str) -> dict | None:
    """A minimal buffer for one whole-op opcode, with the activation declared BEFORE the weight.

    The declaration order is deliberately the one the four failing capsules use, so a renderer that
    silently reordered to weight-first would be caught rather than accidentally agreed with.
    """
    if opcode == "CONV2D":
        return {"abi_version": "0.1",
                "tensors": {"IFM": {"shape": [1, 4, 4, 4], "dtype": "i8", "role": "input"},
                            "W": {"shape": [36, 8], "dtype": "i8", "role": "weight"},
                            "Y0": {"shape": [4, 8], "dtype": "i32", "role": "output"}},
                "commands": [
                    {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"},
                     "attributes": {"layout": "packed_conv_rhs"}},
                    {"opcode": "CONV2D", "operands": {"ifm": "IFM", "weight": "W_res", "dst": "Y0"},
                     "attributes": {"kernel": [3, 3, 4, 8], "stride": [1, 1], "padding": [0, 0, 0, 0],
                                    "dilation": [1, 1], "epilogue": [], "output_dtype": "i32",
                                    "layout": "nhwc"}},
                    {"opcode": "EVICT", "operands": {"handle": "W_res"}}]}
    if opcode in ("ATTENTION_QK", "ATTENTION_PV"):
        a, b = ("q", "k") if opcode == "ATTENTION_QK" else ("p", "v")
        return {"abi_version": "0.1",
                "tensors": {"A": {"shape": [16, 16], "dtype": "i8", "role": "input"},
                            "B": {"shape": [16, 16], "dtype": "i8", "role": "weight"},
                            "Y0": {"shape": [16, 16], "dtype": "i32", "role": "output"}},
                "commands": [{"opcode": opcode, "operands": {a: "A", b: "B", "dst": "Y0"},
                              "attributes": {"output_dtype": "i32"}}]}
    return None


def _probe_resident_matmul() -> dict:
    """Two resident weights, two matmuls each — the shape whose GROUP-MAJOR ordering the flat prose
    string never described. A one-weight probe cannot tell group-major from command order."""
    t = {"W0": {"shape": [16, 16], "dtype": "i8", "role": "weight"},
         "W1": {"shape": [16, 16], "dtype": "i8", "role": "weight"},
         "A0": {"shape": [16, 16], "dtype": "i8", "role": "input"},
         "A1": {"shape": [16, 16], "dtype": "i8", "role": "input"},
         "A2": {"shape": [16, 16], "dtype": "i8", "role": "input"},
         # A fused bias on exactly ONE of the three jobs — the middle one in group-major order — so the
         # trailing bias block is exercised for both what it contains and what it leaves out. Its dtype
         # is the accumulator's, which is what the contract's token says a bias pointee carries.
         "BIAS2": {"shape": [16], "dtype": "i32", "role": "bias"}}
    cmds: list[dict] = [
        {"opcode": "RES_PACK", "operands": {"src": "W0", "dst": "R0"},
         "attributes": {"layout": "packed_rhs"}},
        {"opcode": "RES_PACK", "operands": {"src": "W1", "dst": "R1"},
         "attributes": {"layout": "packed_rhs"}}]
    # Interleave the two groups in COMMAND order, so a group-major emitter and a command-order emitter
    # produce different lists and the token can only match one of them.
    for acc, lhs, res, out, bias in (("acc0", "A0", "R0", "Y0", None), ("acc1", "A1", "R1", "Y1", None),
                                     ("acc2", "A2", "R0", "Y2", "BIAS2")):
        cmds.append({"opcode": "MATMUL_RESIDENT", "operands": {"lhs": lhs, "rhs": res, "dst": acc}})
        attrs: dict = {"epilogue": [], "output_dtype": "i32"}
        if bias is not None:
            attrs = {"epilogue": ["bias_add"], "output_dtype": "i32", "bias": bias}
        cmds.append({"opcode": "COMMIT", "operands": {"src": acc, "dst": out}, "attributes": attrs})
    cmds.append({"opcode": "EVICT", "operands": {"handle": "R0"}})
    cmds.append({"opcode": "EVICT", "operands": {"handle": "R1"}})
    return {"abi_version": "0.1", "tensors": t, "commands": cmds}


# --------------------------------------------------------------------------------------------------
# token resolution — the contract's `order` tokens, resolved against a command buffer
# --------------------------------------------------------------------------------------------------
class Unresolvable(Exception):
    """A token this gate cannot resolve against a buffer: a refusal, never a guessed position."""


def _movement_command(cb: dict) -> dict:
    for cmd in cb.get("commands", []):
        if cmd.get("opcode") == "MOVEMENT":
            return cmd
        if cmd.get("opcode") == "VECTOR_MAP" and (cmd.get("attributes") or {}).get("combine") == "identity":
            return cmd
    raise Unresolvable("no MOVEMENT / identity VECTOR_MAP command in the probe")


def _resident_groups(cb: dict) -> list[tuple[str, list[tuple[str, str]]]]:
    """``[(weight, [(lhs, out), ...]), ...]`` in resident-pack order, group-major within each group."""
    res_to_weight: dict[str, str] = {}
    order: list[str] = []
    for cmd in cb.get("commands", []):
        if cmd.get("opcode") != "RES_PACK":
            continue
        ops = cmd.get("operands") or {}
        dst, src = ops.get("dst"), ops.get("src")
        if not isinstance(dst, str) or not isinstance(src, str):
            raise Unresolvable("a RES_PACK command names no src/dst pair")
        if dst not in res_to_weight:
            res_to_weight[dst] = src
            order.append(dst)
    if not order:
        raise Unresolvable("no RES_PACK command in the probe")
    commit_of = {(cmd.get("operands") or {}).get("src"): (cmd.get("operands") or {}).get("dst")
                 for cmd in cb.get("commands", []) if cmd.get("opcode") == "COMMIT"}
    jobs: dict[str, list[tuple[str, str]]] = {res: [] for res in order}
    for cmd in cb.get("commands", []):
        if cmd.get("opcode") not in ("MATMUL", "MATMUL_RESIDENT"):
            continue
        ops = cmd.get("operands") or {}
        res, lhs, acc = ops.get("rhs"), ops.get("lhs"), ops.get("dst")
        if res not in jobs:
            raise Unresolvable(f"matmul rhs {res!r} resolves to no resident weight")
        out = commit_of.get(acc)
        if not isinstance(lhs, str) or not isinstance(out, str):
            raise Unresolvable(f"matmul {acc!r} has no lhs / committed output")
        jobs[res].append((lhs, out))
    return [(res_to_weight[res], jobs[res]) for res in order]


def resolve_token(token: str, cb: dict) -> list[str]:
    """The pointer arguments one contract token expands to, for ``cb``. Raises on an unknown token."""
    if token == "movement_src":
        ops = _movement_command(cb).get("operands") or {}
        name = ops.get("src") or ops.get("lhs")
        if not isinstance(name, str):
            raise Unresolvable("the movement command names no src/lhs")
        return [name]
    if token == "movement_dst":
        name = (_movement_command(cb).get("operands") or {}).get("dst")
        if not isinstance(name, str):
            raise Unresolvable("the movement command names no dst")
        return [name]
    if token == "interface_external_tensors_in_declaration_order":
        return [name for name, spec in (cb.get("tensors") or {}).items()
                if (spec or {}).get("role") in _EXTERNAL_ROLES]
    if token == "resident_weights_in_resident_pack_order":
        return [weight for weight, _jobs in _resident_groups(cb)]
    if token == "matmul_lhs_group_major":
        return [lhs for _w, jobs in _resident_groups(cb) for lhs, _o in jobs]
    if token == "commit_outputs_group_major":
        return [out for _w, jobs in _resident_groups(cb) for _l, out in jobs]
    if token == "commit_biases_group_major":
        biased: dict[str, str] = {}
        for cmd in cb.get("commands", []):
            if cmd.get("opcode") != "COMMIT":
                continue
            ops, attrs = cmd.get("operands") or {}, cmd.get("attributes") or {}
            if not any(s in ("bias_add", "bias") for s in (attrs.get("epilogue") or [])):
                continue
            dst, name = ops.get("dst"), (attrs.get("bias") or ops.get("bias"))
            if not isinstance(dst, str) or not isinstance(name, str):
                raise Unresolvable("a COMMIT declaring a bias stage names no dst and/or bias tensor")
            biased[dst] = name
        return [biased[out] for _w, jobs in _resident_groups(cb) for _l, out in jobs if out in biased]
    raise Unresolvable(f"unknown arg_order token {token!r} — this gate cannot certify a shape whose "
                       f"order it does not know how to resolve")


# --------------------------------------------------------------------------------------------------
# the emitted harness — parse the entry call out of the rendered C, structurally
# --------------------------------------------------------------------------------------------------
def emitted_call_args(text: str, symbol: str) -> list[str] | None:
    """The argument names of the ``symbol(...)`` CALL in a rendered harness, or None if not parseable.

    Structural: find the call site (not the ``extern`` declaration), take what is inside its
    parentheses — matched by DEPTH, because every argument carries a ``(void*)`` cast whose own close
    paren would end the argument list if the first one were taken — split on commas, and strip each
    argument down to the ``T_<name>`` buffer it passes.
    """
    needle = symbol + "("
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(needle):
            continue                     # a declaration, a comment, or an unrelated mention
        _, _, rest = stripped.partition(needle)
        depth, end = 1, None
        for index, ch in enumerate(rest):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end is None:
            return None                  # an unterminated call: not parseable, never assumed empty
        inner = rest[:end]
        if not inner.strip():
            return []
        names: list[str] = []
        for piece in inner.split(","):
            token = piece.strip()
            _, _, after_cast = token.rpartition(")")   # drop a leading "(void*)" cast if present
            token = (after_cast or token).strip()
            if not token.startswith("T_"):
                return None              # an argument shape this parser does not understand
            names.append(token[2:])
        return names
    return None


# --------------------------------------------------------------------------------------------------
def _renderers() -> list[tuple[str, object, object]]:
    """``[(target, defining_module, render_harness), ...]`` for every registered backend that has one.

    The module reported is the one that DEFINES ``render_harness``, not the package that re-exports it:
    a backend package re-exports with ``from .x import *``, which drops the private roster attribute the
    shape check reads. Following ``__module__`` reaches the definition without naming any target.
    """
    from merlin.runtime.backends import base
    found = []
    for name in base.list_backends():
        try:
            render = base.harness_renderer(name)
        except Exception:                # noqa: BLE001 — no renderer, or an unimportable backend
            continue
        owner = sys.modules.get(getattr(render, "__module__", "")) or base.get_backend(name)
        found.append((name, owner, render))
    return found


def check(verbose: bool = False) -> list[str]:
    import yaml

    problems: list[str] = []
    doc = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8")) or {}
    abi = doc.get("kernel_abi") or {}
    rows = abi.get("arg_order_by_command_shape")
    symbol_pattern = str(abi.get("symbol") or "")
    if not isinstance(rows, list) or not rows or not symbol_pattern:
        return [f"{_CONTRACT.name}: kernel_abi.arg_order_by_command_shape / .symbol could not be read "
                f"— the contract shape changed; this gate cannot certify it and does NOT report success"]

    tokens_doc = abi.get("arg_order_tokens") or {}
    declared_tokens = {t for row in rows for t in (row.get("order") or [])}
    undocumented = sorted(declared_tokens - set(tokens_doc))
    if undocumented:
        problems.append(f"{_CONTRACT.name}: arg_order token(s) {undocumented} are used by a shape row "
                        f"but absent from arg_order_tokens")
    orphan = sorted(set(tokens_doc) - declared_tokens)
    if orphan:
        problems.append(f"{_CONTRACT.name}: arg_order_tokens define(s) {orphan}, which no shape row "
                        f"uses — a vocabulary entry nothing derives from is drift waiting to happen")

    by_shape = {}
    for row in rows:
        shape = row.get("shape")
        if not isinstance(shape, str) or not shape or not isinstance(row.get("order"), list):
            problems.append(f"{_CONTRACT.name}: a shape row is missing `shape` or `order`: {row!r}")
            continue
        by_shape[shape] = row
    for required in ("movement", "native_whole_op", "resident_matmul"):
        if required not in by_shape:
            problems.append(f"{_CONTRACT.name}: no arg_order row for the {required!r} command shape — "
                            f"a renderer dispatches to it, so it cannot go undocumented")

    renderers = _renderers()
    if not renderers:
        problems.append("no registered backend exposes a render_harness — this gate could not run and "
                        "does NOT report success")
        return problems

    for target, backend, render in renderers:
        # The whole-op ROSTER: what the code treats as a whole op vs what the contract declares.
        row = by_shape.get("native_whole_op")
        code_ops = getattr(backend, _WHOLE_OP_ATTR, None)
        if row is not None:
            if code_ops is None:
                problems.append(f"{target}: backend exposes no {_WHOLE_OP_ATTR}, so the whole-op opcode "
                                f"roster cannot be compared with the contract's; this gate cannot "
                                f"certify it and does NOT report success")
            else:
                declared_ops = set(row.get("opcodes") or ())
                if set(code_ops) != declared_ops:
                    problems.append(
                        f"{target}: whole-op opcodes {sorted(set(code_ops))} != contract "
                        f"native_whole_op.opcodes {sorted(declared_ops)} — a command shape's roster "
                        f"drifted from the document")

        # Probe every declared shape and compare the emitted call with the resolved tokens.
        probes: list[tuple[str, dict]] = []
        if "movement" in by_shape:
            probes.append(("movement", _probe_movement()))
        if "native_whole_op" in by_shape:
            for opcode in sorted(set((by_shape["native_whole_op"].get("opcodes") or ()))):
                probe = _probe_native_whole_op(str(opcode))
                if probe is None:
                    problems.append(
                        f"{target}: the contract declares whole-op opcode {opcode!r} that this gate has "
                        f"no probe buffer for; it cannot certify that shape and does NOT report success")
                    continue
                probes.append(("native_whole_op", probe))
        if "resident_matmul" in by_shape:
            probes.append(("resident_matmul", _probe_resident_matmul()))

        symbol = symbol_pattern.replace("{target}", target)
        for shape, cb in probes:
            expected: list[str] = []
            try:
                for token in by_shape[shape]["order"]:
                    expected.extend(resolve_token(str(token), cb))
            except Unresolvable as exc:
                problems.append(f"{target}/{shape}: the contract's order could not be resolved against "
                                f"the probe buffer ({exc}) — NOT certified")
                continue
            try:
                text = render(cb, target=target)
            except Exception as exc:     # noqa: BLE001 — an unrenderable declared shape is a refusal
                problems.append(f"{target}/{shape}: the harness renderer refused the probe buffer "
                                f"({type(exc).__name__}: {exc}) — this gate cannot certify that shape "
                                f"and does NOT report success")
                continue
            got = emitted_call_args(text, symbol)
            if got is None:
                problems.append(f"{target}/{shape}: no parseable {symbol}(...) call in the rendered "
                                f"harness — NOT certified")
                continue
            if got != expected:
                problems.append(f"{target}/{shape}: harness passes {got}, contract declares {expected} "
                                f"(order tokens {list(by_shape[shape]['order'])})")
            elif verbose:
                print(f"  ok {target}/{shape}: {symbol}({', '.join(got)})")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verbose", action="store_true", help="print every order it resolved")
    ap.add_argument("--staged", action="store_true", help="accepted for pre-commit symmetry (no-op)")
    a = ap.parse_args(argv)
    problems = check(verbose=a.verbose)
    if problems:
        print("kernel ABI argument-order drift:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("kernel ABI argument order OK: every declared command shape's harness matches the contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

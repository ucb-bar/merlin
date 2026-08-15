"""``merlin_iface`` interface-grammar: emit a Merlin command buffer as contract text, and
parse it back.

The interface grammar is the *frozen, versioned* input format the experiment ABI hands to an
out-of-tree target-backend package. It is a small, regular MLIR module using a custom
``merlin_iface`` dialect — regular enough that a few-line Python regex parser reads it AND a
registered C++ MLIR dialect parses it natively (``mlir-opt``). The canonical spec is
``merlin/contract/interface_grammar.md``; this module is the reference Python implementation.

Design constraints (see the plan / contract):
- Decoupled from xDSL: emission is plain string templating; parsing is plain regex. No xDSL
  import. (Merlin's lowering plane is xDSL, but the *contract surface* must not be.)
- Carries logical tensor names (``W``, ``A0``, ``Y0`` …) because leaf tensors are materialized
  deterministically by name (:func:`merlin.runtime.commandbuffer.materialize_inputs`), so both
  Merlin and the package regenerate identical data and map outputs by name.
- Round-trips: ``parse_interface_mlir(emit_interface_mlir(cb))`` reproduces ``cb``'s
  abi_version / target / tensors (leaves) / commands (opcode, operands, attributes).
"""
from __future__ import annotations

import re
from typing import Any

GRAMMAR_VERSION = "0.1"

# command-buffer opcode  <->  merlin_iface op mnemonic
_OPCODE_TO_OP = {
    "RES_PACK": "resident_pack",
    "MATMUL_RESIDENT": "matmul",
    "COMMIT": "commit",
    "EVICT": "evict",
}
_OP_TO_OPCODE = {v: k for k, v in _OPCODE_TO_OP.items()}

# Named whole-op mnemonics (op classes with no residency decomposition): each maps its positional
# operands to command-buffer operand keys the target codegen reads. The opcode is the mnemonic
# upper-cased. Extend this table (and the command_buffer opcode enum) as codegen gains op classes;
# the parse/emit machinery is generic over it, so a new class is one table row plus its kernel nest.
_NAMED_OP_OPERAND_KEYS = {
    "rmsnorm": ["src", "gamma"],
    "attention_qk": ["q", "k"],
    "rope": ["src"],
    "matmul_batched": ["a", "w"],
}
_NAMED_OP_TO_OPCODE = {op: op.upper() for op in _NAMED_OP_OPERAND_KEYS}
_NAMED_OPCODE_TO_OP = {v: k for k, v in _NAMED_OP_TO_OPCODE.items()}


# --------------------------------------------------------------------------- emit


def _tensor_type(spec: dict[str, Any]) -> str:
    shape = "x".join(str(d) for d in spec["shape"])
    return f"tensor<{shape}x{spec.get('dtype', 'i8')}>"


# float output dtype -> the float accumulate dtype its datapath commits in (a float MXU accumulates in
# bf16/f32, an integer array widens to i32). Keyed on the dtype, never on a target name.
_FLOAT_ACC = {"bf16": "bf16", "bfloat16": "bf16", "f16": "bf16", "float16": "bf16",
              "f32": "f32", "float32": "f32"}


def _acc_dtype(cb: dict[str, Any]) -> str:
    """The accumulator dtype for this command buffer, DERIVED from its commits' declared ``output_dtype``:
    a float output accumulates in a float (bf16/f32), an integer output widens to i32
    (``widening_integer_accumulate``). No target-name literal — a float MXU (e.g. bf16 output) correctly
    gets ``acc<bf16>`` instead of the old hardcoded ``acc<i32>``."""
    for cmd in cb.get("commands", []):
        if cmd.get("opcode") == "COMMIT":
            odt = cmd.get("attributes", {}).get("output_dtype", "")
            if odt in _FLOAT_ACC:
                return _FLOAT_ACC[odt]
    return "i32"


def _fmt_attrs(attrs: dict[str, Any]) -> str:
    parts: list[str] = []
    for k, v in attrs.items():
        if isinstance(v, bool):
            parts.append(f"{k} = {str(v).lower()}")
        elif isinstance(v, (int,)):
            parts.append(f"{k} = {v} : i64")
        elif isinstance(v, float):
            # repr keeps the exact decimal so the Python round-trip is value-stable; the `: f32`
            # suffix is the honest accumulator-scale precision (requant uses an f32 multiply).
            parts.append(f"{k} = {v!r} : f32")
        elif isinstance(v, str):
            parts.append(f'{k} = "{v}"')
        elif isinstance(v, list):
            inner = ", ".join(f'"{x}"' for x in v)
            parts.append(f"{k} = [{inner}]")
        else:  # pragma: no cover - defensive
            raise TypeError(f"unsupported attribute type for {k!r}: {type(v)}")
    return "{" + ", ".join(parts) + "}"


def emit_interface_mlir(cb: dict[str, Any]) -> str:
    """Render a command buffer as ``merlin_iface`` contract text (grammar v0.1)."""
    tensors: dict[str, Any] = cb.get("tensors", {})
    commands = cb.get("commands", [])

    # leaf inputs declared up front, in declaration order
    lines: list[str] = []
    header = (f'module attributes {{merlin_iface.version = "{GRAMMAR_VERSION}", '
              f'merlin_iface.target = "{cb.get("target", "")}", '
              f'merlin_iface.abi_version = "{cb.get("abi_version", "0.1")}"}} {{')
    lines.append(header)

    for name, spec in tensors.items():
        role = spec.get("role", "input")
        lines.append(f'  %{name} = merlin_iface.tensor '
                     f'{{name = "{name}", role = "{role}"}} : {_tensor_type(spec)}')

    acc_t = _acc_dtype(cb)
    # remember each leaf/handle type for printing operand type lists
    val_type: dict[str, str] = {n: _tensor_type(s) for n, s in tensors.items()}

    for cmd in commands:
        op = cmd["opcode"]
        ops = cmd.get("operands", {})
        attrs = cmd.get("attributes", {})
        if op == "RES_PACK":
            src, dst = ops["src"], ops["dst"]
            a = _fmt_attrs({"layout": attrs.get("layout", "packed_rhs")})
            lines.append(f'  %{dst} = merlin_iface.resident_pack %{src} {a} : '
                         f'({val_type[src]}) -> !merlin_iface.resident')
            val_type[dst] = "!merlin_iface.resident"
        elif op in ("MATMUL_RESIDENT", "MATMUL"):
            lhs, rhs, dst = ops["lhs"], ops["rhs"], ops["dst"]
            lines.append(f'  %{dst} = merlin_iface.matmul %{lhs}, %{rhs} : '
                         f'({val_type[lhs]}, {val_type[rhs]}) -> !merlin_iface.acc<{acc_t}>')
            val_type[dst] = f"!merlin_iface.acc<{acc_t}>"
        elif op == "COMMIT":
            src, dst = ops["src"], ops["dst"]
            # output shape = m x n: m from the matmul lhs, n from the resident weight.
            out_attrs = dict(attrs)
            out_attrs = {"name": dst, **out_attrs}
            odt = attrs.get("output_dtype", "i8")
            m, n = _commit_out_shape(cb, src)
            lines.append(f'  %{dst} = merlin_iface.commit %{src} {_fmt_attrs(out_attrs)} : '
                         f'(!merlin_iface.acc<{acc_t}>) -> tensor<{m}x{n}x{odt}>')
            val_type[dst] = f"tensor<{m}x{n}x{odt}>"
        elif op == "EVICT":
            h = ops["handle"]
            lines.append(f'  merlin_iface.evict %{h} : (!merlin_iface.resident) -> ()')
        else:  # pragma: no cover - defensive
            raise ValueError(f"unsupported opcode {op!r} for interface grammar v{GRAMMAR_VERSION}")

    lines.append("}")
    return "\n".join(lines) + "\n"


def _commit_out_shape(cb: dict[str, Any], acc_name: str) -> tuple[int, int]:
    """(m, n) for a commit: m from the matmul lhs rows, n from the resident weight cols."""
    tensors = cb.get("tensors", {})
    res_src: dict[str, str] = {}
    for cmd in cb.get("commands", []):
        if cmd["opcode"] == "RES_PACK":
            res_src[cmd["operands"]["dst"]] = cmd["operands"]["src"]
    for cmd in cb.get("commands", []):
        if cmd["opcode"] in ("MATMUL_RESIDENT", "MATMUL") and cmd["operands"]["dst"] == acc_name:
            lhs = cmd["operands"]["lhs"]
            rhs = cmd["operands"]["rhs"]
            w = res_src.get(rhs, rhs)
            m = tensors[lhs]["shape"][0]
            n = tensors[w]["shape"][1]
            return int(m), int(n)
    raise ValueError(f"no matmul produces accumulator {acc_name!r}")


# --------------------------------------------------------------------------- parse

_RE_TENSOR = re.compile(r'%(\S+)\s*=\s*merlin_iface\.tensor\s*\{([^}]*)\}\s*:\s*(tensor<[^>]+>)')
_RE_PACK = re.compile(r'%(\S+)\s*=\s*merlin_iface\.resident_pack\s*%(\S+)\s*\{([^}]*)\}')
_RE_MATMUL = re.compile(r'%(\S+)\s*=\s*merlin_iface\.matmul\s*%(\S+),\s*%(\S+)\s*:')
_RE_COMMIT = re.compile(r'%(\S+)\s*=\s*merlin_iface\.commit\s*%(\S+)\s*\{([^}]*)\}\s*:\s*\(.*?\)\s*->\s*(tensor<[^>]+>)')
_RE_EVICT = re.compile(r'merlin_iface\.evict\s*%(\S+)')
_RE_MOD = re.compile(r'module\s+attributes\s*\{([^}]*)\}')
# A whole-op mnemonic line: `%dst = merlin_iface.<mnemonic> %a, %b, ... {attrs} : (types) -> type`.
# The mnemonic (group 2) is looked up in _NAMED_OP_OPERAND_KEYS to decide whether to handle it and how
# to name its operands; matmul/commit/etc. fall through to their own patterns (not in that table).
_RE_NAMED_OP = re.compile(
    r'%(\S+)\s*=\s*merlin_iface\.(\w+)\s+((?:%\S+\s*,?\s*)+)\{([^}]*)\}')


def _parse_attr_block(s: str) -> dict[str, Any]:
    """Parse a ``{k = v, ...}`` attribute body into a dict (strings, ints, floats, lists)."""
    out: dict[str, Any] = {}
    # split on commas that are not inside brackets
    depth = 0
    cur = ""
    parts: list[str] = []
    for ch in s:
        if ch in "[":
            depth += 1
        elif ch in "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        out[k] = _parse_value(v)
    return out


def _parse_value(v: str) -> Any:
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        body = v[1:-1].strip()
        if not body:
            return []
        return [_parse_value(x) for x in body.split(",")]
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    # typed scalar: "<num> : <type>"
    num = v.split(":")[0].strip()
    if re.fullmatch(r"[-+]?\d+", num):
        return int(num)
    try:
        return float(num)
    except ValueError:
        return num


def _shape_dtype(ttype: str) -> tuple[list[int], str]:
    """Parse ``tensor<D0xD1x...xELEM>`` into (dims, element-dtype) STRUCTURALLY (no regex): split the
    shape/elem list on ``x``; the trailing token is the element dtype, the leading tokens are integer dims.
    Deriving the dtype instead of matching a fixed ``i\\d+|f\\d+|bf\\d+`` alternation is what lets it accept
    the OCP MX float spellings (``f8E4M3FN`` / ``f6E3M2FN`` / ``f4E2M1FN``) the old narrow pattern silently
    dropped — the exact "too-narrow regex mis-measures a conformant input" failure this repo forbids."""
    s = ttype.strip()
    lb, rb = s.find("<"), s.rfind(">")
    if not s.startswith("tensor") or lb < 0 or rb <= lb:
        raise ValueError(f"unparseable tensor type {ttype!r}")
    body = s[lb + 1:rb].split(",", 1)[0].strip()   # drop any trailing layout/encoding attribute
    parts = body.split("x")
    if len(parts) < 2:
        raise ValueError(f"unparseable tensor type {ttype!r} (need at least one dim and an element type)")
    *dim_toks, dtype = parts
    try:
        dims = [int(d) for d in dim_toks]
    except ValueError as e:
        raise ValueError(f"unparseable tensor type {ttype!r} (non-integer dim)") from e
    if not dtype:
        raise ValueError(f"unparseable tensor type {ttype!r} (empty element type)")
    return dims, dtype


def parse_interface_mlir(text: str) -> dict[str, Any]:
    """Parse ``merlin_iface`` contract text back into a command-buffer dict.

    Reconstructs abi_version / target / tensors (leaf inputs) / commands so that
    ``parse_interface_mlir(emit_interface_mlir(cb)) == cb`` for the supported op set.
    """
    mod = _RE_MOD.search(text)
    mod_attrs = _parse_attr_block(mod.group(1)) if mod else {}
    cb: dict[str, Any] = {
        "abi_version": mod_attrs.get("merlin_iface.abi_version", "0.1"),
        "target": mod_attrs.get("merlin_iface.target", ""),
        "tensors": {},
        "commands": [],
    }

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        m = _RE_TENSOR.search(line)
        if m:
            attrs = _parse_attr_block(m.group(2))
            name = attrs.get("name", m.group(1))
            shape, dtype = _shape_dtype(m.group(3))
            cb["tensors"][name] = {"shape": shape, "dtype": dtype,
                                   "role": attrs.get("role", "input")}
            continue
        m = _RE_PACK.search(line)
        if m:
            dst, src = m.group(1), m.group(2)
            attrs = _parse_attr_block(m.group(3))
            cb["commands"].append({"opcode": "RES_PACK",
                                   "operands": {"src": src, "dst": dst},
                                   "attributes": {"layout": attrs.get("layout", "packed_rhs")}})
            continue
        m = _RE_MATMUL.search(line)
        if m:
            dst, lhs, rhs = m.group(1), m.group(2), m.group(3)
            cb["commands"].append({"opcode": "MATMUL_RESIDENT",
                                   "operands": {"lhs": lhs, "rhs": rhs, "dst": dst}})
            continue
        m = _RE_NAMED_OP.search(line)
        if m and m.group(2) in _NAMED_OP_OPERAND_KEYS:
            dst_ssa, mnem = m.group(1), m.group(2)
            srcs = [tok.strip().lstrip("%") for tok in m.group(3).split(",") if tok.strip()]
            attrs = _parse_attr_block(m.group(4))
            keys = _NAMED_OP_OPERAND_KEYS[mnem]
            operands = {k: s for k, s in zip(keys, srcs)}
            operands["dst"] = attrs.pop("name", dst_ssa)
            cb["commands"].append({"opcode": _NAMED_OP_TO_OPCODE[mnem],
                                   "operands": operands, "attributes": attrs})
            continue
        m = _RE_COMMIT.search(line)
        if m:
            dst_ssa, src = m.group(1), m.group(2)
            attrs = _parse_attr_block(m.group(3))
            dst = attrs.pop("name", dst_ssa)
            cb["commands"].append({"opcode": "COMMIT",
                                   "operands": {"src": src, "dst": dst},
                                   "attributes": attrs})
            continue
        m = _RE_EVICT.search(line)
        if m:
            cb["commands"].append({"opcode": "EVICT", "operands": {"handle": m.group(1)}})
            continue

    return cb

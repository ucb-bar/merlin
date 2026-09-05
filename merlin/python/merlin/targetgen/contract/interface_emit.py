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
- FAILS CLOSED on an op the grammar does not define (:class:`InterfaceGrammarError`). It used to
  return only the commands it recognised and say nothing about the rest, which silently mis-read 15
  of 160 shipped interface capsules; see the exception's docstring for the measurement.
"""
from __future__ import annotations

from typing import Any

GRAMMAR_VERSION = "0.1"


class InterfaceGrammarError(ValueError):
    """A module uses a ``merlin_iface`` op mnemonic the frozen grammar does not define.

    Raised, never warned, because the alternative was measured and is worse: this parser used to
    return only the commands it recognised and report nothing about the rest, so a module using an
    undefined op parsed "successfully" into a SHORTER command list. Across the shipped corpora that
    silently mis-read 15 of 160 interface capsules — 5 ``movement`` capsules parsed to zero commands
    (their only op vanished) and 7 flash-attention capsules lost their second matmul. A backend
    package built on that output computes the wrong thing with nothing to point at. Subclasses
    ``ValueError`` so existing ``except ValueError`` callers keep working.
    """


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
    # P @ V, the second matmul of flash attention and the sibling of attention_qk (which contracts
    # over the trailing head dim of both operands; this one is a plain [m,s] x [s,d] contraction).
    # Undefined until now, so all 7 shipped flash-attention capsules parsed WITHOUT their second
    # matmul and the parser reported success — the measured wrong-answer this table row closes.
    "attention_pv": ["p", "v"],
    "rope": ["src"],
    "matmul_batched": ["a", "w"],
    "softmax": ["src"],
    # Identity round-trip through the accelerator (load -> store, operand dtype in / accumulate dtype
    # out; see corpus_spec.build_movement). Its capsules carry exactly ONE op, so leaving it undefined
    # made all 5 of them parse to ZERO commands while still reporting success.
    "movement": ["src"],
    # im2col convolution: an NHWC activation contracted against a PRE-im2col'd weight
    # [Kh*Kw*Ci, Co] that has been made resident, producing [N*Ho*Wo, Co] (see
    # corpus_spec.build_conv2d, which writes the only spelling this grammar accepts). The geometry
    # rides in the attributes — ``kernel = [kh, kw, ci, co]``, ``stride``, ``padding``, ``dilation``,
    # ``layout`` — not in extra operands, so the operand list is just the activation and the weight.
    # Undefined until now, so all 3 shipped conv capsules lost their ONLY compute op and were left
    # with resident_pack + evict; the fail-closed parse surfaced that instead of hiding it, and this
    # row is what finally lets them read whole.
    "conv2d": ["ifm", "weight"],
    # A per-column bias added to an already-committed tensor: the `bias_add` COMMIT stage standing on
    # its own, so the L5 fusion claim has an unfused half to be compared against. Its operands are in
    # the accumulator's dtype, because that is the domain the stage runs in.
    #
    # Added with the op, not after it: `boundary.grammar_mnemonics` reads THIS table, so a mnemonic the
    # emitter writes and this table does not define makes the canonical parser refuse the module -- and
    # every capsule using it classifies UNKNOWN. Documenting the op in `interface_grammar.md` without
    # this row is exactly the doc/parser drift the single-source design exists to prevent.
    "bias_add": ["src", "bias"],
}
# Most mnemonics map to their upper-case opcode; a few need an explicit target opcode because the emitter /
# harness / simulator spell it differently (``matmul_batched`` -> the ``BATCHED_MATMUL`` those consume, not
# the auto-upper ``MATMUL_BATCHED``). Keep this the ONE place the spelling is reconciled.
_NAMED_OP_OPCODE_OVERRIDE = {"matmul_batched": "BATCHED_MATMUL"}
_NAMED_OP_TO_OPCODE = {op: _NAMED_OP_OPCODE_OVERRIDE.get(op, op.upper()) for op in _NAMED_OP_OPERAND_KEYS}
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
            # Element-wise, NOT blanket-quoted. Quoting every element turned an integer geometry
            # (``pool_size = [2, 2]``) into ``["2", "2"]``, which the parser reads back as STRINGS --
            # so a round-tripped pooling command lost its window to a typing mismatch reported far from
            # the spelling that caused it. String stages (``epilogue = ["relu"]``) are unchanged.
            inner = ", ".join(f'"{x}"' if isinstance(x, str) else str(x) for x in v)
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
            m, n = _commit_out_shape(cb, src, attrs)
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


def _commit_out_shape(cb: dict[str, Any], acc_name: str,
                      attrs: dict[str, Any] | None = None) -> tuple[int, int]:
    """(m, n) for a commit: m from the matmul lhs rows, n from the resident weight cols.

    A POOLING epilogue reduces m: the accumulator's rows unflatten to a ``pool_in_dims`` plane and the
    window walks it, so the committed tensor has fewer rows than the matmul produced. Printing the
    matmul's row count for a pooled commit would emit a module whose declared result type disagrees
    with what any engine executing it computes -- a type error the grammar cannot catch because the
    grammar is what would be lying."""
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
            m = int(tensors[lhs]["shape"][0])
            n = int(tensors[w]["shape"][1])
            if "maxpool" in ((attrs or {}).get("epilogue") or []):
                from merlin.runtime.commandbuffer import pool_params
                from merlin.runtime.tensor import pool_out_dims
                p = pool_params(attrs or {}, op=f"COMMIT of {acc_name!r}")
                H, W = p["pool_in_dims"]
                Ho, Wo = pool_out_dims(H, W, p["pool_size"], p["pool_stride"], p["pool_padding"])
                m = (m // (H * W)) * Ho * Wo
            return m, n
    raise ValueError(f"no matmul produces accumulator {acc_name!r}")


# --------------------------------------------------------------------------- parse

#: The dialect prefix every op line in this grammar carries. Lines are matched by SHAPE -- an SSA
#: result, this prefix, a mnemonic, %-prefixed operands, an optional {attrs} body, a type after ':' --
#: rather than by one pattern per op form.
#:
#: Why not patterns. A pattern per form is a second grammar definition that drifts from the emitter's,
#: and this file has already shipped the failure that causes: an op the patterns did not cover was
#: silently DROPPED and the parse reported success, so 15 of 160 shipped capsules parsed into a command
#: list missing their work. Shape-matching means an unknown mnemonic reaches the mnemonic check and is
#: refused by name instead of falling through every pattern and vanishing.
#:
#: xDSL is deliberately NOT used here even though it is a dependency elsewhere: this module ships in the
#: agent starter kit, where the contract forbids that dependency.
_DIALECT = "merlin_iface."
_MODULE_KW = "module"
_ATTRIBUTES_KW = "attributes"


def _braced(text: str, start: int = 0) -> tuple[str, int]:
    """The body of the first ``{...}`` at or after ``start``, and the index just past its close.

    Depth-tracked, because an attribute value may itself contain braces; taking the first ``}`` would
    truncate the body and silently drop every attribute after it.
    """
    open_i = text.find("{", start)
    if open_i == -1:
        return "", -1
    depth = 0
    for i in range(open_i, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_i + 1:i], i + 1
    return "", -1


def _split_operands(text: str) -> list[str]:
    """``%a, %b`` -> ``['a', 'b']``. Stops at the first token that does not start with ``%``."""
    out = []
    for tok in text.replace(",", " ").split():
        if not tok.startswith("%"):
            break
        name = tok[1:].strip()
        if name:
            out.append(name)
    return out


def _op_line(line: str) -> dict | None:
    """One ``merlin_iface`` op line, decomposed by shape, or ``None`` when the line carries no op.

    Returns ``{result, mnemonic, operands, attrs_body, tail}`` where ``tail`` is everything after the
    type colon -- enough for every form the grammar defines, without a form-specific pattern.
    """
    at = line.find(_DIALECT)
    if at == -1:
        return None
    head = line[:at]
    # `!merlin_iface.resident` is a TYPE, and a type never opens a statement.
    if head.rstrip().endswith("!"):
        return None
    result = ""
    if "=" in head:
        lhs = head.split("=", 1)[0].strip()
        if lhs.startswith("%"):
            result = lhs[1:].strip()
    rest = line[at + len(_DIALECT):]
    mnemonic = ""
    for i, ch in enumerate(rest):
        if ch.isalnum() or ch == "_":
            mnemonic += ch
        else:
            rest = rest[i:]
            break
    else:
        rest = ""
    operands = _split_operands(rest.split("{")[0].split(":")[0])
    attrs_body, after = _braced(rest)
    tail = rest[after:] if after != -1 else rest
    _, _, tail = tail.partition(":")
    return {"result": result, "mnemonic": mnemonic, "operands": operands,
            "attrs_body": attrs_body, "tail": tail.strip()}


def _module_attrs(text: str) -> str:
    """The module-level attribute body, or ``""``. Found by keyword over the WHOLE text.

    Scanned across the whole string rather than line by line, because MLIR may print the header
    wrapped -- `module attributes {` then one attribute per line then `} {`. A per-line scan finds the
    opening line, fails to find a close on it, and returns nothing, so a conformant module silently
    loses its target and abi_version. Caught by an existing test rather than by the 499-capsule
    equivalence check, since no shipped capsule happens to wrap its header: a narrower accept set that
    no current input exercises is exactly the drift this rule exists to prevent.
    """
    at = text.find(_MODULE_KW)
    while at != -1:
        after = text[at + len(_MODULE_KW):]
        kw = after.find(_ATTRIBUTES_KW)
        brace = after.find("{")
        # The keyword must come before the brace it introduces; otherwise this `module` is something
        # else (a nested region, a word in a comment) and the next occurrence is tried.
        if kw != -1 and (brace == -1 or kw < brace):
            body, _ = _braced(after)
            if body:
                return body
        at = text.find(_MODULE_KW, at + len(_MODULE_KW))
    return ""


def _last_type(tail: str) -> str:
    """The result type a line declares: the text after the last ``->``, else the tail itself."""
    _, arrow, after = tail.rpartition("->")
    return (after if arrow else tail).strip()


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
    # An optionally-signed run of digits. `int()` alone is too permissive here: it accepts "1_000"
    # and surrounding whitespace, which the pattern this replaces did not, and a silently different
    # accept set in an ABI parser is the failure this module has already shipped once.
    _digits = num[1:] if num[:1] in "+-" else num
    if _digits.isdigit() and _digits.isascii():
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


#: Grammar ops that declare structure rather than issue a command — they produce no command-buffer
#: entry, so the per-line parse loop handles them but they are still DEFINED mnemonics.
_STRUCTURAL_OPS = frozenset({"tensor"})

_IFACE_MARKER = "merlin_iface."
#: A type in this grammar always carries the ``!`` sigil (``!merlin_iface.resident``,
#: ``!merlin_iface.acc<i32>``) and a type never opens a statement; an op does. Module attributes
#: (``merlin_iface.version``/``.target``/``.abi_version``) only ever occur on the ``module`` line.
_TYPE_SIGIL = "!"
#: Characters that terminate a mnemonic in this grammar (whitespace, the attribute/operand/type
#: punctuation). Split structurally rather than pattern-matched: a too-narrow mnemonic pattern is
#: precisely how valid-but-differently-spelled input gets dropped.
_MNEMONIC_STOPS = (" ", "\t", "{", "(", ")", "<", ">", ":", ",", "%", "!", '"')


def defined_mnemonics() -> frozenset[str]:
    """The op mnemonics grammar v0.1 defines, read from this module's OWN dispatch tables.

    Derived from the tables the parse loop actually consults, so adding an op is one table row and
    this follows for free — a second hand-maintained list would be a thing that can drift from the
    parser, which is the class of bug this whole function exists to prevent.
    """
    return frozenset(set(_OP_TO_OPCODE) | set(_NAMED_OP_OPERAND_KEYS) | _STRUCTURAL_OPS)


def op_mnemonics(text: str) -> list[str]:
    """Every ``merlin_iface`` OP mnemonic in a module, in program order (a tokenizer, not a parser).

    Structural, per the no-regex rule: partition each line on the dialect marker and cut the mnemonic
    at the first grammar delimiter. Deliberately independent of which ops are *defined*, so a mnemonic
    outside the frozen grammar is still SEEN — that is what lets the parser fail closed on it instead
    of skipping the line.
    """
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        # `module attributes {merlin_iface.version = ...}` carries ATTRIBUTES, not ops.
        if not line or line.startswith("//") or line.startswith("module") or line.startswith("}"):
            continue
        head, sep, rest = line.partition(_IFACE_MARKER)
        while sep:
            if not head.endswith(_TYPE_SIGIL):        # `!merlin_iface.resident` is a type, not an op
                mnem = rest
                for stop in _MNEMONIC_STOPS:
                    mnem = mnem.partition(stop)[0]
                # An ATTRIBUTE is immediately assigned (`merlin_iface.version = "0.1"`); no op in this
                # grammar is. Checking that as well as the `module` line means a header wrapped across
                # lines cannot be misread as three ops named version/target/abi_version — a false
                # "undefined mnemonic" would be as unhelpful as the silent drop it replaces.
                if mnem and not rest[len(mnem):].lstrip().startswith("="):
                    out.append(mnem)
            head, sep, rest = rest.partition(_IFACE_MARKER)
    return out


def undefined_op_mnemonics(text: str) -> list[str]:
    """Mnemonics a module uses that grammar v0.1 does not define, sorted and de-duplicated."""
    known = defined_mnemonics()
    return sorted({m for m in op_mnemonics(text) if m not in known})


def parse_interface_mlir(text: str) -> dict[str, Any]:
    """Parse ``merlin_iface`` contract text back into a command-buffer dict.

    Reconstructs abi_version / target / tensors (leaf inputs) / commands so that
    ``parse_interface_mlir(emit_interface_mlir(cb)) == cb`` for the supported op set.

    Raises :class:`InterfaceGrammarError`, naming the mnemonic, for any ``merlin_iface`` op grammar
    v0.1 does not define. FAIL CLOSED: the returned command list is the program, and a caller cannot
    tell a short list from a complete one, so an unreadable op must stop the parse rather than
    shorten its result. See the class docstring for the 15-of-160 measurement.
    """
    undefined = undefined_op_mnemonics(text)
    if undefined:
        raise InterfaceGrammarError(
            f"interface grammar v{GRAMMAR_VERSION} does not define merlin_iface op(s) "
            f"{', '.join(repr(m) for m in undefined)}; it defines "
            f"{', '.join(repr(m) for m in sorted(defined_mnemonics()))}. Parsing would drop the "
            f"undefined op(s) and return a command list that is missing that work")

    mod_attrs = _parse_attr_block(_module_attrs(text))
    cb: dict[str, Any] = {
        "abi_version": mod_attrs.get("merlin_iface.abi_version", "0.1"),
        "target": mod_attrs.get("merlin_iface.target", ""),
        "tensors": {},
        "commands": [],
    }

    # ONE decomposition per line, then a dispatch on the mnemonic. Previously each op form had its own
    # pattern and the line was tried against each in turn, which is what let an undefined op fall
    # through every one of them and vanish. Here an unknown mnemonic cannot fall through: it either
    # matched the shape and is dispatched, or it did not and the line carries no op at all.
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        op = _op_line(line)
        if op is None or not op["mnemonic"]:
            continue
        mnem, attrs = op["mnemonic"], _parse_attr_block(op["attrs_body"])
        srcs, result = op["operands"], op["result"]

        if mnem == "tensor":
            name = attrs.get("name", result)
            shape, dtype = _shape_dtype(_last_type(op["tail"]))
            cb["tensors"][name] = {"shape": shape, "dtype": dtype,
                                   "role": attrs.get("role", "input")}
        elif mnem == "resident_pack":
            cb["commands"].append({"opcode": "RES_PACK",
                                   "operands": {"src": srcs[0], "dst": result},
                                   "attributes": {"layout": attrs.get("layout", "packed_rhs")}})
        elif mnem == "matmul":
            cb["commands"].append({"opcode": "MATMUL_RESIDENT",
                                   "operands": {"lhs": srcs[0], "rhs": srcs[1], "dst": result}})
        elif mnem in _NAMED_OP_OPERAND_KEYS:
            keys = _NAMED_OP_OPERAND_KEYS[mnem]
            operands = {k: s for k, s in zip(keys, srcs)}
            operands["dst"] = attrs.pop("name", result)
            cb["commands"].append({"opcode": _NAMED_OP_TO_OPCODE[mnem],
                                   "operands": operands, "attributes": attrs})
        elif mnem == "commit":
            dst = attrs.pop("name", result)
            cb["commands"].append({"opcode": "COMMIT",
                                   "operands": {"src": srcs[0], "dst": dst},
                                   "attributes": attrs})
        elif mnem == "evict":
            cb["commands"].append({"opcode": "EVICT", "operands": {"handle": srcs[0]}})

    return cb


def to_generic_form(text: str) -> str:
    """Re-spell a ``merlin_iface`` module in MLIR's GENERIC op syntax, semantics unchanged.

    The pretty form this module emits is the contract surface and stays the contract surface. But an
    MLIR dialect registered DYNAMICALLY from IRDL (``mlir-opt --irdl-file=``, ``irdl::loadDialects``)
    has no custom parser -- a generated dialect has no ``assemblyFormat`` to run -- so it can read
    only the generic spelling. Measured before this existed: ``mlir-opt --irdl-file`` parsed 0 of the
    370 ``merlin_iface`` capsules, and did so with rc=1 and an EMPTY stderr, which is why the gap sat
    unnoticed. This is the one-way bridge that makes the IRDL contract checkable against the real
    corpus instead of against hand-written fixtures.

    Only the SPELLING changes::

        %Y = merlin_iface.commit %acc {name = "Y"} : (!merlin_iface.acc<i32>) -> tensor<4x4xi8>
        %Y = "merlin_iface.commit"(%acc) {name = "Y"} : (!merlin_iface.acc<i32>) -> tensor<4x4xi8>

    Reuses :func:`_op_line`, the module's ONE shape decomposition, so this cannot drift from what
    :func:`parse_interface_mlir` reads. FAILS CLOSED, for the same reason that function does: a line
    it cannot re-spell raises rather than passing through in pretty form, because a partially
    converted module parses as a DIFFERENT, shorter program with nothing to point at.
    """
    undefined = undefined_op_mnemonics(text)
    if undefined:
        raise InterfaceGrammarError(
            f"cannot re-spell in generic form: interface grammar v{GRAMMAR_VERSION} does not define "
            f"merlin_iface op(s) {', '.join(repr(m) for m in undefined)}")

    out: list[str] = []
    for raw in text.splitlines():
        op = _op_line(raw.strip()) if raw.strip() and not raw.strip().startswith("//") else None
        if op is None or not op["mnemonic"] or not _is_op_statement(raw, op["mnemonic"]):
            out.append(raw)
            continue
        indent = raw[: len(raw) - len(raw.lstrip())]
        ftype = _functional_type(op)
        attrs = f' {{{op["attrs_body"]}}}' if op["attrs_body"] else ""
        operands = ", ".join(f"%{s}" for s in op["operands"])
        lhs = f'%{op["result"]} = ' if op["result"] else ""
        out.append(f'{indent}{lhs}"{_IFACE_MARKER}{op["mnemonic"]}"({operands}){attrs} : {ftype}')
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _functional_type(op: dict) -> str:
    """The ``(operands) -> results`` type the generic form requires, from a pretty line's tail.

    Which of the two ODS type formats an op uses is read off the tail itself rather than tabled per
    mnemonic: ``functional-type(operands, results)`` already prints the parenthesised operand list,
    while ``type($result)`` prints a bare result type (``merlin_iface.tensor``, whose operand list is
    empty). Deriving it keeps a new grammar op from needing a new table row.
    """
    tail = op["tail"].strip()
    if tail.startswith("("):
        return tail
    if op["operands"]:
        # A bare result type with operands would silently become `() -> T`, dropping the operand
        # types from the module's own signature. Refuse instead of guessing them.
        raise InterfaceGrammarError(
            f"merlin_iface.{op['mnemonic']} prints a bare result type but takes "
            f"{len(op['operands'])} operand(s); its operand types cannot be recovered from the line")
    return f"() -> {tail}"


def _is_op_statement(line: str, mnemonic: str) -> bool:
    """False when ``merlin_iface.<mnemonic>`` on this line is an ATTRIBUTE KEY, not an op.

    The module header spells its metadata in the same namespace
    (``module attributes {merlin_iface.version = "0.1", ...}``), so the shape decomposition finds a
    "version" op there. The discriminator is the one :func:`op_mnemonics` already uses: an attribute
    key is followed by ``=``, an op never is. Without it the header is rewritten into a bogus
    ``"merlin_iface.version"() : () ->`` and the module loses its version, target and abi_version.
    """
    at = line.find(_IFACE_MARKER)
    return not line[at + len(_IFACE_MARKER) + len(mnemonic):].lstrip().startswith("=")

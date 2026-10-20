"""Explicit host legalization of a proved, bounded binary32 ties-even chain.

This late seam preserves upstream tensor fusion. An explicit typed SSA tokenizer
recognizes the complete clamp/truncate/fraction/parity/sign chain; unrelated
instructions and other uses retain their source bytes. Unknown forms, fast FP,
strict FP and constrained operations are not legalized. No policy is selected by
default. CPU ISA emission is a host codegen policy, independent of an accelerator.
The equivalent bounded ``llvm.roundeven``/integer conversion is also accepted,
so earlier shared legalization does not hide the same host codegen opportunity.
"""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int


def _tokens(text: str) -> list[_Token]:
    """LLVM punctuation, comments, identifiers and quoted strings with spans.

    This is an intentionally small lexer, not a permissive LLVM module parser.
    LLVM's assembler verifies the whole module at the public build seam. The
    matcher below accepts only explicit instruction productions with typed
    operands and a recognized statement boundary.
    """
    result: list[_Token] = []
    i = 0
    punctuation = frozenset("(){}[],=<>:*!")
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        if text[i] == ";":
            end = text.find("\n", i)
            i = len(text) if end < 0 else end + 1
            continue
        start = i
        if text[i] in "%@":
            i += 1
        if i < len(text) and text[i] == '"':
            i += 1
            while i < len(text) and text[i] != '"':
                if text[i] == "\\":
                    # LLVM quoted bytes use two hexadecimal escape digits.
                    i += 3
                else:
                    i += 1
            if i >= len(text):
                raise ValueError("unterminated LLVM quoted token")
            i += 1
        elif i == start and text[i] in punctuation:
            i += 1
        else:
            while i < len(text) and not text[i].isspace() and text[i] not in punctuation and text[i] not in ';"':
                i += 1
        if i == start:
            raise ValueError("unsupported LLVM token")
        result.append(_Token(text[start:i], start, i))
    return result


def _identity(token: str) -> str:
    value = token[1:] if token.startswith(("%", "@")) else token
    if not value.startswith('"'):
        return value
    value = value[1:-1]
    chars: list[str] = []
    i = 0
    while i < len(value):
        if value[i] == "\\":
            chars.append(chr(int(value[i + 1 : i + 3], 16)))
            i += 3
        else:
            chars.append(value[i])
            i += 1
    return "".join(chars)


@dataclass(frozen=True)
class _Instruction:
    result: str
    opcode: str
    dtype: str
    operands: tuple[str, ...]
    start: int
    end: int
    predicate: str = ""
    callee: str = ""
    output_type: str = ""


_STATEMENTS = frozenset(
    [
        "ret",
        "br",
        "switch",
        "store",
        "call",
        "invoke",
        "unreachable",
        "fence",
        "resume",
        "indirectbr",
        "catchret",
        "cleanupret",
        "catchswitch",
        "atomicrmw",
        "cmpxchg",
    ]
)


def _statement_boundary(tokens: list[_Token], i: int) -> bool:
    if i == len(tokens) or tokens[i].text == "}":
        return True
    if tokens[i].text in _STATEMENTS:
        return True
    if i + 1 < len(tokens):
        return tokens[i + 1].text == ":" or (tokens[i].text.startswith("%") and tokens[i + 1].text == "=")
    return False


def _instruction(tokens: list[_Token], i: int) -> _Instruction | None:
    """Recognize one entire supported typed instruction, without FP/int flags."""
    t = [x.text for x in tokens[i : i + 16]]
    if len(t) < 5 or not t[0].startswith("%") or t[1] != "=":
        return None
    opcode = t[2]
    predicate = callee = output_type = ""
    if opcode in ("add", "and", "or", "fsub") and len(t) >= 7 and t[5] == ",":
        dtype, operands, count = t[3], (t[4], t[6]), 7
    elif opcode in ("fcmp", "icmp") and len(t) >= 8 and t[6] == ",":
        predicate, dtype, operands, count = t[3], t[4], (t[5], t[7]), 8
    elif opcode in ("fptosi", "sitofp") and len(t) >= 7 and t[5] == "to":
        dtype, operands, output_type, count = t[3], (t[4],), t[6], 7
    elif opcode == "fneg":
        dtype, operands, count = t[3], (t[4],), 5
    elif opcode == "select" and len(t) >= 11 and t[3] == "i1" and t[5] == "," and t[8] == "," and t[6] == t[9]:
        dtype, operands, count = t[6], (t[4], t[7], t[10]), 11
    elif opcode == "call" and len(t) >= 9 and t[4].startswith("@") and t[5] == "(" and t[8] == ")" and t[3] == t[6]:
        dtype, callee, operands, count = t[3], _identity(t[4]), (t[7],), 9
    elif (
        opcode == "call"
        and len(t) >= 12
        and t[4].startswith("@")
        and t[5] == "("
        and t[8] == ","
        and t[11] == ")"
        and t[3] == t[6] == t[9]
    ):
        dtype, callee, operands, count = t[3], _identity(t[4]), (t[7], t[10]), 12
    else:
        return None
    if not _statement_boundary(tokens, i + count):
        return None
    return _Instruction(
        t[0], opcode, dtype, operands, tokens[i].start, tokens[i + count - 1].end, predicate, callee, output_type
    )


def _functions(tokens: list[_Token]) -> list[list[_Token]]:
    functions: list[list[_Token]] = []
    i = 0
    while i < len(tokens):
        if tokens[i].text != "define":
            i += 1
            continue
        # Return aggregates can contain braces; find the function symbol and
        # balanced argument list first. Exotic prefix/prologue headers refuse.
        while i < len(tokens) and not tokens[i].text.startswith("@"):
            i += 1
        if i + 1 >= len(tokens) or tokens[i + 1].text != "(":
            raise ValueError("unsupported LLVM function header")
        i += 2
        depth = 1
        while i < len(tokens) and depth:
            depth += (tokens[i].text == "(") - (tokens[i].text == ")")
            i += 1
        while i < len(tokens) and tokens[i].text != "{":
            if tokens[i].text in ("prefix", "prologue", "personality"):
                raise ValueError("unsupported LLVM function header contract")
            i += 1
        if i == len(tokens):
            raise ValueError("unterminated LLVM function")
        start = i + 1
        depth = 1
        i += 1
        while i < len(tokens) and depth:
            depth += (tokens[i].text == "{") - (tokens[i].text == "}")
            i += 1
        if depth:
            raise ValueError("unterminated LLVM function")
        functions.append(tokens[start : i - 1])
    return functions


def _number(value: str) -> float | None:
    try:
        if value.startswith("0x") and len(value) == 18:
            return struct.unpack(">d", bytes.fromhex(value[2:]))[0]
        return float(value)
    except (ValueError, OverflowError):
        return None


def _match(result: _Instruction, definitions: dict[str, _Instruction]) -> dict | None:
    def definition(
        value: str, opcode: str, dtype: str, *, predicate="", callee="", output_type=""
    ) -> _Instruction | None:
        op = definitions.get(_identity(value)) if value.startswith("%") else None
        return (
            op
            if (
                op is not None
                and op.opcode == opcode
                and op.dtype == dtype
                and op.predicate == predicate
                and op.callee == callee
                and op.output_type == output_type
            )
            else None
        )

    def same(a: str, b: str) -> bool:
        return a.startswith("%") and b.startswith("%") and _identity(a) == _identity(b)

    dtype = result.dtype
    if result.opcode != "add" or not dtype.startswith("i") or not dtype[1:].isdigit() or not 1 <= int(dtype[1:]) <= 24:
        return None
    bits = int(dtype[1:])
    lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    trunc, bump = result.operands
    t = definition(trunc, "fptosi", "float", output_type=dtype)
    b = definition(bump, "select", dtype)
    if t is None or b is None or _number(b.operands[2]) != 0:
        return None
    x = t.operands[0]
    sign = definition(b.operands[1], "select", dtype)
    trigger = definition(b.operands[0], "or", "i1")
    high = definition(x, "call", "float", callee="llvm.minimum.f32")
    if (
        sign is None
        or trigger is None
        or high is None
        or tuple(_number(v) for v in sign.operands[1:]) != (-1, 1)
        or _number(high.operands[1]) != upper
    ):
        return None
    low = definition(high.operands[0], "call", "float", callee="llvm.maximum.f32")
    negative_sign = definition(sign.operands[0], "fcmp", "float", predicate="olt")
    if (
        low is None
        or _number(low.operands[1]) != lower
        or negative_sign is None
        or not same(negative_sign.operands[0], x)
        or _number(negative_sign.operands[1]) != 0
    ):
        return None
    greater = definition(trigger.operands[0], "fcmp", "float", predicate="ogt")
    tie = definition(trigger.operands[1], "and", "i1")
    if greater is None or tie is None or _number(greater.operands[1]) != 0.5:
        return None
    absolute = definition(greater.operands[0], "call", "float", callee="llvm.maximum.f32")
    equal = definition(tie.operands[0], "fcmp", "float", predicate="oeq")
    parity = definition(tie.operands[1], "icmp", dtype, predicate="ne")
    if (
        absolute is None
        or equal is None
        or parity is None
        or not same(equal.operands[0], greater.operands[0])
        or _number(equal.operands[1]) != 0.5
        or _number(parity.operands[1]) != 0
    ):
        return None
    frac, neg = absolute.operands
    diff = definition(frac, "fsub", "float")
    convert = definition(diff.operands[1], "sitofp", dtype, output_type="float") if diff else None
    negate = definition(neg, "fneg", "float")
    odd = definition(parity.operands[0], "and", dtype)
    if (
        diff is None
        or convert is None
        or negate is None
        or odd is None
        or not same(diff.operands[0], x)
        or not same(convert.operands[0], trunc)
        or not same(negate.operands[0], frac)
        or not same(odd.operands[0], trunc)
        or _number(odd.operands[1]) != 1
    ):
        return None
    return dict(
        result=result.result,
        clamped_input=x,
        raw_input=low.operands[0],
        dtype="f32",
        integer_dtype=dtype,
        bounds=[lower, upper],
        rounding="nearest_ties_even",
        source_nan="poison_from_fptosi",
    )


def _match_intrinsic(result: _Instruction, definitions: dict[str, _Instruction]) -> dict | None:
    """Read the actual bounded source conversion, independently of provenance."""
    dtype = result.output_type
    if (
        result.opcode != "fptosi"
        or result.dtype != "float"
        or not dtype.startswith("i")
        or not dtype[1:].isdigit()
        or not 1 <= int(dtype[1:]) <= 24
    ):
        return None

    def call(value, callee, arity):
        operation = definitions.get(_identity(value)) if value.startswith("%") else None
        return (
            operation
            if (
                operation is not None
                and operation.opcode == "call"
                and operation.dtype == "float"
                and operation.callee == callee
                and len(operation.operands) == arity
            )
            else None
        )

    rounded = call(result.operands[0], "llvm.roundeven.f32", 1)
    lower, upper = -(1 << (int(dtype[1:]) - 1)), (1 << (int(dtype[1:]) - 1)) - 1
    if rounded is None:
        return None
    current = rounded.operands[0]
    remaining = {"llvm.maximum.f32": lower, "llvm.minimum.f32": upper}
    for _ in range(2):
        clamp = definitions.get(_identity(current)) if current.startswith("%") else None
        if (
            clamp is None
            or clamp.opcode != "call"
            or clamp.dtype != "float"
            or clamp.callee not in remaining
            or len(clamp.operands) != 2
        ):
            return None
        bound = remaining.pop(clamp.callee)
        if _number(clamp.operands[1]) == bound:
            current = clamp.operands[0]
        elif _number(clamp.operands[0]) == bound:
            current = clamp.operands[1]
        else:
            return None
    return dict(
        result=result.result,
        clamped_input=rounded.operands[0],
        raw_input=current,
        dtype="f32",
        integer_dtype=dtype,
        bounds=[lower, upper],
        rounding="nearest_ties_even",
        source_nan="poison_from_fptosi",
        source_form="bounded_roundeven_conversion",
    )


def rewrite(
    text: str, *, host_isa: str | None = None, combine_clamp: bool = False, temporary_prefix: str = "merlin.rne"
) -> tuple[str, dict]:
    """Legalize only with explicit ``portable`` or ``rv64gc`` host ISA policy.

    Portable emission is the paired native oracle. RV64GC uses an explicit RNE
    conversion, optionally combining the proved clamp. NaN source poison may
    be refined; no output contract or FP exception-flag promise is introduced.
    ``temporary_prefix`` is printer identity only, for adapter compatibility.
    """
    receipt = dict(
        schema="merlin_late_bounded_rne_v1",
        source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        routes=[],
        host_isa=host_isa,
    )
    if type(combine_clamp) is not bool:
        raise TypeError("combine_clamp must be a bool")
    if host_isa is None:
        if combine_clamp:
            raise ValueError("combined clamp requires explicit rv64gc host ISA policy")
        return text, receipt
    if host_isa not in ("portable", "rv64gc"):
        raise ValueError("unsupported bounded-RNE host ISA policy")
    if combine_clamp and host_isa != "rv64gc":
        raise ValueError("combined clamp requires rv64gc host ISA policy")
    if (
        not isinstance(temporary_prefix, str)
        or not temporary_prefix
        or temporary_prefix[0].isdigit()
        or any(not (c.isascii() and (c.isalnum() or c in "._-$")) for c in temporary_prefix)
    ):
        raise ValueError("temporary_prefix must be an unquoted LLVM local identifier")
    try:
        tokens = _tokens(text)
        if any(
            t.text == "strictfp"
            or (t.text.startswith("@") and _identity(t.text).startswith("llvm.experimental.constrained."))
            for t in tokens
        ):
            receipt["refusal"] = "strict or constrained FP module"
            return text, receipt
        functions = _functions(tokens)
        original_locals = {_identity(t.text) for t in tokens if t.text.startswith("%")}
    except ValueError as exc:
        receipt["refusal"] = str(exc)
        return text, receipt
    edits: list[tuple[int, int, str]] = []
    for function in functions:
        definitions: dict[str, _Instruction] = {}
        # Include arguments (even unused ones) and quoted aliases of identifiers.
        # A conservative module-wide set also avoids named-type collisions.
        used = set(original_locals)
        for i in range(len(function)):
            if not function[i].text.startswith("%") or i + 1 == len(function) or function[i + 1].text != "=":
                continue
            op = _instruction(function, i)
            if op is not None:
                definitions[_identity(op.result)] = op
        serial = 0
        for op in sorted(definitions.values(), key=lambda x: x.start):
            proof = _match(op, definitions) or _match_intrinsic(op, definitions)
            if proof is None:
                continue
            indent = text[text.rfind("\n", 0, op.start) + 1 : op.start]
            if indent.strip():
                continue
            while f"{temporary_prefix}.{serial}" in used:
                serial += 1
            temporary = f"%{temporary_prefix}.{serial}"
            used.add(_identity(temporary))
            serial += 1
            dtype = proof["integer_dtype"]
            if host_isa == "portable":
                replacement = (
                    f"{temporary} = call float @llvm.roundeven.f32(float {proof['clamped_input']})\n"
                    f"{indent}{op.result} = fptosi float {temporary} to {dtype}"
                )
            elif combine_clamp:
                lower, upper = (f"{bound:.6e}" for bound in proof["bounds"])
                replacement = (
                    f'{temporary} = call i32 asm "fmax.s ft0, $1, $2\\0Afmin.s ft0, ft0, $3\\0A'
                    f'fcvt.w.s $0, ft0, rne", "=r,f,f,f,~{{ft0}}"'
                    f"(float {proof['raw_input']}, float {lower}, float {upper})\n"
                    f"{indent}{op.result} = trunc i32 {temporary} to {dtype}"
                )
            else:
                replacement = (
                    f'{temporary} = call i32 asm "fcvt.w.s $0, $1, rne", "=r,f"(float {proof["clamped_input"]})\n'
                    f"{indent}{op.result} = trunc i32 {temporary} to {dtype}"
                )
            edits.append((op.start, op.end, replacement))
            receipt["routes"].append(proof)
    rewritten = text
    for start, end, replacement in reversed(edits):
        rewritten = rewritten[:start] + replacement + rewritten[end:]
    declared = False
    for i, token in enumerate(tokens):
        if token.text == "declare":
            j = i + 1
            while j < len(tokens) and not tokens[j].text.startswith("@"):
                j += 1
            if j < len(tokens) and _identity(tokens[j].text) == "llvm.roundeven.f32":
                declared = True
    if host_isa == "portable" and edits and not declared:
        rewritten += "\ndeclare float @llvm.roundeven.f32(float)\n"
    receipt["rewritten_sha256"] = hashlib.sha256(rewritten.encode()).hexdigest()
    return rewritten, receipt


def merlin_host_llvm_transform(
    llvm_bin: str | Path, *, host_isa: str, combine_clamp: bool = False, temporary_prefix: str = "merlin.rne"
):
    """Verified pre-object callback; normal builder owns objects/link identity."""
    llvm_bin = Path(llvm_bin)

    def transform(source, work):
        source, work = Path(source), Path(work)
        work.mkdir(parents=True, exist_ok=True)
        assembler = llvm_bin / "llvm-as"
        subprocess.run([str(assembler), str(source), "-o", str(work / "source.bc")], check=True)
        original = source.read_text()
        target, proof = rewrite(
            original, host_isa=host_isa, combine_clamp=combine_clamp, temporary_prefix=temporary_prefix
        )
        if not proof["routes"]:
            raise ValueError("no proven bounded RNE chain: " + str(proof.get("refusal", "unrecognized source")))
        selected = work / "model.ll"
        selected.write_text(target)
        subprocess.run([str(assembler), str(selected), "-o", str(work / "model.bc")], check=True)
        native, _ = rewrite(original, host_isa="portable", temporary_prefix=temporary_prefix)
        (work / "model.native.ll").write_text(native)
        subprocess.run([str(assembler), str(work / "model.native.ll"), "-o", str(work / "model.native.bc")], check=True)
        proof.update(
            combine_clamp=combine_clamp,
            legalizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            llvm_as_sha256=hashlib.sha256(assembler.read_bytes()).hexdigest(),
            native_oracle_sha256=hashlib.sha256(native.encode()).hexdigest(),
            scope="pre-object host legalization; normal Merlin compilation and build identity",
        )
        (work / "receipt.json").write_text(json.dumps(proof, indent=2) + "\n")
        return selected

    return transform

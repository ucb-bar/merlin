"""AOT-quantize constant matrices that the W8A8 pass otherwise quantizes per inference.

This is a bundle rewrite, not a compiler arithmetic rewrite.  The stored f32 matrix becomes the
exact symmetric per-output-channel i8 representation used by :mod:`passes_quant_int`, its f32 scale
is appended to the bundle, and the sole contraction use is wrapped in
``quant_ext.dequantize_per_channel``.  The existing integer lowering recognizes that wrapper and
uses the stored i8 bytes directly.

The feature is deliberately structural and fail-closed.  Each selected value must be a stored,
unaliased f32 parameter with one contraction use and no other reader.  It may reach that use through
sole-use collapse/expand metadata only when those reshapes preserve the output-channel partition.
Anything less precise is skipped or refused rather than reported as an applied optimization.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FEATURE = "prequantize_constant_weights"
REWRITE_VERSION = "3"
_KEY_FILES = ("model.mlir", "weights.safetensors", "weights.safetensors.manifest.json")
_NP = {"I8": np.int8, "U8": np.uint8, "I16": np.int16, "I32": np.int32,
       "I64": np.int64, "F16": np.float16, "F32": np.float32, "F64": np.float64,
       "BF16": np.uint16}


class PrequantizeRefused(RuntimeError):
    """The requested bundle rewrite could not be proven semantics-preserving."""


@dataclass(frozen=True)
class WeightPlan:
    arg: int
    weight: str
    shape: tuple[int, ...]
    use_shape: tuple[int, ...]
    scale_axis: int
    layout_lines: tuple[int, ...]
    use_line: int
    use_value: str

    @property
    def scale_name(self) -> str:
        return f"{self.weight}.__merlin_int8_scale"


@dataclass(frozen=True)
class PrequantizePlan:
    bundle: str
    weights: tuple[WeightPlan, ...]
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.weights) and not self.problems


def _header(path: Path) -> tuple[dict[str, Any], int, Any]:
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        raw = json.loads(f.read(hlen))
    meta = raw.pop("__metadata__", None)
    return raw, 8 + hlen, meta


def read_tensors(path: Path | str) -> dict[str, np.ndarray]:
    """Read bundle tensors for rewrite verification and tests."""
    path = Path(path)
    header, base, _ = _header(path)
    out: dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        for name, spec in header.items():
            dtype = _NP.get(spec["dtype"])
            if dtype is None:
                raise PrequantizeRefused(f"{name}: unsupported safetensors dtype {spec['dtype']!r}")
            start, end = spec["data_offsets"]
            f.seek(base + start)
            out[name] = np.frombuffer(f.read(end - start), dtype=dtype).reshape(spec["shape"]).copy()
    return out


_SSA_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$-"
)


@dataclass(frozen=True)
class _InputGroup:
    operands: tuple[str, ...]
    types: tuple[str, ...]
    operands_start: int
    colon: int


@dataclass(frozen=True)
class _LayoutUse:
    result: str
    source: str
    source_type: str
    result_type: str


def _code(line: str) -> str:
    """Return MLIR code before a real ``//`` comment, retaining quoted text verbatim."""
    quote = False
    escaped = False
    for index, char in enumerate(line):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = False
        elif char == '"':
            quote = True
        elif char == "/" and index + 1 < len(line) and line[index + 1] == "/":
            return line[:index]
    return line


def _ssa_tokens(line: str) -> tuple[str, ...]:
    """Lex SSA identifiers, ignoring quoted attributes and comments.

    Weight eligibility depends on the exact use graph.  Treating a provenance string or comment
    containing ``%12`` as a reader is just as unsound as missing a real use, so this deliberately
    implements the small relevant part of MLIR's lexer instead of searching substrings.
    """
    text = _code(line)
    out: list[str] = []
    index = 0
    quote = False
    escaped = False
    while index < len(text):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = False
            index += 1
            continue
        if char == '"':
            quote = True
            index += 1
            continue
        if char != "%":
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in _SSA_CHARS:
            end += 1
        if end > index + 1:
            out.append(text[index:end])
        index = end
    return tuple(out)


def _definition(line: str) -> str | None:
    """Return the sole SSA result of a one-line assignment, if present."""
    text = _code(line).lstrip()
    if not text.startswith("%"):
        return None
    end = 1
    while end < len(text) and text[end] in _SSA_CHARS:
        end += 1
    if end == 1:
        return None
    cursor = end
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return text[:end] if cursor < len(text) and text[cursor] == "=" else None


def _matching(text: str, opening: int, left: str, right: str) -> int | None:
    """Find ``right`` matching ``left`` while respecting strings and nested delimiters."""
    if opening >= len(text) or text[opening] != left:
        return None
    depth = 0
    quote = False
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = False
            continue
        if char == '"':
            quote = True
        elif char == left:
            depth += 1
        elif char == right:
            depth -= 1
            if depth == 0:
                return index
    return None


def _top_level_separator(text: str, separator: str) -> int | None:
    """Locate a separator outside strings and all MLIR grouping delimiters."""
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    stack: list[str] = []
    quote = False
    escaped = False
    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = False
            continue
        if char == '"':
            quote = True
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char == separator and not stack:
            return index
    return None


def _split_top_level(text: str) -> tuple[str, ...]:
    """Split a comma list without splitting nested types, attributes, or expressions."""
    fields: list[str] = []
    start = 0
    while True:
        relative = _top_level_separator(text[start:], ",")
        if relative is None:
            fields.append(text[start:].strip())
            break
        end = start + relative
        fields.append(text[start:end].strip())
        start = end + 1
    return tuple(field for field in fields if field)


def _input_operands(line: str) -> _InputGroup | None:
    """Parse the typed operand list of a one-line linalg ``ins`` clause."""
    text = _code(line)
    marker = text.find("ins(")
    if marker < 0:
        return None
    opening = marker + len("ins")
    closing = _matching(text, opening, "(", ")")
    if closing is None or "outs(" not in text[closing + 1:]:
        return None
    content = text[opening + 1:closing]
    colon_relative = _top_level_separator(content, ":")
    if colon_relative is None:
        return None
    operands_text = content[:colon_relative]
    types_text = content[colon_relative + 1:]
    return _InputGroup(
        _split_top_level(operands_text),
        _split_top_level(types_text),
        opening + 1,
        opening + 1 + colon_relative,
    )


def _parse_layout(line: str) -> _LayoutUse | None:
    """Parse a one-result ``tensor.{collapse,expand}_shape`` use structurally."""
    result = _definition(line)
    if result is None:
        return None
    text = _code(line)
    equal = text.find("=", text.find(result) + len(result))
    if equal < 0:
        return None
    tail = text[equal + 1:].lstrip()
    names = ("tensor.collapse_shape", "tensor.expand_shape")
    op_name = next((name for name in names if tail.startswith(name)), None)
    if op_name is None:
        return None
    after_name = tail[len(op_name):].lstrip()
    source_tokens = _ssa_tokens(after_name)
    if not source_tokens or not after_name.startswith(source_tokens[0]):
        return None
    first = after_name.find("tensor<")
    if first < 0:
        return None
    first_end = _matching(after_name, first + len("tensor"), "<", ">")
    if first_end is None:
        return None
    cursor = first_end + 1
    while cursor < len(after_name) and after_name[cursor].isspace():
        cursor += 1
    if not after_name.startswith("into", cursor):
        return None
    second = after_name.find("tensor<", cursor + len("into"))
    if second < 0:
        return None
    second_end = _matching(after_name, second + len("tensor"), "<", ">")
    if second_end is None:
        return None
    return _LayoutUse(
        result=result,
        source=source_tokens[0],
        source_type=after_name[first + len("tensor<"):first_end],
        result_type=after_name[second + len("tensor<"):second_end],
    )


def _tensor_type(spelling: str) -> tuple[tuple[int, ...], str] | None:
    fields = spelling.split("x")
    if len(fields) < 2 or any(not field.isdigit() for field in fields[:-1]):
        return None
    return tuple(int(field) for field in fields[:-1]), fields[-1]


def _ssa_users(lines: list[str], token: str, signature_index: int) -> list[int]:
    users = []
    for index, line in enumerate(lines):
        if index == signature_index or _definition(line) == token:
            continue
        if token in _ssa_tokens(line):
            users.append(index)
    return users


def _trace_weight(lines: list[str], signature_index: int, arg: int,
                  shape: tuple[int, ...], name: str) -> tuple[WeightPlan | None, str | None]:
    """Trace a constant through sole-use shape metadata to its supported contraction use."""
    token = f"%{arg}"
    current_shape = shape
    layout_lines: list[int] = []
    for _ in range(9):
        users = _ssa_users(lines, token, signature_index)
        if not users:
            return None, None
        if len(users) != 1:
            return None, f"arg {arg}: target {name!r} has {len(users)} readers"
        use_line = users[0]
        line = lines[use_line]
        layout = _parse_layout(line)
        if layout is not None and layout.source == token:
            source = _tensor_type(layout.source_type)
            result = _tensor_type(layout.result_type)
            if (source is None or result is None or source != (current_shape, "f32")
                    or result[1] != "f32"
                    or int(np.prod(source[0])) != int(np.prod(result[0]))):
                return None, f"arg {arg}: target {name!r} has an invalid layout-only use"
            layout_lines.append(use_line)
            token = layout.result
            current_shape = result[0]
            continue

        parsed = _input_operands(line)
        if parsed is None:
            return None, None
        operands, types = parsed.operands, parsed.types
        positions = [i for i, operand in enumerate(operands) if operand == token]
        if len(positions) != 1 or len(types) != len(operands):
            return None, f"arg {arg}: target {name!r} has an ambiguous contraction operand"
        operand_index = positions[0]
        expected_type = "tensor<" + "x".join(map(str, current_shape)) + "xf32>"
        if types[operand_index] != expected_type:
            return None, f"arg {arg}: target {name!r} use type disagrees with its layout chain"
        if " = linalg.matmul" in line and len(current_shape) == 2 and operand_index in (0, 1):
            scale_axis = operand_index
        elif (" = linalg.generic" in line and operand_index == 1
              and 'prov.conv_path = "direct_contraction"' in line
              and len(current_shape) >= 2):
            scale_axis = 0
        else:
            return None, None
        # Collapse/expand preserve row-major element order, but not every reshape preserves the
        # per-output-channel partition. The admitted convolution chain is specifically
        # [O,...] -> [O,K]; a reshape to [O*x,K/x] would require a different scale tensor and is
        # refused. RHS matrices remain direct rank-2 values, where source/use axis 1 is identical.
        if scale_axis == 0 and shape[0] != current_shape[0]:
            return None, f"arg {arg}: target {name!r} layout merges its scale axis"
        if scale_axis == 1 and (len(shape) != 2 or layout_lines
                                or shape[1] != current_shape[1]):
            return None, f"arg {arg}: target {name!r} layout does not preserve its scale axis"
        return WeightPlan(arg, name, shape, current_shape, scale_axis,
                          tuple(layout_lines), use_line, token), None
    return None, f"arg {arg}: target {name!r} layout chain exceeds the eight-op safety bound"


def plan(src: Path | str) -> PrequantizePlan:
    src = Path(src)
    text = (src / "model.mlir").read_text(encoding="utf-8")
    lines = text.splitlines()
    signature_index = next((i for i, line in enumerate(lines)
                            if line.lstrip().startswith("func.func @forward(")), -1)
    manifest = json.loads((src / "weights.safetensors.manifest.json").read_text())
    header, _base, _meta = _header(src / "weights.safetensors")
    problems: list[str] = []
    weights: list[WeightPlan] = []

    ranges = {name: tuple(spec.get("data_offsets", ())) for name, spec in header.items()}
    payload_bytes = (src / "weights.safetensors").stat().st_size - _base
    # Real bundle manifests use numeric keys, but sorting must not throw before the fail-closed
    # checks below get a chance to explain a malformed target binding.
    ordered = sorted(
        manifest.items(),
        key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else item[0]),
    )
    for key, entry in ordered:
        # Eligibility comes from the ABI and use graph, never a model-specific name list.
        # Non-f32/non-matrix/non-parameter arguments cannot be the constant weight stream this
        # rewrite is designed to remove and are simply outside its domain.
        if not key.isdigit():
            continue
        shape = tuple(entry.get("shape", ()))
        if entry.get("kind") != "param" or entry.get("dtype") != "float32" or len(shape) < 2:
            continue
        arg = int(key)
        name = entry.get("weight")
        traced, trace_problem = _trace_weight(lines, signature_index, arg, shape, name)
        if trace_problem is not None:
            problems.append(trace_problem)
            continue
        if traced is None:
            continue
        spec = header.get(name)
        if spec is None:
            problems.append(f"arg {arg}: target {name!r} has no stored safetensors bytes")
            continue
        if spec.get("dtype") != "F32" or tuple(spec.get("shape", ())) != shape:
            problems.append(
                f"arg {arg}: target {name!r} manifest/header disagree or are not F32: "
                f"manifest={list(shape)}, header={spec}")
            continue
        if name + ".__merlin_int8_scale" in header:
            problems.append(f"arg {arg}: target {name!r} is already prequantized")
            continue
        offsets = tuple(spec.get("data_offsets", ()))
        expected_bytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
        if (len(offsets) != 2 or not all(isinstance(x, int) for x in offsets)
                or offsets[0] < 0 or offsets[0] > offsets[1] or offsets[1] > payload_bytes
                or offsets[1] - offsets[0] != expected_bytes):
            problems.append(
                f"arg {arg}: target {name!r} has invalid safetensors byte range {offsets}")
            continue
        readers = [k for k, other in manifest.items()
                   if other.get("weight") == name and k != key]
        if readers:
            problems.append(f"arg {arg}: target {name!r} is also bound by args {readers}")
            continue
        mine = ranges[name]
        overlap = [other for other, theirs in ranges.items()
                   if other != name and len(theirs) == 2
                   and theirs[0] < mine[1] and mine[0] < theirs[1]]
        if overlap:
            problems.append(f"arg {arg}: target {name!r} shares bytes with {overlap}")
            continue
        signature = f"%{arg}: tensor<{'x'.join(map(str, shape))}xf32>"
        signature_fields = _split_top_level(
            lines[signature_index].split("@forward(", 1)[1].rsplit(") ->", 1)[0]
        ) if signature_index >= 0 and ") ->" in lines[signature_index] else ()
        if sum(field == signature for field in signature_fields) != 1:
            problems.append(f"arg {arg}: expected exactly one forward signature token {signature!r}")
            continue
        weights.append(traced)
    return PrequantizePlan(src.name, tuple(weights), tuple(problems))


def _quantize(weight: np.ndarray, name: str, scale_axis: int) -> tuple[np.ndarray, np.ndarray]:
    # This is the current passes_quant_int formula, in the same f32 precision:
    # scale=max(abs(W), reduction axis)/127; q=i8(clamp(roundeven(W/scale), -127, 127)).
    reduce_axes = tuple(axis for axis in range(weight.ndim) if axis != scale_axis)
    scale = np.max(np.abs(weight), axis=reduce_axes) / np.float32(127.0)
    if not np.all(np.isfinite(scale)) or np.any(scale == 0):
        raise PrequantizeRefused(
            f"{name}: non-finite or zero output-channel scale has target-dependent fptosi semantics")
    broadcast = [1] * weight.ndim
    broadcast[scale_axis] = weight.shape[scale_axis]
    quant = np.clip(np.rint(weight / scale.reshape(broadcast)), -127, 127).astype(np.int8)
    return np.ascontiguousarray(quant), np.ascontiguousarray(scale, dtype=np.float32)


def _rewrite_weights(src: Path, dst: Path, weights: tuple[WeightPlan, ...]) -> None:
    path = src / "weights.safetensors"
    header, base, meta = _header(path)
    wanted = {item.weight: item for item in weights}
    replacements: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with open(path, "rb") as f:
        for name, item in wanted.items():
            spec = header[name]
            start, end = spec["data_offsets"]
            f.seek(base + start)
            value = np.frombuffer(f.read(end - start), np.float32).reshape(item.shape)
            replacements[name] = _quantize(value, name, item.scale_axis)

    new_header: dict[str, Any] = {}
    chunks: list[tuple[str, bytes, str, list[int]]] = []
    with open(path, "rb") as f:
        for name, spec in header.items():
            if name in replacements:
                value = replacements[name][0]
                chunks.append((name, value.tobytes(), "I8", list(value.shape)))
            else:
                start, end = spec["data_offsets"]
                f.seek(base + start)
                chunks.append((name, f.read(end - start), spec["dtype"], list(spec["shape"])))
    for item in weights:
        scale = replacements[item.weight][1]
        chunks.append((item.scale_name, scale.tobytes(), "F32", list(scale.shape)))

    offset = 0
    for name, data, dtype, shape in chunks:
        new_header[name] = {"dtype": dtype, "shape": shape,
                            "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    if meta is not None:
        new_header["__metadata__"] = meta
    encoded = json.dumps(new_header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    with open(dst / "weights.safetensors", "wb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for _name, data, _dtype, _shape in chunks:
            out.write(data)


def _rewrite_ir(text: str, weights: tuple[WeightPlan, ...]) -> str:
    lines = text.splitlines()
    by_line = {item.use_line: item for item in weights}
    layout_lines = {line for item in weights for line in item.layout_lines}

    signature_index = next((i for i, line in enumerate(lines)
                            if line.lstrip().startswith("func.func @forward(")), None)
    if signature_index is None or ") ->" not in lines[signature_index]:
        raise PrequantizeRefused("@forward must have a one-line ranked-tensor signature")
    signature = lines[signature_index]
    for item in weights:
        shape = "x".join(map(str, item.shape))
        old = f"%{item.arg}: tensor<{shape}xf32>"
        new = f"%{item.arg}: tensor<{shape}xi8>"
        signature = signature.replace(old, new)
    extra = ", ".join(
        f"%merlin_wq_scale{item.arg}: tensor<{item.shape[item.scale_axis]}xf32>"
        for item in weights)
    head, tail = signature.rsplit(") ->", 1)
    lines[signature_index] = f"{head}, {extra}) ->{tail}"

    out: list[str] = []
    for index, line in enumerate(lines):
        if index in layout_lines:
            out.append(line.replace("xf32>", "xi8>"))
            continue
        item = by_line.get(index)
        if item is None:
            out.append(line)
            continue
        indent = line[:len(line) - len(line.lstrip())]
        arg = item.arg
        shape = "x".join(map(str, item.use_shape))
        scale_len = item.shape[item.scale_axis]
        typ = f"tensor<{shape}"
        out.extend([
            f"{indent}%merlin_wq_zero{arg} = arith.constant 0 : i32",
            f"{indent}%merlin_wq_zps{arg} = tensor.splat %merlin_wq_zero{arg} : "
            f"tensor<{scale_len}xi32>",
            f'{indent}%merlin_wq_deq{arg} = "quant_ext.dequantize_per_channel"('
            f"{item.use_value}, %merlin_wq_scale{arg}, %merlin_wq_zps{arg}) "
            f'<{{axis = {item.scale_axis} : i64, input_dtype = "i8"}}> '
            f"{{prov.op = \"dequantize\", prov.family = \"quantize\"}} : "
            f"({typ}xi8>, tensor<{scale_len}xf32>, tensor<{scale_len}xi32>) -> {typ}xf32>",
        ])
        parsed = _input_operands(line)
        if parsed is None:
            raise PrequantizeRefused(f"arg {arg}: contraction use disappeared during rewrite")
        rewritten = list(parsed.operands)
        positions = [i for i, operand in enumerate(rewritten) if operand == item.use_value]
        if len(positions) != 1:
            raise PrequantizeRefused(f"arg {arg}: contraction operand became ambiguous during rewrite")
        rewritten[positions[0]] = f"%merlin_wq_deq{arg}"
        out.append(
            line[:parsed.operands_start]
            + ", ".join(rewritten)
            + line[parsed.colon:]
        )
    return "\n".join(out) + "\n"


def cache_key(src: Path | str) -> str:
    src = Path(src).resolve()
    parts = [FEATURE, REWRITE_VERSION, str(src)]
    for name in _KEY_FILES:
        stat = (src / name).stat()
        parts.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def prequantized_bundle(src: Path | str, *, cache_root: Path | str | None = None) -> tuple[Path, dict]:
    from ..baselines.bundle_rewrite import (REWRITES_FILE, RewriteRecord, _carry_sidecars,
                                             read_rewrites, record_rewrite, retarget_weights_file)
    from ..common.artifacts import cache_dir

    src = Path(src).resolve()
    root = Path(cache_root) if cache_root is not None else cache_dir("weight_prequant")
    root.mkdir(parents=True, exist_ok=True)
    dst = root / f"{src.name}__{cache_key(src)}"
    if dst.is_dir():
        records = [record for record in read_rewrites(dst)
                   if record.name == "prequantize_constant_weights"]
        if records:
            return dst, {"cached": True, **records[-1].effect}
        shutil.rmtree(dst)

    candidate = plan(src)
    if not candidate.ok:
        reason = "; ".join(candidate.problems) if candidate.problems else "no eligible matrices"
        raise PrequantizeRefused(f"{src.name}: {reason}")
    tmp = root / f".tmp-{src.name}-{os.getpid()}-{cache_key(src)}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        _rewrite_weights(src, tmp, candidate.weights)
        manifest = json.loads((src / "weights.safetensors.manifest.json").read_text())
        next_arg = max(int(key) for key in manifest if key.isdigit()) + 1
        for item in candidate.weights:
            manifest[str(item.arg)]["dtype"] = "int8"
            manifest[str(next_arg)] = {
                "kind": "param", "weight": item.scale_name, "dtype": "float32",
                "shape": [item.shape[item.scale_axis]], "generated_by": FEATURE,
            }
            next_arg += 1
        (tmp / "weights.safetensors.manifest.json").write_text(json.dumps(manifest, indent=2))
        text = _rewrite_ir((src / "model.mlir").read_text(), candidate.weights)
        text, retargeted = retarget_weights_file(text, (tmp / "weights.safetensors").resolve())
        (tmp / "model.mlir").write_text(text)
        writes = {"model.mlir", "weights.safetensors", "weights.safetensors.manifest.json",
                  REWRITES_FILE}
        skipped = _carry_sidecars(src, tmp, writes)
        if (src / REWRITES_FILE).is_file():
            shutil.copy2(src / REWRITES_FILE, tmp / REWRITES_FILE)
        record = RewriteRecord(
            name="prequantize_constant_weights", source_bundle=src.name,
            soundness=("each selected f32 tensor is an unaliased stored parameter whose sole reader "
                       "is a supported contraction, optionally through output-channel-preserving "
                       "collapse/expand metadata; stored q/scale use the compiler's existing "
                       "symmetric per-output-channel round-even formula"),
            effect={"weights_prequantized": len(candidate.weights),
                    "weight_args": [item.arg for item in candidate.weights],
                    "weight_names": [item.weight for item in candidate.weights],
                    "weights_file_retargeted": retargeted,
                    "sidecars_not_carried": skipped},
            caveats=([f"stale, NOT carried over from the source bundle: {skipped}"]
                     if skipped else []),
        )
        record_rewrite(tmp, record)
        try:
            os.replace(tmp, dst)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            if not dst.is_dir():
                raise
            return dst, {"cached": True, **record.effect}
        text, changed = retarget_weights_file((dst / "model.mlir").read_text(),
                                              dst / "weights.safetensors")
        if changed:
            (dst / "model.mlir").write_text(text)
        return dst, {"cached": False, **record.effect}
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _feature():
    from .impr_features import ImprFeature
    return ImprFeature(
        name=FEATURE, action_class="PASS",
        description=("AOT-quantize eligible constant contraction weights with the exact W8A8 formula, "
                     "store i8+scale in a cached rewritten bundle, and let the existing static-weight "
                     "fast path remove their per-inference scale search/broadcast/quantization."))


def ensure_registered() -> str:
    from .impr_features import known, register
    if FEATURE not in known():
        register(_feature())
    return FEATURE

"""Load/validate a Merlin command buffer and materialize its declared tensors.

The execution-oriented command buffer carries a ``tensors`` table (name -> shape/dtype/role)
in addition to the opcode list. Input/weight/bias tensors are materialized deterministically
(see :func:`Tensor.deterministic`) so a run is reproducible without external input files; an
explicit inputs mapping can override them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .tensor import Tensor


def load_command_buffer(path: str | Path) -> dict[str, Any]:
    """Load a command-buffer JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


REQUIRED_KEYS = ("abi_version", "target", "commands")


def validate_command_buffer(cb: dict[str, Any]) -> list[str]:
    """Return a list of problems (empty == valid).

    Beyond the required keys, this catches the one structural error the JSON schema cannot express: an
    operand slot holds the NAME of a buffer, and the schema types it as a bare string, so a value that
    names nothing at all still validates. A submission that emitted commands referencing shape strings
    (``dst: "16x16"``) with no ``tensors`` declared passed schema validation, was told it was valid, and
    was then rejected downstream for a constraint the contract never stated -- so it spent its session
    guessing spellings instead of writing a compiler.

    The check is deliberately the weakest one that is certainly true: commands that reference operands
    need SOMETHING to reference. Names produced by earlier commands (an accumulator, a committed
    intermediate) are legitimately absent from ``tensors``, so per-name resolution is NOT asserted here --
    only that a computing command buffer declares at least one buffer to compute over."""
    problems: list[str] = []
    for k in REQUIRED_KEYS:
        if k not in cb:
            problems.append(f"missing key '{k}'")
    for i, cmd in enumerate(cb.get("commands", [])):
        if "opcode" not in cmd:
            problems.append(f"command {i} missing 'opcode'")
    cmds = cb.get("commands") or []
    referenced = sorted({str(v) for c in cmds for v in (c.get("operands") or {}).values()
                         if isinstance(v, str) and v})
    if referenced and not (cb.get("tensors") or {}) and not cb.get("declined"):
        problems.append(
            f"commands reference operand name(s) {referenced[:6]}"
            f"{' ...' if len(referenced) > 6 else ''} but the command buffer declares no 'tensors'. "
            f"An operand slot holds the NAME of a tensor declared in 'tensors' (e.g. \"Y0\"), not a "
            f"shape, a type, or a dimension list")
    return problems


def conv_out_dims(H: int, W: int, kh: int, kw: int, stride, padding, dilation) -> tuple[int, int]:
    sh, sw = stride
    pt, pl, pb, pr = padding
    dh, dw = dilation
    Ho = (H + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    Wo = (W + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
    return Ho, Wo


def conv_im2col(ifm: Tensor, *, kh: int, kw: int, ci: int, stride, padding, dilation,
                layout: str = "nhwc") -> Tensor:
    """Build the [N*Ho*Wo, Kh*Kw*Ci] im2col matrix from an NHWC activation (zero-pad OOB taps).

    This is the single source of truth shared by the runner harness/reference/simulate (via
    :func:`materialize_inputs`) and the capsule golden. Column order = (kh, kw, ci), matching the
    weight packing [Kh*Kw*Ci, Co].
    """
    if layout != "nhwc":
        raise ValueError(f"conv_im2col layout {layout!r} unsupported (nhwc only)")
    N, H, W, C = ifm.shape
    if C != ci:
        raise ValueError(f"conv_im2col channel mismatch: ifm C={C} != ci={ci}")
    sh, sw = stride
    pt, pl, _, _ = padding
    dh, dw = dilation
    Ho, Wo = conv_out_dims(H, W, kh, kw, stride, padding, dilation)
    a = ifm.data
    rows: list[int] = []

    def at(n, y, x, c):
        if 0 <= y < H and 0 <= x < W:
            return a[((n * H + y) * W + x) * C + c]
        return 0

    for n in range(N):
        for oy in range(Ho):
            for ox in range(Wo):
                by, bx = oy * sh - pt, ox * sw - pl
                for ky in range(kh):
                    for kx in range(kw):
                        for c in range(ci):
                            rows.append(at(n, by + ky * dh, bx + kx * dw, c))
    return Tensor((N * Ho * Wo, kh * kw * ci), rows, ifm.dtype)


def materialize_inputs(cb: dict[str, Any], inputs: dict[str, Any] | None = None) -> dict[str, Tensor]:
    """Create the leaf input tensors declared in the command buffer's ``tensors`` table.

    A tensor is a *leaf* (materialized here) when its role is input/weight/bias, i.e. it is
    not produced by a command. ``inputs`` may supply explicit nested-list data per name.

    If the command buffer declares ``params.im2col_recipes``, each derived activation is built by
    gathering conv windows from its source leaf (so a compiler-lowered conv2d's im2col activation is
    materialized identically for the reference, the simulator, and the device harness).
    """
    inputs = inputs or {}
    produced = set()
    for cmd in cb.get("commands", []):
        ops = cmd.get("operands", {})
        for key in ("dst",):
            if key in ops:
                produced.add(ops[key])
    env: dict[str, Tensor] = {}
    for name, spec in cb.get("tensors", {}).items():
        if name in produced:
            continue
        shape = tuple(spec["shape"])
        dtype = spec.get("dtype", "i8")
        if name in inputs:
            flat = _flatten(inputs[name])
            env[name] = Tensor(shape, flat, dtype)
        else:
            env[name] = Tensor.deterministic(name, shape, dtype)
    # additive: overwrite derived im2col activations from their source leaf
    for r in cb.get("params", {}).get("im2col_recipes", []):
        src = env[r["source"]]
        env[r["target"]] = conv_im2col(
            src, kh=int(r["kh"]), kw=int(r["kw"]), ci=int(r["ci"]),
            stride=tuple(r.get("stride", [1, 1])), padding=tuple(r.get("padding", [0, 0, 0, 0])),
            dilation=tuple(r.get("dilation", [1, 1])), layout=r.get("layout", "nhwc"))
    return env


def _flatten(nested) -> list[int]:
    out: list[int] = []
    if nested and isinstance(nested[0], list):
        for row in nested:
            out.extend(row)
    else:
        out.extend(nested)
    return out

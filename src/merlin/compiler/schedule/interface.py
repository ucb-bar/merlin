"""Adapt an explicit interface command program into checked contraction schedules."""

from __future__ import annotations

from typing import (
    Any,
)

from .block import (
    schedule_contraction,
    schedule_convolution,
)
from .model import (
    BlockSchedule,
    BlockScheduleError,
    Contraction,
    ConvContraction,
    Geometry,
    Knobs,
)


def schedule_interface_program(
    tensors: dict[str, Any], commands: list[dict[str, Any]], geometry: Geometry, knobs: Knobs | None = None
) -> list[BlockSchedule]:
    """Schedule every resident matmul an interface program commits, and every convolution, in program
    order.

    ``tensors`` maps a name to that tensor's spec (anything with ``shape``, or a mapping carrying
    ``"shape"``); ``commands`` is the program's command list. Any other command -- a convolution, a
    movement, an epilogue this pass does not model -- is a refusal, so a caller never receives a
    schedule that silently dropped work.
    """
    resident_of: dict[str, str] = {}
    pending: dict[str, tuple[str, str]] = {}
    schedules: list[BlockSchedule] = []
    for command in commands:
        opcode = command.get("opcode")
        operands = command.get("operands", {})
        if opcode == "RES_PACK":
            resident_of[operands["dst"]] = operands["src"]
        elif opcode == "MATMUL_RESIDENT":
            rhs = operands["rhs"]
            if rhs not in resident_of:
                raise BlockScheduleError(f"{rhs!r} is used as a resident operand before it is packed")
            pending[operands["dst"]] = (operands["lhs"], resident_of[rhs])
        elif opcode == "COMMIT":
            src = operands["src"]
            if src not in pending:
                raise BlockScheduleError(f"commit of {src!r}, which no matmul produced")
            lhs_name, weight_name = pending.pop(src)
            attributes = command.get("attributes", {}) or {}
            epilogue = tuple(attributes.get("epilogue", ()) or ())
            unsupported = [e for e in epilogue if e not in ("relu", "acc_scale")]
            if unsupported:
                raise BlockScheduleError(
                    f"epilogue {unsupported} changes which accumulator block a store drains; this pass "
                    "schedules block moves only and will not guess that mapping"
                )
            m, k = _shape(tensors, lhs_name)
            wk, n = _shape(tensors, weight_name)
            if wk != k:
                raise BlockScheduleError(f"contraction mismatch: {lhs_name} is {m}x{k}, {weight_name} is {wk}x{n}")
            schedules.append(
                schedule_contraction(Contraction(m, k, n, lhs_name, weight_name, operands["dst"]), geometry, knobs)
            )
        elif opcode == "CONV2D":
            schedules.append(_conv_command(tensors, command, resident_of, geometry, knobs))
        elif opcode != "EVICT":
            raise BlockScheduleError(
                f"{opcode!r} is neither a resident matmul nor a convolution; this pass "
                "schedules those only, and refuses rather than dropping the command"
            )
    return schedules


def _conv_command(
    tensors: dict[str, Any],
    command: dict[str, Any],
    resident_of: dict[str, str],
    geometry: Geometry,
    knobs: Knobs | None,
) -> BlockSchedule:
    """One ``CONV2D`` command as a :class:`ConvContraction`, refusing what the nest cannot express."""
    operands, attributes = command.get("operands", {}) or {}, command.get("attributes", {}) or {}
    missing = [key for key in ("ifm", "weight", "dst") if key not in operands]
    missing += [key for key in ("kernel",) if key not in attributes]
    if missing:
        raise BlockScheduleError(f"a CONV2D command missing {missing}; refusing rather than guessing them")
    epilogue = tuple(attributes.get("epilogue", ()) or ())
    unsupported = [e for e in epilogue if e not in ("relu", "acc_scale")]
    if unsupported:
        # Pooling drains the accumulator N-major and defers every drain past the nest: a different
        # accumulator LAYOUT, not an ordering choice, so it is refused rather than approximated.
        raise BlockScheduleError(
            f"epilogue {unsupported} changes which accumulator block a store drains; "
            "this pass schedules block moves only and will not guess that mapping"
        )
    if attributes.get("layout", "nhwc") != "nhwc":
        raise BlockScheduleError(f"conv layout {attributes.get('layout')!r}; this pass schedules NHWC")
    weight = operands["weight"]
    if weight not in resident_of:
        raise BlockScheduleError(f"{weight!r} is used as a resident operand before it is packed")
    ifm = _shape_nd(tensors, operands["ifm"], 4)
    kh, kw, kci, co = (int(v) for v in attributes["kernel"])
    batch, in_h, in_w, ci = ifm
    if kci != ci:
        raise BlockScheduleError(f"kernel expects {kci} input channels, the input has {ci}")
    packed = _shape_nd(tensors, resident_of[weight], 2)
    if packed != (kh * kw * ci, co):
        raise BlockScheduleError(
            f"conv weight {resident_of[weight]!r} is {packed}, not the pre-im2col {(kh * kw * ci, co)}"
        )
    conv = ConvContraction(
        batch,
        in_h,
        in_w,
        ci,
        kh,
        kw,
        co,
        tuple(int(v) for v in attributes.get("stride", (1, 1))),
        tuple(int(v) for v in attributes.get("padding", (0, 0, 0, 0))),
        tuple(int(v) for v in attributes.get("dilation", (1, 1))),
        operands["ifm"],
        resident_of[weight],
        operands["dst"],
    )
    return schedule_convolution(conv, geometry, knobs)


def _shape_nd(tensors: dict[str, Any], name: str, rank: int) -> tuple[int, ...]:
    spec = tensors.get(name)
    if spec is None:
        raise BlockScheduleError(f"the program names no tensor {name!r}")
    shape = spec["shape"] if isinstance(spec, dict) else getattr(spec, "shape", None)
    if shape is None or len(shape) != rank:
        raise BlockScheduleError(f"{name!r} has shape {shape!r}; expected rank {rank}")
    return tuple(int(v) for v in shape)


def _shape(tensors: dict[str, Any], name: str) -> tuple[int, int]:
    spec = tensors.get(name)
    if spec is None:
        raise BlockScheduleError(f"the program names no tensor {name!r}")
    shape = spec["shape"] if isinstance(spec, dict) else getattr(spec, "shape", None)
    if shape is None or len(shape) != 2:
        raise BlockScheduleError(f"{name!r} has shape {shape!r}; this pass schedules 2-D operands")
    return int(shape[0]), int(shape[1])

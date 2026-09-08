"""Normalise a workload into (a) the ABI command buffer and (b) a target-independent kernel plan.

This is the semantic half of `--convert-iface-to-gemmini`.  Everything downstream — the tile
schedule, the gemmini-dialect module, the LLVM artifact — reads the plan, so the interface is
interpreted exactly once and every extent comes from the capsule that was handed in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..frontend.reader import TensorDecl, Workload
from ..tables import rtl_facts as _F
from .native_conv_selector import select_native_loop_conv
from .source_conv_plan import Convolution

DTYPE_BYTES = {"i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8,
               "f16": 2, "bf16": 2, "f32": 4}

#: opcodes the ABI treats as a whole-op command for the kernel argument order
WHOLE_OP_OPCODES = ("ATTENTION_QK", "ATTENTION_PV", "CONV2D")


class LoweringDeclined(Exception):
    """This backend cannot lower the workload; the caller emits a `declined` command buffer."""

    def __init__(self, reason: str, *, op: str = "", shape: list[int] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.op = op
        self.shape = list(shape or [])


#: Element budget for the DERIVED block-diagonal activation a batched contraction expands into.
#: The expansion costs a factor of the batch count in BOTH dimensions, so a large batch is refused
#: with the count stated rather than turned into a gather nothing can hold.
BLOCK_DIAGONAL_ELEM_BUDGET = 4_000_000


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


#: mesh edge, from the RTL facts; the harness lays a buffer out row-major with each row padded
#: up to a whole tile ("edge tiles zero-padded to a multiple of 16 (DIM)" in the kernel ABI).
TILE = _F.DIM


def row_pitch(cols: int) -> int:
    """Elements between the starts of two consecutive rows of a DRAM-resident tensor."""
    return -(-int(cols) // TILE) * TILE


@dataclass
class Buffer:
    """A DRAM buffer the kernel touches."""

    name: str
    shape: list[int]
    dtype: str
    role: str                 # input | weight | bias | output | scratch
    storage_encoding: dict[str, Any] | None = None

    @property
    def pitch(self) -> int:
        if self.storage_encoding is not None:
            encoding = self.storage_encoding
            declared = (encoding.get("logical_shape")
                        if encoding.get("schema") == "permuted_axes_storage_v1"
                        else encoding["physical_shape"])
            if declared != self.shape or encoding["dtype"] != self.dtype:
                raise LoweringDeclined("buffer type changed after storage encoding", op="storage_encoding")
            physical = encoding["physical_shape"]
            return (encoding["strides_elements"][-2] if len(physical) > 1
                    else encoding["storage_elements"])
        return row_pitch(self.shape[-1]) if self.shape else 0

    @property
    def elems(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    @property
    def nbytes(self) -> int:
        if self.storage_encoding is not None:
            _ = self.pitch  # Check that allocation and logical descriptor have not diverged.
            return self.storage_encoding["storage_elements"] * DTYPE_BYTES[self.dtype]
        rows = 1
        for d in self.shape[:-1]:
            rows *= int(d)
        return rows * self.pitch * DTYPE_BYTES[self.dtype]


@dataclass
class Epilogue:
    """The commit epilogue, normalised."""

    stages: list[str] = field(default_factory=list)
    output_dtype: str = "i32"
    acc_scale: float = 1.0
    requant_shift: int | None = None
    bias: str | None = None
    pool_in_dims: list[int] | None = None
    pool_size: list[int] | None = None
    pool_stride: list[int] | None = None
    pool_padding: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    pool_pad_value: int | None = None
    integer_output_policy: str = "saturate"

    @property
    def has_pool(self) -> bool:
        return "maxpool" in self.stages


@dataclass
class Contraction:
    """One `[M, K] x [K, N] -> [M, N]` integer contraction plus its readout."""

    lhs: str
    rhs: str
    dst: str
    m: int
    k: int
    n: int
    lhs_row_elems: int          # DRAM row pitch of the lhs, in elements
    rhs_row_elems: int
    epilogue: Epilogue = field(default_factory=Epilogue)
    #: extra rows/cols offsets applied to the destination (batched matmul stacking)
    dst_row_offset: int = 0
    dst_row_elems: int = 0
    accumulator_temporary: str | None = None


@dataclass
class Movement:
    """An identity load/store round trip through the accelerator."""

    src: str
    dst: str
    rows: int
    cols: int
    src_dtype: str
    dst_dtype: str


@dataclass
class Transpose:
    """A compiler-generated 2-D transpose staged in DRAM (feeds `q @ transpose(k)`)."""

    src: str
    dst: str
    rows: int
    cols: int
    dtype: str


@dataclass
class HostBiasAdd:
    """The standalone `bias_add` whole-op: dst[i, j] = src[i, j] + bias[j]."""

    src: str
    bias: str
    dst: str
    rows: int
    cols: int
    dtype: str


@dataclass
class ResAdd:
    """Exact identity-scale saturating i8 residual addition."""

    lhs: str
    rhs: str
    dst: str
    rows: int
    cols: int
    relu: bool = False


@dataclass
class Plan:
    target: str
    buffers: dict[str, Buffer]
    tasks: list[Any]
    command_buffer: dict[str, Any]
    kernel_args: list[str]


# --------------------------------------------------------------------------------------------


def _epilogue_from(attrs: dict[str, Any]) -> Epilogue:
    stages = [str(s) for s in (attrs.get("epilogue") or [])]
    e = Epilogue(
        stages=stages,
        output_dtype=str(attrs.get("output_dtype", "i32")),
        acc_scale=float(attrs.get("acc_scale", 1.0)),
        bias=attrs.get("bias"),
    )
    if "requant_shift" in attrs:
        e.requant_shift = int(attrs["requant_shift"])
    for key in ("pool_in_dims", "pool_size", "pool_stride"):
        if key in attrs:
            setattr(e, key, [int(v) for v in attrs[key]])
    if "pool_padding" in attrs:
        e.pool_padding = [int(v) for v in attrs["pool_padding"]]
    if "pool_pad_value" in attrs:
        e.pool_pad_value = int(attrs["pool_pad_value"])
    if e.has_pool:
        missing = [k for k in ("pool_in_dims", "pool_size", "pool_stride")
                   if getattr(e, k) is None]
        if missing:
            raise LoweringDeclined(
                f"maxpool epilogue declares no {', '.join(missing)}; the ABI gives it no default",
                op="maxpool")
        if any(p for p in e.pool_padding) and e.pool_pad_value is None:
            raise LoweringDeclined(
                "maxpool epilogue has nonzero pool_padding but declares no pool_pad_value",
                op="maxpool")
    for s in stages:
        if s == "requant" and e.requant_shift is None:
            raise LoweringDeclined(
                "requant epilogue stage declares no requant_shift", op="requant")
    return e


def conv_out_dims(h: int, w: int, kh: int, kw: int, stride, padding, dilation):
    sh, sw = int(stride[0]), int(stride[1])
    pt, pl, pb, pr = (int(v) for v in padding)
    dh, dw = int(dilation[0]), int(dilation[1])
    ho = (h + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    wo = (w + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
    return ho, wo


def pool_out_dims(h: int, w: int, size, stride, padding):
    ph, pw = int(size[0]), int(size[1])
    sh, sw = int(stride[0]), int(stride[1])
    pt, pl, pb, pr = (int(v) for v in padding)
    return (h + pt + pb - ph) // sh + 1, (w + pl + pr - pw) // sw + 1


def epilogue_out_shape(m: int, n: int, e: Epilogue) -> list[int]:
    if not e.has_pool:
        return [m, n]
    ih, iw = e.pool_in_dims
    if ih < 1 or iw < 1 or m % (ih * iw):
        # `pool_in_dims` is the spatial extent the committed ROWS unflatten to, so the row count
        # must be a whole number of [H, W] planes.  Reconciling a remainder would pool across a
        # batch boundary and read as an arithmetic error rather than as a geometry the buffer
        # cannot express, so it is REFUSED with both numbers stated.
        raise LoweringDeclined(
            f"maxpool pool_in_dims {list(e.pool_in_dims)} does not divide the {m} committed rows "
            f"({m} % {ih * iw} != 0), so the rows do not unflatten to whole [H, W] planes",
            op="commit", shape=[m, n])
    ho, wo = pool_out_dims(ih, iw, e.pool_size, e.pool_stride, e.pool_padding)
    if ho < 1 or wo < 1:
        raise LoweringDeclined(
            f"maxpool geometry {list(e.pool_in_dims)} window {list(e.pool_size)} stride "
            f"{list(e.pool_stride)} pad {list(e.pool_padding)} has an empty {ho}x{wo} output",
            op="commit", shape=[m, n])
    batch = m // (ih * iw)
    return [batch * ho * wo, n]


# --------------------------------------------------------------------------------------------


class Builder:
    """Builds the command buffer and kernel plan from a `merlin_iface` workload."""

    def __init__(self, wl: Workload):
        self.wl = wl
        self.buffers: dict[str, Buffer] = {}
        self.tasks: list[Any] = []
        self.commands: list[dict[str, Any]] = []
        self.params: dict[str, Any] = {}
        self.tensor_order: list[str] = []
        self._scratch = 0
        #: resident handle -> the leaf weight name it packs
        self.residents: dict[str, str] = {}
        #: accumulator handle -> the matmul node that produced it
        self.acc_of: dict[str, dict[str, Any]] = {}
        #: leaf tensors re-declared under a different VIEW of the same bytes (batched contraction),
        #: and the DECLARED shape each one was interpreted under before the view was taken
        self._reshaped: dict[str, list[int]] = {}
        self._as_declared: dict[str, list[int]] = {}

    # -- buffer bookkeeping ------------------------------------------------------------------
    def declare(self, name: str, shape: list[int], dtype: str, role: str) -> Buffer:
        buf = Buffer(name, [int(d) for d in shape], dtype, role)
        self.buffers[name] = buf
        if role != "scratch" and name not in self.tensor_order:
            self.tensor_order.append(name)
        return buf

    def _reshape(self, buf: Buffer, shape: list[int]) -> None:
        """Re-declare a leaf under a different VIEW of the same row-major bytes."""
        shape = [int(d) for d in shape]
        if buf.shape == shape:
            return
        prior = self._reshaped.get(buf.name)
        if prior is not None and prior != shape:
            raise LoweringDeclined(
                f"tensor {buf.name!r} is asked for two different views, {prior} and {shape}; a "
                f"declaration carries one shape and reshaping it under the earlier reader would "
                f"change what that reader contracts",
                op="matmul_batched", shape=shape)
        for cmd in self.commands:
            if buf.name in (cmd.get("operands") or {}).values():
                raise LoweringDeclined(
                    f"tensor {buf.name!r} was already read as {buf.shape} by an earlier "
                    f"{cmd.get('opcode')} command, so it cannot be re-declared as {shape}",
                    op="matmul_batched", shape=shape)
        self._as_declared.setdefault(buf.name, list(buf.shape))
        self._reshaped[buf.name] = shape
        self.declare(buf.name, shape, buf.dtype, buf.role)

    def scratch(self, shape: list[int], dtype: str, hint: str) -> Buffer:
        name = f"__scratch_{self._scratch}_{hint}"
        self._scratch += 1
        return self.declare(name, shape, dtype, "scratch")

    # -- entry point -------------------------------------------------------------------------
    def build(self) -> Plan:
        for decl in self.wl.tensors.values():
            self.declare(decl.name, decl.shape, decl.dtype, decl.role)
        for node in self.wl.nodes:
            self._node(node)
        cb = self._command_buffer()
        return Plan(self.wl.target or "gemmini", self.buffers, self.tasks, cb,
                    kernel_args(cb, self.tensor_order))

    # -- per-op lowering ---------------------------------------------------------------------
    def _node(self, node) -> None:
        handler = getattr(self, f"_op_{node.kind}", None)
        if handler is None:
            raise LoweringDeclined(
                f"interface op `{node.kind}` has no gemmini lowering in this backend",
                op=node.kind, shape=node.out_shape)
        handler(node)

    def _op_resident_pack(self, node) -> None:
        src = node.ins[0]
        self.residents[node.name] = src
        self.commands.append({"opcode": "RES_PACK",
                              "operands": {"src": src, "dst": node.name},
                              "attributes": {"layout": str(node.attrs.get("layout", "packed_rhs"))}})

    def _op_evict(self, node) -> None:
        self.commands.append({"opcode": "EVICT", "operands": {"handle": node.ins[0]}})

    def _op_matmul(self, node) -> None:
        lhs, rhs = node.ins
        self.acc_of[node.name] = {"lhs": lhs, "rhs": rhs}
        self.commands.append({"opcode": "MATMUL_RESIDENT",
                              "operands": {"lhs": lhs, "rhs": rhs, "dst": node.name}})

    def _op_commit(self, node) -> None:
        acc = node.ins[0]
        info = self.acc_of.get(acc)
        if info is None:
            raise LoweringDeclined("commit of an accumulator no matmul produced", op="commit")
        lhs = self.buffers[info["lhs"]]
        weight_name = self.residents.get(info["rhs"], info["rhs"])
        rhs = self.buffers[weight_name]
        e = _epilogue_from(node.attrs)
        m, k = _as_2d(lhs.shape)
        k2, n = _as_2d(rhs.shape)
        if k != k2:
            raise LoweringDeclined(
                f"contraction dim mismatch: lhs {lhs.shape} vs weight {rhs.shape}", op="matmul")
        out_shape = epilogue_out_shape(m, n, e)
        if out_shape != node.out_shape:
            raise LoweringDeclined(
                f"commit result {node.out_shape} disagrees with the derived shape {out_shape}",
                op="commit", shape=node.out_shape)
        self.declare(node.name, out_shape, e.output_dtype, "output")
        self.tasks.append(Contraction(lhs.name, weight_name, node.name, m, k, n,
                                      lhs_row_elems=k, rhs_row_elems=n, epilogue=e))
        attrs: dict[str, Any] = {"epilogue": e.stages, "output_dtype": e.output_dtype}
        if "acc_scale" in node.attrs:
            attrs["acc_scale"] = float(node.attrs["acc_scale"])
        if e.requant_shift is not None:
            attrs["requant_shift"] = e.requant_shift
        # MEASURED (docs/iteration_notes.md R2.4): splitting a trailing bias into the ABI's
        # standalone `BIAS_ADD` whole-op is numerically identical and passes L0/L1, but this
        # target's program oracle cannot build that opcode either (it unpacks the rank-1 bias as
        # a 2-D extent).  The FUSED form is what the interface declares and what the kernel does
        # -- the bias is DMA'd into the accumulator rows with a zero DRAM stride ahead of the
        # first compute -- so it is what ships.
        split_bias = False
        commit_ops: dict[str, Any] = {"src": acc, "dst": node.name}
        if e.bias:
            commit_ops["bias"] = e.bias
            attrs["bias"] = e.bias
        if split_bias:
            # This target's program oracle implements the store path's own stages (relu,
            # acc_scale, maxpool) and REJECTS a bias stage by name.  A trailing bias is the ABI's
            # standalone `BIAS_ADD` whole-op standing over the committed tensor, and because it is
            # the LAST stage the two forms are the same computation -- so it is emitted split,
            # in place on the commit's own destination (which keeps the commit output, and hence
            # the kernel ABI's argument list, unchanged).  The kernel still fuses it: the bias is
            # DMA'd into the accumulator rows ahead of the first compute.
            attrs["epilogue"] = [st for st in e.stages if st not in BIAS_STAGES]
            attrs.pop("bias", None)
        if e.has_pool:
            attrs.update(pool_in_dims=e.pool_in_dims, pool_size=e.pool_size,
                         pool_stride=e.pool_stride, pool_padding=e.pool_padding)
            if e.pool_pad_value is not None:
                attrs["pool_pad_value"] = e.pool_pad_value
        self.commands.append({"opcode": "COMMIT",
                              "operands": commit_ops,
                              "attributes": attrs})
        if split_bias:
            self.commands.append(
                {"opcode": "BIAS_ADD",
                 "operands": {"src": node.name, "bias": e.bias, "dst": node.name},
                 "attributes": {"output_dtype": e.output_dtype, "epilogue": ["bias_add"]}})

    def _op_movement(self, node) -> None:
        src = self.buffers[node.ins[0]]
        rows, cols = _as_2d(src.shape)
        out_dtype = str(node.attrs.get("output_dtype", src.dtype))
        self.declare(node.name, list(node.out_shape) or [rows, cols], out_dtype, "output")
        self.tasks.append(Movement(src.name, node.name, rows, cols, src.dtype, out_dtype))
        attrs: dict[str, Any] = {"output_dtype": out_dtype}
        if "semantic" in node.attrs:
            attrs["semantic"] = str(node.attrs["semantic"])
        self.commands.append({"opcode": "MOVEMENT",
                              "operands": {"src": src.name, "dst": node.name},
                              "attributes": attrs})

    def _op_conv2d(self, node) -> None:
        ifm = self.buffers[node.ins[0]]
        weight_name = self.residents.get(node.ins[1], node.ins[1])
        weight = self.buffers[weight_name]
        kernel = [int(v) for v in node.attrs["kernel"]]
        kh, kw, ci, co = kernel
        stride = [int(v) for v in node.attrs.get("stride", [1, 1])]
        padding = [int(v) for v in node.attrs.get("padding", [0, 0, 0, 0])]
        dilation = [int(v) for v in node.attrs.get("dilation", [1, 1])]
        layout = str(node.attrs.get("layout", "nhwc"))
        if layout != "nhwc":
            raise LoweringDeclined(f"conv2d layout {layout!r} is rejected (nhwc only)", op="conv2d")
        if len(ifm.shape) != 4:
            raise LoweringDeclined(f"conv2d activation rank {len(ifm.shape)} is not NHWC",
                                   op="conv2d", shape=ifm.shape)
        bn, h, w, cin = ifm.shape
        if cin != ci:
            raise LoweringDeclined(
                f"conv2d channel mismatch: activation C={cin} vs kernel ci={ci}", op="conv2d")
        ho, wo = conv_out_dims(h, w, kh, kw, stride, padding, dilation)
        if ho <= 0 or wo <= 0:
            raise LoweringDeclined(f"conv2d output extent {ho}x{wo} is empty", op="conv2d")
        e = _epilogue_from(node.attrs)
        if e.has_pool and [int(v) for v in e.pool_in_dims] != [ho, wo]:
            # The ABI states this cross-check outright: a fused pool's `pool_in_dims` is the
            # conv's OWN output extent, and a disagreement with the geometry derived from
            # kernel/stride/padding/dilation is REJECTED, never reconciled -- reconciling it
            # pools over a window the convolution never produced.
            raise LoweringDeclined(
                f"conv2d fused maxpool declares pool_in_dims {list(e.pool_in_dims)} but the "
                f"convolution's derived output extent is [{ho}, {wo}]",
                op="conv2d", shape=[ho, wo])
        rows, kdim = bn * ho * wo, kh * kw * ci
        if weight.shape != [kdim, co]:
            raise LoweringDeclined(
                f"conv2d weight {weight.shape} is not the [{kdim}, {co}] im2col packing",
                op="conv2d", shape=weight.shape)
        out_shape = epilogue_out_shape(rows, co, e)
        if out_shape != node.out_shape:
            raise LoweringDeclined(
                f"conv2d result {node.out_shape} disagrees with the derived shape {out_shape}",
                op="conv2d", shape=node.out_shape)
        candidate_dst = Buffer(node.name, out_shape, e.output_dtype, "output")

        # LOOP_CONV_WS consumes the interface's existing NHWC activation, flattened-HWIO
        # [KH*KW*CI, CO] weight and flattened-NHWC [N*HO*WO, CO] result directly.  Select it
        # only when the target advertises the sequencer and the operation needs none of the
        # richer Contraction epilogue machinery.  Unsupported geometry remains on the exact
        # im2col + Contraction route below; selection is never keyed on a model or layer name.
        candidate = Convolution(
            ifm.name, weight_name, node.name,
            bn, ci, h, w, co, kh, kw, ho, wo,
            stride[0], stride[1], dilation[0], dilation[1],
            padding[0], padding[1], padding[2], padding[3],
            input_layout="NHWC", weight_layout="HWIO", output_layout="NHWC",
            output_dtype=e.output_dtype,
        )
        native_loop_conv, _reason = select_native_loop_conv(
            candidate, ifm, weight, candidate_dst,
            epilogue_stages=e.stages, bias=e.bias)
        if native_loop_conv:
            self.declare(node.name, out_shape, e.output_dtype, "output")
            self.params.setdefault("convolution_lowering", {
                "selection_policy": "capability_geometry_layout_semantics_v1",
                "target_has_loop_conv": bool(_F.HAS_LOOP_CONV),
                "target_has_im2col": bool(_F.HAS_IM2COL),
            })
            self.commands.append({
                "opcode": "CONV2D",
                "operands": {"ifm": ifm.name, "weight": weight_name, "dst": node.name},
                "attributes": {
                    "kernel": kernel, "stride": stride, "padding": padding,
                    "dilation": dilation, "layout": "nhwc", "output_dtype": e.output_dtype,
                },
            })
            self.tasks.append(candidate)
            return

        im2col = self.declare(f"{ifm.name}_im2col", [rows, kdim], ifm.dtype, "input")
        self.params.setdefault("im2col_recipes", []).append(
            {"source": ifm.name, "target": im2col.name, "kh": kh, "kw": kw, "ci": ci,
             "stride": stride, "padding": padding, "dilation": dilation, "layout": layout})
        self.declare(node.name, out_shape, e.output_dtype, "output")
        acc = f"acc_{node.name}"
        handle = node.ins[1]
        if handle not in self.residents:
            # a conv whose weight was not packed by an explicit `resident_pack`
            handle = f"{weight_name}_res"
            self.residents[handle] = weight_name
            self.commands.append({"opcode": "RES_PACK",
                                  "operands": {"src": weight_name, "dst": handle},
                                  "attributes": {"layout": "packed_conv_rhs"}})
        self.commands.append({"opcode": "MATMUL_RESIDENT",
                              "operands": {"lhs": im2col.name, "rhs": handle, "dst": acc}})
        attrs = {"epilogue": e.stages, "output_dtype": e.output_dtype}
        if "acc_scale" in node.attrs:
            attrs["acc_scale"] = float(node.attrs["acc_scale"])
        if e.has_pool:
            attrs.update(pool_in_dims=e.pool_in_dims, pool_size=e.pool_size,
                         pool_stride=e.pool_stride, pool_padding=e.pool_padding)
            if e.pool_pad_value is not None:
                attrs["pool_pad_value"] = e.pool_pad_value
        conv_commit = {"src": acc, "dst": node.name}
        if e.bias:
            conv_commit["bias"] = e.bias
            attrs["bias"] = e.bias
        self.commands.append({"opcode": "COMMIT",
                              "operands": conv_commit,
                              "attributes": attrs})
        self.tasks.append(Contraction(im2col.name, weight_name, node.name, rows, kdim, co,
                                      lhs_row_elems=kdim, rhs_row_elems=co, epilogue=e))

    def _op_attention_qk(self, node) -> None:
        q = self.buffers[node.ins[0]]
        kt = self.buffers[node.ins[1]]
        m, d = _as_2d(q.shape)
        n, d2 = _as_2d(kt.shape)
        if d != d2:
            raise LoweringDeclined(
                f"attention_qk head dim mismatch: q {q.shape} vs k {kt.shape}", op="attention_qk")
        e = _epilogue_from(node.attrs)
        self.declare(node.name, [m, n], e.output_dtype, "output")
        staged = self.scratch([d, n], kt.dtype, "kt")
        self.tasks.append(Transpose(kt.name, staged.name, n, d, kt.dtype))
        self.tasks.append(Contraction(q.name, staged.name, node.name, m, d, n,
                                      lhs_row_elems=d, rhs_row_elems=n, epilogue=e))
        attrs = {"epilogue": e.stages, "output_dtype": e.output_dtype}
        if "acc_scale" in node.attrs:
            attrs["acc_scale"] = float(node.attrs["acc_scale"])
        self.commands.append({"opcode": "ATTENTION_QK",
                              "operands": {"q": q.name, "k": kt.name, "dst": node.name},
                              "attributes": attrs})

    def _op_attention_pv(self, node) -> None:
        p = self.buffers[node.ins[0]]
        v = self.buffers[node.ins[1]]
        m, s = _as_2d(p.shape)
        s2, d = _as_2d(v.shape)
        if s != s2:
            raise LoweringDeclined(
                f"attention_pv contraction mismatch: p {p.shape} vs v {v.shape}",
                op="attention_pv")
        e = _epilogue_from(node.attrs)
        self.declare(node.name, [m, d], e.output_dtype, "output")
        self.tasks.append(Contraction(p.name, v.name, node.name, m, s, d,
                                      lhs_row_elems=s, rhs_row_elems=d, epilogue=e))
        attrs = {"epilogue": e.stages, "output_dtype": e.output_dtype}
        if "acc_scale" in node.attrs:
            attrs["acc_scale"] = float(node.attrs["acc_scale"])
        self.commands.append({"opcode": "ATTENTION_PV",
                              "operands": {"p": p.name, "v": v.name, "dst": node.name},
                              "attributes": attrs})

    def _op_bias_add(self, node) -> None:
        src = self.buffers[node.ins[0]]
        bias = self.buffers[node.ins[1]]
        rows, cols = _as_2d(src.shape)
        out_dtype = str(node.attrs.get("output_dtype", src.dtype))
        self.declare(node.name, [rows, cols], out_dtype, "output")
        self.tasks.append(HostBiasAdd(src.name, bias.name, node.name, rows, cols, out_dtype))
        self.commands.append({"opcode": "BIAS_ADD",
                              "operands": {"src": src.name, "bias": bias.name,
                                           "dst": node.name},
                              "attributes": {"output_dtype": out_dtype,
                                             "epilogue": ["bias_add"]}})

    def _op_matmul_batched(self, node) -> None:
        """A rank-3 stack of independent contractions, lowered to ONE 2-D contraction.

        `out[b, i, j] = sum_k a[b, i, k] * w[b, k, j]` is not a single matmul, because each batch
        contracts against its OWN weight slice.  It becomes one when the activation is expanded
        into its BLOCK-DIAGONAL form `A*[b*M + i, b*K + k] = a[b, i, k]` (zero off the diagonal
        blocks): then `A* @ W*` -- with `W*` the weight stack read as one `[B*K, N]` matrix, which
        is exactly its row-major layout -- reproduces every batch's product, stacked in rows, and
        the cross terms multiply the structural zeros away.

        The block-diagonal operand is DERIVED, not authored: it is expressed as an im2col gather
        over the activation viewed as `[1, B, M, K]` NHWC, which is the ABI's one additive
        derivation mechanism (`params.im2col_recipes`) and is materialised identically by the
        reference, the simulator and the device harness.  The window walks the batch axis with
        `kh = B` taps at dilation `-B` and stride `B + 1`, so tap `t` of output row `b*M + i`
        lands on plane `b + (t - b) * (-B) + b*B`... concretely `h = b*(1 + B) - t*B`, which is
        `b` when `t == b` and outside `[0, B)` for every other tap -- and an out-of-bounds im2col
        tap reads zero, which IS the off-diagonal block.  Nothing here is capsule-specific: B, M,
        K and N are the operands' own extents.
        """
        a = self.buffers[node.ins[0]]
        w = self.buffers[node.ins[1]]
        # A tensor an earlier batched contraction already took a view of is interpreted under the
        # shape it was DECLARED with, not under the view: two batched contractions may legitimately
        # share an activation, and each derives its own gather from the same declaration.
        a_shape = self._as_declared.get(a.name, a.shape)
        w_shape = self._as_declared.get(w.name, w.shape)
        if len(a_shape) != 3 or len(w_shape) != 3:
            raise LoweringDeclined(
                f"matmul_batched needs rank-3 operands, got {a_shape} and {w_shape}",
                op="matmul_batched", shape=a_shape)
        batch, m, k = (int(v) for v in a_shape)
        wb, k2, n = (int(v) for v in w_shape)
        if wb != batch or k2 != k:
            raise LoweringDeclined(
                f"matmul_batched operand mismatch: {a_shape} vs {w_shape}", op="matmul_batched",
                shape=a_shape)
        if batch < 1 or m < 1 or k < 1 or n < 1:
            raise LoweringDeclined(f"matmul_batched degenerate extents {a_shape} x {w_shape}",
                                   op="matmul_batched", shape=a_shape)
        rows, kdim = batch * m, batch * k
        if rows * kdim > BLOCK_DIAGONAL_ELEM_BUDGET:
            raise LoweringDeclined(
                f"matmul_batched block-diagonal operand is {rows}x{kdim} = {rows * kdim} elements, "
                f"past this backend's {BLOCK_DIAGONAL_ELEM_BUDGET}-element budget for the derived "
                f"gather (the expansion costs a factor of the batch count)",
                op="matmul_batched", shape=a_shape)

        e = _epilogue_from(node.attrs)
        if "output_dtype" in node.attrs:
            e.output_dtype = str(node.attrs["output_dtype"])
        out_shape = epilogue_out_shape(rows, n, e)
        declared = [int(v) for v in (node.out_shape or [])]
        if declared and _numel(declared) != _numel(out_shape):
            raise LoweringDeclined(
                f"matmul_batched result {declared} disagrees with the derived shape {out_shape}",
                op="matmul_batched", shape=declared)

        # The activation is re-declared as the rank-4 NHWC view the gather reads it through, and
        # the weight stack as the [B*K, N] matrix it already is in memory.  Both are reshapes:
        # the row-major element order is unchanged, so the harness materialises the same bytes.
        # A tensor an EARLIER command already read under its original rank cannot be reshaped
        # underneath it, and two batched contractions cannot ask for two different views of the
        # same tensor -- either would silently change what that earlier command contracts, so both
        # are refused with the two shapes named.
        self._reshape(a, [1, batch, m, k])
        self._reshape(w, [kdim, n])
        blk = self.declare(f"{node.name}_lhs_batchdiag", [rows, kdim], a.dtype, "input")
        self.params.setdefault("im2col_recipes", []).append(
            {"source": a.name, "target": blk.name, "kh": batch, "kw": 1, "ci": k,
             "stride": [batch + 1, 1], "padding": [0, 0, 0, 0], "dilation": [-batch, 1],
             "layout": "nhwc"})

        self.declare(node.name, out_shape, e.output_dtype, "output")
        handle = f"{w.name}_res"
        if handle not in self.residents:
            self.residents[handle] = w.name
            self.commands.append({"opcode": "RES_PACK",
                                  "operands": {"src": w.name, "dst": handle},
                                  "attributes": {"layout": "packed_batched_rhs"}})
        acc = f"acc_{node.name}"
        self.commands.append({"opcode": "MATMUL_RESIDENT",
                              "operands": {"lhs": blk.name, "rhs": handle, "dst": acc}})
        attrs: dict[str, Any] = {"epilogue": e.stages, "output_dtype": e.output_dtype}
        if "acc_scale" in node.attrs:
            attrs["acc_scale"] = float(node.attrs["acc_scale"])
        if e.requant_shift is not None:
            attrs["requant_shift"] = e.requant_shift
        commit_ops: dict[str, Any] = {"src": acc, "dst": node.name}
        if e.bias:
            commit_ops["bias"] = e.bias
            attrs["bias"] = e.bias
        if e.has_pool:
            attrs.update(pool_in_dims=e.pool_in_dims, pool_size=e.pool_size,
                         pool_stride=e.pool_stride, pool_padding=e.pool_padding)
            if e.pool_pad_value is not None:
                attrs["pool_pad_value"] = e.pool_pad_value
        self.commands.append({"opcode": "COMMIT", "operands": commit_ops, "attributes": attrs})
        self.commands.append({"opcode": "EVICT", "operands": {"handle": handle}})
        self.tasks.append(Contraction(blk.name, w.name, node.name, rows, kdim, n,
                                      lhs_row_elems=kdim, rhs_row_elems=n, epilogue=e))
        self.params.setdefault("batched_contractions", []).append(
            {"lhs": a.name, "rhs": w.name, "dst": node.name, "batch": batch,
             "m": m, "k": k, "n": n, "block_diagonal_lhs": blk.name})

    # -- serialisation -----------------------------------------------------------------------
    def _command_buffer(self) -> dict[str, Any]:
        tensors: dict[str, Any] = {}
        for name in self.tensor_order:
            buf = self.buffers[name]
            tensors[name] = {"shape": buf.shape, "dtype": buf.dtype, "role": buf.role}
        cb: dict[str, Any] = {"abi_version": self.wl.abi_version or "0.1",
                              "target": self.wl.target or "gemmini",
                              "backend": "mlir_oot_xdsl_gemmini",
                              "tensors": tensors,
                              "commands": self.commands}
        if self.params:
            cb["params"] = dict(self.params)
        return cb


def _as_2d(shape: list[int]) -> tuple[int, int]:
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 1:
        return 1, int(shape[0])
    if len(shape) > 2:
        rows = 1
        for d in shape[:-1]:
            rows *= int(d)
        return rows, int(shape[-1])
    raise LoweringDeclined(f"cannot read a 2-D extent out of shape {shape}", shape=shape)


#: The epilogue stages that add a per-column bias (the ABI's own `BIAS_STAGES`).
BIAS_STAGES = ("bias_add", "bias")


def _splittable_bias(e: "Epilogue") -> bool:
    """Is this commit's bias stage the LAST one, so the ABI's standalone BIAS_ADD is equivalent?"""
    stages = list(e.stages)
    bias_at = [i for i, st in enumerate(stages) if st in BIAS_STAGES]
    if not bias_at or e.bias is None:
        return False
    return all(st in BIAS_STAGES for st in stages[bias_at[0]:])


#: The command-buffer opcodes `kernel_abi.arg_order_tokens.matmul_lhs_group_major` resolves: each
#: carries a `lhs` against a resident `rhs`.  A whole-op such as `BATCHED_MATMUL` is deliberately
#: NOT here -- it has no `lhs`/accumulator pair, so the resident-matmul row cannot bind it and the
#: ABI falls through to declaration order.
MATMUL_OPCODES = ("MATMUL_RESIDENT", "MATMUL")


def kernel_args(cb: dict[str, Any], declaration_order: list[str]) -> list[str]:
    """The kernel's pointer arguments, per `mlir_oot_backend_contract.kernel_abi`.

    Rows are tried top-down and the first matching `when` decides, exactly as the contract states.
    """
    abi = cb.get("kernel_abi") or {}
    if abi.get("kind") == "whole_program":
        return [arg["tensor"] for arg in abi["args"]]
    cmds = cb.get("commands", [])
    tensors = cb.get("tensors", {})
    opcodes = [c.get("opcode") for c in cmds]

    # row 1 -- movement
    if "RES_PACK" not in opcodes:
        for c in cmds:
            if c.get("opcode") == "MOVEMENT":
                ops = c.get("operands", {})
                return [ops["src"], ops["dst"]]

    # row 2 -- native whole op
    whole = [c for c in cmds if c.get("opcode") in WHOLE_OP_OPCODES]
    if len(whole) == 1:
        return [n for n in declaration_order
                if tensors.get(n, {}).get("role") in ("input", "weight", "bias", "output")]

    # row 3 -- resident matmul
    residents: list[str] = []
    handle_src: dict[str, str] = {}
    for c in cmds:
        if c.get("opcode") == "RES_PACK":
            ops = c["operands"]
            handle_src[ops["dst"]] = ops["src"]
    rhs_handles = {c.get("operands", {}).get("rhs") for c in cmds
                   if c.get("opcode") in MATMUL_OPCODES}
    for c in cmds:
        if c.get("opcode") == "RES_PACK":
            ops = c["operands"]
            if ops["dst"] in rhs_handles and ops["src"] not in residents:
                residents.append(ops["src"])
    groups: dict[str, list[dict[str, Any]]] = {r: [] for r in residents}
    acc_to_group: dict[str, str] = {}
    for c in cmds:
        if c.get("opcode") in MATMUL_OPCODES:
            ops = c["operands"]
            weight = handle_src.get(ops.get("rhs", ""), ops.get("rhs", ""))
            groups.setdefault(weight, []).append(c)
            acc_to_group[ops["dst"]] = weight
    commit_of: dict[str, dict[str, Any]] = {}
    for c in cmds:
        if c.get("opcode") == "COMMIT":
            commit_of[c["operands"]["src"]] = c
    order = residents or list(groups)
    lhs_args = [c["operands"]["lhs"] for g in order for c in groups.get(g, [])]
    out_args = [commit_of[c["operands"]["dst"]]["operands"]["dst"]
                for g in order for c in groups.get(g, [])
                if c["operands"]["dst"] in commit_of]
    bias_args: list[str] = []
    for group in order:
        for matmul in groups.get(group, []):
            commit = commit_of.get(matmul["operands"]["dst"])
            if commit is None:
                continue
            ops = commit.get("operands", {})
            attrs = commit.get("attributes", {})
            if not any(stage in BIAS_STAGES for stage in (attrs.get("epilogue") or [])):
                continue
            bias = attrs.get("bias") or ops.get("bias")
            if not isinstance(bias, str) or not bias:
                raise LoweringDeclined(
                    f"COMMIT {ops.get('dst')!r} declares a bias stage but names no bias tensor",
                    op="commit")
            bias_args.append(bias)
    args = residents + lhs_args + out_args + bias_args
    if args:
        return args

    # nothing in the contract's three rows resolves: fall back to declaration order, which is
    # what an ABI-less whole-op buffer (e.g. BATCHED_MATMUL) can be called with.
    return [n for n in declaration_order
            if tensors.get(n, {}).get("role") in ("input", "weight", "bias", "output")]

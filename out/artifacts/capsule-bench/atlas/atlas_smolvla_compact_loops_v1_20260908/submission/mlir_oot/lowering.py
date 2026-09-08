"""Interface-to-Atlas xDSL lowering passes and the hardware-derived tile scheduler."""
import io
import json
from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.printer import Printer

from . import dialect as d


@dataclass(frozen=True)
class Tile:
    m0: int
    n0: int
    k0: int
    m1: int
    n1: int
    k1: int


class TileScheduler:
    """Select a complete row-major, K-inner schedule for the RTL-derived 32x32 mesh."""
    DIM = 32

    @classmethod
    def choose(cls, m, k, n):
        return [Tile(m0, n0, k0, min(m0 + cls.DIM, m), min(n0 + cls.DIM, n), min(k0 + cls.DIM, k))
                for m0 in range(0, m, cls.DIM)
                for n0 in range(0, n, cls.DIM)
                for k0 in range(0, k, cls.DIM)]


class ConvertIfaceToAtlasPass:
    """Build verified target operations from the structurally parsed interface graph."""
    name = "convert-iface-to-atlas"

    def apply(self, workload):
        ops = []
        for item in workload.ops:
            kind = item["op"]
            if kind == "matmul":
                tensors = {t.name: t for t in workload.tensors}
                rhs_name = item["rhs"].removesuffix("_resident")
                lhs, rhs = tensors[item["lhs"]], tensors[rhs_name]
                m, k, n = lhs.shape[-2], lhs.shape[-1], rhs.shape[-1]
                for tile in TileScheduler.choose(m, k, n):
                    detail = StringAttr(json.dumps(tile.__dict__, separators=(",", ":")))
                    for cls in (d.DmaLoadOp, d.TensorLoadOp, d.TransposeOp, d.WeightPushOp,
                                d.MatmulOp, d.AccumulatorPopOp, d.TensorStoreOp, d.DmaStoreOp):
                        ops.append(cls.create(attributes={"detail": detail}))
            elif kind in ("movement", "add", "bias_add", "gelu", "silu", "softmax", "reduce_sum"):
                cls = d.VectorBinaryOp if kind in ("add", "bias_add") else d.VectorUnaryOp
                ops.append(d.DmaLoadOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(cls.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.DmaStoreOp.create(attributes={"detail": StringAttr(kind)}))
            elif kind == "rmsnorm":
                ops.append(d.DmaLoadOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.VectorUnaryOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.VectorBinaryOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.DmaStoreOp.create(attributes={"detail": StringAttr(kind)}))
            elif kind == "layernorm":
                ops.append(d.DmaLoadOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.VectorUnaryOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.VectorBinaryOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.DmaStoreOp.create(attributes={"detail": StringAttr(kind)}))
            elif kind == "rope":
                ops.append(d.DmaLoadOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.VectorBinaryOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.DmaStoreOp.create(attributes={"detail": StringAttr(kind)}))
            elif kind == "geglu":
                ops.append(d.DmaLoadOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.MatmulOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.VectorUnaryOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.VectorBinaryOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.DmaStoreOp.create(attributes={"detail": StringAttr(kind)}))
            elif kind == "attention_full":
                ops.append(d.DmaLoadOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.MatmulOp.create(attributes={"detail": StringAttr("qk")}))
                ops.append(d.VectorUnaryOp.create(attributes={"detail": StringAttr("softmax")}))
                ops.append(d.MatmulOp.create(attributes={"detail": StringAttr("pv")}))
                ops.append(d.DmaStoreOp.create(attributes={"detail": StringAttr(kind)}))
            elif kind in ("gemv_batched", "matmul_batched"):
                ops.append(d.DmaLoadOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.MatmulOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.DmaStoreOp.create(attributes={"detail": StringAttr(kind)}))
            elif kind == "depthwise_conv2d":
                ops.append(d.DmaLoadOp.create(attributes={"detail": StringAttr("im2col")}))
                ops.append(d.MatmulOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.VectorBinaryOp.create(attributes={"detail": StringAttr(kind)}))
                ops.append(d.DmaStoreOp.create(attributes={"detail": StringAttr(kind)}))
        ops.append(d.HaltOp.create(attributes={"detail": StringAttr("ecall")}))
        module = ModuleOp(ops)
        module.verify()
        return module


def lower_text(workload):
    module = ConvertIfaceToAtlasPass().apply(workload)
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    return stream.getvalue() + "\n"

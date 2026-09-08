#!/usr/bin/env python3
"""Capture an explicit, target-neutral native-aligned ResNet-50 W8A8 contract.

This is deliberately a NEW numerical contract.  It does not mutate or claim bit-equivalence
with the canonical per-channel PT2E capture.  Every Conv2d/Linear edge is calibrated as:

  symmetric per-tensor i8 activation and weight
  -> i32 reduction
  -> add symmetric per-tensor i32 bias in accumulator units
  -> one precomputed f32 multiplier
  -> round-to-nearest-even and saturate to i8

The converted PT2E graph is used only to obtain independently calibrated qvalues/qparams and the
full model topology.  ``NativeAlignedInterpreter`` executes the declared arithmetic directly; it
does not consume Merlin IR, target code, or target output.  Full logits are never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.fx import Interpreter, Node
from torchvision.models import ResNet50_Weights, resnet50
from torchao.quantization.pt2e import HistogramObserver, MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)


HERE = Path(__file__).resolve().parent
ARTIFACT = HERE.parent
ROOT = ARTIFACT.parents[4]
M2M = Path(os.environ.get("MERLIN_M2M_DIR", "/scratch/agustin/projects/model2MLIR"))
if str(M2M) not in sys.path:
    sys.path.insert(0, str(M2M))

MEASURED = ROOT / "out/artifacts/recaptures/resnet50_pt2e_w8a8_universal_dog_20260906/source/measured_input.npz"
CALIBRATION = ROOT / "out/artifacts/recaptures/resnet50_pt2e_w8a8_universal_dog_20260906/source/independent_synthetic_calibration.npz"
SCHEME = "native_aligned_per_tensor_i8"
SELECTED_FQN = "model.layer1.0.conv1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return sha256_bytes(array.tobytes())


def load_array(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.ascontiguousarray(data[data.files[0]], dtype=np.float32)


class Classifier(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


class NativeAlignedQuantizer(Quantizer):
    """Portable PT2E annotations for the declared scalar-scale integer contract."""

    def __init__(self) -> None:
        eps = 2**-12
        self.activation = QuantizationSpec(
            dtype=torch.int8,
            observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=eps),
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_tensor_symmetric,
        )
        self.weight = QuantizationSpec(
            dtype=torch.int8,
            observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=eps),
            quant_min=-127,
            quant_max=127,
            qscheme=torch.per_tensor_symmetric,
        )
        self.annotated = 0

    @staticmethod
    def _bias_qparams(observers: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        act_scale, act_zp = observers[0].calculate_qparams()
        weight_scale, _weight_zp = observers[1].calculate_qparams()
        return act_scale * weight_scale, torch.zeros_like(act_zp)

    def annotate(self, graph_module: Any) -> Any:
        supported = {torch.ops.aten.conv2d.default, torch.ops.aten.linear.default}
        for node in graph_module.graph.nodes:
            if node.target not in supported or len(node.args) < 2:
                continue
            activation, weight = node.args[:2]
            inputs: dict[Node, Any] = {activation: self.activation, weight: self.weight}
            if len(node.args) > 2 and isinstance(node.args[2], Node):
                bias = node.args[2]
                inputs[bias] = DerivedQuantizationSpec(
                    derived_from=[(activation, node), (weight, node)],
                    derive_qparams_fn=self._bias_qparams,
                    dtype=torch.int32,
                    quant_min=-(2**31),
                    quant_max=2**31 - 1,
                    qscheme=torch.per_tensor_symmetric,
                )
            # An output qspec is intentional: every contraction has an immediate narrow edge.
            # Residual/add/pool semantics remain ordinary graph operations after dequantization.
            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=inputs,
                output_qspec=self.activation,
                _annotated=True,
            )
            self.annotated += 1
        return graph_module

    def validate(self, graph_module: Any) -> None:
        del graph_module
        if self.annotated != 54:
            raise ValueError(f"expected 53 Conv2d + 1 Linear annotations, got {self.annotated}")


def pair(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    values = tuple(int(v) for v in value)
    if len(values) != 2:
        raise ValueError(f"expected pair, got {value!r}")
    return values


def module_fqn(node: Node) -> str:
    stack = node.meta.get("nn_module_stack", {})
    if stack:
        value = list(stack.values())[-1][0]
        return str(value)
    return node.name


class NativeAlignedInterpreter(Interpreter):
    """Independent executor for one-combined-multiply native-aligned arithmetic."""

    Q = torch.ops.quantized_decomposed.quantize_per_tensor.default
    DQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    CONV = torch.ops.aten.conv2d.default
    LINEAR = torch.ops.aten.linear.default

    def __init__(self, graph_module: Any, selected_fqn: str) -> None:
        super().__init__(graph_module, garbage_collect_values=False)
        self.pending: dict[Node, dict[str, Any]] = {}
        self.selected_fqn = selected_fqn
        self.selected: dict[str, torch.Tensor | float | str] = {}
        self.convs = 0
        self.linears = 0
        self.overflow_checks = 0

    def value(self, item: Any) -> Any:
        return self.env[item] if isinstance(item, Node) else item

    def qparams(self, node: Any, dtype: torch.dtype) -> tuple[torch.Tensor, np.float32, int]:
        if not isinstance(node, Node) or node.target != self.DQ or len(node.args) < 3:
            raise ValueError(f"expected per-tensor dequantize operand, got {node!r}")
        qvalue = self.value(node.args[0])
        if qvalue.dtype != dtype:
            raise ValueError(f"expected {dtype} qvalue, got {qvalue.dtype}")
        return qvalue, np.float32(float(self.value(node.args[1]))), int(self.value(node.args[2]))

    @staticmethod
    def requant(acc: torch.Tensor, multiplier: np.float32, qmin: int, qmax: int) -> torch.Tensor:
        # Exactly one runtime f32 multiply. torch.round is round-to-nearest-even.
        scaled = acc.to(torch.float32) * torch.tensor(float(multiplier), dtype=torch.float32)
        return torch.round(scaled).clamp(qmin, qmax).to(torch.int8)

    def run_node(self, node: Node) -> Any:
        if node.op == "call_function" and node.target in (self.CONV, self.LINEAR):
            qx, sx, zx = self.qparams(node.args[0], torch.int8)
            qw, sw, zw = self.qparams(node.args[1], torch.int8)
            if zx != 0 or zw != 0:
                raise ValueError("native-aligned contract requires symmetric activation/weight")
            qb, sb, zb = self.qparams(node.args[2], torch.int32)
            if zb != 0 or np.float32(sb) != np.float32(np.float32(sx) * np.float32(sw)):
                raise ValueError("bias scale is not exactly activation_scale * weight_scale")

            x = qx.to(torch.int32)
            w = qw.to(torch.int32)
            if node.target == self.LINEAR:
                acc = torch.matmul(x, w.transpose(-1, -2)).to(torch.int64)
                acc += qb.to(torch.int64)
                self.linears += 1
            else:
                args, kwargs = self.fetch_args_kwargs_from_env(node)
                stride = pair(args[3] if len(args) > 3 else kwargs.get("stride", 1))
                padding = pair(args[4] if len(args) > 4 else kwargs.get("padding", 0))
                dilation = pair(args[5] if len(args) > 5 else kwargs.get("dilation", 1))
                groups = int(args[6] if len(args) > 6 else kwargs.get("groups", 1))
                kh, kw = int(w.shape[-2]), int(w.shape[-1])
                # unfold copies integer-valued f32 exactly; the reduction itself is integer.
                cols = F.unfold(x.to(torch.float32), (kh, kw), dilation=dilation,
                                padding=padding, stride=stride).to(torch.int32)
                batch, _flat_k, positions = cols.shape
                cin_group, cout_group = int(x.shape[1]) // groups, int(w.shape[0]) // groups
                cols = cols.reshape(batch, groups, cin_group * kh * kw, positions)
                weights = w.reshape(groups, cout_group, cin_group * kh * kw)
                acc = torch.matmul(weights.unsqueeze(0), cols).to(torch.int64)
                oh = (int(x.shape[2]) + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
                ow = (int(x.shape[3]) + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
                acc = acc.reshape(batch, w.shape[0], oh, ow)
                acc += qb.to(torch.int64).reshape(1, -1, 1, 1)
                self.convs += 1
            if int(acc.min()) < -(2**31) or int(acc.max()) > 2**31 - 1:
                raise OverflowError(f"{module_fqn(node)} exceeds the declared i32 accumulator")
            self.overflow_checks += 1
            self.pending[node] = {"acc": acc.to(torch.int32), "sx": sx, "sw": sw,
                                  "qx": qx, "qw": qw, "qb": qb}
            # The only semantic consumer is the immediate output quantizer intercepted below.
            return torch.empty_like(acc, dtype=torch.float32)

        if node.op == "call_function" and node.target == self.Q and isinstance(node.args[0], Node):
            producer = node.args[0]
            if producer in self.pending:
                state = self.pending[producer]
                out_scale = np.float32(float(self.value(node.args[1])))
                out_zp = int(self.value(node.args[2]))
                qmin, qmax = int(self.value(node.args[3])), int(self.value(node.args[4]))
                if out_zp != 0 or qmin != -128 or qmax != 127:
                    raise ValueError("native-aligned output must be symmetric saturated i8")
                multiplier = np.float32(
                    np.float32(state["sx"]) * np.float32(state["sw"]) / out_scale)
                result = self.requant(state["acc"], multiplier, qmin, qmax)
                if module_fqn(producer) == self.selected_fqn:
                    self.selected = {
                        "fqn": self.selected_fqn,
                        "qx": state["qx"].detach().cpu(),
                        "qw": state["qw"].detach().cpu(),
                        "qb": state["qb"].detach().cpu(),
                        "expected": result.detach().cpu(),
                        "sx": float(state["sx"]),
                        "sw": float(state["sw"]),
                        "out_scale": float(out_scale),
                        "multiplier": float(multiplier),
                    }
                return result
        return super().run_node(node)


def make_quantized_model(image: torch.Tensor, calibration: np.ndarray) -> tuple[Any, int]:
    torch.manual_seed(0)
    model = Classifier(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()).eval()
    exported = torch.export.export(model, (image,)).module()
    quantizer = NativeAlignedQuantizer()
    prepared = prepare_pt2e(exported, quantizer)
    with torch.no_grad():
        for i in range(min(2, int(calibration.shape[0]))):
            prepared(torch.from_numpy(calibration[i:i + 1]))
    quantized = convert_pt2e(
        prepared, use_reference_representation=False, fold_quantize=True)
    # model2MLIR defensively calls eval() at its capture boundary. Exported modules reject that
    # call until PyTorch's explicit compatibility shim is installed.
    try:
        from torch.ao.quantization import allow_exported_model_train_eval

        allow_exported_model_train_eval(quantized)
    except (ImportError, AttributeError):  # pragma: no cover - version-dependent API
        pass
    return quantized, quantizer.annotated


def get_attr(graph_module: Any, node: Node) -> torch.Tensor:
    if node.op != "get_attr":
        raise ValueError(f"qvalue is not frozen graph state: {node}")
    value = graph_module
    for part in str(node.target).split("."):
        value = getattr(value, part)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{node.target} is not a tensor")
    return value.detach().cpu()


def node_qparams(graph_module: Any, dequant: Node, dtype: torch.dtype) -> tuple[torch.Tensor, np.float32, int]:
    if dequant.target != torch.ops.quantized_decomposed.dequantize_per_tensor.default:
        raise ValueError(f"not a per-tensor dequant: {dequant}")
    qvalue = get_attr(graph_module, dequant.args[0]) if isinstance(dequant.args[0], Node) else dequant.args[0]
    if qvalue.dtype != dtype:
        raise ValueError(f"expected {dtype}, got {qvalue.dtype}")
    return qvalue, np.float32(float(dequant.args[1])), int(dequant.args[2])


def tensor_shape(node: Node) -> list[int]:
    """Recover a static tensor shape through inserted PT2E Q/DQ wrappers.

    ``convert_pt2e`` preserves metadata on the original producer but does not copy it to the
    quantize/dequantize nodes it inserts.  A census must therefore walk through those transparent
    wrappers instead of assuming every new node owns ``meta["val"]``.
    """
    seen: set[Node] = set()
    current: Any = node
    while isinstance(current, Node) and current not in seen:
        seen.add(current)
        value = current.meta.get("val")
        if value is not None and hasattr(value, "shape"):
            return [int(v) for v in value.shape]
        tensor_meta = current.meta.get("tensor_meta")
        if tensor_meta is not None and hasattr(tensor_meta, "shape"):
            return [int(v) for v in tensor_meta.shape]
        if current.target in (
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ) and current.args:
            current = current.args[0]
            continue
        break
    raise ValueError(f"no static tensor shape reachable from {node}")


def contraction_rows(graph_module: Any) -> list[dict[str, Any]]:
    rows = []
    for node in graph_module.graph.nodes:
        if node.target not in (torch.ops.aten.conv2d.default, torch.ops.aten.linear.default):
            continue
        qx_node, qw_node, qb_node = node.args[:3]
        qx, sx, zx = node_qparams(graph_module, qx_node, torch.int8) if qx_node.args[0].op == "get_attr" else (None, np.float32(qx_node.args[1]), int(qx_node.args[2]))
        qw, sw, zw = node_qparams(graph_module, qw_node, torch.int8)
        qb, sb, zb = node_qparams(graph_module, qb_node, torch.int32)
        uses = list(node.users)
        output_q = uses[0] if len(uses) == 1 else None
        if output_q is None or output_q.target != torch.ops.quantized_decomposed.quantize_per_tensor.default:
            raise ValueError(f"{module_fqn(node)} does not have one immediate output quantizer")
        sout = np.float32(float(output_q.args[1]))
        zout, qmin, qmax = int(output_q.args[2]), int(output_q.args[3]), int(output_q.args[4])
        multiplier = np.float32(np.float32(sx) * np.float32(sw) / sout)
        out_shape = [int(v) for v in node.meta["val"].shape]
        row: dict[str, Any] = {
            "index": len(rows),
            "kind": "conv2d" if node.target == torch.ops.aten.conv2d.default else "linear",
            "fqn": module_fqn(node),
            "input_shape": tensor_shape(qx_node),
            "weight_shape": list(qw.shape),
            "bias_shape": list(qb.shape),
            "output_shape": out_shape,
            "activation_scale_f32": float(sx),
            "weight_scale_f32": float(sw),
            "bias_scale_f32": float(sb),
            "output_scale_f32": float(sout),
            "combined_multiplier_f32": float(multiplier),
            "activation_zero_point": zx,
            "weight_zero_point": zw,
            "bias_zero_point": zb,
            "output_zero_point": zout,
            "output_range": [qmin, qmax],
            "weight_sha256": tensor_sha256(qw),
            "bias_sha256": tensor_sha256(qb),
            "bias_scale_matches_accumulator_scale": bool(
                np.float32(sb) == np.float32(np.float32(sx) * np.float32(sw))),
            "native_scalar_epilogue_eligible": bool(
                zx == zw == zb == zout == 0 and qmin == -128 and qmax == 127
                and np.float32(sb) == np.float32(np.float32(sx) * np.float32(sw))),
        }
        if row["kind"] == "conv2d":
            row.update({
                "stride": list(pair(node.args[3] if len(node.args) > 3 else 1)),
                "padding": list(pair(node.args[4] if len(node.args) > 4 else 0)),
                "dilation": list(pair(node.args[5] if len(node.args) > 5 else 1)),
                "groups": int(node.args[6] if len(node.args) > 6 else 1),
            })
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "native_aligned_capture"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    measured = load_array(MEASURED)
    image = torch.from_numpy(measured[0] if measured.ndim == 5 else measured)
    calibration = load_array(CALIBRATION)
    quantized, annotations = make_quantized_model(image, calibration)
    rows = contraction_rows(quantized)

    independent = NativeAlignedInterpreter(quantized, SELECTED_FQN)
    with torch.no_grad():
        native_logits = independent.run(image)
        qdq_logits = quantized(image)
        fp32_logits = Classifier(
            resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()).eval()(image)
    if len(independent.selected) == 0:
        raise ValueError(f"selected actual layer was not executed: {SELECTED_FQN}")

    # The fixture contains real quantized checkpoint state and the activation observed under the
    # independent native-aligned execution. Expected output stays derivable, not embedded.
    selected = independent.selected
    np.savez_compressed(
        out / "actual_conv_fixture.npz",
        activation_nchw=selected["qx"].numpy(),
        weight_oihw=selected["qw"].numpy(),
        bias_i32=selected["qb"].numpy(),
    )

    import m2m
    from m2m.capture.torchao_pipeline import QuantizationConfig
    from m2m.coverage import opaque_report
    weights = out / "weights.safetensors"
    converted = m2m.convert(
        quantized, (image,), backend="fx_importer",
        quantization=QuantizationConfig(scheme=SCHEME),
        quantization_preapplied=True, level="linalg-on-tensors",
        func_name="forward", weights_path=str(weights))
    (out / "linalg.mlir").write_text(converted.mlir_text)
    opaque = opaque_report(converted.mlir_text)

    def summary(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
        c, r = candidate.detach().float().flatten(), reference.detach().float().flatten()
        return {
            "max_abs": float(torch.max(torch.abs(c - r))),
            "cosine": float(F.cosine_similarity(c, r, dim=0)),
        }

    actual_layer = next(row for row in rows if row["fqn"] == SELECTED_FQN)
    expected = selected["expected"]
    receipt = {
        "schema": "native_aligned_per_tensor_i8_resnet50_capture_v1",
        "status": "passed" if len(rows) == 54 and converted.ok and not opaque else "failed",
        "contract_name": SCHEME,
        "comparison_to_canonical_pt2e": "NOT_BIT_EQUIVALENT_NEW_TARGET_AWARE_QUANTIZATION",
        "checkpoint": "torchvision.models.resnet50(IMAGENET1K_V2)",
        "measured_input": {"path": str(MEASURED), "sha256": sha256_file(MEASURED)},
        "calibration": {"path": str(CALIBRATION), "sha256": sha256_file(CALIBRATION),
                        "samples": min(2, int(calibration.shape[0])),
                        "includes_measured_input": False},
        "arithmetic": {
            "activation": "symmetric_per_tensor_i8",
            "weight": "symmetric_per_tensor_i8",
            "accumulator": "i32",
            "bias": "symmetric_i32_in_accumulator_units",
            "requant": "one_precomputed_f32_multiplier_then_round_nearest_even",
            "saturation": [-128, 127],
        },
        "annotated_contractions": annotations,
        "conv2d_count": sum(row["kind"] == "conv2d" for row in rows),
        "linear_count": sum(row["kind"] == "linear" for row in rows),
        "native_scalar_epilogue_eligible": sum(row["native_scalar_epilogue_eligible"] for row in rows),
        "independent_reference": {
            "implementation": "NativeAlignedInterpreter in this source file",
            "conv2d_executed": independent.convs,
            "linear_executed": independent.linears,
            "i32_overflow_checks": independent.overflow_checks,
            "finite_logits": int(torch.isfinite(native_logits).sum()),
            "top1": int(torch.argmax(native_logits)),
            "logits_sha256": tensor_sha256(native_logits),
            "vs_qdq": summary(native_logits, qdq_logits),
            "vs_fp32": summary(native_logits, fp32_logits),
        },
        "actual_layer_fixture": {
            "fqn": SELECTED_FQN,
            "contract": actual_layer,
            "fixture": "actual_conv_fixture.npz",
            "activation_sha256": tensor_sha256(selected["qx"]),
            "expected_output_sha256": tensor_sha256(expected),
            "expected_elements": int(expected.numel()),
            "expected_output_embedded": False,
        },
        "source_capture": {
            "linalg": "linalg.mlir",
            "weights": "weights.safetensors",
            "weights_manifest": "weights.safetensors.manifest.json",
            "opaque": sum(opaque.values()),
            "opaque_detail": opaque,
            "complete_graph": bool(converted.ok and not opaque),
            "ownership": "Merlin/model2MLIR capture; no TVM host, arena, call graph, or runner",
        },
        "layers": rows,
        "hardware_qualification": "NOT_RUN",
    }
    (out / "capture_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in (
        "status", "conv2d_count", "linear_count", "native_scalar_epilogue_eligible",
        "independent_reference", "source_capture")}, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Explicit accumulator-grid residuals expressed using ordinary contractions.

This is a prospective numerical policy, never an equivalence assertion about an
older QDQ graph. A selected site computes XW + integer_bias + integer_gain*R,
then performs one declared readout. Small output-column components bound the
additional diagonal work; no accelerator instruction or workload is selected.
Framework imports remain inside the selected capture interpreter.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

SCHEMA = "integer_accumulator_residual_v1"
_EXACT_F32 = 1 << 24


class ResidualError(ValueError):
    """The selected integer residual cannot be represented within its budget."""


def check(block: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"schema", "coefficient_rounding", "max_terms", "max_weight_elements", "max_mac_ratio"}
    if not isinstance(block, Mapping) or set(block) != fields or block["schema"] != SCHEMA:
        raise ResidualError("accumulator residual policy has an unknown schema or field")
    if block["coefficient_rounding"] not in ("exact", "half_even", "align_weight_grid"):
        raise ResidualError("residual coefficient policy must be exact, half_even or align_weight_grid")
    for key in ("max_terms", "max_weight_elements"):
        if type(block[key]) is not int or block[key] < 1:
            raise ResidualError("residual expansion limits must be positive integers")
    ratio = block["max_mac_ratio"]
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio < 1:
        raise ResidualError("residual MAC ratio limit must be finite and at least one")
    return dict(block)


def _f32(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ResidualError("selected scales must be positive finite numbers")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ResidualError("selected scale is outside f32") from error
    if not math.isfinite(result) or result <= 0:
        raise ResidualError("selected scale overflows or underflows f32")
    return result


def derive_plan(
    *,
    activation_scale: float,
    weight_scales: Sequence[float],
    skip_scale: float,
    output_scale: float,
    input_channels: int,
    component_width: int,
    weight_max: int,
    block: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive fixed-grid integer coefficients, exact f32 readouts and finite work.

    The half_even policy explicitly rounds the exact rational ratio of the
    represented scales. It does not claim that this preserves an old reference.
    """
    selected = check(block)
    if selected["coefficient_rounding"] == "align_weight_grid":
        raise ResidualError("aligned weight grids need the actual frozen floating weights")
    if any(type(v) is not int or v < 1 for v in (input_channels, component_width, weight_max)):
        raise ResidualError("residual geometry and coefficient range must be positive integers")
    sx, sr, so = map(_f32, (activation_scale, skip_scale, output_scale))
    if not weight_scales:
        raise ResidualError("residual plan has no complete output-channel scale vector")
    steps = [_f32(sx * _f32(sw)) for sw in weight_scales]
    coefficients, errors = [], []
    for step in steps:
        ratio = Fraction(sr) / Fraction(step)
        coefficient = round(ratio)
        if selected["coefficient_rounding"] == "exact" and ratio.denominator != 1:
            raise ResidualError("skip scale is not an integer multiple of its accumulator grid")
        if coefficient < 1:
            raise ResidualError("the selected coefficient policy would drop a nonzero residual")
        coefficients.append(coefficient)
        errors.append(float(Fraction(coefficient) - ratio))
    return _components(steps, coefficients, errors, so, input_channels, component_width, weight_max, selected)


def _components(steps, coefficients, errors, output_scale, inputs, width, weight_max, block):
    components, extra = [], 0
    channels = len(coefficients)
    for start in range(0, channels, width):
        stop = min(start + width, channels)
        terms = (max(coefficients[start:stop]) + weight_max - 1) // weight_max
        if terms > block["max_terms"]:
            raise ResidualError("integer skip decomposition exceeds the selected term budget")
        extra += (stop - start) ** 2 * terms
        components.append(
            {
                "start": start,
                "stop": stop,
                "terms": terms,
                "chunks": [
                    [min(weight_max, max(0, k - term * weight_max)) for k in coefficients[start:stop]]
                    for term in range(terms)
                ],
            }
        )
    original = inputs * channels
    if original + extra > block["max_weight_elements"] or (original + extra) / original > block["max_mac_ratio"]:
        raise ResidualError("component residual expansion exceeds the selected weight or MAC budget")
    return {
        "schema": SCHEMA,
        "accumulator_steps": steps,
        "coefficients": coefficients,
        "coefficient_errors": errors,
        "readout_gains": [_f32(step / output_scale) for step in steps],
        "components": components,
        "original_weight_elements": original,
        "augmented_weight_elements": original + extra,
        "mac_ratio": (original + extra) / original,
        "coefficient_rounding": block["coefficient_rounding"],
    }


def _attr(module, name):
    for part in str(name).split("."):
        module = getattr(module, part)
    return module


def _shape(node):
    import torch

    value = getattr(node, "meta", {}).get("val")
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise ResidualError("residual source needs observed f32 tensor shapes")
    if not value.shape or any(type(d) is not int or d < 1 for d in value.shape):
        raise ResidualError("residual source needs positive static extents")
    return tuple(value.shape)


def _fresh(node):
    import torch

    schema = getattr(getattr(node, "target", None), "_schema", None)
    returns = getattr(schema, "returns", ())
    return (
        isinstance(getattr(node, "target", None), torch._ops.OpOverload)
        and node.target.namespace == "aten"
        and not schema.is_mutable
        and len(returns) == 1
        and returns[0].alias_info is None
        and returns[0].type.kind() == "TensorType"
    )


def _arguments(node):
    """Bind only the actual operator schema, including its declared defaults."""
    schema = node.target._schema
    if len(node.args) > len(schema.arguments):
        raise ResidualError("contraction has extra positional operands")
    result = {}
    for i, arg in enumerate(schema.arguments):
        if i < len(node.args):
            if arg.name in node.kwargs:
                raise ResidualError("contraction repeats an operand")
            result[arg.name] = node.args[i]
        elif arg.name in node.kwargs:
            result[arg.name] = node.kwargs[arg.name]
        elif arg.has_default_value():
            result[arg.name] = arg.default_value
        else:
            raise ResidualError("contraction lacks a required operand")
    if set(node.kwargs) - set(result):
        raise ResidualError("contraction has unknown keyword operands")
    return result


def _geometry(node):
    """Restrict selection before observer ownership changes the graph."""
    import torch

    args = _arguments(node)
    if node.target == torch.ops.aten.conv2d.default:
        if (
            list(args["stride"]) != [1, 1]
            or list(args["padding"]) != [0, 0]
            or list(args["dilation"]) != [1, 1]
            or args["groups"] != 1
        ):
            raise ResidualError("residual convolution requires ungrouped unit-stride1x1 geometry")
        weight = args["weight"].meta.get("val")
        if weight is not None and tuple(weight.shape)[2:] != (1, 1):
            raise ResidualError("residual convolution requires a1x1 kernel")
    return args


def _tensor_pin(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _storage_shape(value):
    """Reopen observed code geometry, or its closed quantizer's f32 source."""
    import torch

    if not isinstance(value, torch.fx.Node):
        raise ResidualError("residual code storage is not an observed graph value")
    observed = value.meta.get("val")
    if observed is not None and observed.dtype == torch.int8:
        shape = tuple(observed.shape)
    elif (
        value.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
        and len(value.args) == 6
        and not value.kwargs
        and value.args[5] == torch.int8
        and type(value.args[2]) is int
        and value.args[2] == 0
        and list(value.args[3:5]) == [-128, 127]
    ):
        _f32(value.args[1])
        shape = _shape(value.args[0])
    else:
        raise ResidualError("residual input storage lacks observed signed-i8 geometry")
    if not shape or any(type(d) is not int or d < 1 for d in shape):
        raise ResidualError("residual input storage lacks static positive geometry")
    return shape


class AccumulatorResidual:
    """One explicitly selected quantizer's pattern, observer and rewrite owner."""

    def __init__(self, block):
        self.block = check(block)
        self.selected: dict[str, dict[str, Any]] = {}
        self.members: dict[str, str] = {}
        self.frozen: dict[str, tuple[Any, Any]] = {}
        self.rewritten: list[dict[str, Any]] = []
        self._selection_closed = False

    def pattern(self, node):
        import torch

        aten = torch.ops.aten
        if self._selection_closed and node.name not in self.selected:
            return []
        if node.target not in (aten.conv2d.default, aten.linear.default) or len(node.users) != 1:
            return []
        add = next(iter(node.users))
        if add.target not in (aten.add.Tensor, aten.add_.Tensor) or len(add.args) != 2:
            return []
        if self.members.get(add.name, node.name) != node.name:
            return []
        if (
            type(add.kwargs.get("alpha", 1)) not in (int, float)
            or add.kwargs.get("alpha", 1) != 1
            or set(add.kwargs) - {"alpha"}
            or sum(arg is node for arg in add.args) != 1
        ):
            return []
        if add.target == aten.add_.Tensor and (add.args[0] is not node or not _fresh(node)):
            return []
        skip = add.args[1] if add.args[0] is node else add.args[0]
        try:
            args = _geometry(node)
            if skip.op == "get_attr":
                return []
            shape = _shape(node)
            if _shape(skip) != shape:
                return []
            xshape = _shape(args["input"])
            if node.target == aten.linear.default:
                if len(shape) != 2 or len(xshape) != 2 or shape[0] != xshape[0]:
                    return []
            elif len(shape) != 4 or len(xshape) != 4 or shape[0] != xshape[0] or shape[2:] != xshape[2:]:
                return []
        except ResidualError:
            return []
        last = add
        if len(add.users) == 1:
            user = next(iter(add.users))
            if user.target in (aten.relu.default, aten.relu_.default) and user.args == (add,) and not user.kwargs:
                last = user
        if len(last.users) != 1:
            return []
        selected = {
            "node": node.name,
            "add": add.name,
            "last": last.name,
            "skip": skip.name,
            "shape": shape,
            "input_shape": xshape,
            "mutable": add.target == aten.add_.Tensor,
        }
        if self._selection_closed and self.selected[node.name] != selected:
            raise ResidualError("closed residual selection changed its source or reader topology")
        self.selected[node.name] = selected
        self.members[add.name] = node.name
        if last is not add:
            self.members[last.name] = node.name
        return [add] if last is add else [add, last]

    def owns(self, member):
        return member.name in self.members

    def close_selection(self):
        """Annotation may revisit admitted sites but cannot revive discarded ones."""
        self._selection_closed = True

    def discard(self, node):
        """Release ownership when the enclosing epilogue keeps a wide exit."""
        self.selected.pop(node.name, None)
        self.frozen.pop(node.name, None)
        self.members = {member: owner for member, owner in self.members.items() if owner != node.name}

    def annotation_inputs(self, member, activation_spec):
        site = self.selected.get(self.members.get(member.name))
        if site is None or member.name != site["add"]:
            return {}
        return {arg: activation_spec for arg in member.args if getattr(arg, "name", None) == site["skip"]}

    def freeze(self, prepared):
        """Snapshot actual float weights/bias after recovery, before conversion."""
        import torch

        if self.block["coefficient_rounding"] != "align_weight_grid":
            return
        nodes = {n.name: n for n in prepared.graph.nodes}
        for name in self.selected:
            node = nodes[name]

            def tensor(value):
                while isinstance(value, torch.fx.Node) and value.op == "call_module" and len(value.args) == 1:
                    value = value.args[0]
                if not isinstance(value, torch.fx.Node) or value.op != "get_attr":
                    raise ResidualError("aligned residual needs frozen original float parameters")
                data = _attr(prepared, value.target)
                if data.dtype != torch.float32 or not bool(torch.isfinite(data).all()):
                    raise ResidualError("aligned residual float parameters must be finite f32")
                return data.detach().cpu().clone()

            args = _arguments(node)
            weight = tensor(args["weight"])
            bias = args["bias"]
            self.frozen[name] = (weight, tensor(bias) if bias is not None else None)

    def rewrite_site(self, gm, node, last, site, epilogue_block):
        """Return None for another pattern; selected invalid patterns raise."""
        name = site.get("node", getattr(node, "name", None))
        selected = self.selected.get(name)
        if selected is None:
            return None
        if node is None or last is None or node.name != name or last.name != selected["last"]:
            raise ResidualError("selected residual source or complete readout is missing")
        return self._rewrite(gm, node, last, selected, epilogue_block)

    def _rewrite(self, gm, node, last, site, epilogue):
        import torch

        aten, qd = torch.ops.aten, torch.ops.quantized_decomposed
        dq = qd.dequantize_per_tensor.default
        args = _geometry(node)
        nodes = {n.name: n for n in gm.graph.nodes}
        add = nodes.get(site["add"])
        if (
            add is None
            or node.target not in (aten.linear.default, aten.conv2d.default)
            or len(node.users) != 1
            or next(iter(node.users)) is not add
            or add.target != (aten.add_.Tensor if site["mutable"] else aten.add.Tensor)
            or len(add.args) != 2
            or sum(a is node for a in add.args) != 1
            or type(add.kwargs.get("alpha", 1)) not in (int, float)
            or add.kwargs.get("alpha", 1) != 1
            or set(add.kwargs) - {"alpha"}
            or last.name != site["last"]
            or _shape(node) != site["shape"]
        ):
            raise ResidualError("selected residual arithmetic or reader topology changed")
        if last is not add and (
            last.target not in (aten.relu.default, aten.relu_.default)
            or last.args != (add,)
            or last.kwargs
            or list(add.users) != [last]
        ):
            raise ResidualError("selected residual ReLU topology changed")
        if site["mutable"] and (add.args[0] is not node or not _fresh(node)):
            raise ResidualError("selected mutable residual lost its fresh exclusive lhs")
        if len(last.users) != 1:
            raise ResidualError("selected residual must leave through exactly one quantize")
        # Moving the contraction to the sum waits for a later skip branch.
        # Only closed pure tensor operations may lie between the two reads.
        ordered = list(gm.graph.nodes)
        pure = {
            aten.linear.default,
            aten.conv2d.default,
            aten.relu.default,
            aten.reshape.default,
            aten.view.default,
            aten.slice.Tensor,
            aten.clone.default,
            aten.contiguous.default,
            aten.permute.default,
            aten.transpose.int,
            aten.add.Tensor,
            aten.mul.Tensor,
            qd.quantize_per_tensor.default,
            qd.dequantize_per_tensor.default,
            qd.quantize_per_channel.default,
            qd.dequantize_per_channel.default,
        }
        for between in ordered[ordered.index(node) + 1 : ordered.index(add)]:
            if between.op != "get_attr" and (between.op != "call_function" or between.target not in pure):
                raise ResidualError("residual scheduling crosses an unsupported effect or mutation")
        quantize = next(iter(last.users))
        if quantize.target != qd.quantize_per_tensor.default:
            raise ResidualError("selected residual has no literal per-tensor quantized output")
        skip = add.args[1] if add.args[0] is node else add.args[0]

        def scalar_dq(value):
            if (
                not isinstance(value, torch.fx.Node)
                or value.target != dq
                or len(value.args) != 6
                or value.args[2] != 0
                or type(value.args[2]) is not int
                or value.args[5] != torch.int8
                or list(value.args[3:5]) != [-128, 127]
                or set(value.kwargs) - {"out_dtype"}
                or value.kwargs.get("out_dtype") not in (None, torch.float32)
            ):
                raise ResidualError("selected residual needs symmetric signed-i8 per-tensor inputs")
            return _f32(value.args[1])

        sx, sr = scalar_dq(args["input"]), scalar_dq(skip)
        if (
            _storage_shape(args["input"].args[0]) != site["input_shape"]
            or _storage_shape(skip.args[0]) != site["shape"]
        ):
            raise ResidualError("residual input or skip code geometry changed")
        if (
            len(quantize.args) != 6
            or quantize.args[2] != 0
            or type(quantize.args[2]) is not int
            or quantize.args[5] != torch.int8
            or quantize.kwargs
            or list(quantize.args[3:5]) != (epilogue.get("requant") or {}).get("clamp")
        ):
            raise ResidualError("selected residual output does not match the declared integer readout")
        so = _f32(quantize.args[1])
        w_dq = args["weight"]
        if (
            not isinstance(w_dq, torch.fx.Node)
            or w_dq.target not in (dq, qd.dequantize_per_channel.default)
            or len(w_dq.args) != (6 if w_dq.target == dq else 7)
            or not isinstance(w_dq.args[0], torch.fx.Node)
            or w_dq.args[0].op != "get_attr"
            or w_dq.kwargs
            or w_dq.args[-1] != torch.int8
        ):
            raise ResidualError("selected residual needs frozen symmetric i8 weight codes")
        weight = _attr(gm, w_dq.args[0].target).detach().cpu()
        conv = node.target == aten.conv2d.default
        shape = tuple(weight.shape)
        if weight.dtype != torch.int8 or len(shape) != (4 if conv else 2) or (conv and shape[2:] != (1, 1)):
            raise ResidualError("residual contraction must be rank-two linear or ungrouped unit-stride1x1 conv")
        channels, inputs = shape[:2]
        if site["input_shape"][1] != inputs or site["shape"][1] != channels:
            raise ResidualError("residual contraction source geometry changed")
        zero = w_dq.args[2]
        zero = _attr(gm, zero.target) if isinstance(zero, torch.fx.Node) else torch.as_tensor(zero)
        low, high = w_dq.args[-3:-1]
        if (
            bool((zero != 0).any())
            or type(low) is not int
            or type(high) is not int
            or low not in (-128, -127)
            or high != 127
            or bool((weight < low).any())
            or zero.dtype not in (torch.int32, torch.int64)
            or (w_dq.target != dq and (type(w_dq.args[3]) is not int or w_dq.args[3] != 0))
        ):
            raise ResidualError("residual weight range, zero point or output-channel axis changed")
        scales = w_dq.args[1]
        scales = _attr(gm, scales.target) if isinstance(scales, torch.fx.Node) else torch.as_tensor(scales)
        scales = scales.reshape(-1).expand(channels) if scales.numel() == 1 else scales.reshape(-1)
        if scales.numel() != channels:
            raise ResidualError("residual weight scales do not cover all output channels")
        width = (epilogue.get("scale") or {}).get("columns_per_scale")
        if type(width) is not int or width < 1:
            raise ResidualError("component width must come from the declared physical readout")
        bias = args["bias"]
        if self.block["coefficient_rounding"] == "align_weight_grid":
            weight, codes, plan, original = self._aligned(node.name, sx, sr, so, inputs, channels, width, low, high)
        else:
            plan = derive_plan(
                activation_scale=sx,
                weight_scales=[float(s) for s in scales],
                skip_scale=sr,
                output_scale=so,
                input_channels=inputs,
                component_width=width,
                weight_max=high,
                block=self.block,
            )
            codes = self._bias(gm, bias, plan["accumulator_steps"], channels)
            original = None
        for part in plan["components"]:
            if len(set(plan["readout_gains"][part["start"] : part["stop"]])) != 1:
                raise ResidualError("residual component gains must share its declared physical readout grid")
        matrix = weight.reshape(channels, inputs).to(torch.int64)
        bounds = matrix.abs().sum(1) * 128 + codes.abs() + torch.tensor(plan["coefficients"]) * 128
        if int(bounds.max()) > _EXACT_F32:
            raise ResidualError("augmented contraction exceeds exact f32 carrier or i32 accumulator bounds")
        if not all(g * int(bounds[i]) <= 3.4028234663852886e38 for i, g in enumerate(plan["readout_gains"])):
            raise ResidualError("residual readout is not finite")
        graph, pieces = gm.graph, []
        tag = len(self.rewritten)

        def stamp(created):
            custom = dict(node.meta.get("custom") or {})
            origins = sorted(
                {
                    origin
                    for member in (node, add, last, skip)
                    for origin in (member.meta.get("custom") or {}).get("m2m_lineage", ())
                }
            )
            if origins:
                custom["m2m_lineage"] = origins
                custom["m2m_transform"] = SCHEMA
                created.meta["custom"] = custom
            return created

        def constant(label, value):
            name = f"_merlin_residual_{tag}_{label}"
            if hasattr(gm, name):
                raise ResidualError("residual constant owner already exists")
            gm.register_buffer(name, value)
            return stamp(graph.get_attr(name))

        with graph.inserting_before(add):
            for index, part in enumerate(plan["components"]):
                start, stop, terms = part["start"], part["stop"], part["terms"]
                span = stop - start
                diagonal = [torch.diag(torch.tensor(chunk, dtype=torch.int8)) for chunk in part["chunks"]]
                packed = torch.cat([matrix[start:stop].to(torch.int8), *diagonal], dim=1)
                if conv:
                    packed = packed.reshape(span, inputs + terms * span, 1, 1)
                kernel = constant(f"{index}_weight", packed)
                residual = stamp(graph.call_function(aten.slice.Tensor, (skip.args[0], 1, start, stop)))
                joined = stamp(graph.call_function(aten.cat.default, ([args["input"].args[0], *[residual] * terms], 1)))
                x_unit = stamp(graph.call_function(dq, (joined, 1.0, 0, -128, 127, torch.int8)))
                w_unit = stamp(graph.call_function(dq, (kernel, 1.0, 0, low, high, torch.int8)))
                b_unit = constant(f"{index}_bias", codes[start:stop].to(torch.float32))
                operands = (x_unit, w_unit, b_unit)
                if conv:
                    operands += (args["stride"], args["padding"], args["dilation"], args["groups"])
                value = stamp(graph.call_function(node.target, operands))
                if last is not add and (epilogue.get("activation") or {}).get("domain") == "accumulator":
                    value = stamp(graph.call_function(aten.relu.default, (value,)))
                gain = torch.tensor(plan["readout_gains"][start:stop], dtype=torch.float32)
                gain = gain.reshape(1, span, 1, 1) if conv else gain
                value = stamp(graph.call_function(aten.mul.Tensor, (value, constant(f"{index}_gain", gain))))
                if last is not add and (epilogue.get("activation") or {}).get("domain") != "accumulator":
                    value = stamp(graph.call_function(aten.relu.default, (value,)))
                pieces.append(value)
            value = pieces[0] if len(pieces) == 1 else stamp(graph.call_function(aten.cat.default, (pieces, 1)))
        # Only the complete replacement has the proved original output geometry.
        if "val" in last.meta:
            value.meta["val"] = last.meta["val"]
        quantize.args = (value, 1.0, 0, *quantize.args[3:])
        # add_/relu_ are intentionally effectful to generic dead-code removal.
        # Their only mutable storage here is the freshly returned contraction
        # (or fresh functional sum), with no alias/reader outside this chain.
        # Erase only these proven owned nodes after replacing their sole exit.
        retired = [last, add, node] if last is not add else [add, node]
        for old in retired:
            if old.users:
                raise ResidualError("retired residual retained an observable reader")
            graph.erase_node(old)
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()
        row = {
            **plan,
            "node": node.name,
            "status": "rewritten",
            "handled": True,
            "carrier_abs_bound": int(bounds.max()),
            "bias_codes": {"channels": channels},
            "device_multiplier": "f32(declared_accumulator_step / s_out)",
            "float_parameter_source": original,
        }
        rows = math.prod(site["shape"]) // channels
        row.update(
            original_macs=rows * plan["original_weight_elements"],
            augmented_macs=rows * plan["augmented_weight_elements"],
            original_weight_bytes=plan["original_weight_elements"],
            augmented_weight_bytes=plan["augmented_weight_elements"],
        )
        self.rewritten.append(row)
        return row

    @staticmethod
    def _bias(gm, bias, steps, channels):
        import torch

        if bias is None:
            return torch.zeros(channels, dtype=torch.int64)
        dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
        if (
            not isinstance(bias, torch.fx.Node)
            or bias.target not in (dq, torch.ops.quantized_decomposed.dequantize_per_channel.default)
            or len(bias.args) != (6 if bias.target == dq else 7)
            or bias.kwargs
            or bias.args[-1] != torch.int32
            or list(bias.args[-3:-1]) != [-(1 << 31), (1 << 31) - 1]
            or not isinstance(bias.args[0], torch.fx.Node)
            or bias.args[0].op != "get_attr"
            or (len(bias.args) == 7 and (type(bias.args[3]) is not int or bias.args[3] != 0))
        ):
            raise ResidualError("residual bias is not a frozen i32 accumulator-grid code")
        codes = _attr(gm, bias.args[0].target).detach().cpu()
        scale = bias.args[1]
        scale = _attr(gm, scale.target) if isinstance(scale, torch.fx.Node) else torch.as_tensor(scale)
        zero = bias.args[2]
        zero = _attr(gm, zero.target) if isinstance(zero, torch.fx.Node) else torch.as_tensor(zero)
        scale = scale.reshape(-1).expand(channels) if scale.numel() == 1 else scale.reshape(-1)
        if (
            codes.dtype != torch.int32
            or codes.numel() != channels
            or bool((zero != 0).any())
            or zero.dtype not in (torch.int32, torch.int64)
            or bool((codes == -(1 << 31)).any())
            or bool((codes == (1 << 31) - 1).any())
            or not torch.equal(scale.to(torch.float32), torch.tensor(steps, dtype=torch.float32))
        ):
            raise ResidualError("residual bias scale differs from its declared accumulator grid")
        return codes.to(torch.int64).reshape(-1)

    def _aligned(self, name, sx, sr, so, inputs, channels, width, low, high):
        import torch

        if name not in self.frozen:
            raise ResidualError("aligned residual requires freeze(prepared) before conversion")
        original, bias = self.frozen[name]
        if tuple(original.shape[:2]) != (channels, inputs):
            raise ResidualError("aligned residual float weight geometry changed")
        chosen = None
        for coefficient in range(high, 0, -1):
            step = _f32(sr / coefficient)
            scale = _f32(step / sx)
            # This bound is stronger than merely observing clamp unchanged on
            # calibration data: every frozen weight fits the selected grid.
            if bool((original.to(torch.float64) >= low * scale).all()) and bool(
                (original.to(torch.float64) <= high * scale).all()
            ):
                chosen = coefficient, step, scale
                break
        if chosen is None:
            raise ResidualError("no positive one-term skip grid represents the frozen float weights without clipping")
        coefficient, step, scale = chosen
        weight = torch.round(original / scale).to(torch.int8)
        quotient = None if bias is None else bias.to(torch.float64) / step
        if quotient is not None and (
            not bool(torch.isfinite(quotient).all()) or bool((quotient.abs() >= _EXACT_F32).any())
        ):
            raise ResidualError("aligned residual bias exceeds exact accumulator/carrier storage")
        codes = torch.zeros(channels, dtype=torch.int64) if quotient is None else torch.round(quotient).to(torch.int64)
        if codes.numel() != channels or bool((codes.abs() >= _EXACT_F32).any()):
            raise ResidualError("aligned residual bias exceeds exact accumulator/carrier storage")
        plan = _components(
            [step] * channels,
            [coefficient] * channels,
            [float(Fraction(coefficient) - Fraction(sr) / Fraction(step))] * channels,
            so,
            inputs,
            width,
            high,
            self.block,
        )
        plan.update(
            aligned_weight_scale=scale,
            input_weight_scale_product_f32=_f32(sx * scale),
            weight_granularity="tensor_aligned_to_skip",
        )
        return (
            weight,
            codes.reshape(-1),
            plan,
            {"weight_sha256": _tensor_pin(original), "bias_sha256": None if bias is None else _tensor_pin(bias)},
        )

    def census(self):
        return {
            "schema": SCHEMA,
            "patterns": len(self.selected),
            "rewritten": len(self.rewritten),
            "extra_contractions": sum(len(row["components"]) - 1 for row in self.rewritten),
            "sites": self.rewritten,
        }

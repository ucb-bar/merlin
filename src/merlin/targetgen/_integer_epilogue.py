#!/usr/bin/env python3
"""A recipe's ``epilogue`` block in TorchAO's own terms -- runs INSIDE the capture venv.

A target's output path (:mod:`merlin.targetgen.output_path`) says what its contraction unit does to
an accumulator on the way out: a value preloaded into the accumulator, a rectifier in the accumulator
or the scaled domain, a conversion, ONE scale per readout command covering ``columns_per_scale``
columns, a rounding, a clamp, a window maximum. :mod:`merlin.targetgen.quant_methodology` derives a
recipe's ``epilogue`` block from that path and the target's declared methodology. This module makes
the quantized graph compute exactly that arithmetic:

* :class:`Epilogue.annotate` puts a contraction and the stages after it that the path performs
  (rectifier, window maximum) under ONE PT2E annotation, in the path's order, with the bias as a
  ``DerivedQuantizationSpec`` (int32, scale ``s_x * s_w``) and the output quantized only where the
  pattern leaves through the requantizing exit;
* :meth:`Epilogue.rewrite`, after ``convert_pt2e``, restates each pattern as the hardware does it:
  integer operands (a dequantize of scale one), the accumulator plus the integer bias code, the
  rectifier, the one device multiplier ``f32(s_x * s_w / s_out)`` (or ``f32(s_x * s_w)`` at the
  full-width exit, where the host scales), and a quantize of scale one: round half to even, clamp.
  The fusion pass then reads back the very arithmetic the unit performs, so its exactness proof holds
  by construction rather than by luck of float rounding;
* the recovery steps the methodology asks for, generically: cross-layer equalization of contraction
  pairs joined only by positively homogeneous steps (:meth:`Epilogue.equalize`, before observers) and
  empirical bias correction (:meth:`Epilogue.track` during calibration, :meth:`Epilogue.correct_biases`
  before conversion).

Nothing here imports merlin, names a target or a model; every number comes from the recipe block.
A pattern the block cannot express is left in PT2E's portable form and listed with the reason.
An explicit logical per-channel policy retains independent source gains while
leaving physical output partitioning to the compiler and its execution gates.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Mapping
from typing import Any

SCHEMA = "integer_epilogue_v1"
#: Rounding modes the graph spells exactly: ``quantize_per_tensor`` rounds half to even.
ROUNDINGS = ("half_even",)
#: The exits a pattern can leave through (``output_path_v1`` ``exits``).
REQUANT, ACCUMULATOR = "requant", "accumulator"
#: Largest integer an f32 carrier holds exactly; a bias code beyond it cannot ride in the graph.
F32_EXACT = 1 << 24
_I32 = (-(1 << 31), (1 << 31) - 1)


class EpilogueError(ValueError):
    """The block cannot be realized; says which field and why."""


def check(block: Mapping[str, Any]) -> dict[str, Any]:
    """The validated ``epilogue`` block, or :class:`EpilogueError`."""
    if not isinstance(block, Mapping) or block.get("schema") != SCHEMA:
        raise EpilogueError(f"the epilogue block is not a {SCHEMA} mapping")
    scale = block.get("scale")
    if not isinstance(scale, Mapping) or scale.get("rounding") not in ROUNDINGS:
        raise EpilogueError("the epilogue scale states a rounding this graph spells exactly (half_even)")
    if "logical_granularity" in scale:
        if scale["logical_granularity"] != "per_channel" or scale.get("granularity") != "per_command":
            raise EpilogueError("logical per-channel scales require a configurable per-command readout")
        if type(scale.get("columns_per_scale")) is not int or scale["columns_per_scale"] < 1:
            raise EpilogueError("logical per-channel scales retain the positive physical scale-vector width")
    requant = block.get("requant")
    if requant is not None:
        clamp = requant.get("clamp")
        if (
            requant.get("rounding") not in ROUNDINGS
            or not isinstance(clamp, list)
            or len(clamp) != 2
            or not all(type(v) is int for v in clamp)
            or clamp[0] >= clamp[1]
        ):
            raise EpilogueError("the epilogue requant states a half_even rounding and an integer clamp [low, high]")
    activation = block.get("activation")
    if activation is not None and (
        activation.get("function") != "relu" or activation.get("domain") not in ("accumulator", "scaled")
    ):
        raise EpilogueError("the epilogue activation is a relu in the accumulator or the scaled domain")
    bias = block.get("bias") or {}
    if bias.get("mode") not in ("accumulator_preload", "host_float"):
        raise EpilogueError("the epilogue bias mode is accumulator_preload or host_float")
    exits = block.get("exits") or {}
    if set(exits) != {REQUANT, ACCUMULATOR} or not any(exits.values()):
        raise EpilogueError("the epilogue states both exits and at least one is open")
    if exits[REQUANT] and requant is None:
        raise EpilogueError("a requantizing exit needs the requant stage")
    return dict(block)


def _attr(module: Any, target: str) -> Any:
    value = module
    for part in str(target).split("."):
        value = getattr(value, part)
    return value


def _set_attr(module: Any, target: str, tensor: Any) -> None:
    import torch

    *owners, name = str(target).split(".")
    owner = module
    for part in owners:
        owner = getattr(owner, part)
    current = getattr(owner, name)
    if isinstance(current, torch.nn.Parameter):
        tensor = torch.nn.Parameter(tensor, requires_grad=False)
    if name in getattr(owner, "_parameters", {}):
        owner._parameters[name] = tensor
    elif name in getattr(owner, "_buffers", {}):
        owner._buffers[name] = tensor
    else:
        setattr(owner, name, tensor)


class Epilogue:
    """One recipe's epilogue: annotation, recovery and the post-conversion rewrite."""

    def __init__(self, block: Mapping[str, Any], *, activation_spec: Any, recovery: Mapping[str, Any] | None = None):
        import torch

        aten = torch.ops.aten
        self.block = check(block)
        self.activation = activation_spec
        self.recovery = dict(recovery or {})
        self.contractions = (aten.conv2d.default, aten.linear.default)
        self.relus = (aten.relu.default, aten.relu_.default)
        self.pools = (aten.max_pool2d.default,)
        self.views = (
            aten.flatten.using_ints,
            aten.view.default,
            aten.reshape.default,
            aten._unsafe_view.default,
            aten.contiguous.default,
            aten.dropout.default,
        )
        self.sites: list[dict[str, Any]] = []
        self.kept: list[dict[str, Any]] = []
        self.equalized: list[dict[str, Any]] = []
        self.bias_corrections: list[dict[str, Any]] = []
        self._means: dict[str, list[Any]] = {}
        self._handles: list[Any] = []

    # ------------------------------------------------------------------------------- annotation

    def _single(self, node: Any) -> Any:
        users = list(node.users)
        return users[0] if len(users) == 1 else None

    def _pool_padded(self, node: Any) -> bool:
        padding = node.args[3] if len(node.args) > 3 else node.kwargs.get("padding", 0)
        values = padding if isinstance(padding, (list, tuple)) else [padding]
        return any(int(v) != 0 for v in values)

    def pattern(self, node: Any) -> list[Any]:
        """The stages after ``node`` the output path performs, in the graph's (and the path's) order."""
        absorbed: list[Any] = []
        last = node
        nxt = self._single(last)
        if self.block.get("activation") is not None and nxt is not None and nxt.target in self.relus:
            absorbed.append(nxt)
            last = nxt
            nxt = self._single(last)
        pool = self.block.get("pool")
        if (
            pool is not None
            and "max" in (pool.get("reductions") or ())
            and nxt is not None
            and nxt.target in self.pools
        ):
            # A padded window needs the path's padding value; an undeclared one keeps the pool on the host.
            # The available full-width exit preserves the host chain before the reader quantizes it.
            if not (self._pool_padded(nxt) and pool.get("padding") in (None, "unknown")):
                absorbed.append(nxt)
        return absorbed

    def _terminal_readers(self, node: Any) -> list[tuple[Any, Any]]:
        """``(reader, edge_source)`` past shape-only views, stopping at every unabsorbed compute step."""
        found, pending = [], [node]
        while pending:
            value = pending.pop()
            for user in value.users:
                if user.op == "call_function" and user.target in self.views:
                    pending.append(user)
                else:
                    found.append((user, value))
        return found

    def exit_of(self, node: Any, last: Any, has_bias: bool, quantized_reader: Callable[[Any, Any], bool]) -> str:
        """``requant`` where every reader requantizes the value anyway, else the full-width exit if open."""
        exits = self.block["exits"]
        bias_ok = not has_bias or self.block["bias"]["mode"] == "accumulator_preload"
        if not exits[REQUANT] or not bias_ok:
            return ACCUMULATOR
        readers = self._terminal_readers(last)
        if readers and all(quantized_reader(reader, source) for reader, source in readers):
            return REQUANT
        return ACCUMULATOR if exits[ACCUMULATOR] else REQUANT

    def annotate(
        self,
        node: Any,
        weight_spec: Any,
        quantized_reader: Callable[[Any, Any], bool],
        owner: str | None,
    ) -> Any:
        """The PT2E annotation of ``node`` and its absorbed stages, as one pattern."""
        import torch
        from torchao.quantization.pt2e.quantizer import (
            DerivedQuantizationSpec,
            QuantizationAnnotation,
        )

        operand, kernel = node.args[:2]
        bias = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
        bias = bias if isinstance(bias, torch.fx.Node) and bias.op == "get_attr" else None
        absorbed = self.pattern(node)
        last = absorbed[-1] if absorbed else node
        exit_ = self.exit_of(node, last, bias is not None, quantized_reader)
        if exit_ == ACCUMULATOR:
            absorbed, last = [], node  # the raw accumulator leaves; the host applies what follows
        inputs: dict[Any, Any] = {operand: self.activation, kernel: weight_spec}
        if bias is not None and self.block["bias"]["mode"] == "accumulator_preload":
            per_channel = weight_spec.qscheme in (torch.per_channel_symmetric, torch.per_channel_affine)

            def derive(observers: list[Any]) -> tuple[Any, Any]:
                act, wt = observers
                s_x, _ = act.calculate_qparams()
                s_w, _ = wt.calculate_qparams()
                scale = (s_x.to(torch.float32).reshape(-1) * s_w.to(torch.float32).reshape(-1)).to(torch.float32)
                return scale, torch.zeros(scale.shape, dtype=torch.int32)

            inputs[bias] = DerivedQuantizationSpec(
                derived_from=[(operand, node), (kernel, node)],
                derive_qparams_fn=derive,
                dtype=torch.int32,
                quant_min=_I32[0],
                quant_max=_I32[1],
                qscheme=torch.per_channel_symmetric if per_channel else torch.per_tensor_symmetric,
                ch_axis=0 if per_channel else None,
            )
        output = self.activation if exit_ == REQUANT else None
        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=inputs, output_qspec=output if last is node else None, _annotated=True
        )
        for member in absorbed:
            member.meta["quantization_annotation"] = QuantizationAnnotation(
                output_qspec=output if member is last else None, _annotated=True
            )
        self.sites.append(
            {
                "node": node.name,
                "owner": owner,
                "operator": str(node.target),
                "stages": [str(m.target) for m in absorbed],
                "last": last.name,
                "exit": exit_,
                "bias": bias.target if bias is not None else None,
                "bias_mode": self.block["bias"]["mode"] if bias is not None else None,
            }
        )
        return node.meta["quantization_annotation"]

    # ------------------------------------------------------------------------------- recovery

    def _weight_attr(self, gm: Any, node: Any) -> Any:
        import torch

        weight = node.args[1] if len(node.args) > 1 else None
        if not isinstance(weight, torch.fx.Node) or weight.op != "get_attr" or len(weight.users) != 1:
            return None
        return weight

    def _homogeneous_path(self, start: Any) -> Any:
        """The single reader chain of ``start`` through steps that commute with a positive per-channel
        scale (relu, max pool, multiplication by a tensor not computed from it); its end, or None."""
        import torch

        aten = torch.ops.aten
        value = start
        for _ in range(8):
            reader = self._single(value)
            if reader is None or reader.op != "call_function":
                return None
            if reader.target in self.contractions:
                return reader if reader.args and reader.args[0] is value else None
            if reader.target in self.relus or reader.target in self.pools:
                value = reader
                continue
            if reader.target == aten.mul.Tensor and len(reader.args) == 2:
                other = reader.args[1] if reader.args[0] is value else reader.args[0]
                if not isinstance(other, torch.fx.Node) or _depends_on(other, start):
                    return None
                value = reader
                continue
            return None
        return None

    def equalize(self, gm: Any, eligible: Callable[[Any], bool], sweeps: int = 4) -> list[dict[str, Any]]:
        """Cross-layer equalization (Nagel et al., 2019) of every eligible contraction pair, in place.

        For ``c2(h(c1(x)))`` with ``h`` positively homogeneous per channel, output channel ``i`` of
        ``c1`` is divided by ``s_i = sqrt(r1_i / r2_i)`` (its bias too) and input channel ``i`` of
        ``c2`` multiplied by it, where ``r1``/``r2`` are the channel's weight ranges: the function is
        unchanged and both layers' per-channel ranges become ``sqrt(r1 * r2)``. New tensors replace
        the stored ones; the source module's parameters are never written.
        """
        import torch

        if not self.recovery.get("cross_layer_equalization"):
            return []
        pairs = []
        for c1 in gm.graph.nodes:
            if c1.op != "call_function" or c1.target not in self.contractions or not eligible(c1):
                continue
            if c1.target == torch.ops.aten.conv2d.default and (c1.args[6] if len(c1.args) > 6 else 1) != 1:
                continue
            c2 = self._homogeneous_path(c1)
            if c2 is None or not eligible(c2):
                continue
            if c2.target == torch.ops.aten.conv2d.default and (c2.args[6] if len(c2.args) > 6 else 1) != 1:
                continue
            if c1.target == torch.ops.aten.linear.default and c2.target != torch.ops.aten.linear.default:
                continue  # a flatten or reshape between them maps channels differently
            w1, w2 = self._weight_attr(gm, c1), self._weight_attr(gm, c2)
            if w1 is None or w2 is None:
                continue
            pairs.append((c1, c2, w1, w2))
        rows = {}
        for _ in range(max(1, int(sweeps))):
            for c1, c2, w1, w2 in pairs:
                a, b = _attr(gm, w1.target).detach(), _attr(gm, w2.target).detach()
                if a.shape[0] != b.shape[1]:
                    continue
                r1 = a.abs().reshape(a.shape[0], -1).amax(1).double()
                r2 = b.abs().transpose(0, 1).reshape(b.shape[1], -1).amax(1).double()
                ok = (r1 > 0) & (r2 > 0)
                s = torch.where(ok, torch.sqrt(r1 / torch.where(ok, r2, torch.ones_like(r2))), torch.ones_like(r1))
                view1 = [-1] + [1] * (a.dim() - 1)
                view2 = [1, -1] + [1] * (b.dim() - 2)
                _set_attr(gm, w1.target, (a.double() / s.reshape(view1)).to(a.dtype))
                _set_attr(gm, w2.target, (b.double() * s.reshape(view2)).to(b.dtype))
                bias = c1.args[2] if len(c1.args) > 2 else None
                if isinstance(bias, torch.fx.Node) and bias.op == "get_attr" and len(bias.users) == 1:
                    stored = _attr(gm, bias.target).detach()
                    _set_attr(gm, bias.target, (stored.double() / s).to(stored.dtype))
                row = rows.setdefault(c1.name, {"from": c1.name, "to": c2.name, "max_factor": 1.0, "min_factor": 1.0})
                row["max_factor"] = max(row["max_factor"], float(s.max()))
                row["min_factor"] = min(row["min_factor"], float(s.min()))
        self.equalized = list(rows.values())
        return self.equalized

    @contextlib.contextmanager
    def track(self, prepared: Any):
        """Record each epilogue site's mean input per channel while calibration runs (bias correction)."""
        if not self.recovery.get("bias_correction"):
            yield self
            return
        import torch

        by_name = {n.name: n for n in prepared.graph.nodes}
        for site in self.sites:
            node = by_name.get(site["node"])
            observer = node.args[0] if node is not None and node.args else None
            if not isinstance(observer, torch.fx.Node) or observer.op != "call_module":
                continue
            module = prepared.get_submodule(str(observer.target))
            axis = 1 if node.target == torch.ops.aten.conv2d.default else -1
            slot: list[Any] = [None, 0]
            self._means[site["node"]] = slot

            def hook(_module: Any, args: tuple, _out: Any, _slot: list = slot, _axis: int = axis) -> None:
                x = args[0].detach().to(torch.float64)
                moved = x.movedim(_axis, -1).reshape(-1, x.shape[_axis])
                total = moved.sum(0)
                _slot[0] = total if _slot[0] is None else _slot[0] + total
                _slot[1] += moved.shape[0]

            self._handles.append(module.register_forward_hook(hook))
        try:
            yield self
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()

    def correct_biases(self, prepared: Any) -> list[dict[str, Any]]:
        """Empirical bias correction: ``b -= (Q(W) - W) E[x]``, before conversion quantizes the bias."""
        import torch

        if not self.recovery.get("bias_correction"):
            return []
        by_name = {n.name: n for n in prepared.graph.nodes}
        for site in self.sites:
            slot = self._means.get(site["node"])
            node = by_name.get(site["node"])
            row = {"node": site["node"]}
            if slot is None or slot[0] is None or node is None:
                self.bias_corrections.append({**row, "status": "kept", "why": "no observed input mean"})
                continue
            bias_obs = node.args[2] if len(node.args) > 2 else None
            bias = (
                bias_obs.args[0] if isinstance(bias_obs, torch.fx.Node) and bias_obs.op == "call_module" else bias_obs
            )
            if not isinstance(bias, torch.fx.Node) or bias.op != "get_attr":
                self.bias_corrections.append({**row, "status": "kept", "why": "the contraction has no stored bias"})
                continue
            w_obs = node.args[1]
            if not isinstance(w_obs, torch.fx.Node) or w_obs.op != "call_module":
                self.bias_corrections.append({**row, "status": "kept", "why": "the weight has no observer"})
                continue
            observer = prepared.get_submodule(str(w_obs.target))
            weight = _attr(prepared, w_obs.args[0].target).detach().to(torch.float64)
            scale, _ = observer.calculate_qparams()
            scale = scale.to(torch.float64).reshape([-1] + [1] * (weight.dim() - 1))
            if scale.numel() == 1:
                scale = scale.reshape([1] * weight.dim())
            codes = torch.clamp(torch.round(weight / scale), observer.quant_min, observer.quant_max)
            error = codes * scale - weight
            mean = slot[0] / max(slot[1], 1)
            if error.dim() == 4:
                delta = (error.sum((2, 3)) * mean.reshape(1, -1)).sum(1)
            else:
                delta = error @ mean
            stored = _attr(prepared, bias.target).detach()
            _set_attr(prepared, bias.target, (stored.to(torch.float64) - delta).to(stored.dtype))
            self.bias_corrections.append(
                {**row, "status": "corrected", "max_abs_shift": float(delta.abs().max()), "samples": int(slot[1])}
            )
        return self.bias_corrections

    # ------------------------------------------------------------------------------- the rewrite

    def rewrite(self, gm: Any) -> list[dict[str, Any]]:
        """Each converted pattern as the output path computes it, in place; every site reported."""
        import torch

        qd = torch.ops.quantized_decomposed
        dq_t, dq_c = qd.dequantize_per_tensor.default, qd.dequantize_per_channel.default
        q_t = qd.quantize_per_tensor.default
        graph = gm.graph
        by_name = {n.name: n for n in graph.nodes}
        done = []
        for index, site in enumerate(self.sites):
            node = by_name.get(site["node"])
            try:
                if node is None:
                    raise EpilogueError("the contraction is not in the converted graph")
                row = self._rewrite_site(gm, node, by_name[site["last"]], site, index, dq_t, dq_c, q_t)
            except (EpilogueError, KeyError) as why:
                self.kept.append({**site, "status": "kept", "why": str(why)})
                continue
            done.append({**site, **row, "status": "rewritten"})
        if done:
            graph.eliminate_dead_code()
            graph.lint()
            gm.recompile()
        self.sites = done + self.kept
        return self.sites

    def _rewrite_site(self, gm, node, last, site, index, dq_t, dq_c, q_t) -> dict[str, Any]:
        import torch

        graph = gm.graph
        x_dq, w_dq = node.args[0], node.args[1]
        if not (isinstance(x_dq, torch.fx.Node) and x_dq.target == dq_t):
            raise EpilogueError("the activation is not a per-tensor dequantized integer")
        if not (isinstance(w_dq, torch.fx.Node) and w_dq.target in (dq_t, dq_c)):
            raise EpilogueError("the weight is not a dequantized integer")
        s_x = x_dq.args[1]
        s_x = float(_attr(gm, s_x.target)) if isinstance(s_x, torch.fx.Node) else float(s_x)
        if x_dq.args[2] != 0:
            raise EpilogueError("the activation has a zero point the path does not apply")
        s_w = w_dq.args[1]
        s_w = (_attr(gm, s_w.target) if isinstance(s_w, torch.fx.Node) else torch.tensor(float(s_w))).detach()
        zero = w_dq.args[2]
        zero = _attr(gm, zero.target) if isinstance(zero, torch.fx.Node) else torch.as_tensor(zero)
        if bool((zero != 0).any()):
            raise EpilogueError("the weight has a zero point the path does not apply")
        if w_dq.target == dq_c and w_dq.args[3] != 0:
            raise EpilogueError("the weight's quantization axis is not the output-channel axis")
        channels = int(node.meta["val"].shape[1 if node.target == torch.ops.aten.conv2d.default else -1])
        s_w = s_w.to(torch.float64).reshape(-1).expand(channels) if s_w.numel() == 1 else s_w.to(torch.float64)
        if s_w.numel() != channels:
            raise EpilogueError("the weight scale does not run along the output channels")
        accumulator_step = torch.tensor(s_x, dtype=torch.float64) * s_w
        bias = node.args[2] if len(node.args) > 2 else None
        codes, host_bias = None, None
        if isinstance(bias, torch.fx.Node) and bias.op == "get_attr" and self.block["bias"]["mode"] == "host_float":
            if site["exit"] != ACCUMULATOR:
                raise EpilogueError("a host bias cannot be added inside a requantizing exit")
            host_bias = _attr(gm, bias.target).detach().to(torch.float32).reshape(-1)
        elif isinstance(bias, torch.fx.Node) and bias.target in (dq_t, dq_c):
            raw = _attr(gm, bias.args[0].target).detach()
            if raw.dtype != torch.int32:
                raise EpilogueError("the bias is not an int32 code")
            low, high = bias.args[-3], bias.args[-2]
            if bool((raw <= low).any()) or bool((raw >= high).any()):
                raise EpilogueError("the bias code reaches the i32 bound, so it may have been clamped")
            scale = bias.args[1]
            scale = (
                _attr(gm, scale.target) if isinstance(scale, torch.fx.Node) else torch.tensor(float(scale))
            ).detach()
            expected = (torch.tensor(s_x, dtype=torch.float32) * s_w.to(torch.float32)).reshape(-1)
            if not torch.equal(scale.to(torch.float32).reshape(-1).expand_as(expected), expected):
                raise EpilogueError("the bias scale is not the accumulator step s_x * s_w")
            if int(raw.abs().max()) >= F32_EXACT:
                raise EpilogueError("a bias code exceeds the integers an f32 carrier holds exactly")
            codes = raw.to(torch.int64).reshape(-1)
        elif bias is not None:
            raise EpilogueError("the bias is not on the accumulator grid")
        exact_bound = None
        if codes is not None:
            # The portable PT2E contraction carries its accumulator and bias in f32.
            # Prove that adding the bias there is the same as preloading it in i32:
            # every partial sum, and the biased sum, must be exactly representable.
            # Checking the bias alone misses rounding above 2**24.
            if w_dq.args[0].op != "get_attr":
                raise EpilogueError("the weight codes are not frozen for an accumulator range proof")
            weight_codes = _attr(gm, w_dq.args[0].target).detach().to(torch.int64)
            operand_bound = max(abs(int(x_dq.args[3])), abs(int(x_dq.args[4])))
            channel_bounds = weight_codes.abs().reshape(channels, -1).sum(1) * operand_bound + codes.abs()
            exact_bound = int(channel_bounds.max())
            if exact_bound > F32_EXACT:
                raise EpilogueError("the biased accumulator exceeds the exact integer range of its f32 carrier")
        quantize = None
        if site["exit"] == REQUANT:
            users = list(last.users)
            if len(users) != 1 or users[0].target != q_t:
                raise EpilogueError("the requantizing exit is not one quantize")
            quantize = users[0]
            s_out, zero = float(quantize.args[1]), quantize.args[2]
            if zero != 0:
                raise EpilogueError("the output has a zero point the path does not apply")
            multiplier = accumulator_step / s_out
        else:
            multiplier = accumulator_step
        multiplier = multiplier.to(torch.float32)
        columns = (self.block.get("scale") or {}).get("columns_per_scale")
        if self.block["scale"].get("logical_granularity") != "per_channel" and isinstance(columns, int) and columns > 0:
            for start in range(0, channels, columns):
                if torch.unique(multiplier[start : start + columns]).numel() != 1:
                    raise EpilogueError(f"the device multiplier differs within a group of {columns} columns")
        if not bool(torch.isfinite(multiplier).all()) or not bool((multiplier > 0).all()):
            raise EpilogueError("the device multiplier is not positive and finite")
        conv = node.target == torch.ops.aten.conv2d.default
        shape = [1, channels, 1, 1] if conv else [channels]
        custom = dict(node.meta.get("custom") or {})

        def stamp(created: Any, source: Any = node) -> Any:
            metadata = dict(source.meta.get("custom") or custom)
            if metadata.get("m2m_lineage"):
                metadata["m2m_transform"] = "integer_epilogue"
                created.meta["custom"] = metadata
            for key in ("nn_module_stack", "source_fn_stack", "stack_trace"):
                if key in source.meta:
                    created.meta[key] = source.meta[key]
            return created

        def constant(name: str, tensor: Any) -> Any:
            target = f"_merlin_epilogue_{index}_{name}"
            gm.register_buffer(target, tensor)
            with graph.inserting_before(node):
                return stamp(graph.get_attr(target))

        x_int, w_int = x_dq.args[0], w_dq.args[0]
        with graph.inserting_before(node):
            x_unit = stamp(graph.call_function(dq_t, (x_int, 1.0, 0, *x_dq.args[3:6])), x_dq)
            w_low, w_high, w_dtype = w_dq.args[-3], w_dq.args[-2], w_dq.args[-1]
            w_unit = stamp(graph.call_function(dq_t, (w_int, 1.0, 0, w_low, w_high, w_dtype)), w_dq)
        args = list(node.args)
        args[0], args[1] = x_unit, w_unit
        if codes is not None:
            args[2] = constant("bias", codes.to(torch.float32))
        elif host_bias is not None:
            args[2] = None  # the raw accumulator leaves; the host scales it, then adds the float bias
        node.args = tuple(args)
        scale_node = constant("scale", multiplier.reshape(shape))
        relu_first = (self.block.get("activation") or {}).get("domain") == "accumulator"
        absorbed = [n for n in _chain(node, last)]
        relus = [n for n in absorbed if n.target in self.relus]
        anchor = relus[-1] if relus and relu_first else node
        users = [u for u in anchor.users]
        with graph.inserting_after(anchor):
            scaled = stamp(graph.call_function(torch.ops.aten.mul.Tensor, (anchor, scale_node)))
        if host_bias is not None:
            bias_node = constant("host_bias", host_bias.reshape(shape))
            with graph.inserting_after(scaled):
                scaled = stamp(graph.call_function(torch.ops.aten.add.Tensor, (scaled, bias_node)))
        for user in users:
            user.replace_input_with(anchor, scaled)
        if quantize is not None:
            low, high = (self.block.get("requant") or {}).get("clamp") or quantize.args[3:5]
            quantize.args = (quantize.args[0], 1.0, 0, int(low), int(high), quantize.args[5])
        return {
            "device_multiplier": "f32(s_x * s_w / s_out)" if quantize is not None else "f32(s_x * s_w)",
            "bias_codes": None
            if codes is None
            else {
                "channels": int(codes.numel()),
                "max_abs": int(codes.abs().max()),
                "accumulator_abs_bound": exact_bound,
            },
            "multipliers_distinct": int(torch.unique(multiplier).numel()),
        }

    def finish(self, gm: Any, island: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """:meth:`rewrite` the converted graph, then hold its float island to fp32."""
        self.rewrite(gm)
        return check_float_island(gm, island)

    def census(self) -> dict[str, Any]:
        rewritten = [s for s in self.sites if s.get("status") == "rewritten"]
        return {
            "schema": SCHEMA,
            "patterns": len(self.sites),
            "rewritten": len(rewritten),
            "by_exit": {e: sum(1 for s in rewritten if s["exit"] == e) for e in (REQUANT, ACCUMULATOR)},
            "absorbing": {
                "relu": sum(1 for s in rewritten if any("relu" in t for t in s["stages"])),
                "pool": sum(1 for s in rewritten if any("pool" in t for t in s["stages"])),
                "bias": sum(1 for s in rewritten if s.get("bias_codes")),
            },
            "kept": [s for s in self.sites if s.get("status") != "rewritten"],
            "equalized_pairs": self.equalized,
            "bias_corrections": {
                "corrected": sum(1 for r in self.bias_corrections if r["status"] == "corrected"),
                "kept": [r for r in self.bias_corrections if r["status"] != "corrected"],
            },
        }


def _chain(start: Any, last: Any) -> Iterable[Any]:
    """The single-user nodes from ``start`` (exclusive) through ``last``."""
    node = start
    while node is not last:
        users = list(node.users)
        if len(users) != 1:
            return
        node = users[0]
        yield node


def _depends_on(node: Any, ancestor: Any, budget: int = 4096) -> bool:
    import torch

    pending, seen = [node], set()
    while pending and budget:
        value = pending.pop()
        budget -= 1
        if value is ancestor:
            return True
        if value in seen:
            continue
        seen.add(value)
        pending.extend(a for a in value.all_input_nodes if isinstance(a, torch.fx.Node))
    return bool(pending)  # out of budget: assume it may, and refuse the pair


#: Framework-neutral names of operations a float island keeps in fp32, and the ATen spellings.
ISLAND_OPS = {
    "softmax": ("_softmax.default", "softmax.int", "_safe_softmax.default"),
    "layer_norm": ("layer_norm.default", "native_layer_norm.default"),
    "rms_norm": ("rms_norm.default", "rsqrt.default"),
    "group_norm": ("group_norm.default", "native_group_norm.default"),
    "gelu": ("gelu.default",),
    "silu": ("silu.default",),
    "rotary": ("cos.default", "sin.default"),
}


def check_float_island(gm: Any, island: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Every island operation computes in at least fp32: a bf16, f16 or narrower value is refused.

    The model's own wider arithmetic (an f64 sinusoid table) is recorded, not narrowed or refused.
    """
    if not island:
        return None
    import torch

    if island.get("dtype") != "fp32":
        raise EpilogueError("a float island computes in fp32")
    spellings = {spelling: name for name in island.get("ops") or () for spelling in ISLAND_OPS.get(str(name), ())}
    unknown = [name for name in island.get("ops") or () if name not in ISLAND_OPS]
    counts: dict[str, int] = {}
    wider: dict[str, int] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        name = spellings.get(str(node.target).removeprefix("aten."))
        if name is None:
            continue
        value = node.meta.get("val")
        dtype = getattr(value, "dtype", None)
        if dtype is not None and dtype.is_floating_point and dtype != torch.float32:
            if dtype.itemsize < torch.float32.itemsize:
                raise EpilogueError(f"float island {name} computes in {dtype} at {node.name}, not fp32")
            wider[name] = wider.get(name, 0) + 1
        counts[name] = counts.get(name, 0) + 1
    return {
        "dtype": "fp32",
        "operations": dict(sorted(counts.items())),
        "wider_than_fp32": dict(sorted(wider.items())),
        "unknown_names": unknown,
    }

"""GPTQ rounding of a recipe's stored integer weights. It runs INSIDE the capture venv.

Round-to-nearest (RTN) puts each weight on the recipe's grid on its own, so the rounding errors of
the weights that read correlated inputs add up in the layer's output. GPTQ (Frantar et al. 2022,
https://arxiv.org/abs/2210.17323) picks the integer codes column by column on the SAME grid. After
each column is rounded, its error is pushed onto the columns not yet rounded, weighted by the
inverse of the inputs' second moment, so that the layer's output error stays small. For each layer
it minimises ``||X W^T - X Q^T||^2`` over the calibration inputs ``X`` that reach that layer.
Measured on a vision-language-action model with per-channel int8 weights and float activations: RTN
weights gave 3.4x the float noise floor, GPTQ weights 1.4x.

WHAT A RECIPE SAYS. ``weight.rounding`` is ``"nearest"`` (the default, also when the key is absent)
or ``"gptq"``. With ``"gptq"``, ``weight.gptq`` records :data:`PARAMETERS`:

* ``calibration_samples``: how many leading samples of the capture's calibration stream feed the
  Hessians. The activation observers still see the whole stream, as they do without this option;
* ``act_order``: round the columns in decreasing order of their input energy (``diag(H)``);
* ``damp``: ``damp * mean(diag(H))`` is added to the diagonal before inverting;
* ``block``: how many columns are updated together (GPTQ's lazy batch). It changes only the order
  in which floating-point updates accumulate, which is why it is recorded.

All four are part of the recipe's content digest, so a capture made under GPTQ codes is never served
for one made under RTN codes.

THE ALGORITHM (:func:`gptq_codes`), standard GPTQ, in float64. For one stored weight seen as a
matrix ``W`` of ``[outputs, reduction]``:

* ``H = 2/n * sum(X^T X)`` over the ``n`` calibration rows of the contraction's input
  (:class:`HessianSum`). A linear's rows are its input tokens. A convolution's rows are its im2col
  patches, in the ``(channel, kh, kw)`` order of ``weight.reshape(out, -1)``. A matmul against a
  stored ``[K, N]`` operand is the linear ``W = operand^T``;
* columns whose input never moved during calibration (``H_jj = 0``) carry no evidence. They get a
  unit diagonal, which decouples them, and are rounded to nearest. The reference implementation
  zeroes them instead, which would change the layer for inputs the calibration did not show;
* act order is a STABLE sort, so equal energies keep their column order;
* after dampening, the upper Cholesky factor of ``H^-1`` drives the column loop. Column ``j`` is
  rounded on its own scales, ``q = clamp(round(w / s), qmin, qmax)``, and the error
  ``(w - q*s) / U_jj`` is subtracted from the remaining columns of the block. Each finished block
  propagates its errors to the columns after it in one product.

THE SCALES ARE NOT GPTQ'S TO CHOOSE. GPTQ only picks integers on a grid that is already fixed. On
the capture path the grid is the one the recipe's weight observer computes (TorchAO's
``calculate_qparams``, including its epsilon floor and the symmetric ``[-127, 127]`` range of a
restricted-range recipe). The granularity is therefore whatever the recipe declares and the observer
realises: one scale per tensor, or one per output channel. :func:`gptq_codes` takes any scale grid
that broadcasts against ``W``. :func:`symmetric_scales` builds the RTN grid for experiments: per
tensor, per output channel, or per block of ``block`` elements along either axis, where the block
length comes from the caller (a recipe's ``weight.block``) and never from this file.

WHERE THE CODES ENTER THE CAPTURED PROGRAM (:class:`PT2EWeightRounding`). With
``fold_quantize=True``, ``convert_pt2e`` replaces each weight observer by ``quantize_per_*`` of the
stored floating weight under the observer's ``calculate_qparams()``, and constant-folds the result
into the int8 ``_frozen_param`` the program carries, followed by a ``dequantize_per_*``. The hook
runs between calibration and convert:

1. while the calibration stream runs, forward pre-hooks on the ACTIVATION observers that feed each
   weight contraction accumulate ``H`` (only for the first ``calibration_samples`` samples). The
   input of an activation observer is the float tensor the contraction reads, and the same pass
   also calibrates the activation ranges, so nothing is run twice;
2. after calibration, the weight observer's own ``calculate_qparams()`` gives the scales. GPTQ
   picks the codes on that grid, and the stored floating weight that the observer's ``get_attr``
   reads is replaced by ``codes * scale``. The observer's state is left alone;
3. ``convert_pt2e`` then computes ``round(codes*s / s) = codes``. This is checked here with the same
   ``quantized_decomposed`` op before returning, and the capture is refused if it fails. The int8
   weights and scales of the converted program, which become Merlin's INT8 reference, are exactly
   GPTQ's.

The replacement is made on the prepared module's own submodule. ``torch.export``'s unlifted module
holds the source model's parameter objects, so a parameter is re-registered, never written in place,
and the source model keeps its weights. Two alternatives were rejected. A custom weight observer or
fake-quant only returns qparams: convert computes the codes from the stored weight, so an observer
cannot carry GPTQ's codes without forking convert. Rewriting the converted ``_frozen_param`` would
mean editing a folded constant whose identity and lineage the capture library owns, after its scale
has already been emitted.

A contraction is LEFT AT NEAREST, with the reason recorded in the report, when GPTQ is not defined
for it here: a grouped convolution, a batched stored operand, a weight read by something other than
its contraction, a weight observed separately by several contractions, a layer no calibration
sample reached, or a damped Hessian that is not positive definite.

DETERMINISM. There is no randomness: the order is fixed by the graph, the sample order and the stable
sort, and everything runs in float64. A recapture with the same inputs, the same framework build and
the same math-library threading reproduces the codes bit for bit. Those are the same conditions under
which the capture's own float forward pass reproduces.

Memory: one ``cols x cols`` float64 Hessian per distinct (input, layout) pair is held during
calibration.

This module imports neither merlin nor torch at import time. The capture venv imports it as a
sibling of ``_recipe_quantizer``, and the derivation side reads its constants as
``merlin.targetgen._gptq``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

#: ``weight.rounding`` values. Absent means :data:`NEAREST`.
NEAREST = "nearest"
GPTQ = "gptq"
#: The GPTQ parameters a recipe records under ``weight.gptq``, all of them, always.
PARAMETERS = ("calibration_samples", "act_order", "damp", "block")
#: The producer's defaults. ``damp`` and ``block`` are the paper's. Sixteen calibration samples with
#: act order is the measured choice: it reached 1.4x the float noise floor where RTN reached 3.4x.
DEFAULT_PARAMETERS = {"calibration_samples": 16, "act_order": True, "damp": 0.01, "block": 128}
#: ``symmetric_scales`` block axes: a run of output channels sharing one scale, or a run of reduced
#: elements within an output channel (the recipe's block format, see ``quant_layer_plan``).
BLOCK_AXES = ("output", "reduction")


class GPTQError(ValueError):
    """GPTQ cannot be applied as asked. The message says which input and why."""


def check_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """``parameters`` validated and in :data:`PARAMETERS` order. Raises :class:`GPTQError`."""
    keys = set(parameters)
    if keys != set(PARAMETERS):
        raise GPTQError(
            f"GPTQ parameters are exactly {list(PARAMETERS)}; missing {sorted(set(PARAMETERS) - keys)}, "
            f"unknown {sorted(keys - set(PARAMETERS))}"
        )
    samples = parameters["calibration_samples"]
    if type(samples) is not int or samples <= 0:
        raise GPTQError(f"calibration_samples must be a positive integer, not {samples!r}")
    act_order, damp, block = _check_algorithm(parameters["act_order"], parameters["damp"], parameters["block"])
    return {"calibration_samples": samples, "act_order": act_order, "damp": damp, "block": block}


def _check_algorithm(act_order: Any, damp: Any, block: Any) -> tuple[bool, float, int]:
    if type(act_order) is not bool:
        raise GPTQError(f"act_order must be true or false, not {act_order!r}")
    if type(damp) not in (int, float) or not math.isfinite(damp) or damp <= 0:
        raise GPTQError(f"damp must be a finite positive number, not {damp!r}")
    if type(block) is not int or block <= 0:
        raise GPTQError(f"block must be a positive integer, not {block!r}")
    return act_order, float(damp), block


def rounding_of(weight: Mapping[str, Any]) -> dict[str, Any] | None:
    """The GPTQ parameters a recipe's weight spec asks for, or ``None`` for round-to-nearest."""
    rounding = weight.get("rounding", NEAREST)
    if rounding == NEAREST:
        if "gptq" in weight:
            raise GPTQError("the weight spec records GPTQ parameters but rounds to nearest")
        return None
    if rounding != GPTQ:
        raise GPTQError(f"weight rounding {rounding!r} is not one this build realises ({NEAREST!r} or {GPTQ!r})")
    parameters = weight.get("gptq")
    if not isinstance(parameters, Mapping):
        raise GPTQError("weight rounding 'gptq' needs its parameters under weight.gptq")
    return check_parameters(parameters)


# --- the algorithm ---------------------------------------------------------------------------------


def symmetric_scales(
    matrix: Any,
    *,
    granularity: str,
    quant_min: int,
    quant_max: int,
    block: int | None = None,
    block_axis: str | None = None,
    eps: float | None = None,
) -> Any:
    """Round-to-nearest's scales for ``matrix`` ``[outputs, reduction]``, broadcastable against it.

    Each scale is ``max|w| / ((quant_max - quant_min) / 2)`` over its group, which is TorchAO's
    symmetric observer formula, floored at ``eps`` (default: the smallest normal float32).
    ``granularity`` is ``"tensor"``, ``"channel"`` (one scale per output row) or ``"block"``: one
    scale per ``block`` consecutive outputs (``block_axis="output"``, a group of output channels
    scaled together) or per ``block`` consecutive reduced elements of one output
    (``block_axis="reduction"``). A reduction that is not a whole number of blocks is refused, as
    ``quant_layer_plan`` refuses it.
    """
    import torch

    magnitude = matrix.detach().to(torch.float32).abs()
    if magnitude.dim() != 2:
        raise GPTQError(f"scales are built for a [outputs, reduction] matrix, not shape {list(matrix.shape)}")
    rows, cols = magnitude.shape
    if granularity == "tensor":
        peak = magnitude.amax().reshape(1, 1)
    elif granularity == "channel":
        peak = magnitude.amax(dim=1, keepdim=True)
    elif granularity == "block":
        if type(block) is not int or block <= 0:
            raise GPTQError(f"a block scale needs a positive block length, not {block!r}")
        if block_axis == "output":
            groups = -(-rows // block)
            per_row = torch.nn.functional.pad(magnitude.amax(dim=1), (0, groups * block - rows))
            peak = per_row.reshape(groups, block).amax(dim=1).repeat_interleave(block)[:rows].reshape(rows, 1)
        elif block_axis == "reduction":
            if cols % block:
                raise GPTQError(f"a reduction of {cols} is not a whole number of blocks of {block}")
            peak = magnitude.reshape(rows, cols // block, block).amax(dim=2).repeat_interleave(block, dim=1)
        else:
            raise GPTQError(f"a block scale runs along one of {list(BLOCK_AXES)}, not {block_axis!r}")
    else:
        raise GPTQError(f"granularity {granularity!r} is not 'tensor', 'channel' or 'block'")
    floor = torch.finfo(torch.float32).tiny if eps is None else eps
    return torch.clamp(peak / (float(quant_max - quant_min) / 2), min=floor)


def nearest_codes(matrix: Any, scales: Any, *, quant_min: int, quant_max: int) -> Any:
    """Round-to-nearest codes of ``matrix`` on ``scales`` (half to even, like the framework's quantize)."""
    import torch

    _check_range(quant_min, quant_max)
    values = matrix.detach().to(torch.float64) / scales.to(torch.float64)
    return torch.clamp(torch.round(values), quant_min, quant_max).to(torch.int8)


def gptq_codes(
    matrix: Any,
    hessian: Any,
    scales: Any,
    *,
    quant_min: int,
    quant_max: int,
    damp: float,
    act_order: bool,
    block: int,
) -> tuple[Any, dict[str, Any]]:
    """GPTQ's int8 codes for ``matrix`` ``[outputs, reduction]`` on the fixed grid ``scales``.

    ``hessian`` is ``[reduction, reduction]`` (:class:`HessianSum`), and ``scales`` broadcasts
    against ``matrix``. Returns ``(codes, info)``: ``codes`` is ``torch.int8`` shaped like
    ``matrix``, and ``info`` records the columns rounded without evidence (``dead_columns``) and the
    dampening added.
    """
    import torch

    _check_range(quant_min, quant_max)
    act_order, damp, block = _check_algorithm(act_order, damp, block)
    weight = matrix.detach().to(torch.float64).clone()
    if weight.dim() != 2:
        raise GPTQError(f"GPTQ rounds a [outputs, reduction] matrix, not shape {list(matrix.shape)}")
    rows, cols = weight.shape
    h = hessian.detach().to(torch.float64).clone()
    if tuple(h.shape) != (cols, cols):
        raise GPTQError(f"a Hessian of shape {list(h.shape)} does not match a reduction of {cols}")
    grid = torch.broadcast_to(scales.detach().to(torch.float64), (rows, cols)).clone()
    if not bool(torch.isfinite(grid).all()) or not bool((grid > 0).all()):
        raise GPTQError("every scale must be finite and positive")
    if not bool(torch.isfinite(h).all()):
        raise GPTQError("the Hessian is not finite")

    dead = (torch.diagonal(h) == 0).nonzero().flatten()
    h[dead, :] = 0
    h[:, dead] = 0
    h[dead, dead] = 1
    if act_order:
        order = torch.argsort(torch.diagonal(h), descending=True, stable=True)
    else:
        order = torch.arange(cols)
    weight, grid, h = weight[:, order], grid[:, order], h[order][:, order]
    added = damp * float(torch.diagonal(h).mean())
    h.diagonal().add_(added)
    try:
        upper = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    except RuntimeError as error:  # torch.linalg.LinAlgError derives from it
        raise GPTQError(f"the damped Hessian is not positive definite ({error})") from error

    codes = torch.zeros_like(weight)
    for start in range(0, cols, block):
        stop = min(start + block, cols)
        part = weight[:, start:stop].clone()
        errors = torch.zeros_like(part)
        local = upper[start:stop, start:stop]
        for i in range(stop - start):
            column, step = part[:, i], grid[:, start + i]
            code = torch.clamp(torch.round(column / step), quant_min, quant_max)
            codes[:, start + i] = code
            error = (column - code * step) / local[i, i]
            part[:, i:] -= error.unsqueeze(1) * local[i, i:].unsqueeze(0)
            errors[:, i] = error
        weight[:, stop:] -= errors @ upper[start:stop, stop:]
    restored = codes[:, torch.argsort(order)]
    return restored.to(torch.int8), {"dead_columns": int(dead.numel()), "damp_added": added}


def output_error(matrix: Any, dequantized: Any, hessian: Any) -> float:
    """``tr((Q - W) H (Q - W)^T) / tr(W H W^T)``: the layer-output error relative to its output energy."""
    import torch

    w = matrix.detach().to(torch.float64)
    delta = dequantized.detach().to(torch.float64) - w
    h = hessian.detach().to(torch.float64)
    energy = float(((w @ h) * w).sum())
    return float(((delta @ h) * delta).sum()) / energy if energy > 0 else 0.0


class HessianSum:
    """``2/n * sum(X^T X)`` over the rows of every ``X`` added, accumulated in float64 in call order."""

    def __init__(self, columns: int) -> None:
        import torch

        self.columns = columns
        self.total = torch.zeros(columns, columns, dtype=torch.float64)
        self.rows = 0

    def add(self, rows: Any) -> None:
        import torch

        x = rows.detach().reshape(-1, self.columns).to(torch.float64)
        self.total.addmm_(x.T, x)
        self.rows += int(x.shape[0])

    def value(self) -> Any:
        if not self.rows:
            raise GPTQError("no calibration row reached this contraction")
        return self.total * (2.0 / self.rows)


def contraction_rows(layout: tuple, x: Any) -> Any:
    """The ``[rows, reduction]`` matrix a contraction of ``layout`` multiplies by its weight."""
    import torch

    if layout[0] == "conv2d":
        _, kernel, stride, padding, dilation = layout
        batched = x if x.dim() == 4 else x.unsqueeze(0)
        patches = torch.nn.functional.unfold(
            batched, kernel_size=kernel, dilation=dilation, padding=padding, stride=stride
        )
        return patches.transpose(1, 2).reshape(-1, patches.shape[1])
    return x.reshape(-1, x.shape[-1])


def _check_range(quant_min: int, quant_max: int) -> None:
    if not (type(quant_min) is int and type(quant_max) is int and -128 <= quant_min < quant_max <= 127):
        raise GPTQError(f"codes in [{quant_min}, {quant_max}] are not an int8 range")


# --- the capture hook --------------------------------------------------------------------------------


def _pairs(value: Any) -> tuple[int, int]:
    items = [value] if isinstance(value, int) else list(value)
    if len(items) == 1:
        items = items * 2
    if len(items) != 2 or not all(type(item) is int for item in items):
        raise GPTQError(f"{value!r} is not a pair of integers")
    return items[0], items[1]


def _bound(node: Any) -> dict[str, Any]:
    """``node``'s arguments by schema name, with the schema's defaults filled in."""
    values = {}
    for index, argument in enumerate(node.target._schema.arguments):
        if index < len(node.args):
            values[argument.name] = node.args[index]
        elif argument.name in node.kwargs:
            values[argument.name] = node.kwargs[argument.name]
        elif argument.has_default_value():
            values[argument.name] = argument.default_value
    return values


class PT2EWeightRounding:
    """GPTQ codes for every stored weight a prepared PT2E program observes, set before ``convert_pt2e``.

    Construct it on the prepared module, iterate the calibration stream through :meth:`track`, then
    call :meth:`apply` before ``convert_pt2e``. ``contractions`` are the ATen operators the recipe
    annotates as contractions. A contraction whose second operand is not a stored weight (an
    activation x activation product) has nothing to round and is ignored.
    """

    def __init__(self, prepared: Any, parameters: Mapping[str, Any], *, contractions: Iterable[Any]) -> None:
        import torch

        self.prepared = prepared
        self.parameters = check_parameters(parameters)
        self._torch = torch
        aten = torch.ops.aten
        self._views = {
            aten.t.default,
            aten.transpose.int,
            aten.permute.default,
            aten.view.default,
            aten.reshape.default,
            aten._unsafe_view.default,
            aten.contiguous.default,
        }
        self._modules = dict(prepared.named_modules())
        self._sums: dict[tuple, HessianSum] = {}
        self._keys_by_observer: dict[str, list[tuple]] = {}
        self._active = False
        self._index = -1
        self._samples: set[int] = set()
        self._handles: list[Any] = []
        self.groups: dict[str, dict[str, Any]] = {}  # stored weight -> its sites, in graph order
        targets = set(contractions)
        for node in prepared.graph.nodes:
            if node.op == "call_function" and node.target in targets and len(node.args) >= 2:
                self._discover(node)
        for observer, keys in self._keys_by_observer.items():
            self._handles.append(self._modules[observer].register_forward_pre_hook(self._hook(keys)))

    # -- discovery

    def _observer(self, node: Any) -> Any:
        if not isinstance(node, self._torch.fx.Node) or node.op != "call_module":
            return None
        module = self._modules.get(str(node.target))
        return module if callable(getattr(module, "calculate_qparams", None)) else None

    def _stored_chain(self, node: Any) -> tuple[Any, list[Any]] | None:
        """``(get_attr node, [view nodes from it to the observed value])`` or ``None`` if not stored."""
        chain = []
        while isinstance(node, self._torch.fx.Node) and node.op == "call_function" and node.target in self._views:
            chain.append(node)
            node = node.args[0] if node.args else None
        if isinstance(node, self._torch.fx.Node) and node.op == "get_attr":
            return node, chain[::-1]
        return None

    def _layout(self, node: Any, rank: int) -> tuple[tuple | None, str | None]:
        aten = self._torch.ops.aten
        if node.target == aten.linear.default and rank == 2:
            return ("linear",), None
        if node.target in (aten.matmul.default, aten.mm.default) and rank == 2:
            return ("matmul",), None
        if node.target == aten.conv2d.default and rank == 4:
            bound = _bound(node)
            if bound.get("groups", 1) != 1:
                return None, f"a grouped convolution (groups={bound.get('groups')}) is not one im2col product"
            try:
                geometry = tuple(_pairs(bound[name]) for name in ("stride", "padding", "dilation"))
            except (GPTQError, KeyError, TypeError) as error:
                return None, f"the convolution's geometry is not numeric ({error})"
            return ("conv2d", None, *geometry), None
        return None, f"{node.target} with a rank-{rank} stored operand has no GPTQ layout here"

    def _discover(self, node: Any) -> None:
        weight_observer = node.args[1]
        if self._observer(weight_observer) is None or not weight_observer.args:
            return  # the operand is not quantized here
        stored = self._stored_chain(weight_observer.args[0])
        if stored is None:
            return  # a run-time operand: nothing stored to round
        attribute, chain = stored
        target = str(attribute.target)
        group = self.groups.setdefault(
            target,
            {"attribute": attribute, "chain": chain, "observer": weight_observer, "sites": [], "why": None},
        )
        group["sites"].append(node.name)
        if group["why"] is not None:
            return
        if weight_observer is not group["observer"]:
            group["why"] = "the stored weight is observed separately by more than one contraction"
            return
        observed = weight_observer.args[0].meta.get("val")
        if not isinstance(observed, self._torch.Tensor):
            group["why"] = "the observed weight has no recorded value"
            return
        layout, why = self._layout(node, observed.dim())
        if layout is not None and layout[0] == "conv2d":
            layout = ("conv2d", tuple(int(extent) for extent in observed.shape[2:]), *layout[2:])
        activation = node.args[0]
        if layout is not None and self._observer(activation) is None:
            layout, why = None, "the contraction's input carries no observer to read it from"
        if layout is None:
            group["why"] = why
            return
        if group.setdefault("layout", layout) != layout:
            group["why"] = "the stored weight is contracted under more than one layout"
            return
        key = (str(activation.target), layout)
        if key not in self._sums:
            columns = int(observed.shape[0] if layout[0] == "matmul" else math.prod(observed.shape[1:]))
            self._sums[key] = HessianSum(columns)
            self._keys_by_observer.setdefault(str(activation.target), []).append(key)
        group.setdefault("keys", [])
        if key not in group["keys"]:
            group["keys"].append(key)

    # -- calibration

    def _hook(self, keys: list[tuple]) -> Any:
        def accumulate(module: Any, args: tuple) -> None:
            del module
            if not self._active or not args or not isinstance(args[0], self._torch.Tensor):
                return
            self._samples.add(self._index)
            for key in keys:
                self._sums[key].add(contraction_rows(key[1], args[0].detach()))

        return accumulate

    def track(self, samples: Iterable[Any]) -> Iterator[Any]:
        """``samples`` unchanged, with Hessians accumulated while the first ``calibration_samples`` run."""
        limit = self.parameters["calibration_samples"]
        try:
            for index, sample in enumerate(samples):
                self._index, self._active = index, index < limit
                yield sample
        finally:
            self._active = False

    # -- rounding

    def apply(self) -> dict[str, Any]:
        """Replace each stored weight by GPTQ codes x scale and return the report. Removes the hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles, self._active = [], False
        rows = []
        for target, group in self.groups.items():
            row = {"weight": target, "contractions": list(group["sites"])}
            why = group["why"]
            if why is None:
                try:
                    row.update(self._round(target, group))
                except GPTQError as error:
                    why = str(error)
            if why is not None:
                row.update({"status": NEAREST, "why": why})
            rows.append(row)
        return {
            "rounding": GPTQ,
            "parameters": dict(self.parameters),
            "calibration_samples_used": len(self._samples),
            "weights_rounded": sum(1 for row in rows if row["status"] == GPTQ),
            "weights_left_nearest": sum(1 for row in rows if row["status"] == NEAREST),
            "weights": rows,
        }

    def _evaluate(self, chain: list[Any], value: Any) -> Any:
        for view in chain:
            value = view.target(value, *view.args[1:], **view.kwargs)
        return value

    def _round(self, target: str, group: Mapping[str, Any]) -> dict[str, Any]:
        torch = self._torch
        attribute, chain, layout = group["attribute"], group["chain"], group["layout"]
        readers = [user for node in (attribute, *chain) for user in node.users]
        expected = [*chain, group["observer"]]
        same_target = [
            node for node in self.prepared.graph.nodes if node.op == "get_attr" and str(node.target) == target
        ]
        consumers = {user.name for user in group["observer"].users}
        if readers != expected or same_target != [attribute] or not consumers <= set(group["sites"]):
            raise GPTQError("the stored weight is also read outside its contraction")
        observer = self._modules[str(group["observer"].target)]
        owner_path, _, name = target.rpartition(".")
        owner = self.prepared.get_submodule(owner_path) if owner_path else self.prepared
        stored = getattr(owner, name)
        observed = self._evaluate(chain, stored.detach())
        scale, zero_point = observer.calculate_qparams()
        per_channel = observer.qscheme in (torch.per_channel_symmetric, torch.per_channel_affine)
        if bool((zero_point != 0).any()):
            raise GPTQError("GPTQ here rounds onto a symmetric grid; the observer reports a zero point")
        shape = [1] * observed.dim()
        if per_channel:
            shape[observer.ch_axis] = -1
        grid = scale.to(torch.float32).reshape(shape).expand(observed.shape)
        quant_min, quant_max = int(observer.quant_min), int(observer.quant_max)

        def to_matrix(tensor: Any) -> Any:
            return tensor.T if layout[0] == "matmul" else tensor.reshape(tensor.shape[0], -1)

        def from_matrix(matrix: Any) -> Any:
            return matrix.T if layout[0] == "matmul" else matrix.reshape(observed.shape)

        total = None
        count = 0
        for key in group["keys"]:
            part = self._sums[key]
            total = part.total.clone() if total is None else total + part.total
            count += part.rows
        if not count:
            raise GPTQError("no calibration sample reached this contraction")
        hessian = total * (2.0 / count)
        matrix, scales = to_matrix(observed), to_matrix(grid)
        codes, info = gptq_codes(
            matrix,
            hessian,
            scales,
            quant_min=quant_min,
            quant_max=quant_max,
            damp=self.parameters["damp"],
            act_order=self.parameters["act_order"],
            block=self.parameters["block"],
        )
        nearest = nearest_codes(matrix, scales, quant_min=quant_min, quant_max=quant_max)
        steps = scales.to(torch.float64)
        errors = {
            NEAREST: output_error(matrix, nearest.to(torch.float64) * steps, hessian),
            GPTQ: output_error(matrix, codes.to(torch.float64) * steps, hessian),
        }
        observed_codes = from_matrix(codes)
        replacement = (observed_codes.to(torch.float32) * grid).to(observed.dtype)
        # What convert_pt2e will fold: the same op on the same tensor must give back GPTQ's codes.
        decomposed = torch.ops.quantized_decomposed
        if per_channel:
            refolded = decomposed.quantize_per_channel.default(
                replacement, scale, zero_point, observer.ch_axis, quant_min, quant_max, observer.dtype
            )
        else:
            refolded = decomposed.quantize_per_tensor.default(
                replacement, float(scale), int(zero_point), quant_min, quant_max, observer.dtype
            )
        if not torch.equal(refolded.to(torch.int8), observed_codes):
            raise RuntimeError(f"{target}: the program would not carry GPTQ's codes after convert")
        self._store(owner, name, stored, self._unview(chain, stored, replacement))
        return {
            "status": GPTQ,
            "shape": list(observed.shape),
            "calibration_rows": count,
            "dead_columns": info["dead_columns"],
            "output_error": errors,
        }

    def _unview(self, chain: list[Any], stored: Any, observed: Any) -> Any:
        """The stored tensor whose views give ``observed``: the view chain is a permutation of elements."""
        torch = self._torch
        if not chain:
            return observed.reshape(stored.shape)
        positions = self._evaluate(chain, torch.arange(stored.numel()).reshape(stored.shape)).reshape(-1)
        if not torch.equal(torch.sort(positions).values, torch.arange(stored.numel())):
            raise GPTQError("the weight's view chain is not a permutation of its elements")
        flat = torch.empty(stored.numel(), dtype=observed.dtype)
        flat[positions] = observed.reshape(-1)
        return flat.reshape(stored.shape)

    def _store(self, owner: Any, name: str, current: Any, value: Any) -> None:
        """Re-register ``owner.name``; the old tensor (shared with the source model) is not written."""
        torch = self._torch
        value = value.to(dtype=current.dtype).contiguous()
        if name in owner._parameters:
            owner._parameters[name] = torch.nn.Parameter(value, requires_grad=current.requires_grad)
        elif name in owner._buffers:
            owner._buffers[name] = value
        else:
            setattr(owner, name, value)

"""Voyager side of the mutation study: seed one fault into Voyager's bufferized program, run it in
Voyager's own stack, apply Voyager's own numeric criterion, and re-emit the faulty program's IR.

Runs inside Voyager's environment (``out/build/voyager-venv``) and imports nothing from merlin, so
every Voyager verdict here is produced by the pinned compiler's own code: the tolerance constants are
read from its ``test/test_codegen.py``, the program runs through its eager ``voyager::*`` kernels
(counting-semaphore oracle on), and the IR comes from its own ``gen_code_bufferized``.

For one fixture workload it

1. rebuilds the program exactly as ``voyager_export.py`` built the fixture (same seed, config and
   quantization flags) and checks that the IR it emits is byte-identical to the checked-in fixture;
2. runs the graph BEFORE ``compile()`` -- the stage at which Voyager's CNN/BERT/ViT harnesses check
   numerics -- and records that verdict once (no bufferization fault can reach it: those nodes do not
   exist yet);
3. for every mutation: recompiles from scratch, rewrites the bufferized FX graph (or its memory plan)
   in place, runs it eagerly, applies ``assert_close(new, old, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)``
   as ``test_codegen.py`` does, measures the error, and re-emits ``model.txt`` / ``model.json`` for
   the merlin side and for the ``run_ci.py`` text-diff gate.

Mutations are located structurally from the emitted IR (which copy reads the input, the weight, the
bias; which copy stores; which tile index both operands share -- the reduction index), never by node
name, so the same catalogue applies to every fixture or reports why it does not.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import operator
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from google.protobuf import json_format, text_format
from torch.utils._pytree import tree_flatten

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import voyager_export  # noqa: E402  (the fixture's own export recipe)


class NotApplicable(Exception):
    """The mutation's target construct does not occur in this program."""


# ------------------------------------------------------------------------------------------------
# Building the program exactly as the fixture was built
# ------------------------------------------------------------------------------------------------


class Program:
    """One freshly compiled Voyager program plus what Voyager's own check compares it against."""

    def __init__(self, manifest: dict, out: Path):
        import test_codegen
        from voyager_compiler import (
            compile,
            convert_pt2e,
            export_model,
            extract_input_preprocessor,
            fuse_operator,
            get_default_quantizer,
            prepare_pt2e,
            transform,
        )
        from voyager_compiler.hardware_config import AcceleratorConfig

        torch.manual_seed(manifest["seed"])
        torch.set_grad_enabled(False)
        fields = dict(manifest["config"])
        fields["pe_array_size"] = tuple(fields["pe_array_size"])
        config = AcceleratorConfig(**fields)
        dtype = torch.bfloat16 if manifest.get("model_dtype", "torch.bfloat16").endswith("bfloat16") else torch.float32
        module, example = voyager_export.build_workload(manifest["workload"], dtype)
        scheme = voyager_export.QUANT_SCHEMES[manifest["quant_scheme"]]
        quantizer = get_default_quantizer(
            output_activation=None,
            force_scale_power_of_two=False,
            **{k: v for k, v in scheme.items() if k != "bias"},
            bias=scheme["bias"],
        )
        gm = export_model(module, (example,))
        if manifest.get("conv2d_im2col"):
            from voyager_compiler import replace_conv2d_with_im2col

            replace_conv2d_with_im2col(gm)
        gm = prepare_pt2e(gm, quantizer)
        for _ in range(manifest["calibration_steps"]):
            gm(torch.randn_like(example))
        convert_pt2e(gm, scheme["bias"])
        self.old_output = gm(example)  # test_codegen's `old_output`: before transform()
        patterns = test_codegen.VECTOR_PIPELINE
        transform(
            gm,
            (example,),
            patterns=patterns,
            config=config,
            skip_op_fusion=True,
            layout_policy=manifest.get("layout_policy", "systolic"),
        )
        gm, preprocess = extract_input_preprocessor(gm)
        self.lowered_input = preprocess(example)
        fuse_operator(gm, patterns)
        # test_codegen's CNN/ViT harnesses take `new_output` HERE, before compile() tiles and
        # bufferizes (test/utils/models/torchvision_models.py, vit.py; bert.py/mobilebert.py
        # likewise after transform()).
        self.pre_tiling_output = gm(self.lowered_input)
        out.mkdir(parents=True, exist_ok=True)
        self.params = compile(
            gm,
            (self.lowered_input,),
            config=config,
            output_dir=str(out),
            output_file=manifest["workload"].get("name", "model"),
            dump_tensors=False,
        )
        self.gm = gm
        self.config = config
        self.model_txt = (out / "model.txt").read_text()
        self.model_dict = json_format.MessageToDict(self.params, preserving_proto_field_name=True)


def _emit(gm, lowered_input):
    from voyager_compiler.codegen.transform.bufferize import gen_code_bufferized

    params = gen_code_bufferized(gm, (lowered_input,), None)
    return (text_format.MessageToString(params), json_format.MessageToDict(params, preserving_proto_field_name=True))


def plan_invariants(gm, config) -> list[str]:
    """Voyager's own memory-plan invariant check (``memory_planning._check_invariants``: two
    simultaneously-live buffers must not share an address range), re-run on the program as it now
    stands. ``plan_memory`` runs it once, at planning time, and only logs a ``[MEM_OVERLAP]``
    WARNING; a plan changed afterwards is judged here by the same code."""
    import logging

    from voyager_compiler.codegen.transform.bufferize import memory_planning as mp

    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            if "MEM_OVERLAP" in record.getMessage():
                records.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    mp.logger.addHandler(handler)
    try:
        buffer_of = mp._buffer_identity(gm)
        bufs = mp._buffer_lifetimes(gm, buffer_of, mp._timestamps(gm), config)
        mp._check_invariants(gm, bufs)
    finally:
        mp.logger.removeHandler(handler)
    return records


def _run_eager(gm, lowered_input):
    """Run the bufferized graph as Voyager's LLM harness and ``--run-lowered`` do. Unwritten
    floating-point allocations start as ``torch.randn`` (``voyager::alloc``), so seed first."""
    torch.manual_seed(0)
    return gm(lowered_input)


# ------------------------------------------------------------------------------------------------
# Voyager's own criterion, verbatim from test/test_codegen.py (constants imported, not copied)
# ------------------------------------------------------------------------------------------------


def voyager_criterion(old_output, new_output) -> dict:
    import test_codegen
    from torch.testing import assert_close

    rtol, atol = test_codegen.OUTPUT_RTOL, test_codegen.OUTPUT_ATOL
    old_flat, _ = tree_flatten(old_output)
    new_flat, _ = tree_flatten(new_output)
    verdict = {"rtol": rtol, "atol": atol}
    try:
        assert len(old_flat) == len(new_flat), f"{len(old_flat)} outputs became {len(new_flat)}"
        worst = 0.0
        for old, new in zip(old_flat, new_flat):
            if not isinstance(old, torch.Tensor):
                assert old == new, f"non-tensor output {old!r} != {new!r}"
                continue
            deviation = (new - old).abs().to(torch.float32) / (old.abs().to(torch.float32) + atol)
            worst = max(worst, deviation.max().item())
            assert_close(new, old, rtol=rtol, atol=atol)
        verdict.update(result="match", log_line=f"Results match (max deviation {worst:.2e})", max_deviation=worst)
    except Exception as exc:  # test_codegen catches everything here and only prints a WARNING
        verdict.update(
            result="warning", log_line=f"WARNING: output verification failed: {str(exc).splitlines()[0][:300]}"
        )
    return verdict


def error_metrics(reference, candidate, rtol: float, atol: float) -> dict:
    ref = torch.as_tensor(reference).to(torch.float64).flatten()
    got = torch.as_tensor(candidate).to(torch.float64).flatten()
    diff = got - ref
    finite = torch.isfinite(got)
    nref = ref.norm().item()
    bound = atol + rtol * ref.abs()
    return {
        "elements": int(ref.numel()),
        "max_abs": float(diff.abs().max().item()) if finite.all() else float("inf"),
        "rel_l2": float(diff.norm().item() / nref) if nref and finite.all() else float("inf"),
        "changed_elements": int((diff != 0).sum().item()),
        "outside_tolerance": int((diff.abs() > bound).sum().item()),
        "nonfinite": int((~finite).sum().item()),
        "reference_abs_mean": float(ref.abs().mean().item()),
    }


# ------------------------------------------------------------------------------------------------
# Locating constructs in the bufferized FX graph
# ------------------------------------------------------------------------------------------------


def _modules(gm) -> dict:
    return {name: mod for name, mod in gm.named_modules() if isinstance(mod, torch.fx.GraphModule)}


def _find(gm, name: str):
    hits = [(mname, mod, node) for mname, mod in _modules(gm).items() for node in mod.graph.nodes if node.name == name]
    if len(hits) != 1:
        raise RuntimeError(f"node {name!r} occurs {len(hits)} times in the bufferized graph")
    return hits[0]


def _walk_ir_copies(ops, found):
    for op in ops or ():
        prim = op.get("prim")
        if prim and prim.get("target") == "voyager::async_copy":
            kw = prim.get("kwargs", {})
            src = kw["src"]["tensor_box"]["box"]
            dst = kw["dst"]["tensor_box"]["box"]
            found.append({"name": prim.get("name") or op["name"], "src": src, "dst": dst})
        for key in ("loop",):
            if key in op:
                body = op[key].get("for_loop") or op[key].get("while_loop") or {}
                _walk_ir_copies(body.get("body", {}).get("ops"), found)
                _walk_ir_copies(body.get("condition", {}).get("ops"), found)
        if "cond" in op:
            _walk_ir_copies(op["cond"].get("true_region", {}).get("ops"), found)
            _walk_ir_copies(op["cond"].get("false_region", {}).get("ops"), found)
        if "async" in op:
            _walk_ir_copies(op["async"].get("body", {}).get("ops"), found)
    return found


def _level(box) -> str:
    return str(box.get("memory", {}).get("level", "MEMORY_LEVEL_IMMEDIATE"))


class Roles:
    """Which FX node plays which part, read from the program's own IR."""

    def __init__(self, gm, model_dict):
        inputs = {b["node"] for b in model_dict.get("inputs", ())}
        params = {b["node"]: b for b in model_dict.get("parameters", ())}
        self.loads: dict[str, list] = {"lhs": [], "weight": [], "bias": []}
        self.stores: list = []
        for copy in _walk_ir_copies(model_dict.get("ops"), []):
            src_dram = "DRAM" in _level(copy["src"])
            dst_dram = "DRAM" in _level(copy["dst"])
            located = _find(gm, copy["name"])
            if src_dram and not dst_dram:
                node = copy["src"]["node"]
                if node in inputs:
                    role = "lhs"
                elif params.get(node, {}).get("dtype", "").startswith("int32"):
                    role = "bias"
                else:
                    role = "weight"
                self.loads[role].append(located)
            elif dst_dram and not src_dram:
                self.stores.append(located)
        for role in ("lhs", "weight"):
            if len(self.loads[role]) != 1:
                raise RuntimeError(f"expected one {role} load, found {len(self.loads[role])}")
        self.lhs = self.loads["lhs"][0]
        self.weight = self.loads["weight"][0]
        self.bias = self.loads["bias"][0] if self.loads["bias"] else None
        # A store inside the rolled loop drains the previous tile; a top-level store drains the last.
        self.loop_stores = [s for s in self.stores if s[0] != ""]
        self.epilogue_stores = [s for s in self.stores if s[0] == ""]
        self.model_dict = model_dict

    def reduction_index(self):
        """The tile index both operand copies share: the reduction (K / input-channel) index."""
        lhs_idx = [a for a in self.lhs[2].args[2] if isinstance(a, torch.fx.Node)]
        w_idx = [a for a in self.weight[2].args[2] if isinstance(a, torch.fx.Node)]
        shared = [a for a in lhs_idx if a in w_idx]
        if len(shared) != 1:
            raise NotApplicable(
                f"the operand copies share {len(shared)} tile indices; no single reduction split to perturb"
            )
        return shared[0]

    def copy_box(self, copy_name: str, which: str) -> dict:
        for copy in _walk_ir_copies(self.model_dict.get("ops"), []):
            if copy["name"] == copy_name:
                return copy[which]
        raise KeyError(copy_name)


def _delinearized(node):
    """(delinearize node, position) if ``node`` is ``getitem(delinearize_index(...), pos)``."""
    if node.op == "call_function" and node.target is operator.getitem:
        source, pos = node.args
        if isinstance(source, torch.fx.Node) and "delinearize_index" in str(source.target):
            return source, pos
    return None


def _extent(index_node) -> int:
    found = _delinearized(index_node)
    if found is None:
        raise NotApplicable(f"{index_node.name} is not a delinearized loop index")
    source, pos = found
    return int(source.args[1][pos])


def _step_counterpart(index_node, offset: int):
    """The same tile-index component for the loop step ``iv + offset`` (``offset`` = -1: the step
    before), reusing Voyager's own node when it exists and creating it the way Voyager does if not."""
    found = _delinearized(index_node)
    if found is None:
        raise NotApplicable(f"{index_node.name} is not a delinearized loop index")
    source, pos = found
    graph = index_node.graph
    iv = source.args[0]
    want_op = operator.sub if offset < 0 else operator.add
    for cand in graph.nodes:
        if (
            cand.op == "call_function"
            and cand.target is source.target
            and isinstance(cand.args[0], torch.fx.Node)
            and cand.args[0].target is want_op
            and cand.args[0].args == (iv, abs(offset))
            and list(cand.args[1]) == list(source.args[1])
        ):
            for user in cand.users:
                if user.target is operator.getitem and user.args[1] == pos:
                    return user
            with graph.inserting_after(cand):
                return graph.call_function(operator.getitem, (cand, pos))
    raise NotApplicable(f"no loop-step {offset:+d} index vector next to {source.name}")


def _current_counterpart(prev_index_node):
    """For ``getitem(delinearize(iv - 1), pos)`` return ``getitem(delinearize(iv), pos)``."""
    found = _delinearized(prev_index_node)
    if found is None:
        return None
    source, pos = found
    step = source.args[0]
    if not (isinstance(step, torch.fx.Node) and step.target is operator.sub and step.args[1] == 1):
        return None
    iv = step.args[0]
    graph = prev_index_node.graph
    for cand in graph.nodes:
        if (
            cand.op == "call_function"
            and cand.target is source.target
            and cand.args[0] is iv
            and list(cand.args[1]) == list(source.args[1])
        ):
            for user in cand.users:
                if user.target is operator.getitem and user.args[1] == pos:
                    return user
            with graph.inserting_after(cand):
                return graph.call_function(operator.getitem, (cand, pos))
    return None


def _scalar_op(before, fn, *args):
    graph = before.graph
    with graph.inserting_before(before):
        return graph.call_function(fn, tuple(args))


def _replace_arg(node, position: int, value) -> None:
    args = list(node.args)
    args[position] = value
    node.args = tuple(args)


def _recompile(gm) -> None:
    for mod in _modules(gm).values():
        mod.graph.lint()
        mod.recompile()


def _scratch_allocs(gm):
    """Top-level on-chip allocations with their planned segment, slot stride and slot count."""
    out = []
    for node in gm.graph.nodes:
        seg = node.meta.get("scratchpad")
        if seg is None or node.op != "call_function":
            continue
        slots = node.args[3] if len(node.args) > 3 else node.kwargs.get("num_slots", 0)
        out.append((node, seg, int(node.meta.get("slot_stride", 0) or 0), max(int(slots), 1)))
    return out


def _slot_intervals(seg, stride, slots):
    span = int(seg.end) - int(seg.start)
    slot_bytes = span - (slots - 1) * stride if slots > 1 else span
    return [(int(seg.start) + s * stride, int(seg.start) + s * stride + slot_bytes) for s in range(slots)], slot_bytes


def _alloc_of(located):
    """The top-level allocation a copy's destination (or source) slot window views."""
    _, _, copy = located
    return copy


def _root_alloc_for_box(gm, box_node_name: str):
    for node in gm.graph.nodes:
        if node.name == box_node_name and "scratchpad" in node.meta:
            return node
    raise RuntimeError(f"no top-level scratchpad allocation named {box_node_name}")


def _rebind_segment(gm, old_seg, new_seg=None, new_stride=None) -> int:
    """Point every node that carries ``old_seg`` (the root and the region placeholders the planner
    stamped) at ``new_seg`` / ``new_stride``. Returns how many nodes changed."""
    changed = 0
    key = (int(old_seg.start), int(old_seg.end), old_seg.memory_space)
    for mod in _modules(gm).values():
        for node in mod.graph.nodes:
            seg = node.meta.get("scratchpad")
            if seg is None or (int(seg.start), int(seg.end), seg.memory_space) != key:
                continue
            if new_seg is not None:
                node.meta["scratchpad"] = dataclasses.replace(seg, start=new_seg[0], end=new_seg[1])
            if new_stride is not None and "slot_stride" in node.meta:
                node.meta["slot_stride"] = new_stride
            changed += 1
    return changed


# ------------------------------------------------------------------------------------------------
# The mutation catalogue
# ------------------------------------------------------------------------------------------------
# Each returns a short statement of what it changed. Classes:
#   control      identity; must leave everything unchanged
#   negative     semantics-preserving; every correct check must pass it
#   schedule     a tiling / bufferization fault in which tile moves where
#   memory_plan  a scratchpad placement fault (addresses, not tensor semantics)
#   hazard       an ordering fault that only concurrency can expose
#   numerics     a quantization-constant fault in the fused tail (not the schedule)

CATALOGUE: dict[str, dict] = {}


def mutation(mid: str, cls: str, title: str):
    def register(fn):
        CATALOGUE[mid] = {"id": mid, "class": cls, "title": title, "apply": fn}
        return fn

    return register


@mutation("C0_identity", "control", "no change (re-emit and re-run the compiled program)")
def _identity(gm, roles, config):
    return "nothing changed"


@mutation("N1_reorder_loads", "negative", "issue the weight-tile copy before the input-tile copy (independent buffers)")
def _reorder_loads(gm, roles, config):
    mname_l, mod_l, lhs = roles.lhs
    mname_w, mod_w, weight = roles.weight
    if mod_l is not mod_w:
        raise NotApplicable("input and weight copies live in different regions")
    moved = [
        a
        for a in weight.args
        if isinstance(a, torch.fx.Node) and a.op == "call_function" and a.target is torch.ops.voyager.subview.default
    ]
    order = {n: i for i, n in enumerate(mod_l.graph.nodes)}
    anchor_candidates = [
        a
        for a in lhs.args
        if isinstance(a, torch.fx.Node) and a.op == "call_function" and a.target is torch.ops.voyager.subview.default
    ] + [lhs]
    anchor = min(anchor_candidates, key=order.get)
    if order[weight] < order[anchor]:
        raise NotApplicable("the weight copy already precedes the input copy")
    for node in moved + [weight]:
        for dep in node.all_input_nodes:
            if dep not in moved and order[dep] >= order[anchor]:
                raise NotApplicable(f"{node.name} depends on {dep.name}, defined after the input copy")
    for node in moved + [weight]:
        anchor.prepend(node)
    return f"moved {', '.join(n.name for n in moved + [weight])} ahead of {anchor.name}"


def _free_start(gm, config, moving, stride, slots, slot_bytes):
    """Highest aligned start where the moved buffer's WHOLE span (slots included) is clear of every
    other buffer's whole span: disjoint bytes are not enough, because Voyager's plan invariant
    compares spans and its planner never interleaves the spans of two bank groups."""
    occupied = []
    for node, seg, s_stride, s_slots in _scratch_allocs(gm):
        if node is moving:
            continue
        occupied.append((int(seg.start), int(seg.end)))
    size = int(config["scratchpad_size"])
    align = int(config["bank_width"])
    span = (slots - 1) * stride + slot_bytes
    start = size - span
    start -= start % align
    while start >= 0:
        if not any(start < oe and os_ < start + span for os_, oe in occupied):
            return start
        start -= align
    raise NotApplicable("no free scratchpad region holds the weight buffer")


@mutation("N2_relocate_weight", "negative", "re-plan the weight buffer into a free, non-overlapping scratchpad region")
def _relocate(gm, roles, config):
    box = roles.copy_box(roles.weight[2].name, "dst")
    alloc = _root_alloc_for_box(gm, box["node"])
    seg = alloc.meta["scratchpad"]
    stride = int(alloc.meta.get("slot_stride", 0) or 0)
    slots = max(int(box.get("bank_count", 1) or 1), 1)
    _, slot_bytes = _slot_intervals(seg, stride, slots)
    start = _free_start(gm, config, alloc, stride, slots, slot_bytes)
    span = int(seg.end) - int(seg.start)
    n = _rebind_segment(gm, seg, new_seg=(start, start + span))
    return f"{alloc.name}: scratchpad {int(seg.start)} -> {start} ({n} nodes), slot stride {stride}"


@mutation(
    "N3_permute_reduction",
    "negative",
    "visit the reduction tiles in rotated order (k -> (k+1) mod nK) in both operand copies",
)
def _permute_reduction(gm, roles, config):
    k = roles.reduction_index()
    nk = _extent(k)
    if nk < 2:
        raise NotApplicable("the reduction is not split")
    for _, _, copy in (roles.lhs, roles.weight):
        rotated = _scalar_op(copy, operator.mod, _scalar_op(copy, operator.add, k, 1), nk)
        _replace_arg(copy, 2, [rotated if a is k else a for a in copy.args[2]])
    return f"reduction index {k.name} -> ({k.name} + 1) % {nk} in the input and weight copies"


@mutation(
    "F1_prev_step_index", "schedule", "input copy uses the previous loop step's row-tile index (pipeline off-by-one)"
)
def _prev_step_index(gm, roles, config):
    _, _, copy = roles.lhs
    k = roles.reduction_index() if _shares_reduction(roles) else None
    others = [a for a in copy.args[2] if isinstance(a, torch.fx.Node) and a is not k]
    if not others:
        raise NotApplicable("the input copy has no non-reduction tile index")
    target = others[0]
    if target.op == "placeholder":
        # The copy sits in a guarded region (load-on-change); its index arrives as a region
        # operand. Perturb the operand the loop body passes in, if the copy is its only reader.
        if list(target.users) != [copy]:
            raise NotApplicable(f"{target.name} feeds more than the input copy")
        cond, position = _region_operand(gm, copy.graph, target)
        outer = cond.args[3][position]
        prev = _step_counterpart(outer, -1)
        operands = list(cond.args[3])
        operands[position] = prev
        _replace_arg(cond, 3, tuple(operands))
        return f"input copy index {target.name} (= {outer.name}) -> {prev.name} (same component, loop step - 1)"
    prev = _step_counterpart(target, -1)
    _replace_arg(copy, 2, [prev if a is target else a for a in copy.args[2]])
    return f"input copy index {target.name} -> {prev.name} (same component, loop step - 1)"


def _region_operand(gm, region_graph, placeholder):
    """(cond node, operand position) that binds ``placeholder`` of a conditional region."""
    placeholders = [n for n in region_graph.nodes if n.op == "placeholder"]
    position = placeholders.index(placeholder)
    for mod in _modules(gm).values():
        for node in mod.graph.nodes:
            if node.op == "call_function" and "cond" in str(node.target):
                for branch in node.args[1:3]:
                    if (
                        isinstance(branch, torch.fx.Node)
                        and branch.op == "get_attr"
                        and getattr(mod, branch.target, None) is not None
                        and getattr(mod, branch.target).graph is region_graph
                    ):
                        return node, position
    raise NotApplicable("the copy's region is not a conditional")


def _shares_reduction(roles) -> bool:
    try:
        roles.reduction_index()
        return True
    except NotApplicable:
        return False


def _copy_kw(copy, name: str, position: int):
    return copy.args[position] if len(copy.args) > position else copy.kwargs.get(name)


def _set_copy_kw(copy, name: str, position: int, value) -> None:
    if len(copy.args) > position:
        _replace_arg(copy, position, value)
    else:
        kwargs = dict(copy.kwargs)
        kwargs[name] = value
        copy.kwargs = kwargs


@mutation("F2_halo_pad", "schedule", "input halo pad one row short on the leading edge")
def _halo_pad(gm, roles, config):
    _, _, copy = roles.lhs
    pad = _copy_kw(copy, "pad", 8)
    if not pad or not any(pad):
        raise NotApplicable("the input copy is not padded")
    dim = next(i for i, p in enumerate(pad) if p)
    new = list(pad)
    new[dim] -= 1
    _set_copy_kw(copy, "pad", 8, new)
    return f"input copy pad {list(pad)} -> {new}"


@mutation("F3_pad_value", "schedule", "halo filled with 1 instead of the zero point 0 (zero-point off by one)")
def _pad_value(gm, roles, config):
    _, _, copy = roles.lhs
    pad = _copy_kw(copy, "pad", 8)
    if not pad or not any(pad):
        raise NotApplicable("the input copy is not padded")
    old = _copy_kw(copy, "pad_value", 9)
    _set_copy_kw(copy, "pad_value", 9, 1.0)
    return f"input copy pad_value {old} -> 1.0"


@mutation(
    "F4_duplicate_partial",
    "schedule",
    "reduction index clamped one short: tile nK-2 accumulated twice, tile nK-1 never",
)
def _duplicate_partial(gm, roles, config):
    k = roles.reduction_index()
    nk = _extent(k)
    if nk < 2:
        raise NotApplicable("the reduction is not split")
    for _, _, copy in (roles.lhs, roles.weight):
        # min(k, nK-2), spelled with the ops Voyager's own loop control emits (ge, sym_ite).
        last = _scalar_op(copy, operator.ge, k, nk - 1)
        clamped = _scalar_op(copy, torch.sym_ite, last, nk - 2, k)
        _replace_arg(copy, 2, [clamped if a is k else a for a in copy.args[2]])
    return f"reduction index {k.name} -> sym_ite({k.name} >= {nk - 1}, {nk - 2}, {k.name}) in both operand copies"


@mutation("F5_drop_partial", "schedule", "a split-reduction combine overwrites the partial instead of adding to it")
def _drop_partial(gm, roles, config):
    candidates = []
    for mname, mod in _modules(gm).items():
        for node in mod.graph.nodes:
            if (
                node.op == "call_function"
                and node.target is torch.ops.aten.add.Tensor
                and len(node.args) == 2
                and isinstance(node.args[1], torch.fx.Node)
                and node.args[1].op == "placeholder"
            ):
                has_relu = any("relu" in str(n.target) for n in mod.graph.nodes)
                candidates.append((has_relu, mname, mod, node))
    if not candidates:
        raise NotApplicable("no split-reduction combine in the program")
    candidates.sort(key=lambda c: (c[0], c[1]))  # prefer a middle part (no epilogue activation)
    _, mname, mod, add = candidates[0]
    add.replace_all_uses_with(add.args[0])
    mod.graph.erase_node(add)
    return f"removed {add.name} in {mname or '<root>'}: the partial it read is dropped"


@mutation(
    "F6_stale_slot",
    "schedule",
    "input copy lands in the other pipeline slot, so each compute reads the previous step's tile",
)
def _stale_slot(gm, roles, config):
    _, mod, copy = roles.lhs
    dst = copy.args[1]
    if not (isinstance(dst, torch.fx.Node) and dst.target is torch.ops.voyager.subview.default):
        raise NotApplicable("the input copy's destination is not a slot window")
    box = roles.copy_box(copy.name, "dst")
    slots = int(box.get("bank_count", 1) or 1)
    if slots < 2:
        raise NotApplicable("the input buffer is not multi-slot")
    offsets = list(dst.args[1])
    other = _scalar_op(copy, operator.mod, _scalar_op(copy, operator.add, offsets[0], 1), slots)
    with mod.graph.inserting_before(copy):
        new_dst = mod.graph.call_function(
            dst.target, (dst.args[0], [other] + offsets[1:], *dst.args[2:]), dict(dst.kwargs)
        )
    new_dst.meta = dict(dst.meta)
    _replace_arg(copy, 1, new_dst)
    return f"input copy destination slot {getattr(offsets[0], 'name', offsets[0])} -> (slot + 1) % {slots}"


@mutation(
    "F7_overlap_live",
    "memory_plan",
    "weight buffer planned on top of the input buffer, so "
    "the weight copy overwrites the input tile the same step's compute reads",
)
def _overlap(gm, roles, config):
    wbox = roles.copy_box(roles.weight[2].name, "dst")
    lbox = roles.copy_box(roles.lhs[2].name, "dst")
    walloc = _root_alloc_for_box(gm, wbox["node"])
    lalloc = _root_alloc_for_box(gm, lbox["node"])
    wseg, lseg = walloc.meta["scratchpad"], lalloc.meta["scratchpad"]
    span = int(wseg.end) - int(wseg.start)
    start = int(lseg.start)
    n = _rebind_segment(gm, wseg, new_seg=(start, start + span), new_stride=int(lalloc.meta.get("slot_stride", 0) or 0))
    return f"{walloc.name}: scratchpad {int(wseg.start)} -> {start} (= {lalloc.name}), {n} nodes"


@mutation(
    "F7b_overlap_across_groups",
    "memory_plan",
    "a buffer of one bank group planned on top of "
    "a simultaneously-live buffer of another group (the case Voyager's plan check covers)",
)
def _overlap_across_groups(gm, roles, config):
    from voyager_compiler.codegen.transform.bufferize import memory_planning as mp

    bufs = mp._buffer_lifetimes(gm, mp._buffer_identity(gm), mp._timestamps(gm), roles.accel_config)
    rows = []
    for root, bf in bufs.items():
        seg = next((m.meta["scratchpad"] for m in bf.members if "scratchpad" in m.meta), None)
        if seg is not None:
            rows.append((root, bf, seg, mp._bank_group_key(bf)))
    for i, (r1, b1, s1, g1) in enumerate(rows):
        for r2, b2, s2, g2 in rows[i + 1 :]:
            if g1 is not None and g1 == g2:
                continue  # _check_overlaps exempts one bank group's members
            if b1.def_t <= b2.last_t and b2.def_t <= b1.last_t:
                span = int(s2.end) - int(s2.start)
                n = _rebind_segment(gm, s2, new_seg=(int(s1.start), int(s1.start) + span))
                return (
                    f"{r2.name} (bank group {g2[1] if g2 else None}): scratchpad "
                    f"{int(s2.start)} -> {int(s1.start)} (= {r1.name}, group "
                    f"{g1[1] if g1 else None}), {n} nodes"
                )
    raise NotApplicable("every simultaneously-live scratchpad buffer is in one bank group")


@mutation(
    "F8_stale_bias",
    "schedule",
    "bias copy index never advances: every output-channel tile reuses the first tile's bias",
)
def _stale_bias(gm, roles, config):
    if roles.bias is None:
        raise NotApplicable("no bias copy")
    _, _, copy = roles.bias
    old = list(copy.args[2])
    _replace_arg(copy, 2, [0 for _ in old])
    return f"bias copy indices {[getattr(a, 'name', a) for a in old]} -> {[0] * len(old)}"


@mutation(
    "F9_store_current_index",
    "schedule",
    "the in-loop drain stores the finished tile at the CURRENT step's tile offset instead of its own",
)
def _store_current(gm, roles, config):
    for mname, mod, store in roles.loop_stores:
        # The store sits in a conditional region; its tile indices come in through the region's
        # operands, which the loop body computes from the previous step's index vector.
        region_owner = None
        for pname, pmod in _modules(gm).items():
            for node in pmod.graph.nodes:
                if (
                    node.op == "call_function"
                    and "cond" in str(node.target)
                    and any(
                        getattr(pmod, a.target, None) is mod
                        for a in node.args[1:3]
                        if isinstance(a, torch.fx.Node) and a.op == "get_attr"
                    )
                ):
                    region_owner = (pname, pmod, node)
        if region_owner is None:
            continue
        _, pmod, cond = region_owner
        operands = list(cond.args[3])
        swapped = []
        for i, value in enumerate(operands):
            if isinstance(value, torch.fx.Node):
                cur = _current_counterpart(value)
                if cur is not None:
                    operands[i] = cur
                    swapped.append(f"{value.name}->{cur.name}")
        if swapped:
            _replace_arg(cond, 3, tuple(operands))
            return f"{store.name}: drained-tile indices {', '.join(swapped)}"
    raise NotApplicable("no in-loop store fed by the previous step's tile index")


@mutation("F10_drain_offset", "schedule", "the epilogue drain of the LAST tile writes one tile short (edge tile)")
def _drain_offset(gm, roles, config):
    for _, _, store in roles.epilogue_stores:
        idx = list(store.args[2])
        if all(isinstance(v, int) for v in idx) and any(idx):
            pos = next(i for i, v in enumerate(idx) if v)
            new = list(idx)
            new[pos] -= 1
            _replace_arg(store, 2, new)
            return f"{store.name}: indices {idx} -> {new}"
    raise NotApplicable("no epilogue store with literal tile indices")


@mutation(
    "H1_drop_wait",
    "hazard",
    "one semaphore wait removed from the loop (a producer/consumer order the hardware relies on)",
)
def _drop_wait(gm, roles, config):
    for mname, mod in _modules(gm).items():
        if mname == "":
            continue
        for node in mod.graph.nodes:
            if node.op == "call_function" and node.target is torch.ops.voyager.async_wait.default:
                mod.graph.erase_node(node)
                return f"erased {node.name} in {mname}"
    raise NotApplicable("no semaphore wait inside the loop")


@mutation(
    "H2_alias_slots",
    "hazard",
    "input buffer's two pipeline slots planned at one address (the next load may overwrite the tile still being read)",
)
def _alias_slots(gm, roles, config):
    box = roles.copy_box(roles.lhs[2].name, "dst")
    if int(box.get("bank_count", 1) or 1) < 2:
        raise NotApplicable("the input buffer is not multi-slot")
    alloc = _root_alloc_for_box(gm, box["node"])
    seg = alloc.meta["scratchpad"]
    stride = int(alloc.meta.get("slot_stride", 0) or 0)
    n = _rebind_segment(gm, seg, new_stride=0)
    return f"{alloc.name}: slot stride {stride} -> 0 ({n} nodes)"


def _fetch_attr(gm, target: str):
    value = gm
    for part in str(target).split("."):  # a get_attr target is a dotted path
        value = getattr(value, part, None)
    return value


def _set_attr(gm, target: str, value) -> None:
    *parents, leaf = str(target).split(".")
    owner = gm
    for part in parents:
        owner = getattr(owner, part)
    setattr(owner, leaf, value)


def _scale_nodes(gm):
    out = []
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        value = _fetch_attr(gm, node.target)
        if (
            isinstance(value, torch.Tensor)
            and value.numel() == 1
            and value.is_floating_point()
            and any("dequantize" in str(u.target) or u.op in ("call_function", "call_module") for u in node.users)
        ):
            out.append((node, value))
    return out


def _scale_fault(gm, factor: float) -> str:
    found = _scale_nodes(gm)
    if not found:
        raise NotApplicable("no scalar dequantize scale")
    notes = []
    for node, value in found:
        new = (value.to(torch.float32) * factor).to(value.dtype)
        _set_attr(gm, node.target, new)
        notes.append(f"{node.target}: {value.item():.6g} -> {new.item():.6g} (x{new.item() / value.item():.4f})")
    return "; ".join(notes)


@mutation("Q1_scale_3pct", "numerics", "dequantize scale off by +3% (within the 5% tolerance)")
def _scale3(gm, roles, config):
    return _scale_fault(gm, 1.03)


@mutation("Q2_scale_10pct", "numerics", "dequantize scale off by +10%")
def _scale10(gm, roles, config):
    return _scale_fault(gm, 1.10)


# ------------------------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _operands(program: Program) -> dict:
    """The DRAM tensors the schedule reads, as exact integers, named as in the IR."""
    gm = program.gm
    out = {}
    names = [b["node"] for b in program.model_dict.get("inputs", ())] + [
        b["node"] for b in program.model_dict.get("parameters", ())
    ]
    for name in names:
        if name in {b["node"] for b in program.model_dict.get("inputs", ())}:
            value = program.lowered_input
        else:
            node = next(n for n in gm.graph.nodes if n.name == name)
            value = _fetch_attr(gm, node.target)
        as64 = value.to(torch.float64)
        if not torch.equal(as64, as64.round()):
            raise RuntimeError(f"{name} is not integral; the exact reference needs integers")
        out[name] = {"array": as64.to(torch.int64).numpy(), "container_dtype": str(value.dtype)}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", required=True, type=Path, help="fixture dir (manifest+model)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--only", nargs="*", help="mutation ids to run (default: all)")
    args = parser.parse_args(argv)

    root = Path(os.environ["MERLIN_EXT_VOYAGER_COMPILER"]).resolve()
    sys.path.insert(0, str(root / "test"))
    manifest = json.loads((args.fixture / "manifest.json").read_text())
    fixture_model = json.loads((args.fixture / "model.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    base = Program(manifest, args.out / "baseline")
    base_lowered = _run_eager(base.gm, base.lowered_input)
    identical = base.model_dict == fixture_model
    pre = voyager_criterion(base.old_output, base.pre_tiling_output)
    post = voyager_criterion(base.old_output, base_lowered)
    operands = _operands(base)
    np.savez(args.out / "operands.npz", **{k: v["array"] for k, v in operands.items()})
    (args.out / "baseline" / "model.json").write_text(
        json.dumps(base.model_dict, sort_keys=True, separators=(",", ":"))
    )
    # Voyager's own quantized reference (float), so the merlin side can check that its exact integer
    # reference and these operands describe the same computation.
    old_flat, _ = tree_flatten(base.old_output)
    np.save(args.out / "voyager_reference.npy", old_flat[0].to(torch.float32).numpy())
    scales = {str(node.target): float(value.item()) for node, value in _scale_nodes(base.gm)}
    torch.save(
        {"old_output": base.old_output, "pre_tiling_output": base.pre_tiling_output, "lowered_output": base_lowered},
        args.out / "baseline" / "outputs.pt",
    )
    summary = {
        "fixture": str(args.fixture),
        "workload": manifest["workload"],
        "config": manifest["config"],
        "fixture_ir_identical": identical,
        "baseline_model_txt_sha256": _sha(base.model_txt),
        "operand_containers": {k: v["container_dtype"] for k, v in operands.items()},
        "dequantize_scales": scales,
        "reference_shape": list(old_flat[0].shape),
        "pre_tiling_check": pre,
        "post_bufferization_check_unmutated": post,
        "plan_invariant_warnings_unmutated": plan_invariants(base.gm, base.config),
        "baseline_seconds": round(time.time() - t0, 2),
        "mutations": [],
    }
    import test_codegen

    rtol, atol = test_codegen.OUTPUT_RTOL, test_codegen.OUTPUT_ATOL

    wanted = args.only or list(CATALOGUE)
    for mid in wanted:
        spec = CATALOGUE[mid]
        row = {"id": mid, "class": spec["class"], "title": spec["title"]}
        mdir = args.out / mid
        t1 = time.time()
        try:
            program = Program(manifest, mdir / "compile")
            row["recompile_model_txt_matches_baseline"] = program.model_txt == base.model_txt
            roles = Roles(program.gm, program.model_dict)
            roles.accel_config = program.config
            try:
                row["change"] = spec["apply"](program.gm, roles, manifest["config"])
            except NotApplicable as exc:
                row.update(status="not_applicable", reason=str(exc))
                summary["mutations"].append(row)
                continue
            _recompile(program.gm)
        except Exception as exc:
            row.update(
                status="harness_error", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-2000:]
            )
            summary["mutations"].append(row)
            continue
        row["status"] = "applied"
        # (a) run the faulty bufferized graph in Voyager's stack
        try:
            lowered = _run_eager(program.gm, program.lowered_input)
            row["eager"] = {"raised": None}
            row["post_bufferization_check"] = voyager_criterion(program.old_output, lowered)
            row["error_vs_voyager_reference"] = error_metrics(program.old_output, lowered, rtol, atol)
            row["error_vs_unmutated_lowered"] = error_metrics(base_lowered, lowered, rtol, atol)
            torch.save(lowered, mdir / "lowered_output.pt")
        except Exception as exc:
            row["eager"] = {"raised": f"{type(exc).__name__}: {str(exc)[:300]}"}
            row["post_bufferization_check"] = {"result": "raised"}
        # (b) re-emit the faulty program's IR with Voyager's own emitter
        try:
            txt, as_dict = _emit(program.gm, program.lowered_input)
            (mdir / "model.txt").write_text(txt)
            (mdir / "model.json").write_text(json.dumps(as_dict, sort_keys=True, separators=(",", ":")))
            row["emit"] = {
                "ok": True,
                "model_txt_sha256": _sha(txt),
                "model_txt_differs_from_baseline": txt != base.model_txt,
            }
            if txt != base.model_txt:
                import difflib

                diff = list(
                    difflib.unified_diff(
                        base.model_txt.splitlines(),
                        txt.splitlines(),
                        "baseline/model.txt",
                        f"{mid}/model.txt",
                        lineterm="",
                        n=2,
                    )
                )
                row["emit"]["diff_lines"] = len(diff)
                (mdir / "model.txt.diff").write_text("\n".join(diff[:400]) + "\n")
        except Exception as exc:
            row["emit"] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        # (c) Voyager's memory-plan invariant check on the faulty plan (last: it only reads meta)
        try:
            row["plan_invariant_warnings"] = plan_invariants(program.gm, program.config)
        except Exception as exc:
            row["plan_invariant_warnings"] = None
            row["plan_invariant_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        row["seconds"] = round(time.time() - t1, 2)
        summary["mutations"].append(row)
        print(
            json.dumps(
                {
                    "id": mid,
                    "status": row["status"],
                    "post": row.get("post_bufferization_check", {}).get("result"),
                    "txt_changed": row.get("emit", {}).get("model_txt_differs_from_baseline"),
                }
            ),
            flush=True,
        )
    (args.out / "voyager_side.json").write_text(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

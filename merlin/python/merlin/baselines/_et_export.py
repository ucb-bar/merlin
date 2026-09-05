"""AOT export helper for the ExecuTorch + XNNPACK baseline arm (runs UNDER the ExecuTorch venv).

Invoked as a subprocess by :mod:`merlin.baselines.executorch` using the dedicated ExecuTorch export
venv (``build/baselines/executorch/et-venv``), because ExecuTorch + its pinned torch are NOT in
merlin's ``.venv``. Dependency-light on merlin (everything arrives via argv) so it can run under a
foreign interpreter.

Flow (on OUR model + OUR captured input/golden, always through the XNNPACK partitioner):

    loader.get_model_and_inputs()      # HF model from model2MLIR/workloads/<model>/loader.py
      -> OVERRIDE example input with the captured inputs.npz (so it matches golden.npy)
      -> torch.export.export(model, (captured_input,))
      -> to_edge_transform_and_lower(partitioner=[XnnpackPartitioner()])
      -> to_executorch(config=external_constants=True)      # weights -> .ptd, program -> small .pte
      -> write model.pte (+ *.ptd next to it) + input0.bin (raw captured input bytes)

We externalize constants (``external_constants=True``) because a whole fp32 LLM's weights (~4 GB)
blow past flatbuffer's 2 GB program-buffer limit if embedded. The board-side stock executor_runner
then loads ``--model_path=model.pte --data_path=<ptd> --inputs=input0.bin`` and dumps the output;
the runner (in merlin's venv) compares that dump to golden.npy off-device — an honest same-input,
same-golden gate with no fabricated numbers.

Prints ``ET_EXPORT_JSON {...}`` with the pte/ptd/input paths, output shape/dtype, and how many
graph nodes XNNPACK delegated vs stayed portable (a coarse delegation signal complementing the
binary-level RVV audit).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path


def _module_fqn(node) -> str | None:
    """The deepest nn.Module FQN a node came from (from ``node.meta['nn_module_stack']``) — the SAME
    key model2MLIR reads for ``prov.fqn``, so a Merlin region and this ET node align on one fqn."""
    nms = node.meta.get("nn_module_stack")
    if not nms:
        return None
    paths = [v[0] for v in nms.values() if isinstance(v, (tuple, list)) and v and v[0]]
    return max(paths, key=lambda p: str(p).count(".")) if paths else None


def extract_fqn_map(exported) -> dict[str, dict]:
    """Map each compute op of the EXPORTED (pre-delegation) graph to its originating model layer.

    Read pre-lowering: after the XNNPACK partitioner delegates the linears they become opaque
    call_delegate nodes, so the per-layer identity only survives here. Keyed by node name; each entry
    carries the fqn + the aten op. This is the ET analogue of Merlin's prov.fqn — the cross-compiler
    join key. Role is derived downstream in the merlin venv (role_from_fqn), not here."""
    out: dict[str, dict] = {}
    for n in exported.graph.nodes:
        if n.op == "call_function":
            fqn = _module_fqn(n)
            if fqn:
                out[n.name] = {"fqn": fqn, "aten": str(n.target).split(".")[-1]}
    return out


def _apply_loader_compat_shims() -> list[str]:
    """Faithful compat shims for loader/framework version drift (NOT numeric changes).

    The m2m loaders were written against older dep versions than the ExecuTorch export venv
    ships. Each shim just re-exposes a symbol under the name the loader expects; none change
    model math. Returns the list of shims that actually applied (recorded in the notes).

      * ``BitNet`` config alias — bitvla's loader builds a Llava config with the (old)
        capitalized ``text_config['model_type']='BitNet'``; current transformers registers
        BitNet under the lowercase key ``'bitnet'``, so ``CONFIG_MAPPING['BitNet']`` KeyErrors
        (fails identically in the m2m venv too — loader-inherent drift). Register the alias.
    """
    applied: list[str] = []
    try:  # torch_compilable_check no-op (lerobot 0.6 eo1 policy vs transformers>=5)
        import transformers.utils as _tu
        if not hasattr(_tu, "torch_compilable_check"):
            # lerobot.policies.__init__ eagerly imports eo1, which imports this compile-guard
            # util removed in transformers>=5. It is a torch.compile guard, a no-op for eager
            # export; stub it so the smolvla policy module (an eo1 sibling) can be imported.
            _tu.torch_compilable_check = lambda *a, **k: None  # noqa: E731
            applied.append("torch_compilable_check-noop")
    except Exception:  # noqa: BLE001
        pass
    try:  # shape-static vision patch-position table (SmolVLM-based VLAs)
        applied += _shape_static_vision_patch_positions()
    except Exception:  # noqa: BLE001
        pass
    try:  # BitNet capitalized-key alias (bitvla)
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        if "BitNet" not in CONFIG_MAPPING:
            from transformers.models.bitnet.configuration_bitnet import BitNetConfig
            CONFIG_MAPPING.register("BitNet", BitNetConfig, exist_ok=True)
            applied.append("bitnet-config-alias")
    except Exception:  # noqa: BLE001 - shim is best-effort; loader failure surfaces the real reason
        pass
    return applied


#: The vision-embedding forward this process replaced, kept so the rewrite can be PROVED an identity
#: against the real captured input rather than argued to be one.
_ORIGINAL_VISION_PATCH_FORWARD = None


def _vision_position_ids(self, pixel_values, patch_attention_mask):
    """SmolVLM's vision patch-position table, computed without a data-dependent SHAPE.

    Upstream derives the table from the patch mask's DATA, in two ways that both mint an unbacked
    symint under ``torch.export``:

      1. ``torch.arange(p_attn_mask[:, 0].sum())`` -- an arange over a 0-dim TENSOR is a
         data-dependent size (transformers 5.0.x; upstream itself vectorised this away in 5.9.0 by
         ranging over the static ``patch_attention_mask.size(1)/size(2)`` and keeping the sum only
         as the arithmetic ``step_h = 1/nb_patches_h``);
      2. ``position_ids[mask] = pos_ids`` -- boolean-mask indexing, whose selected-element count is
         data-dependent.

    ExecuTorch then cannot decide ``u31 < 1`` in ``exir/tensor.py`` ``dim_order_from_stride`` and
    ``to_executorch`` refuses the program, so no ``.pte`` exists at all.

    This body is upstream-5.9.0's vectorised form with (2) written as the shape-preserving identity
    ``position_ids = where(mask, pos_ids, 0)`` -- exact because ``position_ids`` starts as
    ``full(..., 0)``, so the masked assignment writes ``pos_ids`` where the mask is true and leaves
    zero elsewhere, which is what ``where`` computes. Neither rewrite touches a number; the caller
    proves that by running the original forward beside it on the captured input.
    """
    import torch

    batch_size, _, max_im_h, max_im_w = pixel_values.shape
    max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
    boundaries = torch.arange(1 / self.num_patches_per_side, 1.0,
                              1 / self.num_patches_per_side, device=pixel_values.device)
    position_ids = torch.full(size=(batch_size, max_nb_patches_h * max_nb_patches_w),
                              fill_value=0, device=pixel_values.device)

    nb_patches_h = patch_attention_mask[:, :, 0].sum(dim=1)   # a VALUE, never a shape
    nb_patches_w = patch_attention_mask[:, 0, :].sum(dim=1)
    step_h, step_w = 1.0 / nb_patches_h, 1.0 / nb_patches_w

    h_indices = torch.arange(patch_attention_mask.size(1), device=position_ids.device,
                             dtype=torch.float32)
    w_indices = torch.arange(patch_attention_mask.size(2), device=position_ids.device,
                             dtype=torch.float32)
    fractional_coords_h = torch.clamp(h_indices[None, :] * step_h[:, None], max=(1.0 - 1e-6))
    fractional_coords_w = torch.clamp(w_indices[None, :] * step_w[:, None], max=(1.0 - 1e-6))
    fractional_coords_h = fractional_coords_h.to(pixel_values.dtype)
    fractional_coords_w = fractional_coords_w.to(pixel_values.dtype)

    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (bucket_coords_h[:, :, None] * self.num_patches_per_side
               + bucket_coords_w[:, None, :]).reshape(batch_size, -1)

    mask = patch_attention_mask.reshape(batch_size, -1)
    return torch.where(mask, pos_ids.to(position_ids.dtype), torch.zeros_like(position_ids))


def _shape_static_vision_patch_forward(self, pixel_values, patch_attention_mask):
    """The patched forward: static position table + the unchanged patch embedding."""
    patch_embeds = self.patch_embedding(pixel_values)
    embeddings = patch_embeds.flatten(2).transpose(1, 2)
    return embeddings + self.position_embedding(
        _vision_position_ids(self, pixel_values, patch_attention_mask))


def _frozen_vision_patch_forward(self, pixel_values, patch_attention_mask):
    """As above with the position TABLE frozen to the constant it evaluates to.

    Needed because ExecuTorch implements no ``aten::bucketize`` kernel in any of its three kernel
    libraries (portable / optimized / quantized), so a program that still calls it is refused at
    ``Method::load`` with OperatorMissing -- an export that succeeds and a runner that cannot load
    the result. The table depends only on the patch mask, which the caller builds from SHAPES
    (``SmolVLMVisionTransformer.forward`` makes it with ``torch.ones`` when it is passed None, which
    is what SmolVLA's ``embed_image`` does), so freezing it specialises the program to the exported
    configuration exactly the way every other static shape in an exported program already is.
    Installed only after the frozen table is shown to reproduce the ORIGINAL forward bit-for-bit.
    """
    patch_embeds = self.patch_embedding(pixel_values)
    embeddings = patch_embeds.flatten(2).transpose(1, 2)
    return embeddings + self.position_embedding(self._merlin_frozen_position_ids)


def _shape_static_vision_patch_positions() -> list[str]:
    """Install :func:`_shape_static_vision_patch_forward` on SmolVLM's vision embeddings."""
    global _ORIGINAL_VISION_PATCH_FORWARD
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMVisionEmbeddings

    if _ORIGINAL_VISION_PATCH_FORWARD is not None:
        return []
    _ORIGINAL_VISION_PATCH_FORWARD = SmolVLMVisionEmbeddings.forward
    SmolVLMVisionEmbeddings.forward = _shape_static_vision_patch_forward
    return ["smolvlm-vision-pos-ids-shape-static"]


def verify_shape_static_vision_patch(model, captured) -> str:
    """Prove the vision-patch rewrite is an IDENTITY on this model + input, or raise.

    Runs one eager forward with a pre-hook that records what actually reached the patched module,
    then replays the ORIGINAL forward on those same tensors and demands a bit-identical result. A
    rewrite argued to be an identity and never checked is exactly the kind of change that later gets
    attributed to the model instead of to us, so this refuses to export rather than reporting a
    tolerance.
    """
    import torch

    if _ORIGINAL_VISION_PATCH_FORWARD is None:
        return ""
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMVisionEmbeddings

    mods = [m for m in model.modules() if isinstance(m, SmolVLMVisionEmbeddings)]
    if not mods:
        return ""
    seen: dict[int, tuple] = {}

    def _hook(mod, args, kwargs):
        pix = kwargs.get("pixel_values", args[0] if args else None)
        msk = kwargs.get("patch_attention_mask", args[1] if len(args) > 1 else None)
        if pix is not None and msk is not None:
            seen[id(mod)] = (pix.detach().clone(), msk.detach().clone())
        return None

    handles = [m.register_forward_pre_hook(_hook, with_kwargs=True) for m in mods]
    try:
        with torch.no_grad():
            model(*captured)
    finally:
        for h in handles:
            h.remove()

    checked = []
    for mod in mods:
        got = seen.get(id(mod))
        if got is None:
            raise RuntimeError("vision patch-position shim: the patched module was never called on "
                               "the captured input, so the rewrite could not be verified")
        pix, msk = got
        with torch.no_grad():
            ref = _ORIGINAL_VISION_PATCH_FORWARD(mod, pix, msk)
            new = _shape_static_vision_patch_forward(mod, pix, msk)
            table = _vision_position_ids(mod, pix, msk)
        if not torch.equal(ref, new):
            raise RuntimeError(
                "vision patch-position shim CHANGED NUMERICS "
                f"(max |diff| {float((ref.float() - new.float()).abs().max())}); refusing to export")
        # Freeze the table (no aten::bucketize kernel exists in ANY ExecuTorch kernel library), then
        # prove the frozen form too -- a specialisation accepted on an argument is how a wrong
        # constant gets into a program that still looks exported.
        mod.register_buffer("_merlin_frozen_position_ids", table.detach().clone(),
                            persistent=False)
        mod.forward = types.MethodType(_frozen_vision_patch_forward, mod)
        with torch.no_grad():
            frozen = mod.forward(pix, msk)
        if not torch.equal(ref, frozen):
            raise RuntimeError(
                "frozen vision patch-position table CHANGED NUMERICS "
                f"(max |diff| {float((ref.float() - frozen.float()).abs().max())}); refusing")
        checked.append(f"{tuple(int(x) for x in msk.shape)}")
    return ("shape-static + FROZEN vision patch positions, VERIFIED bit-identical to upstream on "
            f"{len(checked)} module(s) with mask shape(s) {', '.join(checked)}; removes the "
            "unbacked symint AND the aten::bucketize ExecuTorch has no kernel for")


def _load_loader(loader_path: Path):
    _apply_loader_compat_shims()
    spec = importlib.util.spec_from_file_location("_m2m_loader", loader_path)
    mod = importlib.util.module_from_spec(spec)
    # Register under the module name BEFORE exec so a strict torch.export (dynamo) that
    # re-imports the defining module of a traced nn.Module class finds it (otherwise the
    # strict fallback dies with ModuleNotFoundError: No module named '_m2m_loader').
    sys.modules["_m2m_loader"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def reconcile_input_arity(captured, keys, example, *, source: str = "inputs.npz"):
    """Reconcile a capture bundle's input tuple with the loader's forward arity.

    A capture bundle records the tensors the capture VARIED. A loader's forward may take more: a
    stateful controller also takes its recurrent hidden/cell state, which the bundle does not store
    because the capture starts it at the value the loader itself defines. Splatting the npz alone
    then raises ``forward() missing N required positional arguments`` and the model reads as
    unexportable when it was only under-fed.

    So the captured tensors fill the leading slots and the loader's own example tuple supplies the
    trailing ones it alone knows the initial value of. Never the reverse: MORE captured tensors than
    the forward accepts is a real bundle/loader ABI disagreement, so fail closed and name it rather
    than guessing which inputs to drop.

    Returns ``(captured, keys, note)`` — ``note`` empty when the arities already matched.
    """
    if len(captured) > len(example):
        raise RuntimeError(
            f"input-arity mismatch: {source} holds {len(captured)} tensors {list(keys)} but the "
            f"loader's forward takes {len(example)} — the bundle and the loader disagree about this "
            "model's ABI (refusing to guess which inputs to drop)")
    if len(captured) == len(example):
        return tuple(captured), list(keys), ""
    tail = tuple(example[len(captured):])
    note = (f"input arity {len(captured)} captured + {len(tail)} loader-initial "
            f"{[tuple(t.shape) for t in tail]}")
    return (tuple(captured) + tail,
            list(keys) + [f"loader_init{i}" for i in range(len(tail))],
            note)


def _dtype_preserving_quantizer(quantizer_cls, qcfg):
    """``quantizer_cls`` with a ``transform_for_annotation`` that does not retype integer arithmetic.

    XNNPACKQuantizer's ``transform_for_annotation`` is ``_convert_scalars_to_attrs``: for every
    ``aten.add.Tensor`` / ``aten.mul.Tensor`` it lifts a scalar argument into a buffer so the scalar
    can carry a quantization annotation. It builds that buffer as ``torch.tensor(float(arg))`` —
    unconditionally **float32**, with no reference to the dtype of the node it is rewriting.

    On any graph whose mask/position arithmetic is integer (every HF ``transformers`` decoder: a
    ``cumsum`` of an int64 attention mask, offset by a literal ``+ 1``) that promotes the result to
    float, and the float tensor then reaches ``aten.index.Tensor`` as an INDEX:

        IndexError: tensors used as indices must be long, int, byte or bool tensors

    raised at calibration, after ``prepare_pt2e`` returned cleanly. It fires with an empty quantizer
    too, so it is the transform and not the annotation, and no ``filter_fn`` / ``set_module_*``
    exclusion reaches it — which is why the whole-model int8 path was a module swap instead.

    The fix is to read the dtype the pass is supposed to preserve rather than assume one: skip nodes
    whose own ``meta['val']`` is not floating point. Those are exactly the nodes the lift cannot
    help — XNNPACK has no integer add/mul ukernel to annotate — so a float graph is transformed
    EXACTLY as before (byte-identical for every model that already exported) and an integer index
    computation is left as the integer computation it is.

    MEASURED with this in place: tiny_llama qd8 goes from failing at calibration to lowering with
    155/155 Linears int8; whisper_tiny reaches ``convert_pt2e`` with 65 int8 Linears (it then fails
    later and separately, inside the XNNPACK partitioner, with a dependency cycle).
    """
    import torch
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer_utils import (
        get_new_attr_name_with_prefix)

    scalar_ops = (torch.ops.aten.add.Tensor, torch.ops.aten.mul.Tensor)

    class _DtypePreserving(quantizer_cls):  # type: ignore[valid-type,misc]
        def transform_for_annotation(self, model):
            for n in model.graph.nodes:
                if n.op != "call_function" or n.target not in scalar_ops:
                    continue
                val = n.meta.get("val")
                dtype = getattr(val, "dtype", None)
                if dtype is not None and not dtype.is_floating_point:
                    continue          # integer arithmetic: leave it integer
                new_args = []
                for arg in n.args:
                    if isinstance(arg, torch.fx.Node):
                        new_args.append(arg)
                        continue
                    name = get_new_attr_name_with_prefix("_tensor_constant_")(model)
                    const = torch.tensor(float(arg))
                    model.register_buffer(name, const)
                    with model.graph.inserting_before(n):
                        attr = model.graph.create_node("get_attr", name, (), {})
                        attr.meta["val"] = n.meta["val"].fake_mode.from_tensor(
                            const, static_shapes=True)
                    new_args.append(attr)
                n.args = tuple(new_args)
            model.recompile()
            return model

    return _DtypePreserving().set_global(qcfg)


def _bound_dynamic_qdq_channels_last_walk() -> str:
    """Keep XNNPACK's NHWC-tagging pass inside the qdq chain it is allowed to relayout.

    ``ChannelsLastTaggedReshapePass.input_to_nhwc`` validates its argument with
    ``can_be_converted_to_nhwc(input_node)`` — which requires a rank-4 tensor — and only then, in
    the ``is_dynamic_qdq(input_node)`` branch, walks ``input_node = input_node.args[0]``. Its own
    comment says the walk finds "the original source node" of the quantize chain, but nothing stops
    it: it runs to the root of the arg chain, and then ``input_node.replace_all_uses_with(...)``
    re-points EVERY consumer of wherever it landed at the inserted
    ``_to_copy(memory_format=torch.channels_last)``. Two distinct failures come out of that, both
    reachable only on qd8 (the branch is gated on ``is_dynamic_qdq``), which is why the model
    exports fine as fp32 and as qs8 and this read as "qd8 is impossible for this model":

    1. The walk REBINDS the name the rank check was made about, so the copy can land on an ancestor
       of a different rank — ``RuntimeError: required rank 4 tensor to use channels_last format``
       at export.
    2. Bounded to the validated rank, the walk still leaves the quantize chain, and the unbounded
       ``replace_all_uses_with`` then re-points consumers that never asked for NHWC — including a
       consumer that is a PARTITION OUTPUT. ``call`` tags nodes in graph order over a snapshot taken
       before any of this, so a node already tagged NCHW silently starts consuming an NHWC value;
       the ``output`` handler later reads that stale NCHW tag and inserts no compensating copy. The
       delegate then hands back an NHWC buffer for a tensor ExecuTorch planned NCHW, and the model
       LOADS but dies at execute:

           Attempted to resize a static tensor. Expected shape (1, 32, 15, 23),
                                                 but received (1, 15, 23, 32).
           Failed to resize output tensor for XNNExecutor
           CALL_DELEGATE execute failed at instruction 31: 0x10

       MEASURED on lstmnetvit: the shared value is ``x.view(1,15,23,-1).permute(0,3,1,2)``, which
       feeds the patchmerge conv AND leaves the partition as a skip connection.

    The fix bounds both. The walk stops at the first non-qdq ancestor — the "original source node"
    the upstream comment names — and still refuses to leave the validated rank; and the rewiring
    re-points only the qdq consumers of that source, so a value that also leaves the partition keeps
    its NCHW identity. Both are invariants the code already assumes; neither was enforced.

    Applied by hand-porting the pinned upstream method (ExecuTorch is pinned by sha and its identity
    is recorded with every result). It is therefore FAIL-CLOSED on drift: if the installed
    ``input_to_nhwc`` no longer contains the unbounded walk this fork was written against, nothing is
    patched and the returned status says so, rather than silently running a stale copy of someone
    else's pass. Returns the status string to record in the export summary.

    MEASURED: lstmnetvit qd8 goes from failing this pass to lowering with 33 delegated nodes of 172
    and EXECUTING (host x86 ``executor_runner``, 3 iterations, output cos 0.99996 vs the eager fp32
    golden); unbounded it raised at export, rank-bounded alone it aborted at CALL_DELEGATE 31.
    """
    import inspect

    try:
        import torch
        from executorch.backends.xnnpack._passes import channels_last_tagged_reshape_pass as clp
        from executorch.backends.xnnpack.utils.utils import is_param_node
        from executorch.exir.dialects._ops import ops as exir_ops
    except ImportError as e:
        # Reached when this module is imported OUTSIDE the ExecuTorch venv (it is also read as a
        # plain module by the board-free tests). Report; never raise — the caller's job is to record
        # whether the fix took effect, and an exception here would be indistinguishable from the
        # export failure the fix exists to prevent.
        return f"channels-last qd8 walk NOT bounded (import failed: {e})"

    cls = clp.ChannelsLastTaggedReshapePass
    try:
        src = inspect.getsource(cls.input_to_nhwc)
    except (OSError, TypeError) as e:
        return f"channels-last qd8 walk NOT bounded (source unavailable: {e})"
    if ("is_dynamic_qdq(input_node)" not in src or "input_node = input_node.args[0]" not in src
            or "input_node.replace_all_uses_with(input_node_nhwc)" not in src):
        return ("channels-last qd8 walk NOT bounded (upstream input_to_nhwc no longer matches the "
                "pinned shape this fix was written against — re-verify before re-applying)")

    # The chain the relayout is allowed to move through, named as OP IDENTITIES rather than as
    # spellings of their names, so a renamed overload fails loudly instead of quietly matching
    # nothing (which would collapse the walk to a no-op and re-open failure 1).
    _qd = exir_ops.edge.quantized_decomposed
    _QDQ_TARGETS = {_qd.quantize_per_tensor.default, _qd.quantize_per_tensor.tensor,
                    _qd.dequantize_per_tensor.default, _qd.dequantize_per_tensor.tensor,
                    _qd.choose_qparams.tensor}

    def _is_qdq(node) -> bool:
        """True iff the node is part of the quantize chain the copy may be pushed under."""
        return getattr(node, "op", None) == "call_function" and node.target in _QDQ_TARGETS

    def _rank(node) -> int | None:
        val = getattr(node, "meta", {}).get("val")
        shape = getattr(val, "shape", None)
        return None if shape is None else len(shape)

    def input_to_nhwc(self, graph_module, input_node, target_node):
        if is_param_node(self.exported_program, input_node):
            if (cls.XNN_NHWC_NODE in input_node.meta and cls.is_nchw_node(input_node)):
                raise AssertionError("The same constant data tensor can't be used in NCHW format "
                                     "in one place and NHWC in another")
            self.mark_as_nhwc_node(input_node)
        if input_node.op == "placeholder":
            if self._is_nhwc(input_node.meta["val"][0]):
                return
        elif cls.is_nhwc_node(input_node):
            return
        if self.input_dim_order(input_node, clp.InputDimOrder.NHWC):
            return
        if not self.can_be_converted_to_nhwc(input_node):
            raise AssertionError("Attempting to convert non-NHWC compatible node to NHWC")

        if cls.PARTNER_NODE in input_node.meta:
            input_node_nhwc = input_node.meta[cls.PARTNER_NODE]
        else:
            is_dynamic_input = clp.is_dynamic_qdq(input_node)
            if is_dynamic_input:
                validated_rank = _rank(input_node)
                while (_is_qdq(input_node) and getattr(input_node, "args", None)
                       and isinstance(input_node.args[0], torch.fx.Node)):
                    candidate = input_node.args[0]
                    if _rank(candidate) != validated_rank:
                        break            # bound 1a: never leave the rank that was validated
                    input_node = candidate   # bound 1b: `while _is_qdq` — stop at the source node
            with graph_module.graph.inserting_after(input_node):
                input_node_nhwc = self.create_call_function_node(
                    graph_module=graph_module, target=exir_ops.edge.aten._to_copy.default,
                    args=(input_node,), memory_format=torch.channels_last)
                cls.mark_as_nhwc_node(input_node_nhwc)
            if is_dynamic_input:
                # Bound 2: only the quantize chain follows the relayout. Any other consumer of the
                # source — notably one that is a partition output — keeps the NCHW value, which is
                # the identity ExecuTorch's memory plan gave it.
                input_node.replace_all_uses_with(input_node_nhwc, delete_user_cb=_is_qdq)
                input_node_nhwc.args = (input_node,)

        self.insert_copy_and_assign_partner_nodes_quantization_sensitive(
            graph_module=graph_module, original_input=input_node,
            copy_node=input_node_nhwc, target_node=target_node)

    cls.input_to_nhwc = input_to_nhwc
    return ("channels-last qd8 walk bounded to the qdq chain; "
            "NHWC rewiring bounded to qdq consumers")


def _linear_subgraph(model):
    """Extract the model's REAL linear-heavy subgraph for the int8 path.

    Whole-model PT2E is IMPOSSIBLE on HF Llama here: ``prepare_pt2e``'s ``transform_for_annotation``
    pass corrupts an integer-index dtype (the position/causal-mask ``aten.index.Tensor`` on a
    ``cumsum``) at calibration — it fails even with an EMPTY quantizer, so it is the transform pass,
    not the observers/annotation, and no ``set_module_*``/``filter_fn`` exclusion can dodge it. So we
    quantize the largest self-contained pure-Linear region: **ALL of a decoder layer's Linears** —
    attention q/k/v/o projections + the SwiGLU MLP (gate/up/down), actual trained weights — driven by
    a seeded hidden state, with the non-linear glue (RoPE index, causal mask, softmax, norms,
    embedding) kept fp32 and OUT of quantization. This is the maximal int8 an HF-Llama export allows
    on this ExecuTorch/PT2E; ``subgraph_note`` records exactly what stayed fp32.

    Returns ``(submodule, (hidden_state_input,), keys, note)``.
    """
    import torch

    # Find a decoder layer with BOTH an attention block (q/k/v/o_proj) and an MLP (gate/up/down_proj).
    attn = mlp = None
    for _name, m in model.named_modules():
        if attn is None and all(hasattr(m, a) for a in ("q_proj", "k_proj", "v_proj", "o_proj")):
            attn = m
        if mlp is None and all(hasattr(m, a) for a in ("gate_proj", "up_proj", "down_proj")):
            mlp = m
        if attn is not None and mlp is not None:
            break

    if attn is not None and mlp is not None:
        d = mlp.gate_proj.in_features
        act = getattr(mlp, "act_fn", torch.nn.SiLU())

        class _AllLinears(torch.nn.Module):
            """All 7 decoder-layer Linears int8; attention math (RoPE/mask/softmax) stays fp32 glue.

            The attention projections are exercised as real matmuls (q/k/v via the projections, o on
            q's output); the k/v outputs are kept live (reduced + broadcast) so every projection is
            quantized. This is NOT the attention numerics — it is a linear-coverage vehicle so all
            of the layer's GEMMs run int8 on XNNPACK qs8 RVV, gated cos vs the SAME-shape fp32."""
            def __init__(self, attn, mlp, act):
                super().__init__()
                self.q, self.k, self.v, self.o = attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
                self.g, self.u, self.dn, self.act = mlp.gate_proj, mlp.up_proj, mlp.down_proj, act

            def forward(self, x):
                a = self.o(self.q(x))                                   # q_proj, o_proj (q_out->d)
                kv = self.k(x).sum(-1, keepdim=True) + self.v(x).sum(-1, keepdim=True)  # keep k,v live
                m = self.dn(self.act(self.g(x)) * self.u(x))            # MLP linears
                return a + m + kv

        sub = _AllLinears(attn, mlp, act).eval()
        torch.manual_seed(0)
        x = torch.randn(1, 8, d)
        return sub, (x,), ["hidden_state"], (
            f"int8-subgraph=decoder-layer-ALL-linears(q/k/v/o+gate/up/down, d={d}, "
            f"h={mlp.gate_proj.out_features}); fp32-glue=embedding+RoPE-index+causal-mask+softmax+"
            "RMSNorm (full-model int8 blocked: prepare_pt2e transform_for_annotation corrupts the "
            "cumsum->index.Tensor dtype even with an empty quantizer)")

    # Generic fallback: stack the first few Linears.
    lins = [m for _, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    if not lins:
        raise RuntimeError("int8-subgraph: model has no Linear layers to quantize")
    d = lins[0].in_features

    class _Seq(torch.nn.Module):
        def __init__(self, lins):
            super().__init__()
            self.lins = torch.nn.ModuleList(lins[:3])

        def forward(self, x):
            for lin in self.lins:
                if lin.in_features == x.shape[-1]:
                    x = torch.nn.functional.silu(lin(x))
            return x

    sub = _Seq(lins).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 8, d)
    return sub, (x,), ["hidden_state"], (
        f"int8-subgraph=first-linears(d={d}); non-linear ops excluded (full-model int8 blocked)")


_TORCH_TO_NP = {}  # filled after torch import


def _int8_whole_model_bias_preserving(model):
    """Whole-model weight-only int8 (per-channel) that PRESERVES nn.Linear bias.

    ExecuTorch's official ``WeightOnlyInt8QuantHandler`` is llama-specific: its
    ``WeightOnlyInt8Linear`` never carries a bias (forward is just
    ``F.linear(x, weight)*scales``) and its ``load_state_dict`` in ``quantized_model()``
    therefore REJECTS any model whose Linears have bias (rdt/rdt2 DiT blocks: qkv/proj/ffn
    all bias=True) with an "Unexpected key(s) ... .bias" error. That is a pure packaging
    limitation, not a quantization-feasibility one.

    This does the SAME math (symmetric per-output-channel int8, fp32 scale) via an eager
    module swap, but keeps the original ``bias`` on the replacement module and adds it back
    in forward. For a bias=False Linear the result is byte-identical to the official handler
    (so llama models are unchanged); for a biased Linear it is the correct int8-weight +
    fp32-bias GEMM. Like the official path it is a module swap, so it never runs PT2E's
    ``transform_for_annotation`` (sidestepping the index-dtype corruption), and its
    ``int8_weight.to(fp32)*scales`` dequant const-folds into a fp32 const weight that
    XNNPACK partitions as a normal fp32 GEMM — arena-collapsing exactly like the llama path.

    Returns ``(model, n_linear, n_biased)``.
    """
    import torch
    import torch.nn.functional as F

    class _Int8LinearBias(torch.nn.Module):
        """int8 weight-only per-channel Linear that carries its original fp32 bias."""

        def __init__(self, lin: torch.nn.Linear):
            super().__init__()
            self.in_features = lin.in_features
            self.out_features = lin.out_features
            w = lin.weight.detach().float()                     # [out, in]
            amax = w.abs().amax(dim=1, keepdim=True)            # per-output-channel
            scales = (amax / 127.0).clamp(min=1e-12)            # symmetric int8
            q = torch.round(w / scales).clamp(-128, 127).to(torch.int8)
            self.register_buffer("weight", q)
            self.register_buffer("scales", scales.squeeze(-1).float())
            if lin.bias is not None:
                self.register_buffer("bias", lin.bias.detach().float())
            else:
                self.bias = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = F.linear(x, self.weight.to(dtype=x.dtype)) * self.scales
            if self.bias is not None:
                y = y + self.bias.to(dtype=x.dtype)
            return y

    def _swap(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                setattr(module, name, _Int8LinearBias(child))
            else:
                _swap(child)

    n_lin = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    n_biased = sum(1 for m in model.modules()
                   if isinstance(m, torch.nn.Linear) and m.bias is not None)
    _swap(model)
    return model.eval(), n_lin, n_biased


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loader", required=True, help="path to model2MLIR workloads/<model>/loader.py")
    ap.add_argument("--inputs-npz", required=True, help="captured inputs.npz (seeds the golden)")
    ap.add_argument("--golden-npy", required=True, help="captured golden.npy (reference output)")
    ap.add_argument("--compute-golden", action="store_true",
                    help="compute the reference by running the eager torch model on the captured "
                         "input (writes it to --golden-npy). Use when the captured golden was made "
                         "with a DIFFERENT model config (e.g. a layer-reduced fit-on-board build), "
                         "so the correctness gate compares ExecuTorch vs eager-torch for THIS model.")
    ap.add_argument("--out", required=True, help="output .pte path (ptd + input0.bin go alongside)")
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--no-xnnpack", action="store_true",
                    help="skip the XNNPACK partitioner (portable-kernel-only baseline)")
    ap.add_argument("--quantize", action="store_true",
                    help="PT2E W8A8 quantize before lowering (exercises XNNPACK's int8 qs8/qd8 RVV "
                         "ukernels). ExecuTorch does its OWN quantization, so pair with "
                         "--compute-golden (gate = eager-vs-ExecuTorch for THIS config). NOTE: this "
                         "is IGNORED when --int8-whole-model is also set (that path is a module "
                         "swap, not PT2E) — pass --qd8 with --quantize and WITHOUT "
                         "--int8-whole-model to get the arithmetic merlin's int8 actually runs.")
    ap.add_argument("--qd8", action="store_true",
                    help="with --quantize: use per-channel weights + DYNAMIC per-row activation "
                         "quantization (XNNPACK qd8) instead of the default static per-tensor qs8. "
                         "This is the mirror of merlin's own int8 datapath (passes_quant_int: each "
                         "activation dynamically quantized to i8, symmetric, per output row, against "
                         "per-channel weight scales) and of the qd8 expert fixture the beam is "
                         "taught from, so an ours-vs-ExecuTorch int8 ratio compares two runs of the "
                         "same arithmetic rather than two different quantization schemes.")
    ap.add_argument("--m2m-root", default="/path/to/model2MLIR",
                    help="model2MLIR repo root (added to sys.path for its deps)")
    ap.add_argument("--int8-whole-model", action="store_true",
                    help="WHOLE-MODEL int8 via ExecuTorch's OFFICIAL llama recipe: source-transform "
                         "weight-only int8 per-channel (WeightOnlyInt8QuantHandler — an eager "
                         "MODULE SWAP, replacing every nn.Linear with a WeightOnlyInt8Linear). "
                         "Because it is a module swap it NEVER runs PT2E's transform_for_annotation "
                         "pass, so it SIDESTEPS the cumsum->index.Tensor dtype corruption that "
                         "blocks generic full-model PT2E on HF Llama — all layers quantize + export "
                         "cleanly. Forces --compute-golden (int8-vs-fp32 gate).")
    ap.add_argument("--int8-subgraph", action="store_true",
                    help="int8 FALLBACK path (only if whole-model won't export): quantize the REAL "
                         "decoder-layer linear subgraph (q/k/v/o + gate/up/down) W8A8 on a seeded "
                         "hidden-state input, embeddings/mask fp32. Honestly labeled.")
    args = ap.parse_args()
    if args.int8_whole_model:
        args.quantize = True
        args.compute_golden = True
    if args.int8_subgraph:
        args.quantize = True
        args.compute_golden = True

    # This script sits in merlin/python/merlin/baselines/ alongside ``executorch.py``, which would
    # SHADOW the installed ``executorch`` package (Python puts the script dir on sys.path[0]).
    _self_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if p not in ("", ".", _self_dir)]
    if args.m2m_root not in sys.path:
        sys.path.insert(0, args.m2m_root)

    import numpy as np
    import torch
    from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
    from torch.export import export

    # OUT VARIANTS for the quantized_decomposed ops. `to_executorch`'s ToOutVarPass needs an
    # out-variant for every operator left OUTSIDE the delegate, and a PT2E graph whose XNNPACK
    # partition does not absorb every quantize/dequantize/choose_qparams keeps portable ones. Without
    # this library the whole program is refused -- "Missing out variants:
    # quantized_decomposed::{quantize_per_tensor,dequantize_per_tensor,choose_qparams}" -- and no
    # .pte is produced. The board runner needs the matching kernel library linked, which
    # `executorch.plan_kernels` already reports as `libraries={'quantized'}` + its cmake option.
    try:
        import executorch.kernels.quantized  # noqa: F401
        _qkernels = "quantized-out-variants=registered"
    except Exception as _e:  # noqa: BLE001
        _qkernels = f"quantized-out-variants=UNAVAILABLE({type(_e).__name__}: {_e})"
    print(f"[{args.model_name}] {_qkernels}", file=sys.stderr)

    # 1. Load model; substitute the CAPTURED input so the export trace matches the captured golden.
    #    Mirror the consistent-capture harness (model2MLIR/workloads/capture_consistent.py): seed the
    #    instantiation so weight init is reproducible, THEN perturb any exactly-zero parameters with
    #    small noise. DiT / flow-matching models (rdt, rdt2) zero-init their adaLN-Zero OUTPUT HEAD,
    #    so a fresh random-init eager forward is EXACTLY ZERO — which made our recomputed golden a
    #    zero tensor and the int8-vs-fp32 cosine degenerate (cos=0.0, rel=0.0: BOTH the golden and the
    #    ExecuTorch output were all-zeros, not an XNNPACK/delegate bug). The capture harness perturbs
    #    those zero params so the captured golden exercises the full numeric path (rdt2 golden
    #    norm=11.4, rdt=52.9); the recompute here MUST do the identical thing or the reference is a
    #    zero tensor for any zero-init-head model. Models with no exactly-zero params (llama/bitvla/
    #    xr0/…) are untouched — the loop is a no-op — so this is non-regressive. The perturbation is
    #    applied to the SAME instance the golden and the int8 swap both derive from, so the gate stays
    #    a self-consistent int8-vs-fp32 comparison.
    loader = _load_loader(Path(args.loader))
    torch.manual_seed(0)
    np.random.seed(0)
    model, _example = loader.get_model_and_inputs()
    model = model.eval()
    with torch.no_grad():
        for _p in model.parameters():
            if float(_p.detach().abs().max()) == 0.0:
                _p.copy_(torch.randn_like(_p) * 0.02)
    # This is an INFERENCE export: nothing downstream differentiates. Leaving ``requires_grad`` on
    # the parameters is not merely wasteful, it makes some models unexportable — ExecuTorch's
    # ``replace_view_ops_with_view_copy_ops_pass`` re-executes each view op eagerly on the real
    # parameter, and a view op whose *output dtype is complex* (``view_as_complex`` -> its
    # ``_copy`` form, used by any spectral/Fourier-filter block that stores a real-valued
    # (..., 2) weight and reinterprets it as complex) has no autograd formula for a complex
    # output: torch raises "does not support automatic differentiation for outputs with complex
    # dtype" and the whole lowering dies. Dropping grad on an eval-mode model removes the
    # autograd path without touching a single number.
    for _p in model.parameters():
        _p.requires_grad_(False)
    for _b in model.buffers():
        if _b.is_floating_point() or _b.is_complex():
            _b.requires_grad_(False)

    npz = np.load(args.inputs_npz)
    keys = list(npz.keys())
    captured = tuple(torch.from_numpy(npz[k]) for k in keys)
    captured, keys, _arity_note = reconcile_input_arity(captured, keys, _example,
                                                        source=args.inputs_npz)
    if _arity_note:
        print(f"[{args.model_name}] {_arity_note}", file=sys.stderr)

    subgraph_note = ""
    _shim_note = verify_shape_static_vision_patch(model, captured)
    if _shim_note:
        print(f"[{args.model_name}] {_shim_note}", file=sys.stderr)
        subgraph_note = _shim_note
    if args.int8_subgraph:
        # Replace the whole model with its REAL linear-heavy subgraph (a genuine slice, actual
        # weights) — the FALLBACK if whole-model int8 won't export.
        model, captured, keys, subgraph_note = _linear_subgraph(model)

    # Golden is the FP32 reference — compute it BEFORE any int8 module swap.
    if args.compute_golden:
        with torch.no_grad():
            eager = model(*captured)
        eager_t = eager[0] if isinstance(eager, (tuple, list)) else eager
        golden_np = eager_t.detach().cpu().to(torch.float32).numpy()
        np.save(args.golden_npy, golden_np)
    else:
        golden_np = np.load(args.golden_npy)

    quantized = False

    # 2a. WHOLE-MODEL int8 via ExecuTorch's OFFICIAL llama recipe (source-transform weight-only int8
    #     per-channel). This is an eager MODULE SWAP (every nn.Linear -> WeightOnlyInt8Linear), so it
    #     never runs PT2E's transform_for_annotation pass and SIDESTEPS the cumsum->index.Tensor
    #     corruption that blocks generic full-model PT2E on HF Llama. ALL layers quantize + export.
    if args.int8_whole_model:
        from executorch.examples.models.llama.source_transformation.quantize import (
            WeightOnlyInt8QuantHandler,
        )

        n_lin = sum(1 for _ in model.modules() if isinstance(_, torch.nn.Linear))
        n_biased = sum(1 for m in model.modules()
                       if isinstance(m, torch.nn.Linear) and m.bias is not None)
        # The official llama handler is bias-free (its WeightOnlyInt8Linear drops bias and its
        # load_state_dict rejects any leftover .bias key). Use it verbatim for the bias-free
        # llama family (identical attribution/numbers as before). For a model with ANY biased
        # Linear (rdt/rdt2 DiT: qkv/proj/ffn all bias=True), the official handler raises
        # "Unexpected key(s) ... .bias" -> use OUR bias-preserving per-channel int8 swap, which
        # does the same math but keeps the fp32 bias. Same const-fold/arena behavior either way.
        if n_biased == 0:
            model = WeightOnlyInt8QuantHandler(model).quantized_model().eval()
            recipe = ("official-llama-recipe(WeightOnlyInt8QuantHandler, weight-only int8 "
                      f"per-channel, ALL {n_lin} nn.Linear swapped -> WeightOnlyInt8Linear)")
        else:
            model, n_lin, n_biased = _int8_whole_model_bias_preserving(model)
            recipe = (f"bias-preserving-int8(weight-only int8 per-channel, ALL {n_lin} nn.Linear "
                      f"swapped, {n_biased} carry fp32 bias; official handler drops bias/rejects "
                      "biased state_dict so this superset is used)")
        quantized = True
        subgraph_note = (subgraph_note + " " if subgraph_note else "") + (
            f"int8-whole-model={recipe}; embeddings/mask/softmax/norm fp32; "
            "module-swap sidesteps PT2E transform_for_annotation index-corruption")

    # 2b. PT2E W8A8 quantization (int8 subgraph path). Only when NOT whole-model int8.
    if args.quantize and not args.int8_whole_model:
        try:
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer, get_symmetric_quantization_config)
            try:
                from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
            except Exception:  # noqa: BLE001
                from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

            cap = export(model, captured, strict=False).module()
            # WHICH int8 this is, named rather than defaulted. `get_symmetric_quantization_config()`
            # with no arguments is STATIC PER-TENSOR qs8, which is not the datapath merlin runs and
            # therefore not a comparand for it: `llvmlower/passes_quant_int` dynamically quantizes
            # each activation to i8, symmetric, PER OUTPUT ROW, against per-channel weight scales --
            # i.e. qd8, the same family as the expert fixture the beam is taught from
            # (merlin/tests/data/cca_asm/xnnpack_qd8_gemm_rvv.objdump). `--qd8` selects that mirror
            # so an ours-vs-ExecuTorch int8 ratio compares two runs of the same arithmetic.
            qcfg = (get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
                    if args.qd8 else get_symmetric_quantization_config())
            # Not a plain XNNPACKQuantizer: its transform_for_annotation retypes INTEGER scalar
            # arithmetic to float, which corrupts the index feeding HF's causal-mask
            # `aten.index.Tensor` and made whole-model PT2E impossible on every transformers
            # decoder. See _dtype_preserving_quantizer — float graphs transform identically.
            quantizer = _dtype_preserving_quantizer(XNNPACKQuantizer, qcfg)
            if args.qd8:
                # Only qd8 reaches the unbounded NHWC walk (its branch is gated on is_dynamic_qdq),
                # so the fix is applied on exactly the path that hits it, and its status is recorded
                # with the result rather than assumed to have taken effect.
                _nhwc_status = _bound_dynamic_qdq_channels_last_walk()
            prepared = prepare_pt2e(cap, quantizer)
            with torch.no_grad():
                prepared(*captured)          # calibrate on the captured input
            model = convert_pt2e(prepared)   # now an int8 graph module
            quantized = True
            # Which recipe produced these numbers, recorded WITH them. Two different int8 recipes
            # (weight-only module swap vs PT2E qd8 vs PT2E qs8) produce walls that are not
            # comparable to each other, and an unlabelled one gets compared anyway.
            _pt2e_recipe = ("pt2e-qd8(symmetric, per-channel weights, DYNAMIC per-row activation "
                            "quant -> XNNPACK qd8 int8 ukernels; mirrors merlin's "
                            "passes_quant_int datapath)" if args.qd8 else
                            "pt2e-qs8(symmetric, per-tensor, STATIC activation quant)")
            if args.qd8:
                _pt2e_recipe += "; " + _nhwc_status
            subgraph_note = (subgraph_note + " " if subgraph_note else "") + _pt2e_recipe
        except Exception as e:  # noqa: BLE001
            # Fail closed and say which recipe failed. A qd8 failure must never silently become a
            # weight-only or qs8 measurement -- the whole point of naming the recipe is that a cell
            # reports the arithmetic it actually ran, or reports nothing.
            _which = "qd8" if args.qd8 else "qs8"
            raise RuntimeError(f"PT2E int8 quantization ({_which}) failed: {e}") from e

    # 3. torch.export (non-strict first; HF causal-LMs trace cleanly non-strict).
    try:
        exported = export(model, captured, strict=False)
    except Exception:  # noqa: BLE001
        exported = export(model, captured, strict=True)

    # 3b. WHOLE-MODEL int8 arena fix: WeightOnlyInt8Linear.forward is
    #     `F.linear(x, weight.to(fp32)) * scales`. That `int8_weight.to(fp32)` dequant is a graph
    #     op whose output is a PLANNED ACTIVATION tensor, so the memory planner lays every layer's
    #     dequantized fp32 weight at a distinct offset -> a 4.25 GB non-const arena for full-depth
    #     TinyLlama (measured), which `std::make_unique<uint8_t[]>(4.25e9)` throws bad_alloc on at
    #     load on the 3.4 GB board. Const-folding the dequant (its inputs are all frozen constants)
    #     turns those fp32 weights into program CONSTANTS: the planned arena collapses from 4.25 GB
    #     to ~2 MB (measured 4250738736 -> 2405952 bytes, ~1770x). The fp32 weights now live in the
    #     .pte program data, which the board mmaps (executor_runner --mmap_model) so their pages
    #     demand-load and stay evictable under the board's RAM ceiling. This is what unblocks the
    #     FULL-DEPTH (not layer-reduced) whole-model int8 on-board run.
    #     const-prop ALSO concretizes data-dependent constant subgraphs: smolvla fp32 otherwise
    #     fails to_executorch with an unbacked-symint "data-dependent" error in dim_order_from_stride
    #     (a bucketize-derived stride the static memory planner can't evaluate); folding the constant
    #     boundaries resolves it, which is why the int8 path (already const-propped) exported cleanly.
    #     So run it for the fp32 whole-model xnnpack path too (not just int8).
    _whole_model = args.int8_whole_model or (not args.no_xnnpack and not args.int8_subgraph
                                             and not args.quantize)
    if _whole_model:
        try:
            from executorch.exir.passes.constant_prop_pass import constant_prop_pass
            exported = constant_prop_pass(exported)
            subgraph_note += (" +const-prop(dequant-weights-folded->const, arena 4.25GB->2MB)"
                              if args.int8_whole_model
                              else " +const-prop(fp32 whole-model; resolves bucketize data-dep symint)")
        except Exception as e:  # noqa: BLE001
            subgraph_note += f" const-prop-skipped({str(e)[:80]})"

    # 4. XNNPACK partitioner (the RVV-microkernel delegate path).
    partitioners = []
    if not args.no_xnnpack:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        partitioners.append(XnnpackPartitioner())

    from executorch.exir import EdgeCompileConfig
    # Disable the advisory core-ATen IR-validity check for ALL models (not just quantized): some
    # real forwards emit ops outside the strict "core ATen opset" (e.g. smolvla fp32 emits
    # aten.bucketize.Tensor) that the verifier rejects even though a runtime kernel exists / can be
    # provided. Whether an op is actually runnable is decided by the runner's registered kernel set,
    # which this arm audits separately (missing kernels surface as a precise method-load gap), so the
    # verifier is redundant here and would spuriously block otherwise-exportable models.
    _cc = EdgeCompileConfig(_check_ir_validity=False)
    edge = to_edge_transform_and_lower(exported, partitioner=partitioners, compile_config=_cc)
    nodes = list(edge.exported_program().graph.nodes)
    delegated = sum(1 for n in nodes
                    if n.op == "call_function" and "call_delegate" in str(n.target))
    total_calls = sum(1 for n in nodes if n.op == "call_function")

    # 4. to_executorch with EXTERNAL constants: weights land in a .ptd next to the .pte, so the
    #    program flatbuffer stays under the 2 GB limit for the big fp32 LLMs.
    et_program = edge.to_executorch(
        config=ExecutorchBackendConfig(external_constants=True)
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- provenance: node -> originating model-layer fqn (the cross-compiler join key) + etrecord ---
    # The fqn map lets the whole-model compare align ET regions to Merlin regions by shared layer; the
    # etrecord lets the devtools Inspector correlate on-board etdump events (debug_handle) back to it.
    fqn_map = extract_fqn_map(exported)
    (out.parent / "et_fqn_map.json").write_text(json.dumps(fqn_map, indent=2))
    etrecord_path = ""
    try:
        from executorch.devtools import generate_etrecord
        etrecord_path = str(out.parent / "etrecord.bin")
        generate_etrecord(etrecord_path, edge, et_program, exported_program=exported)
    except Exception as e:  # noqa: BLE001 - etrecord is for per-region timing; absence is not fatal
        etrecord_path = ""
        print(f"[warn] generate_etrecord failed ({type(e).__name__}: {str(e)[:120]}); "
              "per-region ET timing will be unavailable", file=sys.stderr)

    with open(out, "wb") as fh:
        et_program.write_to_file(fh)
    # .ptd tensor-data files (weights). write_tensor_data_to_file writes <key>.ptd into outdir.
    ptd_files: list[str] = []
    try:
        et_program.write_tensor_data_to_file(str(out.parent))
        ptd_files = sorted(str(p) for p in out.parent.glob("*.ptd"))
    except Exception as e:  # noqa: BLE001
        # If there were no external constants (small model), there is simply no ptd — not an error.
        ptd_files = []

    # 5. Save the model input(s) as raw bytes (the runner memcpy's the file into the input tensor).
    #    Use the actual `captured` tensors (int8-subgraph feeds a seeded fp32 hidden state, not the
    #    npz token ids), so the on-board input matches exactly what golden was computed from.
    input_files = []
    for i, (k, t) in enumerate(zip(keys, captured)):
        arr = np.ascontiguousarray(t.detach().cpu().numpy())
        p = out.parent / f"input{i}.bin"
        p.write_bytes(arr.tobytes())
        input_files.append({"path": str(p), "key": k, "dtype": str(arr.dtype),
                            "shape": list(arr.shape), "nbytes": arr.nbytes})

    # --- AOT PROFILE: what ExecuTorch's ahead-of-time pipeline actually DID to this graph ---------
    # Counts alone ("42 of 60 nodes delegated") say a partition happened, not what it bought. These
    # four say what to copy: which ops XNNPACK swallowed (and therefore prepacks + fuses), which it
    # REFUSED (those run portable and are where a compiler can beat a kernel library), how the
    # weights are encoded, and how much arena ExecuTorch plans statically. The last is directly
    # comparable to our own allocation behaviour -- we emit 209 `tensor.empty` on this model.
    def _aot_profile() -> dict:
        import collections
        prof: dict = {}
        try:
            portable = collections.Counter()
            delegated_backends = collections.Counter()
            for n in nodes:
                if n.op != "call_function":
                    continue
                t = str(n.target)
                if "call_delegate" in t:
                    # the lowered module carries its backend id
                    try:
                        lm = n.args[0]
                        delegated_backends[str(getattr(lm, "backend_id", "") or
                                                getattr(lm, "_backend_id", "") or "unknown")] += 1
                    except Exception:  # noqa: BLE001
                        delegated_backends["unknown"] += 1
                else:
                    portable[t.split(".")[-2] if "." in t else t] += 1
            prof["portable_ops"] = dict(portable.most_common(40))
            prof["n_portable_ops"] = sum(portable.values())
            prof["delegate_backends"] = dict(delegated_backends)
        except Exception as e:  # noqa: BLE001
            prof["partition_error"] = f"{type(e).__name__}: {e}"
        # ExecuTorch's STATIC memory plan: the per-arena byte sizes its planning pass chose. A
        # framework that plans its intermediates has no per-op allocator traffic at run time.
        try:
            plans = et_program.executorch_program.execution_plan
            prof["memory_plan_arenas"] = [list(pl.non_const_buffer_sizes) for pl in plans]
            prof["memory_plan_total_bytes"] = sum(
                sum(x for x in pl.non_const_buffer_sizes if isinstance(x, int) and x > 0)
                for pl in plans)
        except Exception as e:  # noqa: BLE001
            prof["memory_plan_error"] = f"{type(e).__name__}: {e}"
        # Weight ENCODING: how the constants are actually stored (dtype x count x bytes).
        try:
            enc = collections.Counter(); byts = collections.Counter()
            gm = exported.graph_module
            for name, t in list(getattr(exported, "state_dict", {}).items()):
                try:
                    enc[str(t.dtype)] += 1; byts[str(t.dtype)] += t.numel() * t.element_size()
                except Exception:  # noqa: BLE001
                    pass
            prof["weight_dtypes"] = dict(enc)
            prof["weight_bytes_by_dtype"] = dict(byts)
            del gm
        except Exception as e:  # noqa: BLE001
            prof["weight_encoding_error"] = f"{type(e).__name__}: {e}"
        return prof

    summary = {
        "aot_profile": _aot_profile(),
        "model": args.model_name,
        "pte": str(out),
        "pte_bytes": out.stat().st_size,
        "ptd_files": ptd_files,
        "input_files": input_files,
        "xnnpack": not args.no_xnnpack,
        "quantized": quantized,
        "int8_subgraph": bool(args.int8_subgraph),
        "int8_whole_model": bool(args.int8_whole_model),
        "subgraph_note": subgraph_note,
        "delegated_nodes": delegated,
        "total_call_nodes": total_calls,
        "golden_shape": list(golden_np.shape),
        "golden_dtype": str(golden_np.dtype),
        "fqn_map": str(out.parent / "et_fqn_map.json"),
        "n_provenanced_nodes": len(fqn_map),
        "etrecord": etrecord_path,
    }
    print("ET_EXPORT_JSON " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

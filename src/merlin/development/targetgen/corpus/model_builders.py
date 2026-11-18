"""Target-agnostic capsule-corpus builder.

ONE builder turns an abstract capsule *entry* (op + shapes-in-tiles + epilogue — the target-agnostic test
definition, declared per target in ``capsules/<target>/corpus_profile.yaml``) plus a *binding* DERIVED from
the target's descriptor (dtypes, tile dim, compare policy, instruction classes, oracle tiers) into a
concrete capsule dict + interface MLIR. It replaces the two forked per-target generators (one
integer, one float): the LOGIC here is shared and carries no target
name or dtype literal in its control flow; the per-target DATA (numeric datapath, which capsules, dtypes)
lives in the descriptor + the profile.

Golden VALUES are not computed here (the integer engine lives in :mod:`capsule_golden`; the float engine
needs the external ``specir`` refmodel, available only at generation time) — the driver
``experiments/corpora/generate_corpus.py`` computes them and writes ``golden.yaml``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from merlin.development.targetgen.corpus.binding import CorpusBinding
from merlin.development.targetgen.corpus.contraction_builders import (
    build_attention_mx,
    build_attention_qk,
    build_bias_add,
    build_matmul,
    build_movement,
    build_resident_reuse,
    build_residual_add,
    build_residual_seam,
    build_rmsnorm,
    build_rmsnorm_qkv,
)
from merlin.development.targetgen.corpus.corpus_activation import activation_matmul_program
from merlin.development.targetgen.corpus.corpus_pool import build as build_maxpool2d
from merlin.development.targetgen.corpus.corpus_sibling_feed import build as _build_sibling_feed
from merlin.development.targetgen.corpus.declarations import _semantic_block
from merlin.development.targetgen.grading.golden_inputs import dtype_bits

# ---- corpus_spec -------------------------------------------------------------------------------


def build_rope_qkv(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """Fused QKV projection + RoPE (op == rope_qkv): H = X @ Wqkv; Y = rope(H) (GPT-NeoX/Llama rotation,
    theta=10000, position = row). SIMT float; golden driver-composed. Interface chains matmul -> rope."""
    from merlin.development.targetgen.corpus.binding import CorpusBinding, _numeric_policy
    from merlin.development.targetgen.corpus.contraction_builders import _iface_prelude

    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    K = entry.get("K", entry.get("K_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    x, weight, out = entry.get("src", "X"), entry.get("weight", "Wqkv"), entry.get("out", "Y0")
    theta = float(entry.get("rope_theta", 10000.0))
    idt, odt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    attrs = {"src": x, "weight": weight, "out": out, "rope_theta": theta, "semantic": "rope_qkv", "output_dtype": odt}
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [
            {"name": x, "role": "input", "shape": [M, K], "dtype": idt},
            {"name": weight, "role": "weight", "shape": [K, N], "dtype": idt},
        ],
        "operation": {"op": "rope_qkv", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": {
            "instruction_classes": binding.classes_for(op="matmul", output_dtype=odt),
            "modes": entry.get("modes", {"rope": True}),
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{x} = merlin_iface.tensor {{name = "{x}", role = "input"}} : tensor<{M}x{K}x{midt}>',
        f'  %{weight} = merlin_iface.tensor {{name = "{weight}", role = "weight"}} : tensor<{K}x{N}x{midt}>',
        f'  %{weight}_res = merlin_iface.resident_pack %{weight} {{layout = "packed_rhs"}} '
        f": (tensor<{K}x{N}x{midt}>) -> !merlin_iface.resident",
        f"  %acc0 = merlin_iface.matmul %{x}, %{weight}_res "
        f": (tensor<{M}x{K}x{midt}>, !merlin_iface.resident) -> !merlin_iface.acc<{modt}>",
        f'  %H = merlin_iface.commit %acc0 {{name = "H", epilogue = [], output_dtype = "{odt}"}} '
        f": (!merlin_iface.acc<{modt}>) -> tensor<{M}x{N}x{modt}>",
        f'  %{out} = merlin_iface.rope %H {{name = "{out}", theta = {theta} : f64, '
        f'output_dtype = "{odt}"}} : (tensor<{M}x{N}x{modt}>) -> tensor<{M}x{N}x{modt}>',
        f"  merlin_iface.evict %{weight}_res : (!merlin_iface.resident) -> ()",
        "}",
    ]
    return cap, "\n".join(L) + "\n"


def _batched_mx_inputs(
    lhs: str, weight: str, B: int, M: int, H: int, N: int, idt: str, binding: CorpusBinding
) -> list[dict]:
    """Declared leaf operands of a BATCHED matmul; one scale stream per batch where the format has them.

    Same reasoning as :func:`_matmul_inputs` -- the elements alone are half the operand -- with the batch
    dimension leading, because each batch is an independent GEMM with its own block scales."""
    from merlin.development.targetgen.corpus.binding import CorpusBinding, is_block_scaled

    inputs = [
        {"name": weight, "role": "weight", "shape": [B, H, N], "dtype": idt},
        {"name": lhs, "role": "input", "shape": [B, M, H], "dtype": idt},
    ]
    blk = binding.scale_block if is_block_scaled(binding.operand_dtype) else None
    if blk and H % blk == 0:
        g = H // blk
        inputs += [
            {
                "name": f"{lhs}_scale",
                "role": "scale",
                "scale_of": lhs,
                "block": blk,
                "shape": [B, g, M],
                "dtype": "e8m0",
            },
            {
                "name": f"{weight}_scale",
                "role": "scale",
                "scale_of": weight,
                "block": blk,
                "shape": [B, g, N],
                "dtype": "e8m0",
            },
        ]
    return inputs


def build_gemv_batched(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A batched contraction (op == gemv_batched): B independent GEMMs ``A_b[M,H] @ W_b[H,N]``, output
    preserving source-visible rank as ``[B, M, N]``. Interface presents the batched tensors + a batched
    matmul op; a target may iterate the B tiles internally, but may not flatten the operation's ABI.

    DATATYPE-AGNOSTIC, and it was not. It was written for the block-scaled datapath and declared
    ``block_scale = "e8m0"`` unconditionally, so the only rank-3 capsule this corpus could produce was
    an MX one. That is what made ``contraction.batched`` an unfillable requirement on every integer and
    fp8 target: not a missing rewrite, a missing golden and an operand declaration that assumed one
    datapath. The block-scale attribute and the scale streams now appear only where the operand format
    actually carries them (``is_block_scaled``), exactly as ``_batched_mx_inputs`` already decided the
    scale operands."""
    from merlin.development.targetgen.corpus.binding import CorpusBinding, _numeric_policy, is_block_scaled
    from merlin.development.targetgen.corpus.contraction_builders import _iface_prelude

    D = binding.tile_dim
    B = int(entry.get("B", 2))
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    H = entry.get("H", entry.get("K", 2 * D))
    N = entry.get("N", D)
    lhs, weight, out = entry.get("lhs", "A0"), entry.get("weight", "W"), entry.get("out", "Y0")
    idt, odt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    scaled = is_block_scaled(binding.operand_dtype)
    attrs = {"lhs": lhs, "weight": weight, "out": out, "batch": B, "semantic": "gemv_batched", "output_dtype": odt}
    if scaled:
        attrs["block_scale"] = "e8m0"
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": _batched_mx_inputs(lhs, weight, B, M, H, N, idt, binding),
        "operation": {"op": "gemv_batched", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": {
            "instruction_classes": binding.classes_for(op="matmul", output_dtype=odt),
            "modes": entry.get("modes", {}),
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{lhs} = merlin_iface.tensor {{name = "{lhs}", role = "input"}} : tensor<{B}x{M}x{H}x{midt}>',
        f'  %{weight} = merlin_iface.tensor {{name = "{weight}", role = "weight"}} : tensor<{B}x{H}x{N}x{midt}>',
    ]
    _blk = binding.scale_block if is_block_scaled(binding.operand_dtype) else None
    if _blk and H % _blk == 0:
        # The batched form carries one scale stream per batch: the op declares `block_scale = "e8m0"`
        # already, and these are the streams that attribute refers to.
        _g = H // _blk
        L += [
            f'  %{lhs}_scale = merlin_iface.tensor {{name = "{lhs}_scale", role = "scale", '
            f'scale_of = "{lhs}", block = {_blk} : i64}} : tensor<{B}x{_g}x{M}xi8>',
            f'  %{weight}_scale = merlin_iface.tensor {{name = "{weight}_scale", role = "scale", '
            f'scale_of = "{weight}", block = {_blk} : i64}} : tensor<{B}x{_g}x{N}xi8>',
        ]
    _bs = 'block_scale = "e8m0", ' if scaled else ""
    L += [
        f'  %{out} = merlin_iface.matmul_batched %{lhs}, %{weight} {{name = "{out}", batch = {B} : i64, '
        f'{_bs}output_dtype = "{odt}"}} '
        f": (tensor<{B}x{M}x{H}x{midt}>, tensor<{B}x{H}x{N}x{midt}>) -> "
        f"tensor<{B}x{M}x{N}x{modt}>",
        "}",
    ]
    return cap, "\n".join(L) + "\n"


def build_conv2d(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A conv2d capsule (op == conv2d): NHWC IFM and a weight laid out ``[tap_h, tap_w, channel] x Cout``
    (``[KH*KW*Ci, Cout]``), output ``[Ho*Wo, Cout]``. Reuses the runtime's canonical conv geometry so the golden
    (capsule_golden conv2d branch) and the harness agree. Native operand dtype (e.g. int8)."""
    from merlin.development.targetgen.corpus.binding import (
        _BIAS_STAGES,
        CorpusBinding,
        _bias_operand,
        _numeric_policy,
        _pool_epilogue,
        _pool_mlir_attrs,
        _resolve_output_dtype,
        requant_shift_for,
    )
    from merlin.development.targetgen.corpus.contraction_builders import _iface_prelude
    from merlin.runtime.engine.command_buffer import conv_out_dims

    ifm, weight, out = entry.get("ifm", "IFM"), entry.get("weight", "W"), entry.get("out", "Y0")
    ci = entry.get("ci", entry.get("Cin", 4))
    cout = entry.get("N", entry.get("Cout", binding.tile_dim))
    H, W = entry.get("Himg", 8), entry.get("Wimg", 8)
    kh, kw = entry.get("kh", 3), entry.get("kw", 3)
    stride = list(entry.get("stride", [1, 1]))
    padding = list(entry.get("padding", [0, 0, 0, 0]))
    dilation = list(entry.get("dilation", [1, 1]))
    Ho, Wo = conv_out_dims(H, W, kh, kw, stride, padding, dilation)
    Kdim = kh * kw * ci
    epilogue = list(entry.get("epilogue", []))
    output_dtype = _resolve_output_dtype(
        binding,
        epilogue,
        entry,
        available_operand_roles=frozenset({"bias"})
        if any(stage in _BIAS_STAGES for stage in epilogue)
        else frozenset(),
    )
    idt, odt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(output_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(output_dtype)
    weight_spec = {"name": weight, "role": "weight", "shape": [Kdim, cout], "dtype": idt}
    if "weight_storage_encoding" in entry:
        from merlin.analysis.programs.storage_encoding import GroupedAxesStorage

        storage = GroupedAxesStorage.from_dict(entry["weight_storage_encoding"])
        storage.validate()
        if storage.logical_shape != (Kdim, cout) or storage.dtype != midt:
            raise ValueError("convolution weight storage differs from its original logical matrix")
        weight_spec.update(shape=list(storage.physical_shape), storage_encoding=storage.to_dict())
    weight_type = "tensor<" + "x".join(map(str, weight_spec["shape"])) + f"x{midt}>"
    from merlin.target.contract.interfaces.grammar import _fmt_attrs

    weight_attrs = _fmt_attrs({key: value for key, value in weight_spec.items() if key not in ("shape", "dtype")})
    attrs = {
        "ifm": ifm,
        "weight": weight,
        "out": out,
        "ci": ci,
        "kh": kh,
        "kw": kw,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "layout": "nhwc",
        "epilogue": epilogue,
        "output_dtype": odt,
        "semantic": "conv2d",
    }
    # A pooling epilogue is fused onto the conv's store path (the fused conv loop takes the window and
    # stores the pooled result), so it rides on this op's own attributes. ``pool_in_dims`` is DERIVED
    # from the conv's output extent rather than declared: the golden sees only the flat [N*Ho*Wo, Co]
    # product and has to trust the declaration, so deriving it is what keeps the golden and the
    # simulator pooling the same image. The result extent below is the POOLED one.
    pool_attrs, out_rows = _pool_epilogue(entry, rows=Ho * Wo, in_dims=(Ho, Wo), op="conv2d")
    attrs.update(pool_attrs)
    requant_shift = requant_shift_for(entry, binding, epilogue, op="conv2d")
    if requant_shift is not None:
        attrs["requant_shift"] = requant_shift
    # THE READOUT A CONVOLUTION SHARES WITH A CONTRACTION. A bias per output channel and the f32
    # multiplier are what every quantized CNN layer carries, and this builder used to drop both: the
    # bias had no operand, and an `acc_scale` stage was emitted with no multiplier, which every engine
    # then read as 1.0. A stage with no parameter is refused here, where the entry is still in hand.
    bias, bias_dt = _bias_operand(entry, epilogue, cout, binding, "conv2d")
    if bias:
        attrs["bias"] = bias
    from merlin.target.contract.interfaces import matmul as MSE

    readout = MSE.acc_scale_parameters(entry, epilogue, cout, op=entry.get("name", "conv2d"))
    attrs.update(readout)
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [
            weight_spec,
            {"name": ifm, "role": "input", "shape": [1, H, W, ci], "dtype": idt},
            *([{"name": bias, "role": "bias", "shape": [cout], "dtype": bias_dt}] if bias else []),
        ],
        "operation": {"op": "conv2d", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, output_dtype, entry.get("acc_scale")),
        "expected": {
            "instruction_classes": binding.classes_for(
                op="conv2d", output_dtype=odt, epilogue=epilogue, movement=False
            ),
            "modes": dict(entry["modes"]) if "modes" in entry else {"conv2d": True, "k_accumulate": True},
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    epi = ", ".join(f'"{e}"' for e in epilogue)
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{ifm} = merlin_iface.tensor {{name = "{ifm}", role = "input"}} : tensor<1x{H}x{W}x{ci}x{midt}>',
        f"  %{weight} = merlin_iface.tensor {weight_attrs} : {weight_type}",
        *(
            [
                f'  %{bias} = merlin_iface.tensor {{name = "{bias}", role = "bias"}} : '
                f"tensor<{cout}x{binding.mlir_dtype(binding.accum_dtype)}>"
            ]
            if bias
            else []
        ),
        f'  %{weight}_res = merlin_iface.resident_pack %{weight} {{layout = "packed_conv_rhs"}} '
        f": ({weight_type}) -> !merlin_iface.resident",
        f"  %{out} = merlin_iface.conv2d %{ifm}, %{weight}_res {{kernel = [{kh}, {kw}, {ci}, {cout}], "
        f"stride = [{stride[0]}, {stride[1]}], padding = [{padding[0]}, {padding[1]}, {padding[2]}, {padding[3]}], "
        f'dilation = [{dilation[0]}, {dilation[1]}], name = "{out}", epilogue = [{epi}], '
        f'output_dtype = "{odt}"'
        + (f", requant_shift = {requant_shift} : i64" if requant_shift is not None else "")
        + MSE.acc_scale_mlir_attrs(readout)
        + (f', bias = "{bias}"' if bias else "")
        + f', layout = "nhwc"{_pool_mlir_attrs(pool_attrs)}}} '
        f": (tensor<1x{H}x{W}x{ci}x{midt}>, !merlin_iface.resident) -> tensor<{out_rows}x{cout}x{modt}>",
        f"  merlin_iface.evict %{weight}_res : (!merlin_iface.resident) -> ()",
        "}",
    ]
    return cap, "\n".join(L) + "\n"


def build_host_island_seam(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """Two identical contractions with an optional, explicit host-only map between them.

    This is a standard linalg-on-tensors program, not a target command dialect.  The first contraction
    widens into the declared accumulator type, an identical saturating narrow is tagged as an
    accelerator epilogue in both members, and only the ``island`` member applies a scalar XOR before
    the second contraction.  Consequently the pair has identical external operands, contraction work,
    and requantization; its sole semantic difference is the host region whose placement it measures.

    The builder is integer-datapath specific but target-neutral.  Eligibility for actually running the
    mixed program is a separate backend execution-capability gate in the shared performance profile.
    """
    from merlin.development.targetgen.corpus.binding import CorpusBinding, _numeric_policy

    role = str(entry.get("comparison_role") or "")
    transform = str(entry.get("host_transform") or "")
    expected_transform = {"island": "xor_low_bit", "no_island": "none"}
    if role not in expected_transform or transform != expected_transform[role]:
        raise ValueError(
            f"{entry.get('name', 'host_island_seam')}: comparison_role/host_transform must be one of "
            f"{expected_transform}, got {role!r}/{transform!r}"
        )
    idt = binding.cap_dtype(binding.operand_dtype)
    midt = binding.mlir_dtype(binding.operand_dtype)
    adt = binding.cap_dtype(binding.accum_dtype)
    madt = binding.mlir_dtype(binding.accum_dtype)
    if midt != "i8" or madt != "i32":
        raise ValueError(
            f"host_island_seam currently defines exact signed i8->i32 arithmetic; binding declares "
            f"{binding.operand_dtype!r}->{binding.accum_dtype!r}. A different integer format needs "
            f"its own format-generic transform semantics, not a target-name exception"
        )

    M, K = int(entry.get("M", binding.tile_dim)), int(entry.get("K", 2 * binding.tile_dim))
    H, N = int(entry.get("H", binding.tile_dim)), int(entry.get("N", binding.tile_dim))
    if min(M, K, H, N) < 1:
        raise ValueError(f"host_island_seam dimensions must be positive, got M={M} K={K} H={H} N={N}")
    mask = int(entry.get("xor_mask", 1))
    if not (0 < mask < 128):
        raise ValueError(f"host_island_seam xor_mask must be in [1,127], got {mask}")
    a, w0, w1, out = (
        entry.get("lhs", "A0"),
        entry.get("weight0", "W0"),
        entry.get("weight1", "W1"),
        entry.get("out", "Y0"),
    )
    attrs = {
        "lhs": a,
        "weight0": w0,
        "weight1": w1,
        "out": out,
        "M": M,
        "K": K,
        "H": H,
        "N": N,
        "comparison_role": role,
        "host_transform": transform,
        "xor_mask": mask,
        "accelerator_contractions": 2,
        "shared_accelerator_epilogue": "saturating_i32_to_i8",
    }
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "dev"),
        "interface_mlir": "capsule.interface.mlir",
        "linalg_mlir": "capsule.interface.mlir",
        "inputs": [
            {"name": a, "role": "input", "shape": [M, K], "dtype": idt},
            {"name": w0, "role": "weight", "shape": [K, H], "dtype": idt},
            {"name": w1, "role": "weight", "shape": [H, N], "dtype": idt},
        ],
        "operation": {"op": "host_island_seam", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, entry.get("acc_scale")),
        "expected": {
            "instruction_classes": binding.classes_for(op="matmul", output_dtype=adt, epilogue=[], movement=False),
            "modes": {"matmul": True, "mixed_lane_whole_program": role == "island"},
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }

    # Standard MLIR only.  The provenance attributes describe placement obligations to any backend;
    # they do not encode a target command or target name. These node identities
    # name the generated direct-IR operations, not an observed framework capture.
    lines = [
        'builtin.module attributes {prov.level = "linalg-on-tensors"} {',
        f"  func.func @forward(%{a}: tensor<{M}x{K}x{midt}>, "
        f"%{w0}: tensor<{K}x{H}x{midt}>, %{w1}: tensor<{H}x{N}x{midt}>) "
        f"-> tensor<{M}x{N}x{madt}> {{",
        f"    %acc0_empty = tensor.empty() : tensor<{M}x{H}x{madt}>",
        f"    %zero_acc0 = arith.constant 0 : {madt}",
        f'    %acc0_init = linalg.fill {{prov.op = "fill", prov.family = "fill"}} '
        f"ins(%zero_acc0 : {madt}) outs(%acc0_empty : tensor<{M}x{H}x{madt}>) "
        f"-> tensor<{M}x{H}x{madt}>",
        f"    %acc0 = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, "
        f"affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], "
        f'iterator_types = ["parallel", "parallel", "reduction"]}} '
        f"ins(%{a}, %{w0} : tensor<{M}x{K}x{midt}>, tensor<{K}x{H}x{midt}>) "
        f"outs(%acc0_init : tensor<{M}x{H}x{madt}>) attrs = "
        f'{{prov.region_id = "contraction_0", prov.source_node_ids = ["direct_ir.contraction_0"], '
        f'prov.op = "matmul", prov.family = "contraction"}} {{',
        f"    ^bb_acc0(%lhs: {midt}, %rhs: {midt}, %acc: {madt}):",
        f"      %lhs_wide = arith.extsi %lhs : {midt} to {madt}",
        f"      %rhs_wide = arith.extsi %rhs : {midt} to {madt}",
        f"      %product = arith.muli %lhs_wide, %rhs_wide : {madt}",
        f"      %sum = arith.addi %acc, %product : {madt}",
        f"      linalg.yield %sum : {madt}",
        f"    }} -> tensor<{M}x{H}x{madt}>",
        f"    %narrow_empty = tensor.empty() : tensor<{M}x{H}x{midt}>",
        f"    %lo = arith.constant -128 : {madt}",
        f"    %hi = arith.constant 127 : {madt}",
        f"    %narrow = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, "
        f'affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}} '
        f"ins(%acc0 : tensor<{M}x{H}x{madt}>) outs(%narrow_empty : tensor<{M}x{H}x{midt}>) "
        f'attrs = {{prov.region_id = "requant_0", prov.source_node_ids = ["direct_ir.requant_0"], '
        f'prov.op = "requant", '
        f'prov.family = "elementwise_map", prov.placement = "accelerator_epilogue"}} {{',
        f"    ^bb0(%x: {madt}, %unused: {midt}):",
        f"      %clamp_lo = arith.maxsi %x, %lo : {madt}",
        f"      %clamp_hi = arith.minsi %clamp_lo, %hi : {madt}",
        f"      %narrowed = arith.trunci %clamp_hi : {madt} to {midt}",
        f"      linalg.yield %narrowed : {midt}",
        f"    }} -> tensor<{M}x{H}x{midt}>",
    ]
    middle = "%narrow"
    if role == "island":
        lines += [
            f"    %host_empty = tensor.empty() : tensor<{M}x{H}x{midt}>",
            f"    %xor_mask = arith.constant {mask} : {midt}",
            f"    %host = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, "
            f'affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}} '
            f"ins(%narrow : tensor<{M}x{H}x{midt}>) outs(%host_empty : tensor<{M}x{H}x{midt}>) "
            f'attrs = {{prov.region_id = "host_island_0", prov.source_node_ids = ["direct_ir.host_island_0"], '
            f'prov.op = "xor_low_bit", '
            f'prov.family = "host_scalar", prov.placement = "host_required"}} {{',
            f"    ^bb1(%x: {midt}, %unused: {midt}):",
            f"      %mapped = arith.xori %x, %xor_mask : {midt}",
            f"      linalg.yield %mapped : {midt}",
            f"    }} -> tensor<{M}x{H}x{midt}>",
        ]
        middle = "%host"
    lines += [
        f"    %acc1_empty = tensor.empty() : tensor<{M}x{N}x{madt}>",
        f"    %zero_acc1 = arith.constant 0 : {madt}",
        f'    %acc1_init = linalg.fill {{prov.op = "fill", prov.family = "fill"}} '
        f"ins(%zero_acc1 : {madt}) outs(%acc1_empty : tensor<{M}x{N}x{madt}>) "
        f"-> tensor<{M}x{N}x{madt}>",
        f"    %result = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, "
        f"affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], "
        f'iterator_types = ["parallel", "parallel", "reduction"]}} '
        f"ins({middle}, %{w1} : tensor<{M}x{H}x{midt}>, tensor<{H}x{N}x{midt}>) "
        f"outs(%acc1_init : tensor<{M}x{N}x{madt}>) attrs = "
        f'{{prov.region_id = "contraction_1", prov.source_node_ids = ["direct_ir.contraction_1"], '
        f'prov.op = "matmul", prov.family = "contraction"}} {{',
        f"    ^bb_acc1(%lhs: {midt}, %rhs: {midt}, %acc: {madt}):",
        f"      %lhs_wide = arith.extsi %lhs : {midt} to {madt}",
        f"      %rhs_wide = arith.extsi %rhs : {midt} to {madt}",
        f"      %product = arith.muli %lhs_wide, %rhs_wide : {madt}",
        f"      %sum = arith.addi %acc, %product : {madt}",
        f"      linalg.yield %sum : {madt}",
        f"    }} -> tensor<{M}x{N}x{madt}>",
        f"    func.return %result : tensor<{M}x{N}x{madt}>",
        "  }",
        "}",
    ]
    return cap, "\n".join(lines) + "\n"


def scope_chain_region_ops(families: list[str]) -> list[str]:
    """Exact operations emitted by the current standalone scope-chain writer.

    This is an emitter inventory, not a semantic equivalence rule for captured
    source operations. Phase 0 must compare against it before selecting a
    source-derived performance cohort.
    """
    if not (
        isinstance(families, list)
        and len(families) >= 3
        and families[:2] == ["movement", "contraction"]
        and all(family == "elementwise_map" for family in families[2:])
    ):
        raise ValueError("scope_chain needs a movement/contraction/elementwise-map+ family list")
    return ["transpose", "matmul", *(["add"] * (len(families) - 2))]


def build_scope_chain(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A priced contraction in a requirement-selected movement/contraction/map chain.

    The transpose and scalar maps are real tensor regions, not command-buffer
    annotations. Only the contraction contributes MACs. This member makes no claim
    about epilogue fusion or carried configuration; those have separate functional
    witnesses derived from the selected requirements.
    """
    from merlin.development.targetgen.corpus.binding import CorpusBinding, _numeric_policy

    idt = binding.cap_dtype(binding.operand_dtype)
    midt = binding.mlir_dtype(binding.operand_dtype)
    adt = binding.cap_dtype(binding.accum_dtype)
    madt = binding.mlir_dtype(binding.accum_dtype)
    if not (
        midt.startswith("i")
        and midt[1:].isdigit()
        and madt.startswith("i")
        and madt[1:].isdigit()
        and int(madt[1:]) > int(midt[1:])
    ):
        raise ValueError(
            f"scope_chain supports signed integer operands with a wider accumulator; "
            f"binding declares {binding.operand_dtype!r}->{binding.accum_dtype!r}"
        )
    families = entry.get("scope_families")
    region_ops = scope_chain_region_ops(families)
    map_count = len(families) - 2
    M, K, N = (int(entry.get(axis, binding.tile_dim)) for axis in ("M", "K", "N"))
    if min(M, K, N) < 1:
        raise ValueError("scope_chain extents must be positive")
    a, w, out = entry.get("lhs", "A0"), entry.get("weight", "W"), entry.get("out", "Y0")
    attrs = {
        "lhs": a,
        "weight": w,
        "out": out,
        "M": M,
        "K": K,
        "N": N,
        "scope_families": list(families),
        "scope_region_ops": region_ops,
        "scope_signature": " -> ".join(families),
        "map_count": map_count,
        "output_dtype": adt,
    }
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "dev"),
        "interface_mlir": "capsule.interface.mlir",
        "linalg_mlir": "capsule.interface.mlir",
        "inputs": [
            {"name": a, "role": "input", "shape": [M, K], "dtype": idt},
            {"name": w, "role": "weight", "shape": [N, K], "dtype": idt},
        ],
        "operation": {"op": "scope_chain", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": {
            "instruction_classes": binding.classes_for(op="matmul", output_dtype=adt, epilogue=[], movement=True),
            "modes": {"matmul": True},
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    lines = [
        'builtin.module attributes {prov.level = "linalg-on-tensors"} {',
        f"  func.func @forward(%{a}: tensor<{M}x{K}x{midt}>, "
        f"%{w}: tensor<{N}x{K}x{midt}>) -> tensor<{M}x{N}x{madt}> {{",
        f"    %wt_empty = tensor.empty() : tensor<{K}x{N}x{midt}>",
        f"    %wt = linalg.transpose ins(%{w} : tensor<{N}x{K}x{midt}>) "
        f"outs(%wt_empty : tensor<{K}x{N}x{midt}>) permutation = [1, 0]",
        f"    %zero = arith.constant 0 : {madt}",
        f"    %acc_init = arith.constant dense<0> : tensor<{M}x{N}x{madt}>",
        "    %acc = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, "
        "affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], "
        'iterator_types = ["parallel", "parallel", "reduction"]} '
        f"ins(%{a}, %wt : tensor<{M}x{K}x{midt}>, tensor<{K}x{N}x{midt}>) "
        f"outs(%acc_init : tensor<{M}x{N}x{madt}>) "
        'attrs = {prov.region_id = "contraction_0", prov.op = "matmul", prov.family = "contraction"} {',
        f"    ^bb0(%lhs: {midt}, %rhs: {midt}, %old: {madt}):",
        f"      %lhs_wide = arith.extsi %lhs : {midt} to {madt}",
        f"      %rhs_wide = arith.extsi %rhs : {midt} to {madt}",
        f"      %product = arith.muli %lhs_wide, %rhs_wide : {madt}",
        f"      %sum = arith.addi %old, %product : {madt}",
        f"      linalg.yield %sum : {madt}",
        f"    }} -> tensor<{M}x{N}x{madt}>",
    ]
    previous = "%acc"
    for index in range(map_count):
        result = f"%map{index}"
        lines += [
            f"    %map{index}_empty = tensor.empty() : tensor<{M}x{N}x{madt}>",
            f"    %step{index} = arith.constant 1 : {madt}",
            "    " + result + " = linalg.generic {indexing_maps = ["
            "affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], "
            'iterator_types = ["parallel", "parallel"]} '
            f"ins({previous} : tensor<{M}x{N}x{madt}>) "
            f"outs(%map{index}_empty : tensor<{M}x{N}x{madt}>) "
            f'attrs = {{prov.region_id = "map_{index}", prov.op = "add", prov.family = "elementwise"}} {{',
            f"    ^bb{index + 1}(%x: {madt}, %unused: {madt}):",
            f"      %mapped = arith.addi %x, %step{index} : {madt}",
            f"      linalg.yield %mapped : {madt}",
            f"    }} -> tensor<{M}x{N}x{madt}>",
        ]
        previous = result
    lines += [f"    func.return {previous} : tensor<{M}x{N}x{madt}>", "  }", "}"]
    return cap, "\n".join(lines) + "\n"


BUILDERS: dict[str, Callable[[dict, CorpusBinding], tuple[dict, str]]] = {
    "maxpool2d": build_maxpool2d,
    "matmul": build_matmul,
    "linear": build_matmul,
    "fused_matmul_bias": build_matmul,
    "bias_add": build_bias_add,
    "residual_add": build_residual_add,
    # THE CAPTURE'S OWN SPELLING FOR THE SAME REGION, and the reason it needs one. `add` is what a
    # captured model's `prov.op` says for two quantized activations summed elementwise -- 16 such
    # regions on a captured ResNet-50, every one of them left on the host. It reached a capsule only
    # through the PyTorch writer, which the generator refuses outside the float regime, so on an
    # INTEGER target the op a deployable model actually contains could not be written at all: no
    # capsule could demand the work, so no backend could be failed for skipping it.
    #
    # It is the one family of the four the ABI contract's `host_refusal_evidence` adjudicates with a
    # FACET (`readout_facet.operand_sum`) rather than with an epilogue stage -- i.e. the one that has a
    # standalone witness on the device. Quantize, minmax and pool are readout stages;
    # maxpool2d above composes movement with readout, rather than claiming a pool engine.
    "add": build_residual_add,
    "residual_seam": build_residual_seam,
    "resident_reuse": build_resident_reuse,
    "movement": build_movement,
    "attention_qk": build_attention_qk,
    "rmsnorm": build_rmsnorm,
    "conv2d": build_conv2d,
    "attention_mx": build_attention_mx,
    "rmsnorm_qkv": build_rmsnorm_qkv,
    "rope_qkv": build_rope_qkv,
    "gemv_batched": build_gemv_batched,
    "host_island_seam": build_host_island_seam,
    "scope_chain": build_scope_chain,
}


def build_component_program(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """Independent typed DAGs use the same builder/writer admission path."""
    from merlin.development.targetgen.corpus.binding import CorpusBinding
    from merlin.development.targetgen.corpus.components import build as build_program

    return build_program(entry, binding)


BUILDERS["component_program"] = build_component_program


def build_spatial_producer_pool(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """Two separately rounded original source operations; no coefficient collapse."""
    from merlin.development.targetgen.corpus.binding import CorpusBinding
    from merlin.development.targetgen.corpus.corpus_spatial_pair import build as build_pair

    return build_pair(entry, binding)


BUILDERS["spatial_producer_pool"] = build_spatial_producer_pool


def build_quantized_producer_sum(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """Preserve the original producer Q1 and ordered residual Q2/activation."""
    from merlin.development.targetgen.corpus.binding import CorpusBinding
    from merlin.development.targetgen.corpus.corpus_quantized_sum import build as build_pair

    return build_pair(entry, binding)


BUILDERS["quantized_producer_sum"] = build_quantized_producer_sum


BUILDERS["sibling_feed"] = _build_sibling_feed  # sibling contractions, merged, feeding a contraction


def build_activation_matmul(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """An activation x activation contraction followed by a reshape (op == activation_matmul).

    Written as the generic component program so the reference is the independent exact integer
    evaluator and the written program is screened operation by operation like any captured source.
    """
    from merlin.development.targetgen.corpus.binding import CorpusBinding

    return build_component_program({**entry, "program": activation_matmul_program(entry, binding)}, binding)


BUILDERS["activation_matmul"] = build_activation_matmul


COMPONENT_PROGRAM_SOURCES: dict[str, Callable[[dict, CorpusBinding], dict]] = {
    "activation_matmul": activation_matmul_program,
}


def entry_component_program(entry: dict, binding: CorpusBinding) -> dict | None:
    """The typed component program an entry will be written as, or ``None`` when it is not one."""
    from merlin.development.targetgen.corpus.binding import CorpusBinding

    op = entry.get("op")
    if op == "component_program":
        return entry.get("program")
    source = COMPONENT_PROGRAM_SOURCES.get(str(op))
    return source(entry, binding) if source is not None else None


_AXIS_BY_KIND: dict[str, str] = {"isa": "seen", "layer": "composition", "model_slice": "composition", "model": "model"}


def _fused_families(entry: dict, family: str | None, binding: CorpusBinding) -> tuple[set, set]:
    """``(families this capsule's epilogue exercises, stage names that did not resolve)``.

    Empty for a capsule with no epilogue: a plain contraction must not claim the epilogue cell, or every
    capsule in the corpus would cover a family none of them exercises.
    """
    from merlin.development.targetgen.corpus.binding import CorpusBinding
    from merlin.target.capabilities import semantic_families as _sf

    stages = [str(s) for s in (entry.get("epilogue") or []) if s]
    if not stages or not family:
        return set(), set()
    try:
        from merlin.target.capabilities.eligibility import capability_map_for_target

        cap_map = capability_map_for_target(getattr(binding, "target", "") or "")
    except Exception:  # noqa: BLE001 — unresolvable manifest
        return set(), set()
    fused, unresolved = set(), set()
    for stage in stages:
        sfam = _sf.from_op(stage)
        if sfam is None:
            unresolved.add(stage)
            continue
        cap = cap_map.get(sfam)
        if cap is not None and family in (getattr(cap, "composed_with", ()) or ()):
            fused.add(sfam)
    return fused, unresolved


def build(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """Build an operation and enforce the shared palette, epilogue and stimulus contract."""
    from merlin.development.targetgen.corpus.binding import CorpusBinding
    from merlin.development.targetgen.corpus.corpus_builder import build_entry

    return build_entry(entry, binding, builders=BUILDERS, semantic_block=_semantic_block)

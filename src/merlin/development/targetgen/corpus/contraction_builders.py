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

from merlin.development.targetgen.corpus.corpus_activation import activation_matmul_program
from merlin.development.targetgen.corpus.corpus_pool import build as build_maxpool2d
from merlin.development.targetgen.corpus.corpus_sibling_feed import build as _build_sibling_feed
from merlin.development.targetgen.corpus.declarations import _semantic_block
from merlin.development.targetgen.grading.golden_inputs import dtype_bits

# ---- corpus_spec -------------------------------------------------------------------------------


def build_matmul(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A single weight-stationary matmul/linear capsule (op ∈ {matmul, linear, fused_matmul_bias}).

    ``fused_matmul_bias`` is the same weight-stationary contraction with its bias epilogue forced on --
    a distinct op name rather than a flag because the L5 fusion claim compares it against the two
    capsules it replaces, and a group whose members differ only by an attribute cannot be named."""
    from merlin.development.targetgen.corpus.binding import (
        _BIAS_STAGES,
        CorpusBinding,
        _bias_operand,
        _default_modes,
        _matmul_inputs,
        _numeric_policy,
        _pool_epilogue,
        _resolve_output_dtype,
        is_block_scaled,
        requant_shift_for,
    )
    from merlin.target.contract.interfaces import matmul as MSE

    D = binding.tile_dim
    M = entry.get("M_tiles", 1) * D if "M" not in entry else entry["M"]
    K = entry.get("K_tiles", 1) * D if "K" not in entry else entry["K"]
    N = entry.get("N_tiles", 1) * D if "N" not in entry else entry["N"]
    lhs, weight, out = entry.get("lhs", "A0"), entry.get("weight", "W"), entry.get("out", "Y0")
    epilogue = list(entry.get("epilogue", []))
    op = entry.get("op", "matmul")
    if op == "fused_matmul_bias" and not any(s in _BIAS_STAGES for s in epilogue):
        # Ahead of _resolve_output_dtype, which reads the epilogue to decide the committed dtype.
        epilogue.insert(0, "bias_add")
    readout = MSE.acc_scale_parameters(entry, epilogue, N, op=entry.get("name", op))
    acc_scale = readout.get("acc_scale")
    output_dtype = _resolve_output_dtype(
        binding,
        epilogue,
        entry,
        available_operand_roles=frozenset({"bias"})
        if any(stage in _BIAS_STAGES for stage in epilogue)
        else frozenset(),
    )
    odt = binding.cap_dtype(output_dtype)
    idt = binding.cap_dtype(binding.operand_dtype)
    bias, bias_dt = _bias_operand(entry, epilogue, N, binding, op)
    attrs: dict[str, Any] = {"lhs": lhs, "weight": weight, "out": out, "epilogue": epilogue, "output_dtype": odt}
    if bias:
        attrs["bias"] = bias
    # A pooling epilogue rides on the SAME attributes the store path configures, and it changes the
    # committed extent -- the commit result type below is the POOLED row count, not M. A contraction
    # carries no spatial extent of its own, so the entry declares pool_in_dims (see _pool_epilogue).
    pool_attrs, commit_rows = _pool_epilogue(entry, rows=M, in_dims=None, op=op)
    attrs.update(pool_attrs)
    # The integer requant stage's one parameter. Declared on the capsule so the golden reads the SAME
    # number the emitted interface hands the backend (and hence the reference), never two defaults.
    requant_shift = requant_shift_for(entry, binding, epilogue, op=op)
    if requant_shift is not None:
        attrs["requant_shift"] = requant_shift
    attrs.update(readout)
    if entry.get("semantic"):
        attrs["semantic"] = entry["semantic"]
    expected = {
        "instruction_classes": binding.classes_for(op=op, output_dtype=odt, epilogue=epilogue, movement=False),
        "modes": dict(entry["modes"]) if "modes" in entry else _default_modes(binding, output_dtype, epilogue),
    }
    if entry.get("forbidden"):
        expected["forbidden_classes"] = entry["forbidden"]
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": (
            _matmul_inputs(lhs, weight, M, K, N, idt, binding)
            + ([{"name": bias, "role": "bias", "shape": [N], "dtype": bias_dt}] if bias else [])
        ),
        "operation": {"op": op, "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, output_dtype, acc_scale),
        "expected": expected,
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    mlir = MSE.emit_interface_mlir(
        lhs=lhs,
        weight=weight,
        out=out,
        M=M,
        K=K,
        N=N,
        epilogue=epilogue,
        output_dtype=odt,
        acc_scale=acc_scale,
        acc_scale_by_column=readout.get("acc_scale_by_column"),
        comment=entry.get("comment", ""),
        pool_attrs=pool_attrs or None,
        commit_rows=commit_rows,
        requant_shift=requant_shift,
        bias=bias,
        bias_dtype=(binding.mlir_dtype(binding.accum_dtype) if bias else None),
        target=binding.target,
        operand_dtype=binding.mlir_dtype(binding.operand_dtype),
        acc_dtype=binding.mlir_dtype(binding.accum_dtype),
        scale_block=(binding.scale_block if is_block_scaled(binding.operand_dtype) else None),
    )
    return cap, mlir


def _iface_prelude(target: str, comment: str) -> list[str]:
    head = (
        'module attributes {merlin_iface.version = "0.1", '
        f'merlin_iface.target = "{target}", merlin_iface.abi_version = "0.1"}} {{'
    )
    return [f"// {comment}", head] if comment else [head]


def build_resident_reuse(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """One resident weight reused across two matmuls (op == resident_reuse).

    ⚠️ AN EXTENT MAY BE DECLARED EITHER WAY, and reading only the ``_tiles`` spelling silently drops
    the other. `expand_sweeps` resolves an ``axes:`` entry to the BARE name (`K`, `N`, `M`), so a sweep
    declaring ``K: ["4*tile", "8*tile"]`` reached `entry["K"]` while this builder read
    ``entry["K_tiles"]``, defaulted to 1, and emitted the tile edge for BOTH points. Measured: PC00_k64
    and PC01_k128 had BYTE-IDENTICAL interfaces and differed only in their name and a prose string --
    the two "separation regimes" `gate.capacity: at_least_two_separation_regimes` demands were one
    regime with two labels, so a paired differential over them would have measured the same program
    twice. Same failure shape as the holdout sets that turned out to be renames.
    """
    from merlin.development.targetgen.corpus.binding import CorpusBinding, _numeric_policy, _resolve_output_dtype

    D = binding.tile_dim
    K = entry.get("K", entry.get("K_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    weight = entry.get("weight", "W")
    idt, adt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)
    midt, madt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    inputs = [{"name": weight, "role": "weight", "shape": [K, N], "dtype": idt}]
    matmuls = []
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L.append(f'  %{weight} = merlin_iface.tensor {{name = "{weight}", role = "weight"}} : tensor<{K}x{N}x{midt}>')
    for idx, mm in enumerate(entry["matmuls"]):
        an, oname = mm["lhs"], mm["out"]
        M = mm.get("M", mm.get("M_tiles", 1) * D)
        epi = list(mm.get("epilogue", []))
        inputs.append({"name": an, "role": "input", "shape": [M, K], "dtype": idt})
        # DERIVED PER MATMUL, not the accumulator width. This builder is the SECOND place a fused
        # epilogue's commit dtype is chosen, and it was missed when the first was fixed: a resident-reuse
        # capsule carrying `relu` committed at i32, which the grade-time readout check then refused
        # because the full-width readout applies no stage. Same derivation as `_resolve_output_dtype`
        # uses everywhere else, so the two cannot drift apart again.
        _odt = _resolve_output_dtype(binding, epi, mm)
        matmuls.append({"lhs": an, "out": oname, "epilogue": epi, "output_dtype": _odt})
        L.append(f'  %{an} = merlin_iface.tensor {{name = "{an}", role = "input"}} : tensor<{M}x{K}x{midt}>')
    L.append(
        f'  %{weight}_res = merlin_iface.resident_pack %{weight} {{layout = "packed_rhs"}} '
        f": (tensor<{K}x{N}x{midt}>) -> !merlin_iface.resident"
    )
    for idx, mm in enumerate(entry["matmuls"]):
        an, oname = mm["lhs"], mm["out"]
        M = mm.get("M", mm.get("M_tiles", 1) * D)
        _epi_list = list(mm.get("epilogue", []))
        epi = ", ".join(f'"{e}"' for e in _epi_list)
        # The emitted interface must state the SAME commit dtype as the record above, or the capsule
        # declares one width and hands the compiler another.
        _odt = _resolve_output_dtype(binding, _epi_list, mm)
        _modt = binding.mlir_dtype(_odt)
        L.append(
            f"  %acc{idx} = merlin_iface.matmul %{an}, %{weight}_res "
            f": (tensor<{M}x{K}x{midt}>, !merlin_iface.resident) -> !merlin_iface.acc<{madt}>"
        )
        L.append(
            f'  %{oname} = merlin_iface.commit %acc{idx} {{name = "{oname}", epilogue = [{epi}], '
            f'output_dtype = "{_odt}"}} : (!merlin_iface.acc<{madt}>) -> tensor<{M}x{N}x{_modt}>'
        )
    L.append(f"  merlin_iface.evict %{weight}_res : (!merlin_iface.resident) -> ()")
    L.append("}")
    attrs = {"weight": weight, "matmuls": matmuls, "semantic": entry.get("semantic", "resident_reuse")}
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": inputs,
        "operation": {"op": "resident_reuse", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": {
            "instruction_classes": binding.classes_for(op="matmul", output_dtype=adt),
            "modes": {
                "resident_reuse": sum((mm.get("M", mm.get("M_tiles", 1) * D) + D - 1) // D for mm in entry["matmuls"])
                > 1
            },
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    return cap, "\n".join(L) + "\n"


def build_movement(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A load->store dequant movement capsule (op == movement): operand dtype in, accumulate dtype out."""
    from merlin.development.targetgen.corpus.binding import CorpusBinding, _numeric_policy

    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    src, out = entry.get("src", "X"), entry.get("out", "Y0")
    idt, odt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    attrs = {"out": out, "src": src, "semantic": entry.get("semantic", "mvin_mvout"), "output_dtype": odt}
    expected = {"instruction_classes": binding.classes_for(op="movement", movement=True), "modes": {"movement": True}}
    if entry.get("forbidden"):
        expected["forbidden_classes"] = entry["forbidden"]
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": src, "role": "input", "shape": [M, N], "dtype": idt}],
        "operation": {"op": "movement", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": expected,
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{src} = merlin_iface.tensor {{name = "{src}", role = "input"}} : tensor<{M}x{N}x{midt}>',
        f'  %{out} = merlin_iface.movement %{src} {{name = "{out}", semantic = "mvin_mvout", '
        f'output_dtype = "{odt}"}} : (tensor<{M}x{N}x{midt}>) -> tensor<{M}x{N}x{modt}>',
        "}",
    ]
    return cap, "\n".join(L) + "\n"


def build_attention_qk(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """Q @ K^T attention scores (op == attention_qk): the device does the transpose internally."""
    from merlin.development.targetgen.corpus.binding import CorpusBinding, _numeric_policy

    D = binding.tile_dim
    # Both spellings, for the reason spelled out in `build_resident_reuse`: a sweep declares the bare
    # axis name, and reading only `_tiles` turns a declared extent into the tile edge without a word.
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    Kd = entry.get("K", entry.get("K_tiles", 1) * D)
    # Queries and keys need not have the same sequence length (prefill against
    # a carried KV cache is rectangular). Preserve the old square default.
    N = entry.get("N", entry["N_tiles"] * D if "N_tiles" in entry else M)
    q, k, out = entry.get("q", "Q"), entry.get("k", "K"), entry.get("out", "Y0")
    idt, odt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    attrs = {"q": q, "k": k, "out": out, "epilogue": [], "output_dtype": odt}
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [
            {"name": q, "role": "input", "shape": [M, Kd], "dtype": idt},
            {"name": k, "role": "input", "shape": [N, Kd], "dtype": idt},
        ],
        "operation": {"op": "attention_qk", "attributes": attrs},
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
        f'  %{q} = merlin_iface.tensor {{name = "{q}", role = "input"}} : tensor<{M}x{Kd}x{midt}>',
        f'  %{k} = merlin_iface.tensor {{name = "{k}", role = "input"}} : tensor<{N}x{Kd}x{midt}>',
        f'  %{out} = merlin_iface.attention_qk %{q}, %{k} {{name = "{out}", output_dtype = "{odt}"}} '
        f": (tensor<{M}x{Kd}x{midt}>, tensor<{N}x{Kd}x{midt}>) -> tensor<{M}x{N}x{modt}>",
        "}",
    ]
    return cap, "\n".join(L) + "\n"


def build_rmsnorm(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A row RMSNorm capsule (op == rmsnorm): X[M,K] * rsqrt(mean(X^2)+eps) * gamma[1,K] -> [M,K]. A SIMT
    (elementwise/reduction) op — its golden is ordinary IEEE float, computed by the driver."""
    from merlin.development.targetgen.corpus.binding import CorpusBinding, _numeric_policy

    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    K = entry.get("K", entry.get("K_tiles", 1) * D)
    x, gamma, out = entry.get("src", "X"), entry.get("gamma", "G"), entry.get("out", "Y0")
    eps = entry.get("eps", 1.0 / 65536.0)
    idt = binding.cap_dtype(binding.operand_dtype)
    odt = binding.cap_dtype(binding.accum_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    attrs = {"src": x, "gamma": gamma, "out": out, "eps": eps, "semantic": "rmsnorm", "output_dtype": odt}
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [
            {"name": x, "role": "input", "shape": [M, K], "dtype": idt},
            {"name": gamma, "role": "weight", "shape": [1, K], "dtype": idt},
        ],
        "operation": {"op": "rmsnorm", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": {"instruction_classes": [], "modes": {"rmsnorm": True}},
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{x} = merlin_iface.tensor {{name = "{x}", role = "input"}} : tensor<{M}x{K}x{midt}>',
        f'  %{gamma} = merlin_iface.tensor {{name = "{gamma}", role = "weight"}} : tensor<1x{K}x{midt}>',
        f'  %{out} = merlin_iface.rmsnorm %{x}, %{gamma} {{name = "{out}", eps = {eps:.9e} : f64, '
        f'output_dtype = "{odt}"}} : (tensor<{M}x{K}x{midt}>, tensor<1x{K}x{midt}>) -> tensor<{M}x{K}x{modt}>',
        "}",
    ]
    return cap, "\n".join(L) + "\n"


def build_bias_add(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A standalone per-column bias add (op == bias_add): X[M,N] + B[N] -> [M,N].

    The unfused half of the L5 fusion pair. It exists so ``fused_matmul_bias`` has something to be
    compared against: the claim is that the fused capsule costs less than the two capsules it replaces,
    and a comparison group with only a fused member is a declaration with no content.

    ⚠️ ITS OPERANDS DEFAULT TO THE ACCUMULATOR DOMAIN, not the operand one. As the unfused half of the
    fusion pair this op IS the bias stage of the fused capsule standing on its own, and that stage runs
    on the accumulator before any requant -- ``Tensor.add_bias`` returns ``self.dtype`` for exactly that
    reason. Declaring X in the operand dtype would price a different computation from the one the fused
    member performs, visibly so on an int8/i32 datapath, and the two members' cycles would then not be
    summable. The fusion-pair entries declare no dtype, so they keep that domain.

    AN ENTRY OUTSIDE A COMPARISON GROUP OVERRIDES IT with its own declared dtype, because a capsule
    synthesized for a conformance cell is ABOUT that cell's operand dtype. Measured on atlas: the three
    ``elementwise_map/fp8_e4m3`` cells chose this op (the only elementwise builder writable at fp8) and
    it emitted bf16 tensors -- so the capsules were built, named for the cells, and covered none of
    them, while the corpus separately reported an ``elementwise_map/bf16`` cell nothing had asked for.
    A capsule that silently answers a different dtype than the one it is named for is worse than a
    missing one.

    The comparison group is the exception and it is the ONE case the accumulator paragraph above is
    about: there this capsule exists to be subtracted from a fused one, and a member priced in a
    different domain from the stage it stands in for makes the two members' cycles unsummable.
    """
    from merlin.development.targetgen.corpus.binding import CorpusBinding, _numeric_policy

    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    src, bias, out = entry.get("src", "X"), entry.get("bias", "B"), entry.get("out", "Y0")
    _dom = binding.accum_dtype if entry.get("comparison_group") else (entry.get("operand_dtype") or binding.accum_dtype)
    adt, madt = binding.cap_dtype(_dom), binding.mlir_dtype(_dom)
    epilogue = list(entry.get("epilogue", [])) or ["bias_add"]
    attrs: dict[str, Any] = {"src": src, "bias": bias, "out": out, "epilogue": epilogue, "output_dtype": adt}
    if entry.get("semantic"):
        attrs["semantic"] = entry["semantic"]
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [
            {"name": src, "role": "input", "shape": [M, N], "dtype": adt},
            {"name": bias, "role": "bias", "shape": [N], "dtype": adt},
        ],
        "operation": {"op": "bias_add", "attributes": attrs},
        # The policy is written for the dtype the OUTPUT is actually in, which is `_dom` -- the same
        # value the tensors carry. Pinning it to the accumulator while the tensors moved would grade an
        # fp8 output against an integer/bf16 comparison.
        "numeric_policy": _numeric_policy(binding, _dom, None),
        # Derived like every other op's: on a target with a binary tensor-compute role this resolves to
        # that class, and on one that has no such role it resolves to the data motion alone rather than
        # to the systolic sequence this op does not use.
        "expected": {
            "instruction_classes": binding.classes_for(op="bias_add", output_dtype=adt),
            "modes": dict(entry["modes"]) if "modes" in entry else {},
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{src} = merlin_iface.tensor {{name = "{src}", role = "input"}} : tensor<{M}x{N}x{madt}>',
        f'  %{bias} = merlin_iface.tensor {{name = "{bias}", role = "bias"}} : tensor<{N}x{madt}>',
        f'  %{out} = merlin_iface.bias_add %{src}, %{bias} {{name = "{out}", '
        f'output_dtype = "{adt}"}} : (tensor<{M}x{N}x{madt}>, tensor<{N}x{madt}>) '
        f"-> tensor<{M}x{N}x{madt}>",
        "}",
    ]
    return cap, "\n".join(L) + "\n"


def build_residual_add(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A quantized residual add (op == residual_add): ``X0[M,N] , X1[M,N] -> Y[M,N]``.

    Both operands are in the OPERAND dtype (they are two activations, each already quantized), each
    is carried into the output's domain by its own multiplier, and the reference rounds the sum
    once (``Tensor.residual_add_scaled``). ``bound_lsb`` is REQUIRED and has no default: it is the
    declared number of output steps a target's result may lie from that reference, and it becomes
    the comparison tolerance (``compare: bounded_int``). Zero demands the capture's arithmetic; one
    admits a unit that scales and rounds each operand on the way in and then adds integers, which
    is what a systolic unit's accumulating load does. A bound chosen after seeing a result is not a
    gate, so the entry states it.
    """
    from merlin.development.targetgen.corpus.binding import CorpusBinding
    from merlin.runtime.engine.command_buffer import residual_add_params

    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    lhs, rhs, out = entry.get("lhs", "X0"), entry.get("rhs", "X1"), entry.get("out", "Y0")
    epilogue = list(entry.get("epilogue", []))
    attrs: dict[str, Any] = {
        "lhs": lhs,
        "rhs": rhs,
        "out": out,
        "epilogue": epilogue,
        "lhs_scale": entry.get("lhs_scale"),
        "rhs_scale": entry.get("rhs_scale"),
        "bound_lsb": entry.get("bound_lsb"),
    }
    lhs_scale, rhs_scale, bound = residual_add_params(attrs, op=str(entry.get("name", "residual_add")))
    ordered_scales = entry.get("source_scales")
    ordered_text = ""
    if ordered_scales is not None:
        import math

        if (
            type(ordered_scales) is not list
            or len(ordered_scales) != 3
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in ordered_scales
            )
        ):
            raise ValueError("residual source_scales requires three positive finite ordered scalars")
        attrs["source_scales"] = list(ordered_scales)
        ordered_text = "source_scales = [" + ", ".join(f"{float(value)!r} : f32" for value in ordered_scales) + "], "
    idt, midt = binding.cap_dtype(binding.operand_dtype), binding.mlir_dtype(binding.operand_dtype)
    attrs["output_dtype"] = idt
    if entry.get("semantic"):
        attrs["semantic"] = entry["semantic"]
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [
            {"name": lhs, "role": "input", "shape": [M, N], "dtype": idt},
            {"name": rhs, "role": "input", "shape": [M, N], "dtype": idt},
        ],
        "operation": {"op": "residual_add", "attributes": attrs},
        # The declared bound IS the comparison: absolute, in output steps, nothing relative.
        "numeric_policy": {"compare": "bounded_int", "dtype": idt, "atol": bound, "rtol": 0},
        "expected": {
            "instruction_classes": binding.classes_for(op="residual_add", output_dtype=idt),
            "modes": dict(entry["modes"]) if "modes" in entry else {"relu": "relu" in epilogue},
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    epi = ", ".join(f'"{e}"' for e in epilogue)
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{lhs} = merlin_iface.tensor {{name = "{lhs}", role = "input"}} : tensor<{M}x{N}x{midt}>',
        f'  %{rhs} = merlin_iface.tensor {{name = "{rhs}", role = "input"}} : tensor<{M}x{N}x{midt}>',
        f'  %{out} = merlin_iface.residual_add %{lhs}, %{rhs} {{name = "{out}", epilogue = [{epi}], '
        f"lhs_scale = {lhs_scale} : f32, rhs_scale = {rhs_scale} : f32, bound_lsb = {bound} : i64, "
        f"{ordered_text}"
        f'output_dtype = "{idt}"}} : (tensor<{M}x{N}x{midt}>, tensor<{M}x{N}x{midt}>) '
        f"-> tensor<{M}x{N}x{midt}>",
        "}",
    ]
    return cap, "\n".join(L) + "\n"


def build_residual_seam(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A contraction and a residual add as TWO regions, the way a captured model contains them.

    ``A0[M,K] @ W[K,N] -> commit(epilogue) -> Yc[M,N]``, then ``Yc + X1[M,N] -> Y0[M,N]`` as a region
    of its own. Every other capsule that evidences a fused-only family hands the compiler an interface
    where the stage is ALREADY written into the contraction's ``epilogue`` attribute -- the compiler is
    told to fuse, not asked. That tests the emitter and not the decision, and the decision is what a
    deployable compiler was measured getting wrong: on a captured ResNet-50 it left 116 regions on the
    host, 16 of them the residual adds this shape contains, refusing each with "no datapath" while a
    hand-built route ran all of them on the same hardware.

    So the two regions are SEPARATE here and the capsule declares nothing about how they are placed.
    Its demand comes from the profile entry's ``generalization.must_accelerate`` / ``lanes.require``,
    which is the existing machinery; what this builder adds is a capsule those can be asserted ON. A
    backend that fuses the add onto the readout passes; one that computes it on the host is refused by
    the same gate that refuses a declined contraction, because the region is eligible and was declined.

    ``epilogue`` is the CONTRACTION's readout and must narrow, because the add's two operands have to
    meet in one domain and ``X1`` is a declared operand-dtype leaf. That is not a restriction invented
    here -- it is what makes the capsule carry the requant/scale family as well as the add.
    """
    from merlin.development.targetgen.corpus.binding import (
        CorpusBinding,
        _default_modes,
        _resolve_output_dtype,
        requant_shift_for,
    )
    from merlin.runtime.engine.command_buffer import residual_add_params

    D = binding.tile_dim
    M = int(entry.get("M", entry.get("M_tiles", 1) * D))
    K = int(entry.get("K", entry.get("K_tiles", 1) * D))
    N = int(entry.get("N", entry.get("N_tiles", 1) * D))
    lhs, weight = entry.get("lhs", "A0"), entry.get("weight", "W")
    rhs, mid, out = entry.get("rhs", "X1"), entry.get("mid", "Yc"), entry.get("out", "Y0")
    epilogue = list(entry.get("epilogue", []))
    add_epilogue = list(entry.get("add_epilogue", []))
    # The readout has to land in the operand's domain, and saying so here -- with the entry in hand --
    # is the difference between a refusal naming the stage and a shape error inside an engine.
    commit_dtype = _resolve_output_dtype(binding, epilogue, entry)
    idt = binding.cap_dtype(binding.operand_dtype)
    cdt = binding.cap_dtype(commit_dtype)
    if cdt != idt:
        raise ValueError(
            f"{entry.get('name', 'residual_seam')}: the contraction commits {cdt!r} but the residual "
            f"operand {rhs!r} is {idt!r}. The add's operands must meet in one domain, so the entry's "
            f"epilogue {epilogue} has to narrow the accumulator -- there is no implied cast here"
        )
    attrs: dict[str, Any] = {
        "lhs": lhs,
        "weight": weight,
        "rhs": rhs,
        "mid": mid,
        "out": out,
        "epilogue": epilogue,
        "add_epilogue": add_epilogue,
        "output_dtype": idt,
        "lhs_scale": entry.get("lhs_scale"),
        "rhs_scale": entry.get("rhs_scale"),
        "bound_lsb": entry.get("bound_lsb"),
        "semantic": entry.get("semantic", "contraction_then_residual"),
    }
    # The add's own arithmetic is validated by the ABI's reader, exactly as a standalone add's is: the
    # multipliers and the bound have no defaults in the builder or in any engine.
    lhs_scale, rhs_scale, bound = residual_add_params(
        {"lhs_scale": attrs["lhs_scale"], "rhs_scale": attrs["rhs_scale"],
         "bound_lsb": attrs["bound_lsb"], "epilogue": add_epilogue},
        op=str(entry.get("name", "residual_seam")),
    )  # fmt: skip
    requant_shift = requant_shift_for(entry, binding, epilogue, op="residual_seam")
    if requant_shift is not None:
        attrs["requant_shift"] = requant_shift
    acc_scale = entry.get("acc_scale")
    if "acc_scale" in epilogue:
        if acc_scale is None:
            raise ValueError(
                f"{entry.get('name', 'residual_seam')}: an 'acc_scale' epilogue stage needs its "
                f"`acc_scale` multiplier; a stage declared without one would be applied at an assumed 1.0"
            )
        attrs["acc_scale"] = acc_scale
    elif acc_scale is not None:
        raise ValueError(
            f"{entry.get('name', 'residual_seam')}: declares acc_scale = {acc_scale!r} but no "
            f"'acc_scale' epilogue stage; the multiplier would be carried into the capsule and read by nothing"
        )
    midt, madt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [
            {"name": weight, "role": "weight", "shape": [K, N], "dtype": idt},
            {"name": lhs, "role": "input", "shape": [M, K], "dtype": idt},
            {"name": rhs, "role": "input", "shape": [M, N], "dtype": idt},
        ],
        "operation": {"op": "residual_seam", "attributes": attrs},
        # The add is the last arithmetic, so the capsule's tolerance is the add's declared bound.
        "numeric_policy": {"compare": "bounded_int", "dtype": idt, "atol": bound, "rtol": 0},
        "expected": {
            "instruction_classes": binding.classes_for(
                op="matmul", output_dtype=idt, epilogue=epilogue, movement=False
            ),
            "modes": dict(entry["modes"]) if "modes" in entry else _default_modes(binding, commit_dtype, epilogue),
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    epi = ", ".join(f'"{e}"' for e in epilogue)
    add_epi = ", ".join(f'"{e}"' for e in add_epilogue)
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{lhs} = merlin_iface.tensor {{name = "{lhs}", role = "input"}} : tensor<{M}x{K}x{midt}>',
        f'  %{weight} = merlin_iface.tensor {{name = "{weight}", role = "weight"}} : tensor<{K}x{N}x{midt}>',
        f'  %{rhs} = merlin_iface.tensor {{name = "{rhs}", role = "input"}} : tensor<{M}x{N}x{midt}>',
        f'  %{weight}_res = merlin_iface.resident_pack %{weight} {{layout = "packed_rhs"}} '
        f": (tensor<{K}x{N}x{midt}>) -> !merlin_iface.resident",
        f"  %acc = merlin_iface.matmul %{lhs}, %{weight}_res "
        f": (tensor<{M}x{K}x{midt}>, !merlin_iface.resident) -> !merlin_iface.acc<{madt}>",
        f'  %{mid} = merlin_iface.commit %acc {{name = "{mid}", epilogue = [{epi}], '
        f'output_dtype = "{idt}"'
        + (f", requant_shift = {requant_shift} : i64" if requant_shift is not None else "")
        + (f", acc_scale = {acc_scale} : f32" if "acc_scale" in epilogue else "")
        + f"}} : (!merlin_iface.acc<{madt}>) -> tensor<{M}x{N}x{midt}>",
        f'  %{out} = merlin_iface.residual_add %{mid}, %{rhs} {{name = "{out}", '
        f"epilogue = [{add_epi}], lhs_scale = {lhs_scale} : f32, rhs_scale = {rhs_scale} : f32, "
        f'bound_lsb = {bound} : i64, output_dtype = "{idt}"}} '
        f": (tensor<{M}x{N}x{midt}>, tensor<{M}x{N}x{midt}>) -> tensor<{M}x{N}x{midt}>",
        f"  merlin_iface.evict %{weight}_res : (!merlin_iface.resident) -> ()",
        "}",
    ]
    return cap, "\n".join(L) + "\n"


def _attention_mx_inputs(
    q: str, k: str, v: str, M: int, H: int, Skv: int, Dv: int, idt: str, binding: CorpusBinding
) -> list[dict]:
    """Declared leaf operands of a fused block-scaled attention capsule.

    Two MX stages contract over different axes, so they carry different scale streams: the QK stage over
    the head dim ``H`` (scaling q's rows and k's rows), the PV stage over the key count ``Skv`` (scaling
    v's columns). The PV stage's LHS is ``P`` -- the softmax output, an INTERMEDIATE the kernel produces
    and requantizes -- and the exponent it is requantized against was chosen when the golden was built.
    It is therefore declared too: a value the kernel cannot derive is an input, whatever produced it."""
    from merlin.development.targetgen.corpus.binding import CorpusBinding, is_block_scaled

    inputs = [
        {"name": q, "role": "input", "shape": [M, H], "dtype": idt},
        {"name": k, "role": "input", "shape": [Skv, H], "dtype": idt},
        {"name": v, "role": "input", "shape": [Skv, Dv], "dtype": idt},
    ]
    blk = binding.scale_block if is_block_scaled(binding.operand_dtype) else None
    if blk and H % blk == 0 and Skv % blk == 0:
        hg, sg = H // blk, Skv // blk
        inputs += [
            {"name": f"{q}_scale", "role": "scale", "scale_of": q, "block": blk, "shape": [hg, M], "dtype": "e8m0"},
            {"name": f"{k}_scale", "role": "scale", "scale_of": k, "block": blk, "shape": [hg, Skv], "dtype": "e8m0"},
            {"name": f"{v}_scale", "role": "scale", "scale_of": v, "block": blk, "shape": [sg, Dv], "dtype": "e8m0"},
            {"name": "P_scale", "role": "scale", "scale_of": "P", "block": blk, "shape": [sg, M], "dtype": "e8m0"},
        ]
    return inputs


def build_attention_mx(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A fused MX flash-attention capsule (op == attention_mx): O = softmax(Q @ K^T / sqrt(H) [+ soft-cap])
    @ V, with the two matmuls on the block-scaled MX PE (E8M0 per 32-element K group, bf16 accumulate) and
    a bf16 row-softmax between them. The interface is a fused merlin_iface program (attention_qk -> softmax
    -> attention_pv) in mxfp8; the golden is composed by the driver from mx_ref (QK & PV) + a numpy bf16
    softmax. Dims: M queries, H head dim (%32), Skv keys (%32), Dv value dim (%16)."""
    from merlin.development.targetgen.corpus.binding import CorpusBinding, _numeric_policy, is_block_scaled

    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    H = entry.get("H", entry.get("head_dim", 2 * D))
    Skv = entry.get("Skv", entry.get("keys", 2 * D))
    Dv = entry.get("Dv", D)
    q, k, v, out = entry.get("q", "Q"), entry.get("k", "K"), entry.get("v", "V"), entry.get("out", "Y0")
    idt = binding.cap_dtype(binding.operand_dtype)  # mxfp8
    odt = binding.cap_dtype(binding.accum_dtype)  # bf16
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    import math

    scale = float(entry.get("scale", 1.0 / math.sqrt(H)))
    softcap = entry.get("softcap")
    attrs: dict[str, Any] = {
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "scale": scale,
        "block_scale": "e8m0",
        "output_dtype": odt,
    }
    if softcap is not None:
        attrs["softcap"] = float(softcap)
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": _attention_mx_inputs(q, k, v, M, H, Skv, Dv, idt, binding),
        "operation": {"op": "attention_mx", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        # two matmuls (QK, PV) drive the MX PE; the softmax rides the SIMT/vector lanes (no matmul classes).
        "expected": {
            "instruction_classes": binding.classes_for(op="matmul", output_dtype=odt),
            "modes": entry.get("modes", {}),
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    sc_attr = f", scale = {scale} : f32"
    cap_attr = f", softcap = {float(softcap)} : f32" if softcap is not None else ""
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{q} = merlin_iface.tensor {{name = "{q}", role = "input"}} : tensor<{M}x{H}x{midt}>',
        f'  %{k} = merlin_iface.tensor {{name = "{k}", role = "input"}} : tensor<{Skv}x{H}x{midt}>',
        f'  %{v} = merlin_iface.tensor {{name = "{v}", role = "input"}} : tensor<{Skv}x{Dv}x{midt}>',
    ]
    _blk = binding.scale_block if is_block_scaled(binding.operand_dtype) else None
    if _blk and H % _blk == 0 and Skv % _blk == 0:
        _hg, _sg = H // _blk, Skv // _blk
        L += [
            f'  %{q}_scale = merlin_iface.tensor {{name = "{q}_scale", role = "scale", '
            f'scale_of = "{q}", block = {_blk} : i64}} : tensor<{_hg}x{M}xi8>',
            f'  %{k}_scale = merlin_iface.tensor {{name = "{k}_scale", role = "scale", '
            f'scale_of = "{k}", block = {_blk} : i64}} : tensor<{_hg}x{Skv}xi8>',
            f'  %{v}_scale = merlin_iface.tensor {{name = "{v}_scale", role = "scale", '
            f'scale_of = "{v}", block = {_blk} : i64}} : tensor<{_sg}x{Dv}xi8>',
            f'  %P_scale = merlin_iface.tensor {{name = "P_scale", role = "scale", '
            f'scale_of = "P", block = {_blk} : i64}} : tensor<{_sg}x{M}xi8>',
        ]
    L += [
        f'  %S = merlin_iface.attention_qk %{q}, %{k} {{name = "S", block_scale = "e8m0", '
        f'output_dtype = "{odt}"}} : (tensor<{M}x{H}x{midt}>, tensor<{Skv}x{H}x{midt}>) '
        f"-> tensor<{M}x{Skv}x{modt}>",
        f'  %P = merlin_iface.softmax %S {{name = "P", axis = 1 : i64{sc_attr}{cap_attr}}} '
        f": (tensor<{M}x{Skv}x{modt}>) -> tensor<{M}x{Skv}x{modt}>",
        f'  %{out} = merlin_iface.attention_pv %P, %{v} {{name = "{out}", block_scale = "e8m0", '
        f'output_dtype = "{odt}"}} : (tensor<{M}x{Skv}x{modt}>, tensor<{Skv}x{Dv}x{midt}>) '
        f"-> tensor<{M}x{Dv}x{modt}>",
        "}",
    ]
    return cap, "\n".join(L) + "\n"


def build_rmsnorm_qkv(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """Fused pre-norm QKV projection (op == rmsnorm_qkv): H = rmsnorm(X, gamma); Y = H @ Wqkv. A SIMT
    op (rmsnorm on the vector lanes feeding a matmul); golden is ordinary IEEE float (driver-composed).
    Interface chains the two EXISTING merlin_iface ops (rmsnorm -> matmul)."""
    from merlin.development.targetgen.corpus.binding import CorpusBinding, _numeric_policy

    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    K = entry.get("K", entry.get("K_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    x, gamma, weight, out = (
        entry.get("src", "X"),
        entry.get("gamma", "G"),
        entry.get("weight", "Wqkv"),
        entry.get("out", "Y0"),
    )
    eps = entry.get("eps", 1.0 / 65536.0)
    idt, odt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    attrs = {
        "src": x,
        "gamma": gamma,
        "weight": weight,
        "out": out,
        "eps": eps,
        "semantic": "rmsnorm_qkv",
        "output_dtype": odt,
    }
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [
            {"name": x, "role": "input", "shape": [M, K], "dtype": idt},
            {"name": gamma, "role": "weight", "shape": [1, K], "dtype": idt},
            {"name": weight, "role": "weight", "shape": [K, N], "dtype": idt},
        ],
        "operation": {"op": "rmsnorm_qkv", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": {
            "instruction_classes": binding.classes_for(op="matmul", output_dtype=odt),
            "modes": entry.get("modes", {"rmsnorm": True}),
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{x} = merlin_iface.tensor {{name = "{x}", role = "input"}} : tensor<{M}x{K}x{midt}>',
        f'  %{gamma} = merlin_iface.tensor {{name = "{gamma}", role = "weight"}} : tensor<1x{K}x{midt}>',
        f'  %{weight} = merlin_iface.tensor {{name = "{weight}", role = "weight"}} : tensor<{K}x{N}x{midt}>',
        f'  %H = merlin_iface.rmsnorm %{x}, %{gamma} {{name = "H", eps = {eps:.9e} : f64, '
        f'output_dtype = "{odt}"}} : (tensor<{M}x{K}x{midt}>, tensor<1x{K}x{midt}>) -> tensor<{M}x{K}x{modt}>',
        f'  %{weight}_res = merlin_iface.resident_pack %{weight} {{layout = "packed_rhs"}} '
        f": (tensor<{K}x{N}x{midt}>) -> !merlin_iface.resident",
        f"  %acc0 = merlin_iface.matmul %H, %{weight}_res "
        f": (tensor<{M}x{K}x{modt}>, !merlin_iface.resident) -> !merlin_iface.acc<{modt}>",
        f'  %{out} = merlin_iface.commit %acc0 {{name = "{out}", epilogue = [], output_dtype = "{odt}"}} '
        f": (!merlin_iface.acc<{modt}>) -> tensor<{M}x{N}x{modt}>",
        f"  merlin_iface.evict %{weight}_res : (!merlin_iface.resident) -> ()",
        "}",
    ]
    return cap, "\n".join(L) + "\n"

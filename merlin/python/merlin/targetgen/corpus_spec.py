"""Target-agnostic capsule-corpus builder.

ONE builder turns an abstract capsule *entry* (op + shapes-in-tiles + epilogue — the target-agnostic test
definition, declared per target in ``capsules/<target>/corpus_profile.yaml``) plus a *binding* DERIVED from
the target's descriptor (dtypes, tile dim, compare policy, instruction classes, oracle tiers) into a
concrete capsule dict + interface MLIR. It replaces the two forked generators (``generate_corpus.py``
gemmini/integer + ``generate_atlas_corpus.py`` atlas/float): the LOGIC here is shared and carries no target
name or dtype literal in its control flow; the per-target DATA (numeric datapath, which capsules, dtypes)
lives in the descriptor + the profile.

Golden VALUES are not computed here (the integer engine lives in :mod:`capsule_golden`; the float engine
needs the external ``specir`` refmodel, available only at generation time) — the driver
``merlin/contract/capsules/generate_corpus.py`` computes them and writes ``golden.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# canonical dtype token -> (capsule.yaml spelling, MLIR spelling, byte width, is_integer). Keyed on the
# dtype, never on a target — a target selects its dtypes via its compute-unit contract.
_DTYPE = {
    "int8": ("i8", "i8", 1, True),
    "i8": ("i8", "i8", 1, True),
    "int32": ("i32", "i32", 4, True),
    "i32": ("i32", "i32", 4, True),
    "fp8_e4m3": ("fp8_e4m3", "f8E4M3FN", 1, False),
    "fp8_e5m2": ("fp8_e5m2", "f8E5M2", 1, False),
    # MX microscaling operand widths (block-scaled). ``mxfp*`` are the manifest tokens; ``fp*_e*m*`` the
    # canonical layout spellings — same code<->value map, so both resolve to the OCP MX MLIR type names.
    "mxfp8": ("mxfp8", "f8E4M3FN", 1, False),
    "mxfp6": ("mxfp6", "f6E3M2FN", 1, False),
    "mxfp4": ("mxfp4", "f4E2M1FN", 1, False),
    "fp6_e3m2": ("fp6_e3m2", "f6E3M2FN", 1, False),
    "fp4_e2m1": ("fp4_e2m1", "f4E2M1FN", 1, False),
    "e8m0": ("e8m0", "f8E8M0FNU", 1, False),
    "fp16": ("fp16", "f16", 2, False),
    "f16": ("fp16", "f16", 2, False),
    "bf16": ("bf16", "bf16", 2, False),
    "f32": ("f32", "f32", 4, False),
}


def dtype_info(token: str) -> tuple[str, str, int, bool]:
    """(capsule spelling, MLIR spelling, byte width, is_integer) for a canonical dtype token."""
    if token not in _DTYPE:
        raise KeyError(f"unknown dtype token {token!r} (extend corpus_spec._DTYPE)")
    return _DTYPE[token]


@dataclass(frozen=True)
class CorpusBinding:
    """Per-target axes DERIVED from the descriptor (nothing hand-set per target)."""
    target: str
    tile_dim: int
    operand_dtype: str          # canonical token, e.g. "int8" / "fp8_e4m3"
    accum_dtype: str            # canonical token, e.g. "i32" / "bf16"
    integer: bool               # numeric regime (drives compare policy + golden engine)
    tiers: list[str]            # required_oracle_tiers for this target
    compare: str                # "exact_int" | "tolerance_float"
    atol: float | None = None
    rtol: float | None = None
    scaling: str | None = None  # compute-unit SCALE_KIND (block_e8m0 -> the MX numeric regime) or None
    requant_output_dtype: str | None = None   # narrow output an acc_scale epilogue requants to (e.g. i8)
    # instruction-class source: a callable (op, output_dtype, epilogue, movement) -> [class,...]
    classes_for: Callable[..., list[str]] = field(default=lambda **_: [])

    def cap_dtype(self, token: str) -> str:
        return dtype_info(token)[0]

    def mlir_dtype(self, token: str) -> str:
        return dtype_info(token)[1]


def _tile_dim(target: str, contract: dict) -> int:
    """Systolic tile/mesh dim: ``capabilities.mesh.rows`` if the manifest carries it, else the CIRCT
    ``arrays[mesh].rows`` fact, else 16. No target literal — both sources are keyed on ``target``."""
    mesh = ((contract.get("capabilities") or {}).get("mesh") or {})
    if mesh.get("rows"):
        return int(mesh["rows"])
    try:
        from merlin.targetgen.rtl.facts import load_facts
        arrays = (load_facts(target).get("facts") or {}).get("arrays") or []
        m = next((a for a in arrays if a.get("name") == "mesh"), {})
        if m.get("rows"):
            return int(m["rows"])
    except Exception:  # noqa: BLE001 — no facts for this target -> fall back to the mesh default
        pass
    return 16


def _accum_dtype(contract: dict, operand: str) -> str:
    """Accumulate dtype from the compute unit's declared ``accumulate`` matrix, else the widening default
    for an integer operand (i32) — a family property (``widening_integer_accumulate``), not a target one."""
    cu = (contract.get("compute_units") or [{}])[0]
    for acc in (cu.get("accumulate") or []):
        if acc.get("acc"):
            return str(acc["acc"])
    return "i32" if dtype_info(operand)[3] else "f32"


def _classes_source(te, contract: dict) -> Callable[..., list[str]]:
    """The instruction-class deriver for a matmul-family op. Two regimes, chosen by what the target ships:
    a self-hosted-ISA target (an ``isa_definition.py`` is present) derives its classes from the taxonomy;
    a RoCC/command target derives the RoCC semantic classes from its ``encoding`` map. Never hardcoded."""
    from merlin.targetgen import isa_taxonomy as IT
    try:
        tax = IT.derive_isa_taxonomy(te)
    except Exception:  # noqa: BLE001
        tax = {}
    if tax.get("by_class"):
        def _from_taxonomy(*, op="matmul", output_dtype=None, epilogue=(), movement=False):
            return IT.required_classes_for_op(tax, op=op, output_dtype=output_dtype,
                                              epilogue=tuple(epilogue), movement=movement)
        return _from_taxonomy
    # RoCC command target: the matmul-relevant semantic classes for a weight-stationary tile, with the
    # single CONFIG class expanded to its declared config subtypes, in RoCC issue order. `pool` = what
    # this target's encoding actually defines (so a target missing a class simply drops it); the order is
    # the RoCC weight-stationary matmul sequence, filtered by `pool`.
    enc = contract.get("encoding") or {}
    sem_vals = set((enc.get("semantic_class") or {}).values())
    sub_vals = set((enc.get("config_subtype") or {}).values())
    pool = (sem_vals - {"CONFIG"}) | sub_vals
    order = ["FLUSH", "CONFIG_EX", "CONFIG_LD", "MVIN", "CONFIG_ST", "PRELOAD", "COMPUTE_PRELOADED", "MVOUT"]
    classes = [c for c in order if c in pool]

    def _from_encoding(*, op="matmul", output_dtype=None, epilogue=(), movement=False):
        return list(classes)
    return _from_encoding


def derive_binding(te, datapath: dict) -> CorpusBinding:
    """Derive the per-target binding from the descriptor + the profile's ``datapath`` block (compare +
    tolerances + optional requant-output dtype — the numeric contract the manifest does not yet carry)."""
    from merlin.targetgen.target_experiment import load_capability_manifest
    from merlin.targetgen import capsule_runner as CR
    m = load_capability_manifest(te.target)
    c = m.contract
    cu = (c.get("compute_units") or [{}])[0]
    # The profile may pin the DEFAULT operand/accumulate dtypes (a target with several compute units — e.g.
    # radiance's simt_cluster + contained mx_pe — needs the profile to say which regime a capsule set drives);
    # both fall back to the primary compute unit's declared datapath, never a target literal.
    operand = datapath.get("operand_dtype") or (cu.get("dtypes") or ["int8"])[0]
    accum = datapath.get("accum_dtype") or _accum_dtype(c, operand)
    integer = dtype_info(accum)[3]
    scaling = datapath.get("scaling") or cu.get("scaling")
    tiers = datapath.get("required_oracle_tiers") \
        or sorted((CR.oracle_adapters(te.target, te.sim_via) or {}).keys())
    return CorpusBinding(
        target=te.target,
        tile_dim=_tile_dim(te.target, c),
        operand_dtype=operand,
        accum_dtype=accum,
        integer=integer,
        tiers=list(tiers),
        compare=datapath.get("compare", "exact_int" if integer else "tolerance_float"),
        atol=datapath.get("atol"),
        rtol=datapath.get("rtol"),
        scaling=(scaling if scaling and scaling != "none" else None),
        requant_output_dtype=datapath.get("requant_output_dtype"),
        classes_for=_classes_source(te, c),
    )


# ---------------------------------------------------------------------------------------------------
# capsule builders — one per op shape; each takes the abstract entry + binding and returns the capsule
# dict + interface MLIR. Shapes are given in TILE units in the profile and scaled by binding.tile_dim.
# ---------------------------------------------------------------------------------------------------

def _numeric_policy(binding: CorpusBinding, output_dtype: str, acc_scale: float | None) -> dict:
    np_: dict[str, Any] = {"compare": binding.compare, "dtype": binding.cap_dtype(output_dtype)}
    if not binding.integer:
        if binding.atol is not None:
            np_["atol"] = binding.atol
        if binding.rtol is not None:
            np_["rtol"] = binding.rtol
    if acc_scale is not None:
        np_["acc_scale"] = acc_scale
    return np_


def _resolve_output_dtype(binding: CorpusBinding, epilogue: list[str]) -> str:
    """Output dtype: the accumulate dtype, unless an acc_scale epilogue requants to a narrow output (the
    target declares that narrow dtype in its datapath as ``requant_output_dtype``, e.g. i8 for gemmini)."""
    if "acc_scale" in epilogue and binding.requant_output_dtype:
        return binding.requant_output_dtype
    return binding.accum_dtype


def _default_modes(binding: CorpusBinding, output_dtype: str, epilogue: list[str]) -> dict:
    """The mode flags a capsule gets when its profile entry declares none: relu/acc_scale from the
    epilogue (+ the integer-narrow-output flag for an integer target). An entry MAY instead declare
    ``modes`` verbatim (e.g. ``{k_accumulate: true}``), which is used as-is."""
    modes = {"relu": "relu" in epilogue, "acc_scale": "acc_scale" in epilogue}
    if binding.integer:
        modes["i8"] = binding.cap_dtype(output_dtype) == "i8"
    return modes


def build_matmul(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A single weight-stationary matmul/linear capsule (op ∈ {matmul, linear})."""
    D = binding.tile_dim
    M = entry.get("M_tiles", 1) * D if "M" not in entry else entry["M"]
    K = entry.get("K_tiles", 1) * D if "K" not in entry else entry["K"]
    N = entry.get("N_tiles", 1) * D if "N" not in entry else entry["N"]
    lhs, weight, out = entry.get("lhs", "A0"), entry.get("weight", "W"), entry.get("out", "Y0")
    epilogue = list(entry.get("epilogue", []))
    acc_scale = entry.get("acc_scale")
    output_dtype = _resolve_output_dtype(binding, epilogue)
    op = entry.get("op", "matmul")
    odt = binding.cap_dtype(output_dtype)
    idt = binding.cap_dtype(binding.operand_dtype)
    attrs: dict[str, Any] = {"lhs": lhs, "weight": weight, "out": out,
                             "epilogue": epilogue, "output_dtype": odt}
    if acc_scale is not None:
        attrs["acc_scale"] = acc_scale
    if entry.get("semantic"):
        attrs["semantic"] = entry["semantic"]
    expected = {"instruction_classes": binding.classes_for(op=op, output_dtype=odt,
                                                           epilogue=epilogue, movement=False),
                "modes": dict(entry["modes"]) if "modes" in entry
                else _default_modes(binding, output_dtype, epilogue)}
    if entry.get("forbidden"):
        expected["forbidden_classes"] = entry["forbidden"]
    cap = {
        "name": entry["name"], "kind": entry["kind"],
        "source_role": entry["source_role"], "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"), "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": weight, "role": "weight", "shape": [K, N], "dtype": idt},
                   {"name": lhs, "role": "input", "shape": [M, K], "dtype": idt}],
        "operation": {"op": op, "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, output_dtype, acc_scale),
        "expected": expected, "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional", "firesim": "optional",
    }
    from merlin.targetgen import model_slice_export as MSE
    mlir = MSE.emit_interface_mlir(
        lhs=lhs, weight=weight, out=out, M=M, K=K, N=N, epilogue=epilogue, output_dtype=odt,
        acc_scale=acc_scale, comment=entry.get("comment", ""),
        target=binding.target, operand_dtype=binding.mlir_dtype(binding.operand_dtype),
        acc_dtype=binding.mlir_dtype(binding.accum_dtype))
    return cap, mlir


def _iface_prelude(target: str, comment: str) -> list[str]:
    head = ('module attributes {merlin_iface.version = "0.1", '
            f'merlin_iface.target = "{target}", merlin_iface.abi_version = "0.1"}} {{')
    return ([f"// {comment}", head] if comment else [head])


def build_resident_reuse(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """One resident weight reused across two matmuls (op == resident_reuse)."""
    D = binding.tile_dim
    K = entry.get("K_tiles", 1) * D
    N = entry.get("N_tiles", 1) * D
    weight = entry.get("weight", "W")
    idt, adt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)
    midt, madt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    inputs = [{"name": weight, "role": "weight", "shape": [K, N], "dtype": idt}]
    matmuls, mlir_mm = [], []
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L.append(f'  %{weight} = merlin_iface.tensor {{name = "{weight}", role = "weight"}} : tensor<{K}x{N}x{midt}>')
    for idx, mm in enumerate(entry["matmuls"]):
        an, oname = mm["lhs"], mm["out"]
        M = mm.get("M_tiles", 1) * D
        epi = list(mm.get("epilogue", []))
        inputs.append({"name": an, "role": "input", "shape": [M, K], "dtype": idt})
        matmuls.append({"lhs": an, "out": oname, "epilogue": epi, "output_dtype": adt})
        L.append(f'  %{an} = merlin_iface.tensor {{name = "{an}", role = "input"}} : tensor<{M}x{K}x{midt}>')
    L.append(f'  %{weight}_res = merlin_iface.resident_pack %{weight} {{layout = "packed_rhs"}} '
             f': (tensor<{K}x{N}x{midt}>) -> !merlin_iface.resident')
    for idx, mm in enumerate(entry["matmuls"]):
        an, oname = mm["lhs"], mm["out"]
        M = mm.get("M_tiles", 1) * D
        epi = ", ".join(f'"{e}"' for e in mm.get("epilogue", []))
        L.append(f'  %acc{idx} = merlin_iface.matmul %{an}, %{weight}_res '
                 f': (tensor<{M}x{K}x{midt}>, !merlin_iface.resident) -> !merlin_iface.acc<{madt}>')
        L.append(f'  %{oname} = merlin_iface.commit %acc{idx} {{name = "{oname}", epilogue = [{epi}], '
                 f'output_dtype = "{adt}"}} : (!merlin_iface.acc<{madt}>) -> tensor<{M}x{N}x{adt}>')
    L.append(f'  merlin_iface.evict %{weight}_res : (!merlin_iface.resident) -> ()')
    L.append("}")
    attrs = {"weight": weight, "matmuls": matmuls, "semantic": entry.get("semantic", "resident_reuse")}
    cap = {
        "name": entry["name"], "kind": entry["kind"], "source_role": entry["source_role"],
        "source_reference": entry["source_reference"], "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir", "inputs": inputs,
        "operation": {"op": "resident_reuse", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": {"instruction_classes": binding.classes_for(op="matmul", output_dtype=adt),
                     "modes": {"resident_reuse": True}},
        "required_oracle_tiers": list(binding.tiers), "vcs": "optional", "firesim": "optional",
    }
    return cap, "\n".join(L) + "\n"


def build_movement(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A load->store dequant movement capsule (op == movement): operand dtype in, accumulate dtype out."""
    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    src, out = entry.get("src", "X"), entry.get("out", "Y0")
    idt, odt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    attrs = {"out": out, "src": src, "semantic": entry.get("semantic", "mvin_mvout"), "output_dtype": odt}
    expected = {"instruction_classes": binding.classes_for(op="movement", movement=True),
                "modes": {"movement": True}}
    if entry.get("forbidden"):
        expected["forbidden_classes"] = entry["forbidden"]
    cap = {
        "name": entry["name"], "kind": entry["kind"], "source_role": entry["source_role"],
        "source_reference": entry["source_reference"], "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": src, "role": "input", "shape": [M, N], "dtype": idt}],
        "operation": {"op": "movement", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": expected, "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional", "firesim": "optional",
    }
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
         f'  %{src} = merlin_iface.tensor {{name = "{src}", role = "input"}} : tensor<{M}x{N}x{midt}>',
         f'  %{out} = merlin_iface.movement %{src} {{name = "{out}", semantic = "mvin_mvout", '
         f'output_dtype = "{odt}"}} : (tensor<{M}x{N}x{midt}>) -> tensor<{M}x{N}x{modt}>', "}"]
    return cap, "\n".join(L) + "\n"


def build_attention_qk(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """Q @ K^T attention scores (op == attention_qk): the device does the transpose internally."""
    D = binding.tile_dim
    M = entry.get("M_tiles", 1) * D
    Kd = entry.get("K_tiles", 1) * D
    q, k, out = entry.get("q", "Q"), entry.get("k", "K"), entry.get("out", "Y0")
    idt, odt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(binding.accum_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(binding.accum_dtype)
    attrs = {"q": q, "k": k, "out": out, "epilogue": [], "output_dtype": odt}
    cap = {
        "name": entry["name"], "kind": entry["kind"], "source_role": entry["source_role"],
        "source_reference": entry["source_reference"], "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": q, "role": "input", "shape": [M, Kd], "dtype": idt},
                   {"name": k, "role": "input", "shape": [M, Kd], "dtype": idt}],
        "operation": {"op": "attention_qk", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": {"instruction_classes": binding.classes_for(op="matmul", output_dtype=odt),
                     "modes": entry.get("modes", {})},
        "required_oracle_tiers": list(binding.tiers), "vcs": "optional", "firesim": "optional",
    }
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
         f'  %{q} = merlin_iface.tensor {{name = "{q}", role = "input"}} : tensor<{M}x{Kd}x{midt}>',
         f'  %{k} = merlin_iface.tensor {{name = "{k}", role = "input"}} : tensor<{M}x{Kd}x{midt}>',
         f'  %{out} = merlin_iface.attention_qk %{q}, %{k} {{name = "{out}", output_dtype = "{odt}"}} '
         f': (tensor<{M}x{Kd}x{midt}>, tensor<{M}x{Kd}x{midt}>) -> tensor<{M}x{M}x{modt}>', "}"]
    return cap, "\n".join(L) + "\n"


def build_rmsnorm(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A row RMSNorm capsule (op == rmsnorm): X[M,K] * rsqrt(mean(X^2)+eps) * gamma[1,K] -> [M,K]. A SIMT
    (elementwise/reduction) op — its golden is ordinary IEEE float, computed by the driver."""
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
        "name": entry["name"], "kind": entry["kind"], "source_role": entry["source_role"],
        "source_reference": entry["source_reference"], "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": x, "role": "input", "shape": [M, K], "dtype": idt},
                   {"name": gamma, "role": "weight", "shape": [1, K], "dtype": idt}],
        "operation": {"op": "rmsnorm", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": {"instruction_classes": [], "modes": {"rmsnorm": True}},
        "required_oracle_tiers": list(binding.tiers), "vcs": "optional", "firesim": "optional",
    }
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
         f'  %{x} = merlin_iface.tensor {{name = "{x}", role = "input"}} : tensor<{M}x{K}x{midt}>',
         f'  %{gamma} = merlin_iface.tensor {{name = "{gamma}", role = "weight"}} : tensor<1x{K}x{midt}>',
         f'  %{out} = merlin_iface.rmsnorm %{x}, %{gamma} {{name = "{out}", eps = {eps:.9e} : f64, '
         f'output_dtype = "{odt}"}} : (tensor<{M}x{K}x{midt}>, tensor<1x{K}x{midt}>) -> tensor<{M}x{K}x{modt}>',
         "}"]
    return cap, "\n".join(L) + "\n"


def build_conv2d(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """An im2col conv2d capsule (op == conv2d): NHWC IFM + pre-im2col'd weight [KH*KW*Ci, Cout] -> a resident
    matmul over conv windows, output [Ho*Wo, Cout]. Reuses the runtime's canonical conv geometry so the golden
    (capsule_golden conv2d branch) and the harness agree. Native operand dtype (e.g. gemmini int8)."""
    from merlin.runtime.commandbuffer import conv_out_dims
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
    output_dtype = _resolve_output_dtype(binding, epilogue)
    idt, odt = binding.cap_dtype(binding.operand_dtype), binding.cap_dtype(output_dtype)
    midt, modt = binding.mlir_dtype(binding.operand_dtype), binding.mlir_dtype(output_dtype)
    attrs = {"ifm": ifm, "weight": weight, "out": out, "ci": ci, "kh": kh, "kw": kw,
             "stride": stride, "padding": padding, "dilation": dilation, "layout": "nhwc",
             "epilogue": epilogue, "output_dtype": odt, "semantic": "conv2d_im2col"}
    cap = {
        "name": entry["name"], "kind": entry["kind"], "source_role": entry["source_role"],
        "source_reference": entry["source_reference"], "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": weight, "role": "weight", "shape": [Kdim, cout], "dtype": idt},
                   {"name": ifm, "role": "input", "shape": [1, H, W, ci], "dtype": idt}],
        "operation": {"op": "conv2d", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, output_dtype, entry.get("acc_scale")),
        "expected": {"instruction_classes": binding.classes_for(op="conv2d", output_dtype=odt,
                                                                epilogue=epilogue, movement=False),
                     "modes": dict(entry["modes"]) if "modes" in entry
                     else {"conv2d": True, "k_accumulate": True}},
        "required_oracle_tiers": list(binding.tiers), "vcs": "optional", "firesim": "optional",
    }
    epi = ", ".join(f'"{e}"' for e in epilogue)
    L = _iface_prelude(binding.target, entry.get("comment", ""))
    L += [
        f'  %{ifm} = merlin_iface.tensor {{name = "{ifm}", role = "input"}} : tensor<1x{H}x{W}x{ci}x{midt}>',
        f'  %{weight} = merlin_iface.tensor {{name = "{weight}", role = "weight"}} : tensor<{Kdim}x{cout}x{midt}>',
        f'  %{weight}_res = merlin_iface.resident_pack %{weight} {{layout = "packed_conv_rhs"}} '
        f': (tensor<{Kdim}x{cout}x{midt}>) -> !merlin_iface.resident',
        f'  %{out} = merlin_iface.conv2d %{ifm}, %{weight}_res {{kernel = [{kh}, {kw}, {ci}, {cout}], '
        f'stride = [{stride[0]}, {stride[1]}], padding = [{padding[0]}, {padding[1]}, {padding[2]}, {padding[3]}], '
        f'dilation = [{dilation[0]}, {dilation[1]}], name = "{out}", epilogue = [{epi}], '
        f'output_dtype = "{odt}", layout = "nhwc"}} '
        f': (tensor<1x{H}x{W}x{ci}x{midt}>, !merlin_iface.resident) -> tensor<{Ho * Wo}x{cout}x{modt}>',
        f'  merlin_iface.evict %{weight}_res : (!merlin_iface.resident) -> ()', "}"]
    return cap, "\n".join(L) + "\n"


# op -> builder, for the driver to dispatch on the entry's declared op.
BUILDERS: dict[str, Callable[[dict, CorpusBinding], tuple[dict, str]]] = {
    "matmul": build_matmul, "linear": build_matmul,
    "resident_reuse": build_resident_reuse, "movement": build_movement,
    "attention_qk": build_attention_qk, "rmsnorm": build_rmsnorm, "conv2d": build_conv2d,
}


def build(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """Dispatch an abstract capsule entry to its op builder -> (capsule dict, interface MLIR)."""
    op = entry.get("op", "matmul")
    if op not in BUILDERS:
        raise ValueError(f"no corpus builder for op {op!r} (have {sorted(BUILDERS)})")
    return BUILDERS[op](entry, binding)

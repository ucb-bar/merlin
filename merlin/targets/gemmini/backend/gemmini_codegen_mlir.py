"""Merlin-FAITHFUL Gemmini codegen: the native RoCC sequence emitted as MLIR
(llvm-dialect `llvm.inline_asm` `.insn` ops), lowered by merlin's compiler — NOT C.

Generalizes the C0 proof to resident matmul groups, including relu/acc-scale readout and the target's
native int8 max-pooling store path. CONV2D uses the same backend after a shared im2col materialization.
Every Gemmini instruction is a custom-3 (0x7b) `.insn` op; target-specific encodings and packed pooling
fields come from the capability manifest and RTL facts. DRAM tile addresses are compile-time
pointer-offset arithmetic on the function's pointer args; the tile sequence is unrolled (shapes static).

requant (C2/C3) is intentionally rejected — Gemmini's acc_scale is not bit-exact with merlin's
integer requant (see docs/gemmini_requant_reconciliation.md). The kernel is pure llvm-dialect
MLIR lowered via `lower_to_llvm_ir` -> object; a thin C harness embeds data + calls it + prints.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import SimpleNamespace

from .gemmini_codegen import _ceil_dim, _pad_rowmajor, _parse, CodegenError   # sibling — moves together
from merlin.runtime.commandbuffer import (BIAS_STAGES, bias_tensor_name, conv_out_dims,
                                          materialize_inputs, pool_params)
from merlin.runtime.tensor import pool_out_dims

GARBAGE = 0xFFFFFFFF                  # universal, not target-specific — no derivation needed

# ISA/ABI encoding constants — DERIVED from the single source of truth (gemmini's capability manifest +
# the RTL fact bundle), not hand-copied. Byte-parity with the former literals is pinned by
# test_codegen_constants_single_source; the emitted .insn is proven byte-identical by
# test_codegen_emit_byte_identical. This retires the emitter's copy of the triplicated constants.
#
# Resolved on FIRST USE, not at import. Importing this module used to load a capability manifest and an
# RTL fact bundle as a side effect, which made a target's data a prerequisite for merely NAMING this
# module: the generic contract-compile path imports `_harness_c` from here, so on a checkout without
# those facts it failed at import rather than at the point of use, and the traceback pointed at an
# import line rather than at the thing that actually needed the facts.
@cache
def _isa() -> SimpleNamespace:
    from merlin.targetgen.target_experiment import load_capability_manifest
    from merlin.targetgen.address_space import derive_address_space
    from merlin.targetgen.rtl.facts import load_facts
    m = load_capability_manifest("gemmini")
    enc, rb = m.encoding, m.encoding["readout_bits"]
    code_of = {cls: code for code, cls in enc["semantic_class"].items()}
    facts_rec = load_facts("gemmini")
    facts = facts_rec["facts"]
    fd = next(i for i in facts["interfaces"] if i.get("name") == "funct_decode_table")
    layouts = next((i for i in facts["interfaces"]
                    if i.get("name") == "register_bundle_layouts"), {})
    build_features = next((i for i in facts["interfaces"]
                           if i.get("name") == "elaborated_rtl_features"), {})
    # DIM (systolic mesh dimension) is a CIRCT-extracted FACT, not a hand-declared manifest field.
    mesh = next((a for a in facts.get("arrays", []) if a.get("name") == "mesh"), {})
    # On-chip operand-store depth (scratchpad rows) is a CIRCT-extracted memory FACT — it bounds how
    # much of an operand can be kept resident (see the capacity-fit residency in emit_kernel_mlir).
    # Fail-closed: absent -> None (no capacity-fit residency, the per-tile schedule stands).
    # A memory fact's ``depth`` is per bank. LocalAddr indexes the banked space as one flat row range,
    # so code generation must use the address-space derivation's TOTAL rows (bytes / RTL-derived row
    # width), not the per-bank depth. Using depth rejected valid retained pool planes and made operand
    # residency four times more conservative on the pinned 4-bank scratchpad.
    address_space = derive_address_space("gemmini", facts=facts_rec)
    sp_store = address_space.store("scratchpad")
    acc_store = address_space.store("accumulator")
    sp_rows = sp_store.total_rows if sp_store is not None else None
    acc_rows = acc_store.total_rows if acc_store is not None else None
    # The CONTAINER the accumulator holds, derived from the same CIRCT memory fact that gives its row
    # count (`array cols 16 x i32 (32 bits)`). A fused bias is moved INTO the accumulator, so its DRAM
    # element width and C type are this dtype's — never the operand dtype, and never a literal 4. Absent
    # -> None, and the bias path refuses rather than guessing a width for a DMA.
    acc_dtype = acc_store.element_dtype if acc_store is not None else None
    acc_bits = acc_store.element_bits if acc_store is not None else None
    config_code_of = {name: int(code) for code, name in enc.get("config_subtype", {}).items()}
    # The ISA bundle proves pooling is encodable; only the configuration which produced the RTL says
    # whether StoreController built it. Require a literal derived True. A human capability declaration,
    # an absent fact, or an UNKNOWN extraction can no longer enable code generation.
    max_pool_supported = ((build_features.get("features") or {}).get("max_pool")
                          if build_features.get("status") == "derived" else None)
    if not isinstance(max_pool_supported, bool):
        max_pool_supported = None
    pool_capable = max_pool_supported is True
    mesh_rows, mesh_cols = mesh.get("rows"), mesh.get("cols")
    if not isinstance(mesh_rows, int) or not isinstance(mesh_cols, int) or mesh_rows <= 0 or mesh_cols <= 0:
        raise CodegenError("CIRCT facts do not contain a positive mesh row/column geometry; refusing "
                           "to substitute a conventional tile dimension")
    if mesh_rows != mesh_cols:
        raise CodegenError(f"this emitter requires a square mesh, but CIRCT derived "
                           f"{mesh_rows}x{mesh_cols}; a single DIM cannot represent it")
    isa = SimpleNamespace(
        DIM=mesh_rows, ADDR_LEN=enc["addr_len"],
        F1=rb["f1"],                        # float 1.0 bits
        C_ACC=rb["c_acc"],                  # 0xA0000000 full-i32 accumulator readout base
        ACC_ACCUM=rb["acc_accum"],          # 0x40000000 accumulate-onto for K-tiles after the first
        ACC_I8=rb["acc_i8"],                # 0x80000000 acc addr WITHOUT full_C: scaled i8 readout
        FULL_C_BIT=rb["full_c_bit"],        # 0x20000000 selects the full-i32 (vs scaled-i8) readout
        ACC_ELEM_DTYPE=acc_dtype,           # CIRCT-derived accumulator container ('i32'); None if absent
        ACC_ELEM_BITS=acc_bits,             # ...and its width in bits; None if absent
        K_CONFIG=code_of["CONFIG"], K_MVIN=code_of["MVIN"], K_MVOUT=code_of["MVOUT"],
        K_COMPUTE_PRELOADED=code_of["COMPUTE_PRELOADED"], K_PRELOAD=code_of["PRELOAD"],
        K_FLUSH=code_of["FLUSH"],
        CUSTOM_OPCODE=fd["custom_opcode"],  # RoCC custom-3 (0x7b)
        FUNCT3=fd["funct3"],                # 0x3
        SCRATCHPAD_ROWS=sp_rows,             # total banked operand-store rows; None if underived
        ACCUMULATOR_ROWS=acc_rows,           # total banked accumulator rows for a retained output plane
        CONFIG_ST_TYPE=config_code_of.get("CONFIG_ST"),
        CONFIG_ST_LAYOUT=(layouts.get("bundles") or {}).get("ConfigMvoutRs1"),
        MAX_POOL_SUPPORTED=max_pool_supported,
        POOL_CAPABLE=pool_capable,
    )
    # config_ex (WS, no activation, shift 0, identity scales, strides 1) and config_ld RS1 (block
    # stride defaults to DIM + pixel_repeats 1 — the RTL LoadController asserts block_stride>=rows).
    isa.CFG_EX_RS1 = (isa.F1 << 32) | (1 << 16) | (1 << 2)          # CONFIG_EX=0, dataflow WS=1
    isa.CFG_EX_RS2 = (1 << 48)
    isa.CFG_LD_RS1 = (isa.F1 << 32) | (isa.DIM << 16) | (1 << 8) | 1  # CONFIG_LD=1
    return isa


#: Names resolved lazily through :func:`_isa`. Exposed as module attributes via ``__getattr__`` (PEP
#: 562) so ``gemmini_codegen_mlir.DIM`` keeps working for callers and tests — the laziness is an
#: implementation change, not an API change.
_DERIVED = ("DIM", "ADDR_LEN", "F1", "C_ACC", "ACC_ACCUM", "ACC_I8", "K_CONFIG", "K_MVIN", "K_MVOUT",
            "K_COMPUTE_PRELOADED", "K_PRELOAD", "K_FLUSH", "CUSTOM_OPCODE", "FUNCT3",
            "CFG_EX_RS1", "CFG_EX_RS2", "CFG_LD_RS1", "SCRATCHPAD_ROWS", "ACCUMULATOR_ROWS",
            "CONFIG_ST_TYPE", "MAX_POOL_SUPPORTED", "POOL_CAPABLE",
            "FULL_C_BIT", "ACC_ELEM_BITS")
#: Derived names whose value is NOT an integer encoding (a packed-field layout, a dtype token).
_DERIVED_LAYOUTS = ("CONFIG_ST_LAYOUT", "ACC_ELEM_DTYPE")


def __getattr__(name: str):
    if name in _DERIVED or name in _DERIVED_LAYOUTS:
        return getattr(_isa(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _pack(addr: int, *, cols: int | None = None, rows: int | None = None) -> int:
    isa = _isa()
    cols = isa.DIM if cols is None else int(cols)
    rows = isa.DIM if rows is None else int(rows)
    return (rows << (isa.ADDR_LEN + 16)) | (cols << isa.ADDR_LEN) | (addr & 0xFFFFFFFF)


def _f32_bits(scale: float) -> int:
    import struct
    return struct.unpack("<I", struct.pack("<f", float(scale)))[0]


@dataclass(frozen=True)
class PoolSpec:
    in_rows: int
    in_cols: int
    size: int
    stride: int
    out_rows: int
    out_cols: int


@dataclass(frozen=True)
class Job:
    lhs: str
    out: str
    epilogue: tuple[str, ...]
    input_rows: int
    output_rows: int
    output_dtype: str
    scale: float
    pool: PoolSpec | None
    #: Name of the DRAM tensor a fused bias epilogue consumes, or None when the job has no bias stage.
    bias: str | None = None


_CONV2D_ATTRS = frozenset({
    "kernel", "stride", "padding", "dilation", "layout", "epilogue", "output_dtype",
    "acc_scale", "requant_shift", "pool_in_dims", "pool_size", "pool_stride", "pool_padding",
    "pool_pad_value", "semantic",
})


def _normalize_command_buffer(cb: dict) -> dict:
    """Lower a whole-op CONV2D to the same im2col/resident-matmul/commit path used by matmul capsules.

    The im2col recipe is executed by :func:`materialize_inputs`, whose gather is shared with the command
    simulator and capsule golden. This transformation changes only the target codegen representation; it
    does not invent a second convolution geometry or a host fallback.
    """
    if not any(c.get("opcode") == "CONV2D" for c in cb.get("commands", [])):
        return cb
    out = deepcopy(cb)
    tensors = out.setdefault("tensors", {})
    params = out.setdefault("params", {})
    recipes = params.setdefault("im2col_recipes", [])
    packed = {c.get("operands", {}).get("dst"): c.get("operands", {}).get("src")
              for c in out.get("commands", []) if c.get("opcode") == "RES_PACK"}
    commands: list[dict] = []
    conv_index = 0
    for command in out.get("commands", []):
        if command.get("opcode") != "CONV2D":
            commands.append(command)
            continue
        ops, attrs = command.get("operands", {}), command.get("attributes", {})
        unknown = sorted(set(attrs) - _CONV2D_ATTRS)
        if unknown:
            raise CodegenError(f"CONV2D does not implement attribute(s) {unknown}")
        ifm, rhs, dst = ops.get("ifm"), ops.get("weight"), ops.get("dst")
        if rhs not in packed:
            raise CodegenError(f"CONV2D weight {rhs!r} is not a resident-packed handle")
        spec = tensors.get(ifm, {})
        shape = spec.get("shape")
        if not isinstance(shape, list) or len(shape) != 4:
            raise CodegenError(f"CONV2D activation {ifm!r} must be rank-4 NHWC, got {shape!r}")
        batch, height, width, channels = (int(x) for x in shape)
        if batch != 1:
            raise CodegenError(f"CONV2D native v1 requires batch 1, got {batch}")
        kernel = attrs.get("kernel")
        if not isinstance(kernel, list) or len(kernel) != 4:
            raise CodegenError(f"CONV2D requires kernel = [kh, kw, ci, co], got {kernel!r}")
        kh, kw, ci, co = (int(x) for x in kernel)
        if channels != ci:
            raise CodegenError(f"CONV2D channel mismatch: activation has {channels}, kernel declares {ci}")
        if attrs.get("layout", "nhwc") != "nhwc":
            raise CodegenError(f"CONV2D native v1 requires layout 'nhwc', got {attrs.get('layout')!r}")
        stride = [int(x) for x in attrs.get("stride", [1, 1])]
        padding = [int(x) for x in attrs.get("padding", [0, 0, 0, 0])]
        dilation = [int(x) for x in attrs.get("dilation", [1, 1])]
        if len(stride) != 2 or len(padding) != 4 or len(dilation) != 2:
            raise CodegenError("CONV2D stride/dilation must have two entries and padding four entries")
        try:
            ho, wo = conv_out_dims(height, width, kh, kw, stride, padding, dilation)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise CodegenError(f"CONV2D {dst!r} has invalid geometry: {exc}") from exc
        if ho <= 0 or wo <= 0:
            raise CodegenError(f"CONV2D geometry produces invalid output extent {ho}x{wo}")
        if "maxpool" in (attrs.get("epilogue") or []):
            try:
                declared = pool_params(attrs, op=f"CONV2D {dst!r}")["pool_in_dims"]
            except ValueError as exc:
                raise CodegenError(str(exc)) from exc
            if declared != (ho, wo):
                raise CodegenError(
                    f"CONV2D {dst!r} pool_in_dims {list(declared)} disagrees with derived [{ho}, {wo}]")
        lhs = f"{ifm}__im2col_{conv_index}"
        acc = f"__conv_acc_{conv_index}"
        conv_index += 1
        tensors[lhs] = {"shape": [batch * ho * wo, kh * kw * ci],
                        "dtype": spec.get("dtype", "i8"), "role": "input"}
        recipes.append({"source": ifm, "target": lhs, "kh": kh, "kw": kw, "ci": ci,
                        "stride": stride, "padding": padding, "dilation": dilation, "layout": "nhwc"})
        commands.extend([
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": lhs, "rhs": rhs, "dst": acc}},
            {"opcode": "COMMIT", "operands": {"src": acc, "dst": dst},
             "attributes": {k: deepcopy(v) for k, v in attrs.items()
                            if k in {"epilogue", "output_dtype", "acc_scale", "requant_shift",
                                     "pool_in_dims", "pool_size", "pool_stride", "pool_padding",
                                     "pool_pad_value"}}},
        ])
    out["commands"] = commands
    return out


def _structurally_empty(cb: dict) -> bool:
    """Recognise only the explicit zero-tensor, zero-command calibration input."""
    tensors, commands = cb.get("tensors"), cb.get("commands")
    return isinstance(tensors, dict) and not tensors and isinstance(commands, list) and not commands


def _native_pool_spec(attrs: dict, *, rows: int, output_dtype: str, op: str) -> PoolSpec:
    """Validate and derive the exact native CONFIG_ST pooling subset."""
    if output_dtype != "i8":
        raise CodegenError(f"{op}: native maxpool requires i8 output, got {output_dtype!r}")
    try:
        params = pool_params(attrs, op=op)
    except ValueError as exc:
        raise CodegenError(str(exc)) from exc
    h, w = params["pool_in_dims"]
    ph, pw = params["pool_size"]
    sh, sw = params["pool_stride"]
    padding = params["pool_padding"]
    if ph != pw:
        raise CodegenError(f"{op}: native v1 requires square pool_size, got {[ph, pw]}")
    if sh != sw:
        raise CodegenError(f"{op}: native v1 requires square pool_stride, got {[sh, sw]}")
    if any(padding):
        raise CodegenError(f"{op}: native v1 requires zero pool_padding, got {list(padding)}")
    if ph <= 0 or sh <= 0:
        raise CodegenError(f"{op}: pool size and stride must be positive")
    plane = h * w
    if plane <= 0 or rows % plane:
        raise CodegenError(f"{op}: {rows} rows are not whole {h}x{w} planes")
    batch = rows // plane
    if batch != 1:
        raise CodegenError(f"{op}: native v1 requires batch 1, got {batch}")
    ho, wo = pool_out_dims(h, w, (ph, pw), (sh, sw), padding)
    if ho <= 0 or wo <= 0:
        raise CodegenError(f"{op}: pool geometry produces invalid output extent {ho}x{wo}")
    return PoolSpec(h, w, ph, sh, ho, wo)


def _pack_register_fields(layout: dict | None, values: dict[str, int], *, register: str) -> int:
    if not layout or not isinstance(layout.get("fields"), dict):
        raise CodegenError(f"{register} packed-field layout is absent from the target RTL facts")
    fields = layout["fields"]
    unknown = sorted(set(values) - set(fields))
    if unknown:
        raise CodegenError(f"{register} RTL layout has no field(s) {unknown}")
    packed = 0
    for name, value in values.items():
        spec = fields[name]
        width, offset = int(spec["width"]), int(spec["offset"])
        value = int(value)
        if value < 0 or value >= (1 << width):
            raise CodegenError(
                f"{register}.{name}={value} does not fit the RTL-derived {width}-bit field")
        packed |= value << offset
    return packed


def _pool_config_rs1(pool: PoolSpec | None, *, acc_act: int) -> int:
    isa = _isa()
    if pool is None:
        if isa.CONFIG_ST_TYPE is None:
            raise CodegenError("CONFIG_ST subtype is absent from the target capability manifest")
        return (int(acc_act) << 2) | int(isa.CONFIG_ST_TYPE)
    if isa.MAX_POOL_SUPPORTED is False:
        raise CodegenError("exact elaborated RTL facts show native max-pool was compiled out")
    if isa.MAX_POOL_SUPPORTED is not True:
        raise CodegenError("exact elaborated RTL facts do not establish native max-pool capability")
    if isa.CONFIG_ST_TYPE is None:
        raise CodegenError("CONFIG_ST subtype is absent from the target capability manifest")
    return _pack_register_fields(isa.CONFIG_ST_LAYOUT, {
        "cmd_type": isa.CONFIG_ST_TYPE,
        "activation": acc_act,
        "pool_stride": pool.stride,
        "pool_size": pool.size,
        "upad": 0,
        "lpad": 0,
        "pool_out_dim": pool.out_cols,
        "porows": pool.out_rows,
        "pocols": pool.out_cols,
        "orows": pool.in_rows,
        "ocols": pool.in_cols,
    }, register="ConfigMvoutRs1")


def _acc_base() -> int:
    """The accumulator ADDRESS base — the acc-select bit alone, with the readout-width selector cleared.

    A fused bias is moved into the accumulator BEFORE the contraction runs (the target's own
    ``sp_tiled_matmul_ws`` does exactly this: ``D_sp_addr_start = 1 << (ADDR_LEN-1)``, an mvin with
    neither the accumulate bit nor the full-C bit, after which every k-tile accumulates ONTO it). The
    base is DERIVED by clearing the derived full-C selector out of the derived full-i32 readout base,
    and cross-checked against the manifest's independently derived acc-only base. A mismatch means the
    two encodings no longer describe one address space, and is a REFUSAL: writing a bias to the wrong
    accumulator address does not fail, it silently biases someone else's tile.
    """
    isa = _isa()
    base = isa.C_ACC & ~isa.FULL_C_BIT
    if base != isa.ACC_I8:
        raise CodegenError(
            f"the target's derived accumulator encodings disagree: clearing full_C ({isa.FULL_C_BIT:#x}) "
            f"from the full-i32 readout base ({isa.C_ACC:#x}) gives {base:#x}, but the acc-only base is "
            f"{isa.ACC_I8:#x}; refusing to address the accumulator on a guess")
    return base


def _bias_container(dtype: str, *, op: str) -> tuple[int, str]:
    """``(element bytes, C type)`` for a fused bias operand, derived from the ACCUMULATOR's container.

    The command-buffer ABI is explicit that a bias is in the accumulator's dtype, not the operand's,
    because that is where the hardware puts it. The width is therefore read off the CIRCT-derived
    accumulator memory fact — never assumed to be four bytes — and the buffer's declared bias dtype must
    AGREE with it. Fail closed on both an underived accumulator container and a disagreeing declaration:
    an mvin issued with the wrong element width reads the right pointer with the wrong pitch, which
    produces plausible-looking wrong numbers rather than an error.
    """
    isa = _isa()
    acc_dtype, acc_bits = isa.ACC_ELEM_DTYPE, isa.ACC_ELEM_BITS
    if not isinstance(acc_dtype, str) or not isinstance(acc_bits, int) or acc_bits <= 0:
        raise CodegenError(
            f"{op}: a fused bias lands in the accumulator, but this target's CIRCT facts do not derive "
            f"an accumulator element container; refusing to pick a bias width")
    if acc_bits % 8:
        raise CodegenError(f"{op}: the derived accumulator container is {acc_bits} bits, which is not a "
                           f"whole number of bytes a DMA can address")
    if str(dtype) != acc_dtype:
        raise CodegenError(
            f"{op}: the bias operand is declared {dtype!r}, but a fused bias is moved into the "
            f"accumulator, whose derived container is {acc_dtype!r} (command-buffer ABI: a bias is in "
            f"the accumulator's dtype, not the operand dtype)")
    if not acc_dtype.startswith("i") or not acc_dtype[1:].isdigit():
        raise CodegenError(f"{op}: the derived accumulator container {acc_dtype!r} is not an integer "
                           f"container this harness can declare")
    return acc_bits // 8, f"int{acc_bits}_t"


def _parse_groups(cb: dict):
    """Parse the cb into resident-weight GROUPS for the MLIR-faithful path (decoupled from the C-path
    _parse, which is full-i32-only). Returns ``[(weight, k, n, list[Job])]`` in resident-pack order,
    each with ``jobs = [(lhs, out, epi, m, out_dtype, scale)]`` — the matmuls that reuse THAT weight.

    Supports MULTIPLE resident weights (a real multi-layer model): the emitter processes the groups
    sequentially — mvin one weight, run its matmuls, mvout — reusing the scratchpad for the next group, so
    a whole model is ONE co-scheduled kernel rather than one kernel per layer. relu + acc_scale epilogues
    and i8/i32 readout; integer requant(shift) is host-side (round-half-up) — NOT a Gemmini readout — so it
    is rejected here in favour of acc_scale."""
    cb = _normalize_command_buffer(cb)
    cmds = cb.get("commands", [])
    packs = [c for c in cmds if c["opcode"] == "RES_PACK"]
    matmuls = [c for c in cmds if c["opcode"] in ("MATMUL_RESIDENT", "MATMUL")]
    commits = [c for c in cmds if c["opcode"] == "COMMIT"]
    if not packs or not matmuls or len(matmuls) != len(commits):
        raise CodegenError(f"expected RES_PACK(s) + matmuls==commits>=1, got "
                           f"{len(packs)}/{len(matmuls)}/{len(commits)}")
    tensors = cb.get("tensors", {})
    acc_to_commit = {c["operands"]["src"]: c for c in commits}
    res_to_weight = {p["operands"]["dst"]: p["operands"]["src"] for p in packs}
    jobs_by_res: dict = {}                      # res id -> [job, ...]
    order: list = []                            # res ids in resident-pack (first-use) order
    for p in packs:
        res = p["operands"]["dst"]
        if res not in jobs_by_res:
            jobs_by_res[res] = []
            order.append(res)
    for mm in matmuls:
        ops = mm["operands"]
        res = ops["rhs"]
        if res not in jobs_by_res:
            raise CodegenError(f"matmul rhs {res!r} reuses no resident weight (have {list(jobs_by_res)})")
        weight = res_to_weight[res]
        k, _n = tensors[weight]["shape"]
        lhs = ops["lhs"]
        m, k2 = tensors[lhs]["shape"]
        if k2 != k:
            raise CodegenError(f"matmul lhs k={k2} != weight k={k}")
        commit = acc_to_commit.get(ops["dst"])
        if commit is None:
            raise CodegenError(f"matmul dst {ops['dst']} has no commit")
        attrs = commit.get("attributes", {})
        epi = list(attrs.get("epilogue", []))
        for s in epi:
            if s == "requant":
                raise CodegenError("integer requant(shift) is a host-side op (round-half-up); "
                                   "Gemmini's i8 readout uses acc_scale (round-near-even) — use "
                                   "epilogue ['acc_scale'] (see results/gemmini/requant_status.yaml)")
            if s not in ("relu", "acc_scale", "maxpool") and s not in BIAS_STAGES:
                raise CodegenError(f"unsupported epilogue stage {s!r} (have: "
                                   f"{', '.join((*BIAS_STAGES, 'relu', 'acc_scale', 'maxpool'))})")
        # A fused bias is the accumulator's INITIAL value (mvin D, then accumulate A@B onto it), so it
        # necessarily happens before every other stage. Declared anywhere but first, the emitted order
        # would not be the declared order — refuse rather than silently reassociate someone's epilogue.
        bias_stages = [s for s in epi if s in BIAS_STAGES]
        bias = None
        if bias_stages:
            op_name = f"COMMIT {commit['operands']['dst']!r}"
            if len(bias_stages) > 1:
                raise CodegenError(f"{op_name}: declares {len(bias_stages)} bias stages {bias_stages}; "
                                   f"the accumulator carries one initial value")
            if epi[0] not in BIAS_STAGES:
                raise CodegenError(
                    f"{op_name}: a fused bias is the accumulator's initial value, so it can only be the "
                    f"FIRST epilogue stage; declared epilogue is {epi}")
            try:
                bias = bias_tensor_name(commit.get("operands", {}), attrs, op=op_name)
            except ValueError as exc:
                raise CodegenError(str(exc)) from exc
            spec = tensors.get(bias)
            if not isinstance(spec, dict):
                raise CodegenError(f"{op_name}: bias tensor {bias!r} is not declared in the buffer")
            bshape = [int(x) for x in (spec.get("shape") or [])]
            if bshape not in ([_n], [1, _n]):
                raise CodegenError(
                    f"{op_name}: bias {bias!r} has shape {bshape}, but a fused bias is one value per "
                    f"OUTPUT COLUMN, broadcast over rows — expected [{_n}]")
            _bias_container(str(spec.get("dtype")), op=op_name)     # derive-or-refuse, before emission
        out_dtype = attrs.get("output_dtype", "i32")
        scale = float(attrs.get("acc_scale", 1.0))
        pool = None
        output_rows = m
        if "maxpool" in epi:
            if epi[-1] != "maxpool":
                raise CodegenError("native maxpool must be the final epilogue stage")
            pool = _native_pool_spec(attrs, rows=m, output_dtype=out_dtype,
                                     op=f"COMMIT {commit['operands']['dst']!r}")
            output_rows = pool.out_rows * pool.out_cols
        jobs_by_res[res].append(Job(lhs, commit["operands"]["dst"], tuple(epi), m,
                                     output_rows, out_dtype, scale, pool, bias))
    groups = []
    for res in order:
        weight = res_to_weight[res]
        k, n = tensors[weight]["shape"]
        groups.append((weight, k, n, jobs_by_res[res]))
    return groups


def emit_kernel_mlir(cb: dict) -> tuple[str, list[str]]:
    """Return (mlir_text, arg_order) for the command buffer.

    arg_order is the func's pointer-arg order: [weights] + [matmul activations] + [outputs].
    Handles full-i32 readout AND Gemmini's scaled/clamped i8 readout (float acc_scale), and MULTIPLE
    resident weights — each weight group is mvin'd, matmul'd, and mvout in turn, reusing the scratchpad,
    so a whole multi-layer model lowers to ONE co-scheduled kernel.
    """
    cb = _normalize_command_buffer(cb)
    if _structurally_empty(cb):
        # This is still compiled and called by the production harness.  It executes no accelerator
        # command, so its measured window is a legitimate shared runner/compiler baseline.
        return ('module {\n  llvm.func @gemmini_kernel() {\n'
                '    llvm.inline_asm has_side_effects "fence", "" : () -> ()\n'
                '    llvm.return\n  }\n}\n'), []
    isa = _isa()
    # Bind the RTL-derived facts as LOCALS. They used to be module-level constants; the lazy refactor
    # moved them behind a PEP 562 module __getattr__, which serves `gemmini_codegen_mlir.DIM` from
    # OUTSIDE but is never consulted for a bare global lookup inside this module -- so every use below
    # raised NameError the moment this path ran. Reading them off `isa` once keeps the laziness and makes
    # the dependency explicit.
    DIM, F1, C_ACC = isa.DIM, isa.F1, isa.C_ACC
    ACC_ACCUM, ACC_I8, SCRATCHPAD_ROWS = isa.ACC_ACCUM, isa.ACC_I8, isa.SCRATCHPAD_ROWS
    K_CONFIG, K_MVIN, K_MVOUT = isa.K_CONFIG, isa.K_MVIN, isa.K_MVOUT
    K_PRELOAD, K_COMPUTE_PRELOADED = isa.K_PRELOAD, isa.K_COMPUTE_PRELOADED
    groups = [] if _structurally_empty(cb) else _parse_groups(cb)
    weights = [g[0] for g in groups]
    lhss = [j.lhs for g in groups for j in g[3]]
    outs = [j.out for g in groups for j in g[3]]
    # Fused-bias operands are a FOURTH, TRAILING block, group-major like the other per-job blocks (see
    # `kernel_abi.arg_order_by_command_shape` in the OOT backend contract). Trailing and skipped-when-
    # absent: a buffer with no bias stage has the identical three-block signature it always had, so
    # every already-certified kernel keeps its ABI byte for byte.
    biases = [j.bias for g in groups for j in g[3] if j.bias is not None]
    args = weights + lhss + outs + biases
    arg_decl = ", ".join(f"%a{i}: !llvm.ptr" for i in range(len(args)))

    body: list[str] = []
    ctr = [0]

    def fresh() -> str:
        ctr[0] += 1
        return f"%v{ctr[0]}"

    def konst(v: int) -> str:
        s = fresh()
        body.append(f"    {s} = llvm.mlir.constant({int(v)} : i64) : i64")
        return s

    def rocc(funct: int, rs1: str, rs2: str) -> None:
        body.append(f'    llvm.inline_asm has_side_effects ".insn r {hex(isa.CUSTOM_OPCODE)}, {hex(isa.FUNCT3)}, '
                    f'{funct}, x0, $0, $1", "r,r" {rs1}, {rs2} : (i64, i64) -> ()')

    pint = {}
    for i, name in enumerate(args):
        s = fresh()
        body.append(f"    {s} = llvm.ptrtoint %a{i} : !llvm.ptr to i64")
        pint.setdefault(name, s)

    def addr(name: str, byte_off: int) -> str:
        if byte_off == 0:
            return pint[name]
        s = fresh()
        body.append(f"    {s} = llvm.add {pint[name]}, {konst(byte_off)} : i64")
        return s

    last_ld = [None]

    def config_ld(stride: int) -> None:        # mvin row stride (bytes); re-emit only on change
        if last_ld[0] != stride:
            rocc(isa.K_CONFIG, konst(isa.CFG_LD_RS1), konst(stride))
            last_ld[0] = stride

    body.append('    llvm.inline_asm has_side_effects "fence", "" : () -> ()')
    rocc(isa.K_FLUSH, konst(0), konst(0))
    rocc(isa.K_CONFIG, konst(isa.CFG_EX_RS1), konst(isa.CFG_EX_RS2))

    # Each resident-weight GROUP is emitted in turn: mvin the weight, run its matmuls, mvout — then the
    # next group reuses the scratchpad. A single-weight model is one group (byte-identical to before); a
    # multi-layer model is several groups in ONE kernel, its layers co-scheduled in one address space.
    for (weight, k, n, jobs) in groups:
        kp, np_ = _ceil_dim(k), _ceil_dim(n)
        Kt, Nt = kp // DIM, np_ // DIM
        a_slot = Kt * Nt * DIM                  # A tile slot, after this group's resident W tiles

        # Resident weight: mvin all Kt x Nt tiles once (config_ld stride = np_ bytes, elem_t=1B).
        config_ld(np_)
        for kt in range(Kt):
            for nj in range(Nt):
                w_row = (kt * Nt + nj) * DIM
                off = (kt * DIM) * np_ + nj * DIM
                rocc(K_MVIN, addr(weight, off), konst(_pack(w_row)))

        # config_ld for activations (stride = kp bytes).
        config_ld(kp)

        # Operand stationarity (capacity-fit): the resident weight already lives in the first Kt*Nt
        # scratchpad tiles; the activation row-panel A[mi, 0:Kt] adds Kt tiles. When both fit the
        # on-chip operand store (scratchpad depth, an RTL-derived fact) we mvin each row-panel ONCE and
        # REUSE its Kt activation tiles across the whole N sweep — the old schedule re-moved every
        # activation tile for every output column (Mt*Nt*Kt mvins), re-fetching the same DRAM tile Nt
        # times. Panel residency cuts that to Mt*Kt mvins. Fail-safe: when the panel would not fit (or
        # the depth is underived) we keep the per-(mi,nj) single-slot schedule, byte-identical to before.
        panel_resident = SCRATCHPAD_ROWS is not None and (Kt * Nt + Kt) * DIM <= SCRATCHPAD_ROWS

        for job in jobs:
            lhs, out, epi = job.lhs, job.out, job.epilogue
            m, out_dtype, scale, pool = job.input_rows, job.output_dtype, job.scale, job.pool
            mp = _ceil_dim(m)
            Mt = mp // DIM
            i8_out = out_dtype == "i8"              # i8 readout = Gemmini float acc_scale + clamp
            acc_act = 1 if "relu" in epi else 0     # config_st acc_act (RELU=1)
            scale_bits = _f32_bits(scale) if i8_out else F1
            elt = 1 if i8_out else 4               # output element bytes
            read_base = ACC_I8 if i8_out else C_ACC  # i8 readout drops the full_C bit -> scale applies
            # Bias element width comes from the CIRCT-derived ACCUMULATOR container (that is where the
            # bias is moved to), validated against the buffer's declaration in _parse_groups.
            bias_bytes = (0 if job.bias is None
                          else _bias_container(cb["tensors"][job.bias]["dtype"],
                                               op=f"COMMIT {out!r}")[0])

            def c_row_of(mi: int, nj: int, _pool=pool, _mp=mp) -> int:
                """Accumulator row this (mi, nj) output tile occupies. Without pooling every tile is
                read out immediately and reuses row 0; pooling RETAINS whole planes, so each tile owns
                its own rows. The bias mvin and the preload must name the SAME row."""
                return nj * _mp + mi * DIM if _pool is not None else 0

            # Pooling reads a whole retained spatial plane from consecutive accumulator rows. One plane
            # per channel tile therefore needs ``mp`` rows; refuse rather than alias accumulator storage.
            if pool is not None:
                needed = Nt * mp
                if isa.ACCUMULATOR_ROWS is None:
                    raise CodegenError("native maxpool needs an RTL-derived accumulator row capacity")
                if needed > int(isa.ACCUMULATOR_ROWS):
                    raise CodegenError(
                        f"native maxpool needs {needed} retained accumulator rows, target has "
                        f"{isa.ACCUMULATOR_ROWS}")
            # config_st RS1 field placement is derived from ConfigMvoutRs1 in the target's Chisel ISA.
            # RS2 retains the established acc-scale/out-row-stride layout.
            rocc(K_CONFIG, konst(_pool_config_rs1(pool, acc_act=acc_act)),
                 konst((scale_bits << 32) | (np_ * elt)))
            for mi in range(Mt):
                if panel_resident:
                    # Row-panel mvin: the Kt activation tiles of row mi, into Kt distinct slots, ONCE.
                    # The stride is (re)asserted HERE rather than once before the loop nest, because a
                    # fused bias reconfigures the load unit to a zero row stride between row panels.
                    # Idempotent: when nothing intervened it emits nothing, so a buffer without a bias
                    # is byte-identical to before.
                    config_ld(kp)
                    for kt in range(Kt):
                        a_off = ((mi * DIM) * kp + kt * DIM)
                        rocc(K_MVIN, addr(lhs, a_off), konst(_pack(a_slot + kt * DIM)))
                for nj in range(Nt):
                    if job.bias is not None:
                        # FUSED BIAS = the accumulator's initial value. One mvin of this output tile's
                        # bias columns into the accumulator tile, with the DRAM row stride configured to
                        # ZERO so all DIM accumulator rows read the SAME bias row — the target's own
                        # repeating-bias move-in. The destination carries neither the accumulate bit
                        # (this OVERWRITES the tile) nor the readout-width bit; the k-loop below then
                        # keeps the accumulate bit set from k=0 so A@B lands ON TOP of it.
                        config_ld(0)
                        rocc(K_MVIN, addr(job.bias, nj * DIM * bias_bytes),
                             konst(_pack(_acc_base() | c_row_of(mi, nj))))
                    for kt in range(Kt):
                        if panel_resident:
                            a_addr = a_slot + kt * DIM              # reuse the resident panel tile
                        else:
                            a_off = ((mi * DIM) * kp + kt * DIM)    # legacy: re-mvin per output column
                            config_ld(kp)                           # see the panel branch: bias resets it
                            rocc(K_MVIN, addr(lhs, a_off), konst(_pack(a_slot)))
                            a_addr = a_slot
                        c_row = c_row_of(mi, nj)
                        cad = C_ACC | c_row
                        # Accumulate rather than overwrite for every k-tile after the first — and for
                        # the first one TOO when a bias was moved in, so A@B lands on top of D instead
                        # of erasing it (the target's own `no_bias_new_matrix` condition, inverted).
                        if kt != 0 or job.bias is not None:
                            cad |= ACC_ACCUM                         # accumulator is always i32
                        rocc(K_PRELOAD, konst(_pack((kt * Nt + nj) * DIM)), konst(_pack(cad)))
                        rocc(K_COMPUTE_PRELOADED, konst(_pack(a_addr)), konst(_pack(GARBAGE)))
                    if pool is None:
                        c_off = ((mi * DIM) * np_ + nj * DIM) * elt
                        rocc(K_MVOUT, addr(out, c_off), konst(_pack(read_base)))
            if pool is not None:
                # StoreController walks the configured HxW plane from localaddr and emits the Ho*Wo
                # result. Its pooling path requires a single channel block (rows is ignored there; the
                # target header's own pool call passes zero), hence one MVOUT per channel tile.
                for nj in range(Nt):
                    channels = min(DIM, n - nj * DIM)
                    c_base = read_base | (nj * mp)
                    rocc(K_MVOUT, addr(out, nj * DIM * elt),
                         konst(_pack(c_base, cols=channels, rows=0)))

    body.append('    llvm.inline_asm has_side_effects "fence", "" : () -> ()')
    text = ("module {\n  llvm.func @gemmini_kernel(" + arg_decl + ") {\n"
            + "\n".join(body) + "\n    llvm.return\n  }\n}\n")
    return text, args


def build_object(cb: dict, workdir: str | Path) -> Path:
    """Lower the MLIR kernel for ``cb`` to a rv64gcv object; return the .o path."""
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower import codegen
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    text, _ = emit_kernel_mlir(cb)
    ll = lower_to_llvm_ir(text, workdir=work)
    (work / "gemmini_kernel.ll").write_text(ll, encoding="utf-8")
    return Path(codegen.compile_ll(work / "gemmini_kernel.ll", work / "gemmini_kernel.o", "riscv"))


def rocc_instruction_count(obj: str | Path) -> int:
    from merlin.llvmlower.custom_isa import disassemble
    dis = disassemble(obj)
    return sum(1 for ln in dis.splitlines()
               if ".insn" in ln and len(ln.split()) > 1 and ln.split()[1].endswith("7b"))


@cache
def _counter_slots() -> dict:
    """Physical counter capacity, extracted from this target's elaborated CIRCT HW.

    The identifiers below select the target-owned counter structure, just as a top-module name selects
    an RTL design.  The capacity itself is never copied here: the generic reader cross-checks three
    independently elaborated state families and returns no number when they are absent or inconsistent.
    """
    from merlin.perf.hw_counters import counter_slots_from_circt
    from merlin.targetgen.rtl import mlc_bridge

    path = mlc_bridge.core_hw_mlir("gemmini")
    if path is None or not Path(path).is_file():
        return {"status": "unknown", "slots": None,
                "why": "the elaborated CIRCT core HW artifact could not be located"}
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return counter_slots_from_circt(
        text, module="CounterFile",
        state_families=("counter_config", "counter_snapshot", "counters"),
        source=str(path))


def _counters_requested() -> bool:
    """Whether this harness should carry the joint-occupancy bracket.

    OPT-IN by environment, because the default must stay byte-identical: this is the graded harness, and
    a change to every run would make a round's verdicts incomparable with the rounds before it.
    """
    import os

    return str(os.environ.get("MERLIN_HW_COUNTERS", "")).strip().lower() in ("1", "true", "yes", "on")


def _counter_unit_requested() -> str | None:
    """Optional unit family selected from the target's own counter names (for example ``BYTES``)."""
    import os

    value = str(os.environ.get("MERLIN_HW_COUNTER_UNIT", "")).strip().upper()
    return value or None


def _read_discovered_counter_header(discovery: dict) -> str:
    """Read once and verify the discovery-time digest before using mutable external headers."""
    import hashlib

    text = Path(discovery["header"]).read_text(encoding="utf-8", errors="replace")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != discovery.get("header_sha256"):
        raise CodegenError("counter header changed after discovery; refusing mixed campaign evidence")
    return text


def _cache_state_requested() -> str:
    """Select a requested pre-measurement protocol, without claiming cache-state observability."""
    import os

    state = str(os.environ.get("MERLIN_CACHE_STATE", "cold")).strip().lower()
    if state not in ("cold", "warm"):
        raise CodegenError(f"unsupported cache-state measurement condition {state!r}")
    return state


def _measurement_c_fragments(warmup_work: str) -> dict[str, str]:
    """Counter and cache-condition fragments shared by every target-owned harness shape."""
    cpro, cepi, include = "", "", ""
    if _counters_requested():
        try:
            from merlin.perf import hw_counters as hc

            discovered = hc.counters_for_target("gemmini")
            if discovered.get("status") != "derived":
                raise CodegenError(discovered.get(
                    "why", "the requested counter header was not derived"))
            text = _read_discovered_counter_header(discovered)
            codes = hc.event_codes(text)
            unit = _counter_unit_requested()
            capacity = _counter_slots()
            if capacity.get("status") != "derived":
                raise CodegenError(capacity.get(
                    "why", "counter capacity is not derivable from CIRCT"))
            disabled_code = codes.get("DISABLE")
            if (isinstance(disabled_code, bool) or not isinstance(disabled_code, int)
                    or disabled_code < 0):
                raise CodegenError(
                    "the target counter header does not derive a non-negative DISABLE event code")
            if unit is None:
                selected = hc.derive_occupancy_counters(text)
                bracket = hc.counter_bracket_c(
                    selected, codes, slots=int(capacity["slots"]), padding_code=disabled_code)
            else:
                selected = hc.counters_with_unit(text, unit)
                if not selected:
                    raise CodegenError(
                        f"the target counter header declares no {unit!r} unit counters")
                bracket = hc.counter_bracket_for_names(
                    tuple(selected), codes, slots=int(capacity["slots"]),
                    padding_code=disabled_code)
            cpro = (f'  printf("{hc.COUNTER_SCHEMA_MARKER} '
                    f'{discovered["header_sha256"]}\\n");\n' + bracket["prologue"])
            cepi = bracket["epilogue"]
            include = '#include "include/gemmini_counter.h"\n'
        except Exception as exc:                    # noqa: BLE001 — normalize requested-instrumentation failure
            raise CodegenError(f"requested counter instrumentation unavailable: {exc}") from exc
    warmup = ""
    if _cache_state_requested() == "warm":
        warmup = warmup_work.rstrip() + "\n  // merlin: warmup completed outside the measured/counter window.\n"
    requested = _cache_state_requested()
    return {"include": include, "prologue": cpro, "epilogue": cepi, "warmup": warmup,
            "cache_state": "unknown", "cache_state_observed": False,
            "cache_protocol": ("one_unmeasured_predecessor" if requested == "warm"
                               else "fresh_elf_process"),
            "requested_cache_condition": requested}


#: Element count at or above which a CONSTANT operand is linked in as a binary blob instead of being
#: written as a C initializer list. Below it nothing changes and the emitted harness stays
#: byte-identical, which matters because this harness is on the graded L0/L1/L3 path and a change to
#: every run would make a round's verdicts incomparable with the rounds before it.
#:
#: Above it the initializer form is not merely large but unbuildable. The sibling SIMT harness measured
#: the same wall on its own element-wise form: a 2048x2048 operand is 4.19 million initializers and
#: 124.7 MB of C, and the compiler ran 45+ minutes without finishing. The census this corpus is
#: supposed to represent puts 99.4% of its MAC mass in shapes far above that, so the ceiling is not a
#: tuning parameter -- it is the reason no member of those shapes can be built at all.
_BLOB_MIN_ELEMS = 1024


def _blob_asm(symbol: str, payload: Path, *, align: int, elems: int) -> str:
    """An assembler stub defining ``symbol`` from ``payload``'s bytes, aligned as the C form was.

    ``.incbin`` rather than a ``.byte`` list: the whole point is to stop handing the toolchain one
    statement per element, and a huge ``.byte`` directive reintroduces exactly that cost in the
    assembler instead of the compiler.

    The alignment is DERIVED from the same expression the C attribute uses (the tile edge times the
    element width), not chosen. An under-aligned operand is not a build error -- the DMA simply reads
    from an address the row stride does not expect, which shows up as wrong data much later.
    """
    return (f"  .section .rodata\n"
            f"  .balign {int(align)}\n"
            f"  .globl {symbol}\n"
            f"{symbol}:\n"
            f"  .incbin \"{payload}\"\n"
            f"  .size {symbol}, . - {symbol}\n"
            f"  /* {elems} elements */\n")


def _const_operand(symbol: str, ctype: str, data: list, *, dtype: str,
                   blobs: dict | None, align_macro: str = "row_align(1)") -> str:
    """One constant operand: an initializer list, or an ``extern`` filled from a blob.

    ``blobs is None`` means the caller cannot link an extra object, so the inline form is the only
    legal one and a large operand is emitted as it always was -- slow, but never silently wrong.

    ``align_macro`` is the target header's own row-alignment macro for this operand's CONTAINER
    (``row_align`` is an operand-element row, ``row_align_acc`` an accumulator-element row). The blob
    form already derived the same number from the element width, so the two forms agree; the default
    keeps every operand that was emitted before this parameter existed byte-identical.
    """
    from merlin.runtime.tensor import DTYPE_BYTES

    if blobs is None or len(data) < _BLOB_MIN_ELEMS:
        return (f"static const {ctype} {symbol}[{len(data)}] {align_macro} = "
                f"{{{','.join(str(int(v)) for v in data)}}};")
    width = int(DTYPE_BYTES[dtype])
    blobs[symbol] = {
        "bytes": b"".join(int(v).to_bytes(width, "little", signed=True) for v in data),
        # `row_align(1)` expands to `aligned(1 * DIM * sizeof(elem_t))`, so the blob must match it.
        "align": _ceil_dim(1) * width,
        "elems": len(data),
    }
    return f"extern const {ctype} {symbol}[{len(data)}];"


def _harness_c(cb: dict, inputs: dict | None = None, *,
               blobs: dict | None = None) -> str:
    """Thin C harness: embed padded leaf data, call the MLIR kernel, print outputs (cropped).

    ``inputs`` (name -> nested-list) INJECTS explicit operand values so the device runs the model's real
    activations/weights; absent, each leaf is materialized deterministically from its name (reproducible)."""
    cb = _normalize_command_buffer(cb)
    groups = [] if _structurally_empty(cb) else _parse_groups(cb)
    leaves = materialize_inputs(cb, inputs)
    weights = [g[0] for g in groups]
    lhss = [j.lhs for g in groups for j in g[3]]
    # (out_name, committed rows, n, out_dtype) — pooling changes the committed row extent.
    outs = [(j.out, j.output_rows, g[2], j.output_dtype) for g in groups for j in g[3]]

    decls = []
    for weight, k, n, jobs in groups:
        kp, np_ = _ceil_dim(k), _ceil_dim(n)
        wpad = _pad_rowmajor(list(leaves[weight].data), k, n, kp, np_)
        decls.append(_const_operand(f"T_{weight}", "elem_t", wpad,
                                    dtype=leaves[weight].dtype, blobs=blobs))
        for job in jobs:
            lhs, m = job.lhs, job.input_rows
            mp = _ceil_dim(m)
            ap = _pad_rowmajor(list(leaves[lhs].data), m, k, mp, kp)
            decls.append(_const_operand(f"T_{lhs}", "elem_t", ap,
                                        dtype=leaves[lhs].dtype, blobs=blobs))
    # Fused-bias operands: one padded ROW of accumulator-container values per biased job, in the same
    # group-major order the kernel's trailing argument block uses. Padded to the output's column tiling
    # (like every other operand) so the zero-stride mvin reads a whole DIM-wide tile of real storage.
    bias_decls = []
    biases = []
    for _weight, _k, n, jobs in groups:
        np_ = _ceil_dim(n)
        for job in jobs:
            if job.bias is None:
                continue
            _width, ctype = _bias_container(cb["tensors"][job.bias]["dtype"],
                                            op=f"COMMIT {job.out!r}")
            bpad = _pad_rowmajor(list(leaves[job.bias].data), 1, n, 1, np_)
            bias_decls.append(_const_operand(f"T_{job.bias}", ctype, bpad,
                                             dtype=leaves[job.bias].dtype, blobs=blobs,
                                             align_macro="row_align_acc(1)"))
            biases.append(job.bias)
    for out, m, n, out_dtype in outs:
        np_ = _ceil_dim(n)
        mp = _ceil_dim(m)
        if out_dtype == "i8":               # scaled/clamped i8 readout buffer (elem_t)
            decls.append(f"static elem_t T_{out}[{mp * np_}] row_align(1);")
        else:
            # Full-i32 accumulator readout is 4 bytes/elem — this is what the RoCC sequence
            # emits for (config_st out-row-stride = np_*4, mvout tile offset *4). Declare the
            # buffer as int32_t, NOT `acc_t`: the gemmini-rocc-tests gemmini_params.h typedefs
            # acc_t=uint64_t (8 B) while the spike --extension=gemmini libgemmini was built with
            # acc_t=int32_t (4 B, the value it actually DMAs). Reading an 8-byte type from the
            # 4-byte readout halved every row (Y[i][j] read C[i][2j]) and zeroed the bottom half.
            decls.append(f"static int32_t T_{out}[{mp * np_}] row_align_acc(1);")

    decls.extend(bias_decls)
    # must mirror emit_kernel_mlir: weights + lhss + outs + biases
    args = weights + lhss + [o[0] for o in outs] + biases
    call = ", ".join(f"(void*)T_{a}" for a in args)
    prints = []
    for out, m, n, _ in outs:
        npo = _ceil_dim(n)                          # this output's own padded column stride
        prints.append(f'  printf("OUT {out} {m} {n}");')
        prints.append(f"  for (long i = 0; i < {m}; i++) for (long j = 0; j < {n}; j++)"
                      f" printf(\" %d\", (int)T_{out}[i * {npo} + j]);")
        prints.append('  printf("\\n");')
    # Print METRIC cycles BEFORE the (possibly huge) OUT tensor dump: large-output kernels flood the UART
    # and the per-ELF FireSim capture truncates mid-dump, losing a trailing METRIC line. Emitting the tiny
    # cycle metric first guarantees it is always captured; the OUT dump follows for correctness.
    # OPTIONAL joint-occupancy counters, bracketing the SAME window the cycle metric already measures.
    #
    # Off unless asked for, and the default path is byte-identical to what it was: this harness is on
    # the graded L0/L1/L3 path, and a change that altered every run would make a round's verdicts
    # incomparable to the rounds before it. Asked for, it configures this target's combination counters
    # before the kernel and reads them back after, so realised overlap is measured on the same window as
    # the cycles -- the two numbers describe one run rather than two.
    #
    # Derived, never typed: the counter set and its event codes come from the target's own shipped
    # header, and the emitter refuses rather than emitting a partial bracket when a code is missing or
    # the set exceeds the hardware's slots. Measured through this path on real RTL: eta 0.8207 against
    # 0.7717 for a bit-exact reordering of the same work, which is the comparison a correctness gate
    # cannot make.
    measured_call = f"  gemmini_kernel({call});\n  gemmini_fence();"
    fragments = _measurement_c_fragments(measured_call)
    return ("#include <stdint.h>\n#include <stdio.h>\n#include \"include/gemmini_testutils.h\"\n"
            + fragments["include"] +
            "extern void gemmini_kernel();\n" + "\n".join(decls) + "\nint main() {\n"
            + fragments["warmup"] +
            fragments["prologue"] +
            "  uint64_t c0 = read_cycles();\n"
            + measured_call + "\n"
            "  uint64_t c1 = read_cycles();\n"
            + fragments["epilogue"] +
            '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));\n'
            '  printf("METRIC cycle_window_gemmini_region 1\\n");\n'
            + "\n".join(prints) + "\n"
            '  printf("DONE\\n");\n  return 0;\n}\n')


def run_on_spike(cb: dict, workdir: str | Path | None = None, *, simulator: str = "spike",
                 timeout: int = 600) -> dict:
    """Build the MLIR kernel object + thin harness, run on the Gemmini oracle, gate vs reference."""
    import subprocess
    import tempfile
    from . import gemmini as gem
    from merlin.runtime.reference import outputs_match, reference_outputs

    cb = _normalize_command_buffer(cb)
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="gemmini_mlir_run_"))
    work.mkdir(parents=True, exist_ok=True)
    obj = build_object(cb, work)
    # THE LINK IS WHAT MAKES THE BLOB FORM LEGAL. `_harness_c` will only move an operand out of line
    # when it is handed somewhere to put it, so a caller that cannot add an object to the link gets
    # the inline form and a correct (if slow) build rather than an undefined symbol.
    blobs: dict = {}
    (work / "harness.c").write_text(_harness_c(cb, blobs=blobs), encoding="utf-8")
    blob_sources = []
    for symbol, spec in sorted(blobs.items()):
        payload = work / f"{symbol}.bin"
        payload.write_bytes(spec["bytes"])
        stub = work / f"{symbol}.S"
        stub.write_text(_blob_asm(symbol, payload, align=spec["align"], elems=spec["elems"]),
                        encoding="utf-8")
        blob_sources.append(str(stub))
    rt, common = gem.rocc_tests_dir(), gem._common_dir()
    elf = work / "gemmini_mlir.elf"
    cmd = [str(gem.gcc_path()), "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-mcmodel=medany",
           "-std=gnu99", "-O2", "-ffast-math", "-fno-common", "-fno-builtin-printf",
           "-fno-tree-loop-distribute-patterns", "-march=rv64gc", "-Wa,-march=rv64gc",
           "-lm", "-lgcc", "-I", str(rt / "riscv-tests"), "-I", str(rt / "riscv-tests/env"),
           "-I", str(rt), "-I", str(common), "-DID_STRING=", "-DPRINT_TILE=0",
           "-nostdlib", "-nostartfiles", "-static", "-T", str(common / "test.ld"), "-DBAREMETAL=1",
           str(work / "harness.c"), str(obj), *blob_sources, "-o", str(elf),
           *(str(p) for p in sorted(common.glob("*.c"))),
           *(str(p) for p in sorted(common.glob("*.S")))]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise gem.GemminiError(f"link failed:\n{proc.stderr[-2000:]}")
    console = gem.run_elf(elf, simulator=simulator, timeout=timeout)
    outputs, raw = gem.parse_output(console)
    result = {"outputs": outputs, "correct": outputs_match(outputs, reference_outputs(cb)),
              "metrics": {"cycles": raw.get("cycles", 0)}, "path": "mlir_inline_asm_rocc",
              "oracle": gem.ORACLE[simulator], "elf": str(elf), "console": console,
              "measurement_conditions": {
                  "cache_state": "unknown", "cache_state_observed": False,
                  "cache_protocol": ("one_unmeasured_predecessor"
                                     if _cache_state_requested() == "warm" else "fresh_elf_process"),
                  "requested_cache_condition": _cache_state_requested(),
                  "cycle_window": "gemmini_region"}}
    if _counters_requested():
        from merlin.perf import hw_counters as hc

        discovery = hc.counters_for_target("gemmini")
        selected_unit = _counter_unit_requested()
        counter_report = {"discovery": discovery, "capacity": _counter_slots(),
                          "selection": {"kind": "unit" if selected_unit else "joint_occupancy",
                                        "unit": selected_unit},
                          "readings": hc.parse_counter_output(console)}
        measured_schema = hc.parse_counter_schema(console)
        counter_report["measured_header_sha256"] = measured_schema
        if (discovery.get("status") == "derived"
                and measured_schema == discovery.get("header_sha256")):
            header = _read_discovered_counter_header(discovery)
            if selected_unit is None:
                occupancy = hc.derive_occupancy_counters(header)
                counter_report["occupancy"] = occupancy.to_dict()
                partition = gem.counter_partition_inputs()
                if partition.get("status") == "available":
                    counter_report["overlap"] = hc.eta_from_counters(
                        counter_report["readings"], occupancy,
                        hw_text=partition["hw_text"], codes=hc.event_codes(header),
                        module=partition["module"], counter_module=partition["counter_module"],
                        measurement_cycles=raw.get("cycles"),
                        source=partition["source"])
                else:
                    counter_report["overlap"] = {
                        "state": "unknown", "eta": None,
                        "why": partition.get("why", "CIRCT partition evidence is unavailable")}
            else:
                counter_report["selected_counters"] = hc.counters_with_unit(header, selected_unit)
                counter_report["overlap"] = {
                    "state": "not_measured", "eta": None,
                    "why": f"this run selected the {selected_unit} counter family, not joint occupancy"}
        else:
            counter_report["overlap"] = {
                "state": "unknown", "eta": None,
                "why": (discovery.get("why", "the target counter set was not derived")
                        if discovery.get("status") != "derived" else
                        "the measured ELF counter-schema digest does not match current discovery")}
        result["counters"] = counter_report
    return result

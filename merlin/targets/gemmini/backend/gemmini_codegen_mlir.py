"""Merlin-FAITHFUL Gemmini codegen: the full (non-requant) RoCC sequence emitted as MLIR
(llvm-dialect `llvm.inline_asm` `.insn` ops), lowered by merlin's compiler — NOT C.

Generalizes the C0 proof to the whole non-requant conformance battery: 1 RES_PACK (resident
weight, mvin'd once), N MATMUL_RESIDENT (each tiled into 16x16 blocks with K-accumulation,
zero-padded edges), N COMMIT (empty epilogue or [relu]), full-i32 mvout. Every Gemmini
instruction is a custom-3 (0x7b) `.insn` op with operands packed exactly per `gemmini.h`
(verified against the libgemmini macros and the Verilator RTL). DRAM tile addresses are
compile-time pointer-offset arithmetic on the func's pointer args; the tile sequence is
unrolled (shapes are static).

requant (C2/C3) is intentionally rejected — Gemmini's acc_scale is not bit-exact with merlin's
integer requant (see docs/gemmini_requant_reconciliation.md). The kernel is pure llvm-dialect
MLIR lowered via `lower_to_llvm_ir` -> object; a thin C harness embeds data + calls it + prints.
"""
from __future__ import annotations

from functools import cache
from pathlib import Path
from types import SimpleNamespace

from .gemmini_codegen import _ceil_dim, _pad_rowmajor, _parse, CodegenError   # sibling — moves together
from merlin.runtime.commandbuffer import materialize_inputs

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
    from merlin.targetgen.rtl.facts import load_facts
    m = load_capability_manifest("gemmini")
    enc, rb = m.encoding, m.encoding["readout_bits"]
    code_of = {cls: code for code, cls in enc["semantic_class"].items()}
    facts = load_facts("gemmini")["facts"]
    fd = next(i for i in facts["interfaces"] if i.get("name") == "funct_decode_table")
    # DIM (systolic mesh dimension) is a CIRCT-extracted FACT, not a hand-declared manifest field.
    mesh = next((a for a in facts.get("arrays", []) if a.get("name") == "mesh"), {})
    # On-chip operand-store depth (scratchpad rows) is a CIRCT-extracted memory FACT — it bounds how
    # much of an operand can be kept resident (see the capacity-fit residency in emit_kernel_mlir).
    # Fail-closed: absent -> None (no capacity-fit residency, the per-tile schedule stands).
    sp = next((mem for mem in facts.get("memories", []) if mem.get("name") == "scratchpad"), {})
    isa = SimpleNamespace(
        DIM=mesh.get("rows", 16), ADDR_LEN=enc["addr_len"],
        F1=rb["f1"],                        # float 1.0 bits
        C_ACC=rb["c_acc"],                  # 0xA0000000 full-i32 accumulator readout base
        ACC_ACCUM=rb["acc_accum"],          # 0x40000000 accumulate-onto for K-tiles after the first
        ACC_I8=rb["acc_i8"],                # 0x80000000 acc addr WITHOUT full_C: scaled i8 readout
        K_CONFIG=code_of["CONFIG"], K_MVIN=code_of["MVIN"], K_MVOUT=code_of["MVOUT"],
        K_COMPUTE_PRELOADED=code_of["COMPUTE_PRELOADED"], K_PRELOAD=code_of["PRELOAD"],
        K_FLUSH=code_of["FLUSH"],
        CUSTOM_OPCODE=fd["custom_opcode"],  # RoCC custom-3 (0x7b)
        FUNCT3=fd["funct3"],                # 0x3
        SCRATCHPAD_ROWS=sp.get("depth"),    # on-chip operand-store depth (rows); None if underived
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
            "CFG_EX_RS1", "CFG_EX_RS2", "CFG_LD_RS1", "SCRATCHPAD_ROWS")


def __getattr__(name: str):
    if name in _DERIVED:
        return getattr(_isa(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _pack(addr: int) -> int:
    isa = _isa()
    return (isa.DIM << (isa.ADDR_LEN + 16)) | (isa.DIM << isa.ADDR_LEN) | (addr & 0xFFFFFFFF)


def _f32_bits(scale: float) -> int:
    import struct
    return struct.unpack("<I", struct.pack("<f", float(scale)))[0]


def _parse_groups(cb: dict):
    """Parse the cb into resident-weight GROUPS for the MLIR-faithful path (decoupled from the C-path
    _parse, which is full-i32-only). Returns ``[(weight, k, n, jobs)]`` in resident-pack (command) order,
    each with ``jobs = [(lhs, out, epi, m, out_dtype, scale)]`` — the matmuls that reuse THAT weight.

    Supports MULTIPLE resident weights (a real multi-layer model): the emitter processes the groups
    sequentially — mvin one weight, run its matmuls, mvout — reusing the scratchpad for the next group, so
    a whole model is ONE co-scheduled kernel rather than one kernel per layer. relu + acc_scale epilogues
    and i8/i32 readout; integer requant(shift) is host-side (round-half-up) — NOT a Gemmini readout — so it
    is rejected here in favour of acc_scale."""
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
            if s not in ("relu", "acc_scale"):
                raise CodegenError(f"unsupported epilogue stage {s!r} (have: relu, acc_scale)")
        out_dtype = attrs.get("output_dtype", "i32")
        scale = float(attrs.get("acc_scale", 1.0))
        jobs_by_res[res].append((lhs, commit["operands"]["dst"], epi, m, out_dtype, scale))
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
    isa = _isa()
    groups = _parse_groups(cb)                  # [(weight, k, n, jobs)], jobs: (lhs,out,epi,m,odt,scale)
    weights = [g[0] for g in groups]
    lhss = [j[0] for g in groups for j in g[3]]
    outs = [j[1] for g in groups for j in g[3]]
    args = weights + lhss + outs
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

        for (lhs, out, epi, m, out_dtype, scale) in jobs:
            mp = _ceil_dim(m)
            Mt = mp // DIM
            i8_out = out_dtype == "i8"              # i8 readout = Gemmini float acc_scale + clamp
            acc_act = 1 if "relu" in epi else 0     # config_st acc_act (RELU=1)
            scale_bits = _f32_bits(scale) if i8_out else F1
            elt = 1 if i8_out else 4               # output element bytes
            read_base = ACC_I8 if i8_out else C_ACC  # i8 readout drops the full_C bit -> scale applies
            # config_st: RS1=(acc_act<<2)|CONFIG_ST ; RS2=(acc_scale_bits<<32)|out_row_stride_bytes
            rocc(K_CONFIG, konst((acc_act << 2) | 2), konst((scale_bits << 32) | (np_ * elt)))
            for mi in range(Mt):
                if panel_resident:
                    # Row-panel mvin: the Kt activation tiles of row mi, into Kt distinct slots, ONCE.
                    for kt in range(Kt):
                        a_off = ((mi * DIM) * kp + kt * DIM)
                        rocc(K_MVIN, addr(lhs, a_off), konst(_pack(a_slot + kt * DIM)))
                for nj in range(Nt):
                    for kt in range(Kt):
                        if panel_resident:
                            a_addr = a_slot + kt * DIM              # reuse the resident panel tile
                        else:
                            a_off = ((mi * DIM) * kp + kt * DIM)    # legacy: re-mvin per output column
                            rocc(K_MVIN, addr(lhs, a_off), konst(_pack(a_slot)))
                            a_addr = a_slot
                        cad = C_ACC if kt == 0 else (C_ACC | ACC_ACCUM)   # accumulator is always i32
                        rocc(K_PRELOAD, konst(_pack((kt * Nt + nj) * DIM)), konst(_pack(cad)))
                        rocc(K_COMPUTE_PRELOADED, konst(_pack(a_addr)), konst(_pack(GARBAGE)))
                    c_off = ((mi * DIM) * np_ + nj * DIM) * elt
                    rocc(K_MVOUT, addr(out, c_off), konst(_pack(read_base)))

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


def _harness_c(cb: dict, inputs: dict | None = None) -> str:
    """Thin C harness: embed padded leaf data, call the MLIR kernel, print outputs (cropped).

    ``inputs`` (name -> nested-list) INJECTS explicit operand values so the device runs the model's real
    activations/weights; absent, each leaf is materialized deterministically from its name (reproducible)."""
    groups = _parse_groups(cb)                  # [(weight, k, n, jobs)]
    leaves = materialize_inputs(cb, inputs)
    weights = [g[0] for g in groups]
    lhss = [j[0] for g in groups for j in g[3]]
    # (out_name, m, n, out_dtype) — n rides along so the printed column count is the group's own n.
    outs = [(j[1], j[3], g[2], j[4]) for g in groups for j in g[3]]

    decls = []
    for weight, k, n, jobs in groups:
        kp, np_ = _ceil_dim(k), _ceil_dim(n)
        wpad = _pad_rowmajor(list(leaves[weight].data), k, n, kp, np_)
        decls.append(f"static const elem_t T_{weight}[{kp * np_}] row_align(1) = "
                     f"{{{','.join(str(int(v)) for v in wpad)}}};")
        for lhs, _, _, m, _, _ in jobs:
            mp = _ceil_dim(m)
            ap = _pad_rowmajor(list(leaves[lhs].data), m, k, mp, kp)
            decls.append(f"static const elem_t T_{lhs}[{mp * kp}] row_align(1) = "
                         f"{{{','.join(str(int(v)) for v in ap)}}};")
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

    args = weights + lhss + [o[0] for o in outs]   # must mirror emit_kernel_mlir: weights + lhss + outs
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
    return ("#include <stdint.h>\n#include <stdio.h>\n#include \"include/gemmini_testutils.h\"\n"
            "extern void gemmini_kernel();\n" + "\n".join(decls) + "\nint main() {\n"
            "  uint64_t c0 = read_cycles();\n"
            f"  gemmini_kernel({call});\n  gemmini_fence();\n"
            "  uint64_t c1 = read_cycles();\n"
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

    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="gemmini_mlir_run_"))
    work.mkdir(parents=True, exist_ok=True)
    obj = build_object(cb, work)
    (work / "harness.c").write_text(_harness_c(cb), encoding="utf-8")
    rt, common = gem.rocc_tests_dir(), gem._common_dir()
    elf = work / "gemmini_mlir.elf"
    cmd = [str(gem.gcc_path()), "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-mcmodel=medany",
           "-std=gnu99", "-O2", "-ffast-math", "-fno-common", "-fno-builtin-printf",
           "-fno-tree-loop-distribute-patterns", "-march=rv64gc", "-Wa,-march=rv64gc",
           "-lm", "-lgcc", "-I", str(rt / "riscv-tests"), "-I", str(rt / "riscv-tests/env"),
           "-I", str(rt), "-I", str(common), "-DID_STRING=", "-DPRINT_TILE=0",
           "-nostdlib", "-nostartfiles", "-static", "-T", str(common / "test.ld"), "-DBAREMETAL=1",
           str(work / "harness.c"), str(obj), "-o", str(elf),
           *(str(p) for p in sorted(common.glob("*.c"))),
           *(str(p) for p in sorted(common.glob("*.S")))]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise gem.GemminiError(f"link failed:\n{proc.stderr[-2000:]}")
    console = gem.run_elf(elf, simulator=simulator, timeout=timeout)
    outputs, raw = gem.parse_output(console)
    return {"outputs": outputs, "correct": outputs_match(outputs, reference_outputs(cb)),
            "metrics": {"cycles": raw.get("cycles", 0)}, "path": "mlir_inline_asm_rocc",
            "oracle": gem.ORACLE[simulator], "elf": str(elf), "console": console}

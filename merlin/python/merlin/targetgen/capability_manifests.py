"""Capability manifests for the two vertical-slice targets: ``rvv`` and ``mx_gemmini``.

These are target **definitions** (a `target_contract.yaml` with `compute_units`) generated as `out/`
artifacts and plugged into the routing tooling — the same shape a real out-of-tree target
(radiance-mlir) will ship. They exist to prove datatype -> compute-unit routing across two very
different targets:

- ``rvv`` (SpacemiT K1 vector unit): the regular formats fp32/fp16/bf16 + int8 — no fp4/fp6/native-fp8
  datapath, so those route to an honest gap.
- ``mx_gemmini`` (microscaling systolic PE): mxfp4/mxfp6/mxfp8 + int8/bf16 with an E8M0 block-scale
  requant — accepts the low-bit formats RVV rejects.

Both are provenance-tagged prototypes flagged ``requires_human_review`` — the numbers/modes trace to a
source but are NOT RTL-certified. The requant reference is opaque (target-specific lowering lives
elsewhere: RVV's software requant in Merlin; gemmini-mx's MxRequantizer out-of-tree).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from merlin.common import schemas as _schemas
from merlin.common.yaml import write_yaml
from merlin.targetgen import compute_units as _cu
from merlin.targetgen import target_registry as _tr

_COMMON: dict[str, Any] = {
    "version": "0.1",
    "status": "prototype",
    "requires_human_review": True,
}


def rvv_manifest() -> dict[str, Any]:
    """K1/RVV vector unit — regular floats + int8 (fp8 is decode-to-f32, not a native format)."""
    return {
        **_COMMON,
        "name": "rvv",
        "family": "cpu_vector",
        "provenance": "SpacemiT K1 X60 rv64gcv (VLEN=256); RVV datapaths per merlin.rvvgen.k1 + "
                      "llvmlower.passes_quant_int (int8 W8A8 vwmacc) + passes_xdsl.lower_bf16_matmul_f32acc.",
        "features": ["vector_unit"],
        "capabilities": {"ops": ["matmul", "elementwise"]},
        "memory_model": {"resident": False},
        "compiler_obligations": [],
        "hardware_promises": ["ieee_float", "widening_integer_accumulate"],
        "runtime_promises": ["metrics"],
        "legality": [],
        "runtime": {"default_backend": "baremetal"},
        "compute_units": [
            {
                "name": "vector",
                "kind": "vector",
                "dtypes": ["fp32", "fp16", "bf16", "int8"],
                "ops": ["matmul", "elementwise"],
                "accumulate": [
                    {"in": "fp32", "weight": "fp32", "acc": "f32"},
                    {"in": "fp16", "weight": "fp16", "acc": "f32"},
                    {"in": "bf16", "weight": "bf16", "acc": "f32"},
                    {"in": "int8", "weight": "int8", "acc": "i32"},
                ],
                "scaling": "per_channel",
                # RVV requant is OUR software lowering (scale-multiply), not a hardware unit.
                "requant": {"ref": "merlin.llvmlower.passes_quant_int"},
            }
        ],
    }


def mx_gemmini_manifest() -> dict[str, Any]:
    """Gemmini-MX microscaling systolic PE — fp4/fp6/fp8 (MX, E8M0 block scale) + int8/bf16."""
    return {
        **_COMMON,
        "name": "mx_gemmini",
        "family": "tensor_resident",
        "provenance": "chipyard generators/gemmini (branch gemmini-mx) MxParameters.scala: "
                      "mxGemminiConfig=[mode0(fp4/fp4), mode4(fp6/fp6), mode8(fp8/fp8)]; block scale via "
                      "ScaleFactorMem + MxExp (E8M0) + QuantLut. TypeSupport permits independent act/wei "
                      "modes (not enumerated here — shipped config is symmetric).",
        "features": ["systolic_array", "microscaling", "resident_packed_tensor", "accumulator_commit"],
        "capabilities": {"ops": ["matmul"], "mesh": {"rows": 16, "cols": 16}},
        "memory_model": {"resident": True, "accumulators": True, "block_scale_memory": True},
        "compiler_obligations": ["must_tile_to_mesh_shape", "must_supply_e8m0_block_scales"],
        "hardware_promises": ["microscaling_block_exponent", "widening_accumulate"],
        "runtime_promises": ["command_buffer", "metrics"],
        "legality": ["mx block-scale granularity == 32"],
        "runtime": {"default_backend": "firesim"},
        "compute_units": [
            {
                "name": "mx_pe",
                "kind": "systolic",
                "dtypes": ["mxfp4", "mxfp6", "mxfp8", "int8", "bf16"],
                "ops": ["matmul"],
                "accumulate": [
                    {"in": "mxfp4", "weight": "mxfp4", "acc": "f32"},   # PE_MxMode mode0
                    {"in": "mxfp6", "weight": "mxfp6", "acc": "f32"},   # mode4
                    {"in": "mxfp8", "weight": "mxfp8", "acc": "f32"},   # mode8
                    {"in": "int8", "weight": "int8", "acc": "i32"},
                ],
                "scaling": "block_e8m0",
                # gemmini-mx requant (MxRequantizer + E8M0 exponent add + QuantLut) is target-specific,
                # out-of-tree — Merlin only references it.
                "requant": {"ref": "mx_gemmini.mx_requantizer"},
            }
        ],
    }


def radiance_manifest() -> dict[str, Any]:
    """Radiance (Muon SIMT tensor core) — regular fp32/fp16/bf16, COMPOSING the gemmini-mx MX PE.

    Demonstrates the composition the hardware allows: gemmini-mx is a compute unit that can stand
    alone (see :func:`mx_gemmini_manifest`) OR be embedded inside a radiance cluster. Here the SIMT
    cluster ``contains`` the ``mx_pe`` unit, so the target's effective capability is the union: regular
    half/single floats on the SIMT lanes + the low-bit MX formats on the embedded PE. Carries a
    ``plugin`` block for the out-of-tree discovery seam (the real dialect + lowering live in a
    radiance-mlir repo; this artifact is the first materialization under out/artifacts/targets).
    """
    mx_pe = mx_gemmini_manifest()["compute_units"][0]   # reuse the exact gemmini-mx PE (composition)
    return {
        **_COMMON,
        "name": "radiance",
        "family": "simt_tensor",
        "provenance": "chipyard generators/radiance (Muon SIMT): CVFPU FP32/FP16/BF16 + INT32 lanes "
                      "(muon backend/fp/FPU.scala; muon_introspect fp_datapath); tensor-core matmul via "
                      "nu.invoke. Embeds the gemmini-mx MX PE as a contained low-bit unit. Not RTL-certified.",
        "features": ["simt", "cvfpu", "tensor_core", "microscaling"],
        "capabilities": {"ops": ["matmul", "elementwise"], "simt": {"lanes_per_warp": 16}},
        "memory_model": {"resident": True, "shared_memory": True},
        "compiler_obligations": ["must_map_to_warps"],
        "hardware_promises": ["ieee_float", "fp32_accumulate"],
        "runtime_promises": ["metrics"],
        "legality": [],
        "runtime": {"default_backend": "simulator"},
        "plugin": {
            "dialect_module": "radiance_mlir.dialect",
            "lowering_entrypoint": "radiance_mlir.lowering:lower",
        },
        "compute_units": [
            {
                "name": "simt_cluster",
                "kind": "simt",
                "dtypes": ["fp32", "fp16", "bf16"],
                "ops": ["matmul", "elementwise"],
                "accumulate": [
                    {"in": "fp32", "weight": "fp32", "acc": "f32"},
                    {"in": "fp16", "weight": "fp16", "acc": "f32"},
                    {"in": "bf16", "weight": "bf16", "acc": "f32"},
                ],
                "scaling": "none",
                "requant": {"ref": "none"},
                "contains": ["mx_pe"],   # gemmini-mx embedded as a sub-unit
            },
            mx_pe,
        ],
    }


def atlas_manifest() -> dict[str, Any]:
    """ATLAS NPU systolic MXU — a 32x32 FP8 (E4M3) systolic array accumulating in BF16 (or FP32),
    with an E8M0 block-scale requant. Facts DERIVED from the atlas-npu chipyard generator RTL config
    (frozen MxuParams formats + PEArchitecture set) and confirmed by mlc arc discovery (dim=32).
    """
    return {
        **_COMMON,
        "name": "atlas",
        "family": "tensor_resident",
        "provenance": "atlas-npu chipyard generator (tmp/dse/atlas-npu) src/main/scala/atlas: "
                      "MxuParams.scala FROZEN formats — inputFmt=E4M3 (fp8, activation+weight), "
                      "accumFmt=outputFmt=BF16; 32x32 systolic mesh (MxuParams arrayRows=arrayCols=32, "
                      "SystolicArrayParams/InnerProductTreeParams isOriginalConfig 32/32/32). "
                      "PEArchitecture={HardFloatFMA (BF16 FMA), FP32Addition (BF16 mul, FP32 add), "
                      "CustomFMA=default ((FP8xFP8)+BF16)}; E8M0 block scale via FPUtils.BF16ScaleToE4M3 "
                      "+ ScalingFactorRegFile. mlc arc discovery confirms dim=32 and a 42-entry legal "
                      "opcode set (arc_available=True). NOT RTL-certified; mesh/dtype facts from the "
                      "generator config, not silicon.",
        "features": ["systolic_array", "fp8_mxu", "microscaling", "bf16_accumulate"],
        "capabilities": {"ops": ["matmul"], "mesh": {"rows": 32, "cols": 32}},
        "memory_model": {"resident": True, "accumulators": True, "block_scale_memory": True},
        "compiler_obligations": ["must_tile_to_mesh_shape", "must_supply_e8m0_block_scales"],
        "hardware_promises": ["fp8_multiply", "bf16_accumulate"],
        "runtime_promises": ["metrics"],
        "legality": ["matmul tiles to the 32x32 mesh"],
        "runtime": {"default_backend": "simulator"},
        "compute_units": [
            {
                "name": "mxu",
                "kind": "systolic",
                # inputFmt is the frozen E4M3 activation+weight format; BF16 is the accum/output element
                # format (and the HardFloat FMA internal precision) — both from MxuParams.scala.
                "dtypes": ["fp8_e4m3", "bf16"],
                "ops": ["matmul"],
                "accumulate": [
                    {"in": "fp8_e4m3", "weight": "fp8_e4m3", "acc": "bf16"},  # CustomFMA(default)/HardFloatFMA
                    {"in": "fp8_e4m3", "weight": "fp8_e4m3", "acc": "f32"},   # FP32Addition PE variant
                ],
                "scaling": "block_e8m0",
                # atlas requant (FPUtils BF16<->E4M3 + E8M0 shared block scale via ScalingFactorRegFile)
                # is target-specific, out-of-tree — Merlin only references it.
                "requant": {"ref": "atlas.e4m3_block_requant"},
            }
        ],
    }


MANIFESTS = {"rvv": rvv_manifest, "mx_gemmini": mx_gemmini_manifest,
             "radiance": radiance_manifest, "atlas": atlas_manifest}


def validate(manifest: dict[str, Any]) -> dict[str, Any]:
    """Schema-validate the contract and parse its compute_units (raises on any problem)."""
    _schemas.validate_or_raise(manifest, "target_contract")
    _cu.compute_units(manifest)   # validates kinds/dtypes/scaling
    return manifest


def write(name: str, base: Path | None = None) -> Path:
    """Write a manifest to ``<base or resolve(name).base>/contracts/target_contract.yaml``."""
    manifest = validate(MANIFESTS[name]())
    root = base if base is not None else _tr.resolve(name).base
    path = root / "contracts" / "target_contract.yaml"
    write_yaml(path, manifest, header=f"GENERATED capability manifest for {name} "
                                       "(merlin.targetgen.capability_manifests). Provenance-tagged; "
                                       "requires_human_review. Regenerable.")
    return path


def write_all(base_root: Path | None = None) -> list[Path]:
    """Write all manifests (rvv, mx_gemmini, radiance). ``base_root`` overrides the per-target base dir."""
    return [write(n, base=(base_root / n) if base_root is not None else None) for n in MANIFESTS]


def dialect_plan_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Derive a schema-valid dialect_plan from a manifest's compute_units (ops + a type per unit).

    The generated dialect exposes one ``!<target>.<unit>_tensor`` type per compute unit and the union
    of the units' ops — the machine-readable dialect spec the out-of-tree repo formalizes into real
    xDSL/MLIR. Types/ops are DERIVED from the capability model, not invented.
    """
    name = manifest["name"]
    units = _cu.compute_units(manifest)
    ops = sorted({op for u in units for op in u.ops})
    types = [{"name": f"{u.name}_tensor"} for u in units]
    return {
        "target": name,
        "dialect_name": name.replace("_", ""),
        "ops": [{"name": op} for op in ops],
        "types": types,
        "lowering": [{"op": op, "to": f"{name.replace('_', '')}.{op}"} for op in ops],
        "tests": [],
    }


def write_oot_target(name: str, root: Path) -> Path:
    """Materialize a complete out-of-tree target package at ``root`` (discoverable via MERLIN_TARGET_PATH).

    Writes ``contracts/target_contract.yaml`` (capability manifest + plugin block),
    ``contracts/dialect_plan.yaml`` (derived from the compute units), and an ``AGENT.md`` describing the
    compute units / datatypes — the first materialization of the out-of-tree target repo (e.g. the
    radiance-mlir repo will host the real dialect + lowering the plugin block references).
    """
    manifest = validate(MANIFESTS[name]())
    root = Path(root)
    write_yaml(root / "contracts" / "target_contract.yaml", manifest,
               header=f"GENERATED out-of-tree target manifest for {name} — plug in via MERLIN_TARGET_PATH.")
    write_yaml(root / "contracts" / "dialect_plan.yaml", dialect_plan_from_manifest(manifest),
               header=f"GENERATED dialect plan for {name} (derived from compute_units).")
    units = _cu.compute_units(manifest)
    lines = [f"# {name} — out-of-tree target package", "",
             f"Generated by `merlin.targetgen.capability_manifests` (provenance: "
             f"{manifest.get('provenance', 'n/a')}).", "",
             "Plug in: `MERLIN_TARGET_PATH=<this dir>`; the dialect + lowering the `plugin` block names "
             "live in the out-of-tree repo (e.g. radiance-mlir).", "",
             "## Compute units (datatype -> unit -> op)"]
    for u in units:
        eff = _cu.effective(u, units)
        lines.append(f"- **{u.name}** ({u.kind}): dtypes {sorted(eff.dtypes)}; ops {sorted(eff.ops)}; "
                     f"scaling {u.scaling}; requant {u.requant}"
                     + (f"; contains {list(u.contains)}" if u.contains else ""))
    (root / "AGENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root

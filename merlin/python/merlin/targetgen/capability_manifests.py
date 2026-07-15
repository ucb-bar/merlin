"""Capability manifests for the two vertical-slice targets: ``rvv`` and ``gemmini_mx``.

These are target **definitions** (a `target_contract.yaml` with `compute_units`) generated as `out/`
artifacts and plugged into the routing tooling — the same shape a real out-of-tree target
(radiance-mlir) will ship. They exist to prove datatype -> compute-unit routing across two very
different targets:

- ``rvv`` (SpacemiT K1 vector unit): the regular formats fp32/fp16/bf16 + int8 — no fp4/fp6/native-fp8
  datapath, so those route to an honest gap.
- ``gemmini_mx`` (microscaling systolic PE): mxfp4/mxfp6/mxfp8 + int8/bf16 with an E8M0 block-scale
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


def gemmini_mx_manifest() -> dict[str, Any]:
    """Gemmini-MX microscaling systolic PE — fp4/fp6/fp8 (MX, E8M0 block scale) + int8/bf16."""
    return {
        **_COMMON,
        "name": "gemmini_mx",
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
                "requant": {"ref": "gemmini_mx.mx_requantizer"},
            }
        ],
    }


MANIFESTS = {"rvv": rvv_manifest, "gemmini_mx": gemmini_mx_manifest}


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
    """Write both slice manifests (rvv, gemmini_mx). ``base_root`` overrides the per-target base dir."""
    return [write(n, base=(base_root / n) if base_root is not None else None) for n in MANIFESTS]

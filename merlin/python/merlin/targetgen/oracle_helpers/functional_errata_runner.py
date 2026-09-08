#!/usr/bin/env python3
"""Run a model-owned functional oracle with reviewed ISA corrections applied.

The target's shipped ``isa_definition.py`` is evidence and deliberately remains
read-only.  When RTL establishes that one of its fixed instruction fields is
wrong, Merlin records the adjudication in ``merlin/contract/isa_errata.yaml``.
This helper runs inside the target model's virtualenv, applies only those
reviewed corrections to the live instruction classes, and then invokes the
model-owned functional runner unchanged.

The correction payload contains complete declared and hardware words.  Fixed
field values are therefore derived from those words; this module contains no
target opcode or funct value.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType


# Positions in the 32-bit RISC-V instruction format.  A field is patched only
# when the class currently carries the value extracted from the declared word.
_FIXED_FIELD_SPANS = {
    "opcode": (0, 7),   # derived-ok: RISC-V base instruction format, opcode = inst[6:0]
    "funct3": (12, 3),  # derived-ok: RISC-V base instruction format, funct3 = inst[14:12]
    "funct7": (25, 7),  # derived-ok: RISC-V base instruction format, funct7 = inst[31:25]
}


def _field(word: int, lo: int, width: int) -> int:
    return (word >> lo) & ((1 << width) - 1)


def apply_reviewed_errata(
    isa_module: ModuleType,
    corrections: dict[str, dict],
) -> list[dict]:
    """Patch live ISA classes, failing closed on stale or malformed evidence."""
    applied: list[dict] = []
    for class_name, correction in sorted(corrections.items()):
        if str(correction.get("authoritative", "")).lower() != "rtl":
            continue
        try:
            declared = int(str(correction["declared"]), 0)
            hardware = int(str(correction["hardware"]), 0)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{class_name}: malformed reviewed ISA correction") from exc
        cls = getattr(isa_module, class_name, None)
        if cls is None:
            raise ValueError(f"{class_name}: reviewed ISA correction names no model class")

        changed: dict[str, dict[str, int]] = {}
        for field_name, (lo, width) in _FIXED_FIELD_SPANS.items():
            old = _field(declared, lo, width)
            new = _field(hardware, lo, width)
            if old == new:
                continue
            current = getattr(cls, field_name, None)
            if current is None or int(current) != old:
                raise ValueError(
                    f"{class_name}.{field_name}: model value {current!r} does not match "
                    f"reviewed declared value {old}"
                )
            setattr(cls, field_name, new)
            changed[field_name] = {"declared": old, "hardware": new}

        if declared != hardware and not changed:
            raise ValueError(
                f"{class_name}: correction changes no supported fixed instruction field"
            )
        if changed:
            applied.append({
                "class": class_name,
                "declared": f"0x{declared:08x}",
                "hardware": f"0x{hardware:08x}",
                "fields": changed,
                "sources_against_spec": list(
                    correction.get("sources_against_spec") or []
                ),
            })
    return applied


def _wrap_vmem_base_unit(cls, unit_bytes: int) -> None:
    original_exec = cls.exec

    def exec_with_address_unit(self, state):
        original_read = state.read_vmem
        original_write = state.write_vmem
        state.read_vmem = lambda base, offset, length: original_read(
            int(base) * unit_bytes, offset, length
        )
        state.write_vmem = lambda base, offset, data: original_write(
            int(base) * unit_bytes, offset, data
        )
        try:
            return original_exec(self, state)
        finally:
            state.read_vmem = original_read
            state.write_vmem = original_write

    cls.exec = exec_with_address_unit


def _wrap_e8m0_scale(cls, bias: int, minimum: int, maximum: int) -> None:
    original_exec = cls.exec

    def exec_with_scale(self, state):
        original_read = state.read_erf

        def read_scale(register):
            code = int(original_read(register)) & 0xFF
            exponent = min(max(code - bias, minimum), maximum)
            # The shipped methods multiply while packing and divide while
            # unpacking.  Returning the reciprocal power-of-two scale makes
            # both operations match RTL's exponent subtraction/addition.
            return math.ldexp(1.0, -exponent)

        state.read_erf = read_scale
        try:
            return original_exec(self, state)
        finally:
            state.read_erf = original_read

    cls.exec = exec_with_scale


def _wrap_weight_buffer_lane_major(cls) -> None:
    """Present an RTL lane-major weight buffer to matmul as reduction x output."""
    original_exec = cls.exec

    def exec_with_weight_view(self, state):
        original_read = state.read_wb_fp8

        def read_weight_for_matmul(unit, slot):
            lane_major = original_read(unit, slot)
            if getattr(lane_major, "ndim", None) != 2:
                raise ValueError("weight-buffer matmul view must be rank two")
            reduction_major = lane_major.swapaxes(0, 1)
            contiguous = getattr(reduction_major, "contiguous", None)
            return contiguous() if callable(contiguous) else reduction_major.copy()

        state.read_wb_fp8 = read_weight_for_matmul
        try:
            return original_exec(self, state)
        finally:
            state.read_wb_fp8 = original_read

    cls.exec = exec_with_weight_view


def apply_reviewed_model_errata(
    isa_module: ModuleType,
    corrections: dict[str, dict],
) -> list[dict]:
    """Apply reviewed functional-semantic corrections to model classes."""
    applied: list[dict] = []
    for name, correction in sorted(corrections.items()):
        if str(correction.get("authoritative", "")).lower() != "rtl":
            continue
        kind = correction.get("correction")
        class_names = list(correction.get("model_classes") or [])
        if not class_names:
            raise ValueError(f"{name}: functional-model correction names no classes")
        classes = []
        for class_name in class_names:
            cls = getattr(isa_module, class_name, None)
            if cls is None or not callable(getattr(cls, "exec", None)):
                raise ValueError(
                    f"{name}: functional-model class {class_name!r} is absent or has no exec"
                )
            classes.append(cls)

        parameters: dict[str, int | str]
        if kind == "vmem_base_unit_bytes":
            declared = int(correction.get("declared_unit_bytes", 0))
            hardware = int(correction.get("hardware_unit_bytes", 0))
            if declared != 1 or hardware <= 0:
                raise ValueError(f"{name}: invalid VMEM address-unit review")
            for cls in classes:
                _wrap_vmem_base_unit(cls, hardware)
            parameters = {"declared_unit_bytes": declared,
                          "hardware_unit_bytes": hardware}
        elif kind == "e8m0_biased_exponent":
            bias = int(correction["exponent_bias"])
            minimum = int(correction["exponent_min"])
            maximum = int(correction["exponent_max"])
            if not (0 <= bias <= 255 and minimum <= maximum):
                raise ValueError(f"{name}: invalid E8M0 exponent review")
            for cls in classes:
                _wrap_e8m0_scale(cls, bias, minimum, maximum)
            parameters = {"exponent_bias": bias, "exponent_min": minimum,
                          "exponent_max": maximum}
        elif kind == "weight_buffer_output_lane_major":
            declared = str(correction.get("declared_matmul_view") or "")
            hardware = str(correction.get("hardware_matmul_view") or "")
            if declared != "output_lane_by_reduction" or hardware != "reduction_by_output_lane":
                raise ValueError(f"{name}: invalid weight-buffer layout review")
            for cls in classes:
                _wrap_weight_buffer_lane_major(cls)
            parameters = {"declared_matmul_view": declared,
                          "hardware_matmul_view": hardware}
        else:
            raise ValueError(f"{name}: unsupported functional-model correction {kind!r}")
        applied.append({
            "name": name,
            "correction": kind,
            "model_classes": class_names,
            "parameters": parameters,
            "sources_against_model": list(
                correction.get("sources_against_model") or []
            ),
        })
    return applied


def _load_runner(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_merlin_model_functional_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load functional runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    request = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    corrections = request.pop("reviewed_isa_errata", {})
    model_corrections = request.pop("reviewed_functional_model_errata", {})
    isa_module_name = request.pop("functional_model_isa_module", "")
    Path(args.infile).write_text(json.dumps(request), encoding="utf-8")

    if not isa_module_name:
        raise ValueError("functional-model ISA module was not derived from the target contract")
    isa_definition = importlib.import_module(isa_module_name)

    applied = apply_reviewed_errata(isa_definition, corrections)
    model_applied = apply_reviewed_model_errata(isa_definition, model_corrections)
    runner = _load_runner(Path(args.runner))
    if not hasattr(runner, "_cli"):
        raise RuntimeError(f"functional runner exposes no _cli entry point: {args.runner}")
    rc = int(runner._cli(["--in", args.infile, "--out", args.out]))
    if rc == 0:
        result_path = Path(args.out)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["reviewed_isa_errata_applied"] = applied
        result["reviewed_functional_model_errata_applied"] = model_applied
        result["isa_errata_policy"] = "reviewed RTL-authoritative runtime overlay"
        result_path.write_text(json.dumps(result), encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())

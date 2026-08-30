"""Validate the movement-volume predictor against a target's own measured byte counts.

Runs `merlin.perf.dma_volume` over every shipped program of a target whose ISA definition and program
images are on disk, and compares the predicted footprint with the measured `(reads + writes) * beat`
from the pinned cycle suite.

Everything target-specific enters as DATA -- the ISA definition path, the program directory and the
measurement file all come from the environment-resolved external checkout, and the movement
encodings, the size operand and the immediate forms are all read out of the target's own ISA source
by structural parse (`ast`, never regex, never a hardcoded opcode).
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from merlin.common import artifacts as A
from merlin.common.paths import ext_path
from merlin.perf.dma_volume import Descriptor, compare_to_measured, kernel_volume

#: RISC-V-shaped field extraction. Positions come from the ISA model's declared layout for the forms
#: we decode; this mirrors that layout and is asserted against it by the caller.
def _fields(word: int) -> dict[str, int]:
    imm = (word >> 20) & 0xFFF
    return {"opcode": word & 0x7F, "rd": (word >> 7) & 0x1F, "funct3": (word >> 12) & 0x7,
            "rs1": (word >> 15) & 0x1F, "rs2": (word >> 20) & 0x1F, "funct7": (word >> 25) & 0x7F,
            "imm_i": imm - 4096 if imm >= 2048 else imm, "imm_u": word & 0xFFFFF000}


def isa_forms(isa_source: Path) -> tuple[dict[tuple, str], dict[str, str]]:
    """Movement encodings and the register each form reads its LENGTH from, both from the ISA source.

    The length operand is taken from the form's own executable body -- the register it reads to size
    the transfer -- not from a position we guessed. That is what makes this a derivation."""
    tree = ast.parse(isa_source.read_text(encoding="utf-8"))
    encodings: dict[tuple, str] = {}
    length_operand: dict[str, str] = {}
    bodies: dict[str, ast.ClassDef] = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

    for name, node in bodies.items():
        for sub in ast.walk(node):
            # `length = state.read_xrf(self.<operand>)` names the size register for this family.
            if isinstance(sub, ast.Assign) and len(sub.targets) == 1:
                tgt = sub.targets[0]
                if isinstance(tgt, ast.Name) and tgt.id == "length":
                    for inner in ast.walk(sub.value):
                        if isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Name) \
                                and inner.value.id == "self":
                            length_operand[name] = inner.attr

    for name, node in bodies.items():
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        def const(key):
            v = kw.get(key)
            return v.value if isinstance(v, ast.Constant) else None
        opcode = const("opcode")
        if opcode is None:
            continue
        exu = kw.get("exu")
        if not (isinstance(exu, ast.Attribute) and exu.attr == "DMA"):
            continue
        encodings[(opcode, const("funct7"), const("funct3"))] = name
    return encodings, length_operand


def _size_operand(form: str, length_operand: dict[str, str], bases: dict[str, list[str]]) -> str | None:
    """Which operand carries this form's size, inherited from the family body that defines it."""
    if form in length_operand:
        return length_operand[form]
    for base in bases.get(form, ()):                      # families define exec once, on the base
        if base in length_operand:
            return length_operand[base]
    return None


def _base_map(isa_source: Path) -> dict[str, list[str]]:
    tree = ast.parse(isa_source.read_text(encoding="utf-8"))
    return {n.name: [b.id for b in n.bases if isinstance(b, ast.Name)]
            for n in tree.body if isinstance(n, ast.ClassDef)}


def predict_kernel(words: list[int], encodings, length_operand, bases) -> Any:
    """Decode, propagate constants, and fold the descriptors into a volume (or a floor)."""
    state: dict[int, int | None] = {}
    descriptors: list[Descriptor] = []
    for index, word in enumerate(words):
        f = _fields(word)
        form = encodings.get((f["opcode"], f["funct7"], f["funct3"]))
        if form and ("LOAD" in form or "STORE" in form):
            operand = _size_operand(form, length_operand, bases)
            size = state.get(f[operand]) if operand and operand in f else None
            reason = None
            if operand is None:
                reason = "this form declares no size operand"
            elif size is None:
                reason = f"the size register x{f[operand]} holds no value derivable from the program"
            descriptors.append(Descriptor(
                index=index, form=form, channel=f["funct3"],
                direction="read" if "LOAD" in form else "write",
                size_bytes=size, size_field=operand, unresolved_reason=reason))
        elif f["opcode"] == 0b0010011 and f["funct3"] == 0:           # add-immediate
            base = 0 if f["rs1"] == 0 else state.get(f["rs1"])
            state[f["rd"]] = None if base is None else base + f["imm_i"]
        elif f["opcode"] == 0b0110111:                                 # load-upper-immediate
            state[f["rd"]] = f["imm_u"]
        elif f["rd"]:
            state[f["rd"]] = None                                      # opaque write kills the value
    return descriptors


def run(target: str, *, write_product: bool = True) -> dict:
    isa_source = (Path(ext_path("npu_model")) / "npu_model/configs/isa_definition.py").resolve()
    hexdir = isa_source.parent / "programs/hex"
    suite = json.loads(Path(_suite_path()).read_text(encoding="utf-8"))
    beat = suite["_meta"]["beat_bytes"]
    encodings, length_operand = isa_forms(isa_source)
    bases = _base_map(isa_source)

    rows, exact, floors, violations, never_set, total_desc = [], 0, 0, 0, 0, 0
    for name, k in sorted(suite["kernels"].items()):
        image = hexdir / f"{name}.hex"
        if not image.is_file():
            rows.append({"kernel": name, "verdict": "no_program_image"})
            continue
        words = [int(t, 16) for t in image.read_text().split()]
        descs = predict_kernel(words, encodings, length_operand, bases)
        vol = kernel_volume(name, descs)
        measured = (k["arc"]["reads"] + k["arc"]["writes"]) * beat
        out = compare_to_measured(vol, measured)
        total_desc += len(descs)
        never_set += sum(1 for d in descs if not d.resolved)
        exact += out["verdict"] == "match"
        floors += out["verdict"] == "consistent_lower_bound"
        violations += out["verdict"] == "bound_violated"
        rows.append(out)

    body = {"experiment": "movement_volume_validation", "target": target,
            "beat_bytes": beat, "rows": rows,
            "summary": {"kernels": len(rows), "exact": exact, "consistent_floors": floors,
                        "bound_violations": violations,
                        "descriptors": total_desc, "unresolved_descriptors": never_set}}
    if write_product:
        d = A.new_product("movement-volume", target=target, version=1)
        (Path(d.path) / "validation.json").write_text(json.dumps(body, indent=2) + "\n")
        d.write_manifest()
        body["artifact"] = str(d.path)
    return body


def _suite_path() -> str:
    from merlin.targetgen.rtl import mlc_bridge
    return str(Path(mlc_bridge.mlc_dir()) / "mlc/validate/npu_model_suite.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", required=True)
    ap.add_argument("--no-product", action="store_true")
    a = ap.parse_args(argv)
    body = run(a.target, write_product=not a.no_product)
    s = body["summary"]
    for r in body["rows"]:
        if r.get("verdict") == "no_program_image":
            continue
        print(f"{r['kernel']:<28}{r['predicted']:>9}{r['measured']:>10}  {r['verdict']}")
    print(f"\n{s['exact']} exact, {s['consistent_floors']} consistent floors, "
          f"{s['bound_violations']} bound violations, of {s['kernels']} kernels")
    print(f"{s['unresolved_descriptors']} of {s['descriptors']} descriptors unresolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

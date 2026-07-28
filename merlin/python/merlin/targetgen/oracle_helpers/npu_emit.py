#!/usr/bin/env python3
"""Model-venv helper for the generic ``external_backend`` program-oracle — runs INSIDE the target model's
own venv (e.g. npu_model's py3.14 venv), NOT merlin's. Its job is the DRAM byte layout that needs the
model's torch dtypes (fp8_e4m3 / bf16), which merlin's venv lacks. It emits a JSON+base64 bundle merlin
reads back across the venv gap:

    {"words": [int32,...],
     "inputs": [{"base": int, "b64": "..."}],
     "output": {"base": int, "shape": [...], "dtype": "bf16", "physical": {...}}|null,
     "golden": {"b64": "...", "shape": [...], "dtype": "bf16"}|null}

Two modes:
  * ``--program NAME`` (self-contained validation): assemble a named ``npu_model`` ``Program`` via the
    MODEL's own assembler + lay out its memory regions/golden. This is the ONLY path that touches the
    model's assembler/ISA — merlin holds no opcode table.
  * ``--inputs JSON`` (the capsule/agent path): lay out the input tensors' DRAM bytes ONLY. The agent's
    ``kernel.S`` is assembled to IMEM words merlin-side by STOCK LLVM (``llvm-mc`` — see
    :mod:`targetgen.program_oracle`), so ``words`` stays empty here and no model assembler is imported.

⚠️ ``--fix-itype-rd`` (``--program`` only) applies the npu_model IType rd-encoding shim: upstream
``IType.to_bytecode`` clobbers ``rd`` with ``imm`` (functional sim hides it; the RTL decoder reads rd
from bits[11:7] → garbage). One-line upstream fix pending in ucb-ee194-tapeout/npu_model; until it lands
this shim keeps the SHARED external tree untouched. Only IType is affected. The capsule path never
assembles here, so it never needs the shim (the agent emits the encoded ``.word``/``.insn`` directly).
"""
from __future__ import annotations

import argparse
import base64
import json
import sys


def _install_itype_shim() -> None:
    from npu_model.isa import IType, _mask  # the MODEL's ISA definition (never a merlin opcode table)

    def _rd_fixed(self):  # correct field packing: rd from self.rd (not self.imm)
        return ((_mask(self.imm, 12) << 20) | (_mask(self.rs1, 5) << 15)
                | (_mask(self.funct3, 3) << 12) | (_mask(getattr(self, "rd", 0), 5) << 7)
                | _mask(self.opcode, 7))
    IType.to_bytecode = _rd_fixed


def _tensor_bytes(t) -> bytes:
    import torch
    # reinterpret to uint8 in torch first — .numpy() rejects fp8/bf16 scalar types.
    return t.flatten().contiguous().view(torch.uint8).numpy().tobytes()


def _layout_inputs(specs: list[dict]) -> list[dict]:
    """Convert [{name,base,dtype,values(nested list)}] to [{base,b64}] DRAM bytes in the model's dtypes."""
    import numpy as np
    import torch
    _NP = {"int8": np.int8, "i8": np.int8}
    out = []
    for spec in specs:
        dt = spec["dtype"]
        vals = np.asarray(spec["values"])
        if dt in ("fp8_e4m3",):
            bts = torch.from_numpy(vals.astype(np.float32)).to(torch.float8_e4m3fn).view(
                torch.uint8).numpy().tobytes()
        elif dt in ("bf16",):
            bts = torch.from_numpy(vals.astype(np.float32)).to(torch.bfloat16).view(
                torch.uint8).numpy().tobytes()
        else:
            bts = vals.astype(_NP.get(dt, np.int8)).tobytes()
        out.append({"base": int(spec["base"]), "b64": base64.b64encode(bts).decode()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--program", help="a Program class in npu_model.configs.programs (self-contained)")
    ap.add_argument("--inputs", help="JSON: [{name,base,dtype,values}] to lay out as DRAM bytes")
    ap.add_argument("--fix-itype-rd", action="store_true", help="apply the IType rd shim (--program only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch  # noqa: F401 — present in the model venv; used for dtype byte layout

    bundle: dict = {"words": [], "inputs": [], "output": None, "golden": None}

    if a.program:
        if a.fix_itype_rd:
            _install_itype_shim()
        import importlib
        mod = importlib.import_module("npu_model.configs.programs")
        prog = getattr(mod, a.program)()
        bundle["words"] = [int(w) & 0xFFFFFFFF for w in prog.assemble()]
        for base, arr in getattr(prog, "memory_regions", []):
            bundle["inputs"].append({"base": int(base),
                                     "b64": base64.b64encode(_tensor_bytes(arr)).decode()})
        g = getattr(prog, "golden_result", None)
        if g is not None:
            gbase, gt = g
            bundle["output"] = {"base": int(gbase), "shape": list(gt.shape), "dtype": str(gt.dtype)}
            bundle["golden"] = {"b64": base64.b64encode(_tensor_bytes(gt)).decode(),
                                "shape": list(gt.shape), "dtype": str(gt.dtype)}
    else:
        # capsule path: lay out the emitted-capsule input tensors ONLY (words come from stock LLVM,
        # merlin-side). No model assembler is imported on this path.
        bundle["inputs"] = _layout_inputs(json.loads(a.inputs or "[]"))

    with open(a.out, "w") as f:
        json.dump(bundle, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())

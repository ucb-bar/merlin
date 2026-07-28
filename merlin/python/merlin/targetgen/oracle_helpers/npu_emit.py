#!/usr/bin/env python3
"""ISA-side helper for the generic ``external_backend`` program-oracle — runs INSIDE the target model's
own venv (e.g. npu_model's py3.14 venv), NOT merlin's. It is the ONLY place the target's ISA is touched,
and it touches it through the MODEL PROJECT's own assembler/ISA definition — merlin hardcodes no opcodes.

Given either a named self-contained ``Program`` (validation) or an emitted ``kernel.S`` + input tensors
(the capsule path), it: assembles to 32-bit IMEM words via the model's assembler, converts each input
tensor to its on-DRAM bytes (dtype layout via torch), and emits a JSON+base64 bundle merlin reads back
across the venv gap:

    {"words": [int32,...],
     "inputs": [{"base": int, "b64": "..."}],
     "output": {"base": int, "shape": [...], "dtype": "bf16", "physical": {...}}|null,
     "golden": {"b64": "...", "shape": [...], "dtype": "bf16"}|null}

⚠️ Applies the npu_model IType rd-encoding shim (``--fix-itype-rd``): upstream ``IType.to_bytecode``
clobbers ``rd`` with ``imm`` (functional sim hides it; the RTL decoder reads rd from bits[11:7] → garbage).
One-line upstream fix pending in ucb-ee194-tapeout/npu_model; until it lands this shim keeps the SHARED
external tree untouched. Only IType is affected.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--program", help="a Program class in npu_model.configs.programs (self-contained)")
    ap.add_argument("--kernel-s", help="path to an emitted atlas-ISA kernel .S")
    ap.add_argument("--inputs", help="JSON: [{name,base,dtype,values(nested list)}] for the .S path")
    ap.add_argument("--fix-itype-rd", action="store_true", help="apply the IType rd-encoding shim")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.fix_itype_rd:
        _install_itype_shim()

    import torch  # noqa: F401 — present in the model venv; used for dtype byte layout

    bundle: dict = {"words": [], "inputs": [], "output": None, "golden": None}

    if a.program:
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
        # capsule path: assemble the agent's emitted kernel.S via the model's own assembler
        import io
        from npu_model.util.converter import input_to_program  # the model's assembler front-end
        # The model's assembler rejects any non-mnemonic line, but a real emitter's kernel.S carries
        # `//` comment headers/annotations. Strip `//` comments + blank lines here (merlin-side, not by
        # patching the external model) so a well-formed commented artifact assembles; the mnemonics
        # themselves still go through the model's OWN ISA (no opcode hardcoded here).
        with open(a.kernel_s) as f:
            _lines = []
            for _ln in f.read().splitlines():
                _cut = _ln.find("//")
                if _cut != -1:
                    _ln = _ln[:_cut]
                if _ln.strip():
                    _lines.append(_ln)
        prog = input_to_program(io.StringIO("\n".join(_lines) + "\n"))
        bundle["words"] = [int(w) & 0xFFFFFFFF for w in prog.assemble()]
        # inputs materialized merlin-side (deterministic capsule leaves); here we only lay out bytes
        import numpy as np
        _NP = {"int8": np.int8, "i8": np.int8, "fp8_e4m3": None, "bf16": None}
        for spec in json.loads(a.inputs or "[]"):
            dt = spec["dtype"]
            vals = np.asarray(spec["values"])
            if dt in ("fp8_e4m3",):
                bts = vals.astype(np.float32)
                bts = torch.from_numpy(bts).to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes()
            elif dt in ("bf16",):
                bts = torch.from_numpy(vals.astype(np.float32)).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
            else:
                bts = vals.astype(_NP.get(dt, np.int8)).tobytes()
            bundle["inputs"].append({"base": int(spec["base"]), "b64": base64.b64encode(bts).decode()})

    with open(a.out, "w") as f:
        json.dump(bundle, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())

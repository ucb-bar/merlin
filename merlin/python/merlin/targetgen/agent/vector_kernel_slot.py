"""Agent-autonomy test: synthesize the RVV vector kernel with LESS hand-holding than Gemmini.

The agent gets the RVV intrinsic *names* (signatures), the command-buffer ABI, the output
contract, and VEC0/VEC1 (elementwise) examples — but NOT the stripmine loop structure and NOT
the reduction algorithm. It must derive `generate_driver` and, in particular, figure out the
reduction codegen for the held-out VEC2 (dot product = mul -> reduce), which the examples do
not demonstrate. Repair loop on the visible rungs; final unbiased cert on the held-out rung.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from . import claude_cli
from ..eval.saturn_vec_conformance import build   # target-ok: saturn reference vector-kernel agent flow; pending OV11 eviction
from ...runtime.backends import saturn_vec        # target-ok: saturn reference vector backend; pending OV11 eviction

RVV_ISA = r"""
RVV (rv64gcv) C intrinsics available after `#include <riscv_vector.h>` (LMUL=1, e32):
  size_t __riscv_vsetvl_e32m1(size_t avl);             // returns the granted vector length
  vint32m1_t __riscv_vle32_v_i32m1(const int32_t *p, size_t vl);
  void       __riscv_vse32_v_i32m1(int32_t *p, vint32m1_t v, size_t vl);
  vint32m1_t __riscv_vadd_vv_i32m1(vint32m1_t a, vint32m1_t b, size_t vl);
  vint32m1_t __riscv_vmul_vv_i32m1(vint32m1_t a, vint32m1_t b, size_t vl);
  vint32m1_t __riscv_vmax_vx_i32m1(vint32m1_t a, int32_t s, size_t vl);   // elementwise max(a,s)
  vint32m1_t __riscv_vmv_v_x_i32m1(int32_t s, size_t vl);                 // splat scalar
  vint32m1_t __riscv_vredsum_vs_i32m1_i32m1(vint32m1_t v, vint32m1_t init, size_t vl); // lane-sum -> lane0
  int32_t    __riscv_vmv_x_s_i32m1_i32(vint32m1_t v);                     // extract lane0
The vector length granted by vsetvl may be smaller than the requested count (the hardware
chooses it); a correct kernel must work for any length, not assume the whole vector fits.
""".strip()

OUTPUT_CONTRACT = r"""
Use the bare-metal HTIF harness (`#include "htif.h"`): htif_puts(const char*), htif_putc(char),
htif_putd(long), htif_exit(int). main has signature `int main(long hart)` and only hart 0 works
(others spin). For each output tensor OUT_NAME of length n, print:
  htif_puts("VOUT OUT_NAME n"); then for each element htif_putc(' '); htif_putd((long)val);
  then htif_putc('\n');
End with htif_puts("METRIC cycles "); htif_putd(cycles); htif_putc('\n'); htif_puts("DONE\n");
htif_exit(0);  (bracket cycles with `asm volatile("csrr %0, mcycle":"=r"(c))`).
Embed input tensors as `static int32_t T_<name>[...] = {...};` and produced tensors as buffers.
You are given each tensor's deterministic data.
""".strip()

TASK = r"""
Write a Python function:  def generate_driver(cb: dict, *, mode: str = "explicit") -> str:
It returns bare-metal C (a full program) implementing the command buffer on RVV. The cb:
  cb["tensors"]: {name: {"shape":[n], "dtype":"i32", "role": "input"|"output"}}
  cb["commands"]: ordered list of {"opcode","operands","attributes"}:
    VECTOR_MAP {lhs, rhs, dst} attrs{combine: "add"|"mul", activation: [] or ["relu"]}
        -> dst[i] = activation(combine(lhs[i], rhs[i]))  (elementwise, equal-length 1-D vectors)
    VREDUCE    {src, dst}      attrs{op: "sum"}
        -> dst[0] = sum over all elements of src    (dst is length 1)
Tensors with role "output" are the results to print (VOUT). Get tensor VALUES with:
    from merlin.runtime.commandbuffer import materialize_inputs
    leaves = materialize_inputs(cb)        # {name: Tensor}; list(leaves[name].data) is row-major
HARD RULES (output rejected if violated): do NOT import/call reference_outputs, simulate, or
numpy; do NOT hardcode expected outputs — compute on the vector hardware only. Output ONLY one
```python code block with the function.
""".strip()

CHEAT = ("reference_outputs", "simulate(", "import numpy", "from numpy")


def _example(rung: str) -> str:
    import json
    return f"Example visible command buffer ({rung}):\n{json.dumps(build(rung), indent=1)}"


def build_prompt(visible, feedback):
    parts = ["You are generating a Merlin RVV vector kernel (NON-matmul, SIMD family).",
             "", "## RVV intrinsics", RVV_ISA, "", "## Output contract", OUTPUT_CONTRACT,
             "", "## Task", TASK, "", "## Visible examples (structure only, NOT outputs)"]
    parts += [_example(r) for r in visible]
    if feedback:
        parts += ["", "## Fix this from the previous attempt", feedback]
    return "\n".join(parts)


def _load(code: str, path: Path):
    path.write_text(code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"vk_{path.stem}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.generate_driver


def _certify(rung, gen, tmp):
    cb = build(rung)
    res = saturn_vec.run_command_buffer(cb, workdir=tmp, timeout=180, driver_src=gen(cb))
    return res


def generate_vector_kernel(*, visible=("VEC0", "VEC1"), heldout=("VEC2",), rounds=4,
                           workdir: str | Path, model: str = "opus", timeout: int = 900):
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    feedback = None
    history = []
    for r in range(rounds):
        out = claude_cli.run_agent(build_prompt(visible, feedback), model=model, timeout=timeout,
                                   workdir=work, save_transcript=work / f"t_r{r}.json")
        try:
            code = claude_cli.extract_code_block(out["text"])
        except claude_cli.AgentError as e:
            feedback = f"{e}. Output one ```python block."
            history.append({"r": r, "stage": "nocode"})
            continue
        bad = [t for t in CHEAT if t in code]
        if bad:
            feedback = f"Rejected: forbidden tokens {bad}; compute on hardware only."
            history.append({"r": r, "stage": "cheat", "bad": bad})
            continue
        try:
            gen = _load(code, work / f"vk_r{r}.py")
        except Exception as e:
            feedback = f"import failed: {e!r}"
            history.append({"r": r, "stage": "load", "e": repr(e)})
            continue
        vis, ok = {}, True
        for rung in visible:
            try:
                res = _certify(rung, gen, work / f"run_{rung}_{r}")
            except Exception as e:
                vis[rung] = f"err:{e!r}"
                ok = False
                feedback = f"{rung} compile/run failed: {e!r}"
                break
            vis[rung] = res["correct"]
            if not res["correct"]:
                ok = False
                from ...runtime import reference_outputs
                feedback = (f"{rung}: got {res['outputs']} expected "
                            f"{reference_outputs(build(rung))} — fix the vector logic.")
                break
        history.append({"r": r, "stage": "certify", "visible": vis})
        if not ok:
            continue
        held = {}
        hok = True
        for rung in heldout:
            try:
                res = _certify(rung, gen, work / f"run_{rung}_{r}")
                held[rung] = res["correct"]
                hok = hok and res["correct"]
            except Exception as e:
                held[rung] = f"err:{e!r}"
                hok = False
        return {"success": hok, "round": r, "code_path": str(work / f"vk_r{r}.py"),
                "visible": vis, "heldout": held, "usage": out["usage"], "history": history}
    return {"success": False, "rounds": rounds, "last_feedback": feedback, "history": history}

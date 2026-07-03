"""Kernel slot: agent synthesizes the Gemmini command-buffer -> C kernel, gated by the oracle.

The agent (Claude Code CLI, Opus) is given the command-buffer ABI, a Gemmini ISA reference, an
output contract, and the *structure* of a couple of visible example command buffers — but
NEVER the reference outputs. It proposes a `generate_driver(cb)` Python function. The harness:
  1. structurally scans the code (no peeking at reference/simulate, no hardcoded golden),
  2. iterates the propose->gate->repair loop on the VISIBLE rungs (localized diff feedback),
  3. once visible passes, certifies ONCE on HELD-OUT rungs (shapes the agent never tuned to)
     — that generalization result is the headline.

This is abc-testing's discipline: agent autonomy on visible data, deterministic oracle gate,
held-out certification, structural cheat detection. The agent's semantic claims are never
trusted — only the executable consequences (run on the Gemmini oracle) are checked.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

from . import claude_cli
from ..eval.gemmini_conformance import build
from ...runtime.backends import gemmini

ISA_REFERENCE = r"""
Gemmini (chipyard) bare-metal C, via libgemmini intrinsics. Constants/macros available after
`#include "include/gemmini_testutils.h"`:
  DIM = 16; elem_t = int8 (inputs); acc_t = int32 (accumulator/output); ADDR_LEN = 32.
  gemmini_flush(0);
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);   // dataflow
  gemmini_config_ld(stride_bytes);                          // DRAM row stride for the NEXT mvin(s)
  gemmini_extended_config_st(stride_bytes, acc_act, ACC_SCALE_IDENTITY); // acc_act = NO_ACTIVATION or RELU
  gemmini_mvin((void*)dram_ptr, spad_row);                 // move a DIMxDIM tile DRAM->scratchpad row
  gemmini_preload(weight_spad_row, c_acc_addr);            // set stationary weight tile + output acc
  gemmini_compute_preloaded(act_spad_row, GARBAGE_ADDR);   // C(acc) = A(act) * W(preloaded)
  gemmini_mvout((void*)dram_ptr, c_acc_addr);              // acc tile -> DRAM
  gemmini_fence();
  read_cycles();                                           // rdcycle
Accumulator addressing (full i32 readout):
  overwrite tile: c_ovw = ((3u<<(ADDR_LEN-2))|(1u<<(ADDR_LEN-3))) & ~(1u<<(ADDR_LEN-2));
  accumulate-onto-existing (for K-tiles after the first): c_acc = c_ovw | (1u<<(ADDR_LEN-2));
Notes: matrices are row-major; M/K/N may exceed DIM -> tile into 16x16 blocks and accumulate
over K (overwrite on the first K-tile, accumulate after). Non-multiples of 16 -> zero-pad the
edges and crop on output. RELU (acc_act) applies to the i32 accumulator on mvout.
""".strip()

OUTPUT_CONTRACT = r"""
The generated C `main()` must, for each committed output tensor named OUT_NAME (m x n), print:
  printf("OUT OUT_NAME m n"); then for each element in row-major order printf(" %d",(int)val);
  printf("\n");
and finally:
  printf("METRIC cycles %lu\n", (unsigned long)(c1 - c0));   // c0/c1 bracket the gemmini region
  printf("METRIC cycle_window_gemmini_region 1\n");
  printf("DONE\n");
Embed input tensors as `static const elem_t T_<name>[...] row_align(1) = {...};` and outputs as
`static acc_t T_<name>[...] row_align_acc(1);`. You are given each tensor's deterministic data.
""".strip()

TASK = r"""
Write a Python function with EXACTLY this signature:

    def generate_driver(cb: dict, *, mode: str = "explicit") -> str:

It takes a Merlin command buffer `cb` and returns bare-metal Gemmini C source (a complete
program with main()). The command buffer has:
  cb["tensors"]: {name: {"shape":[r,c], "dtype":"i8", "role":...}}
  cb["commands"]: ordered list of {"opcode","operands","attributes"} with opcodes
    RES_PACK {src,dst}              -- make weight `src` resident as `dst`
    MATMUL_RESIDENT {lhs,rhs,dst}   -- dst(acc) = lhs(activation) @ rhs(resident weight)
    COMMIT {src,dst} attrs{epilogue:[], output_dtype:"i32"}  -- write acc `src` to output `dst`
    EVICT {handle}                  -- free the resident weight
There is one RES_PACK (weight W, k x n), one or more MATMUL_RESIDENT each reusing W with its own
activation A (m x k), each followed by a COMMIT producing an m x n i32 output, then one EVICT.
`attributes.epilogue` is [] or ["relu"]. Support arbitrary m,k,n (tile into 16x16 with
K-accumulation; zero-pad edges). Get the tensor VALUES with:

    from merlin.runtime.commandbuffer import materialize_inputs
    leaves = materialize_inputs(cb)        # {name: Tensor};  list(leaves[name].data) is row-major

HARD RULES (your output is rejected if violated):
  - Do NOT import or call reference_outputs, simulate, or numpy; do NOT hardcode expected
    outputs. The kernel must COMPUTE the result on the Gemmini hardware only.
  - Output ONLY one ```python code block containing the function (plus any helpers/imports it
    needs). It will be imported and certified on the real Gemmini simulator.
""".strip()

CHEAT_TOKENS = ("reference_outputs", "simulate(", "import numpy", "from numpy", "golden")


def _example(rung: str) -> str:
    cb = build(rung)
    return f"Example visible command buffer ({rung}):\n{json.dumps(cb, indent=1)}"


def build_prompt(visible: tuple[str, ...], feedback: str | None) -> str:
    parts = ["You are generating a Merlin runtime kernel for the Gemmini accelerator.",
             "", "## Gemmini ISA reference", ISA_REFERENCE,
             "", "## Output contract", OUTPUT_CONTRACT,
             "", "## Task", TASK,
             "", "## Visible examples (structure only — NOT the expected outputs)"]
    parts += [_example(r) for r in visible]
    if feedback:
        parts += ["", "## Feedback from the previous attempt (fix this)", feedback]
    return "\n".join(parts)


def _scan_cheat(code: str) -> list[str]:
    return [t for t in CHEAT_TOKENS if t in code]


def _load_generate_driver(code: str, path: Path) -> Callable:
    path.write_text(code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"agent_kernel_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "generate_driver"):
        raise AttributeError("module has no generate_driver")
    return mod.generate_driver


def _certify(rung: str, gen_fn: Callable, simulator: str, timeout: int) -> dict:
    cb = build(rung)
    src = gen_fn(cb)
    res = gemmini.run_command_buffer(cb, simulator=simulator, timeout=timeout, driver_src=src)
    return res


def _diff_feedback(rung: str, res: dict) -> str:
    """A localized hint (one mismatching element) — does not reveal the full golden."""
    from ...runtime import reference_outputs
    cb = build(rung)
    ref = reference_outputs(cb)
    got = res.get("outputs", {})
    for name, exp_rows in ref.items():
        got_rows = got.get(name)
        if got_rows != exp_rows:
            for i, row in enumerate(exp_rows):
                grow = got_rows[i] if got_rows and i < len(got_rows) else []
                for j, ev in enumerate(row):
                    gv = grow[j] if j < len(grow) else None
                    if gv != ev:
                        return (f"Rung {rung}: output {name}[{i}][{j}] expected {ev} but kernel "
                                f"produced {gv}. Likely a tiling/transpose/accumulator-address "
                                f"or layout bug. Recompute on hardware (no peeking).")
    return f"Rung {rung}: outputs differ from reference (and/or run failed)."


def generate_kernel(*, visible: tuple[str, ...] = ("C0", "C1"),
                    heldout: tuple[str, ...] = ("C4", "C4e"), rounds: int = 4,
                    simulator: str = "spike", workdir: str | Path,
                    model: str = "opus", timeout: int = 900) -> dict[str, Any]:
    """Run the gated kernel-synthesis loop. Returns a structured result with the verdict."""
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    feedback: str | None = None
    history: list[dict] = []

    for r in range(rounds):
        prompt = build_prompt(visible, feedback)
        out = claude_cli.run_agent(prompt, model=model, timeout=timeout, workdir=work,
                                   save_transcript=work / f"transcript_r{r}.json")
        try:
            code = claude_cli.extract_code_block(out["text"])
        except claude_cli.AgentError as e:
            feedback = f"{e}. Output exactly one ```python code block."
            history.append({"round": r, "stage": "no_code_block"})
            continue

        cheats = _scan_cheat(code)
        if cheats:
            feedback = (f"Rejected: code referenced forbidden tokens {cheats}. The kernel must "
                        f"compute on hardware, never read the reference/golden.")
            history.append({"round": r, "stage": "cheat", "tokens": cheats})
            continue

        try:
            gen_fn = _load_generate_driver(code, work / f"agent_codegen_r{r}.py")
        except Exception as e:
            feedback = f"Your code failed to import/define generate_driver: {e!r}"
            history.append({"round": r, "stage": "load_error", "error": repr(e)})
            continue

        # Repair loop runs on VISIBLE rungs only.
        visible_ok, vis_results = True, {}
        for rung in visible:
            try:
                res = _certify(rung, gen_fn, simulator, timeout)
            except Exception as e:
                vis_results[rung] = f"error: {e!r}"
                visible_ok = False
                feedback = f"Rung {rung} failed to compile/run: {e!r}"
                break
            vis_results[rung] = res["correct"]
            if not res["correct"]:
                visible_ok = False
                feedback = _diff_feedback(rung, res)
                break

        history.append({"round": r, "stage": "certify", "visible": vis_results})
        if not visible_ok:
            continue

        # Visible passed -> FINAL held-out certification (unbiased generalization test).
        held_results = {}
        held_ok = True
        for rung in heldout:
            try:
                res = _certify(rung, gen_fn, simulator, timeout)
            except Exception as e:
                held_results[rung] = f"error: {e!r}"
                held_ok = False
                continue
            held_results[rung] = res["correct"]
            held_ok = held_ok and res["correct"]
        return {"success": held_ok, "round": r, "code_path": str(work / f"agent_codegen_r{r}.py"),
                "visible": vis_results, "heldout": held_results, "usage": out["usage"],
                "history": history}

    return {"success": False, "rounds": rounds, "last_feedback": feedback, "history": history}

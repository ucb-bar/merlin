"""Emit a Muon SIMT C++ kernel from a Merlin command buffer (fp32).

This is the Muon analog of :mod:`merlin.runtime.backends.gemmini_codegen_mlir`: it turns a command
buffer into a device program clang-muon compiles. The reference backend uses it to produce a *correct*
(if unoptimized) kernel -- the ceiling the agentic backend must re-derive and then beat on cycles.

Design (faithful + simple):
  * Leaf inputs are materialized on the host via the SAME deterministic rule the reference uses
    (:func:`merlin.runtime.commandbuffer.materialize_inputs`) and baked as ``const float`` arrays, so
    device outputs gate bit-for-bit against :func:`merlin.runtime.reference.reference_outputs`.
  * Only the first threadblock computes (``threadblock_id == 0``); within it every thread runs a
    grid-stride loop over the output, with an ``mu_barrier`` between dependent matmuls. Other
    threadblocks/cores no-op or redundantly recompute identical values -- correct, not yet tiled.
  * fp32 epilogues supported: ``relu`` and ``bias_add``. (requant / i8 are Gemmini-only.)

The console (OUT/METRIC/DONE) helpers come from :data:`merlin.runtime.backends.muon.MUON_CONSOLE`.
"""
from __future__ import annotations

from typing import Any

from .muon import MUON_CONSOLE
from ..commandbuffer import materialize_inputs


class MuonCodegenError(RuntimeError):
    pass


def _ident(name: str) -> str:
    return "t_" + "".join(c if c.isalnum() else "_" for c in name)


def _carray(name: str, flat: list) -> str:
    body = ", ".join(f"{float(v)!r}f" for v in flat)
    return f"static const float {_ident(name)}[{len(flat)}] = {{{body}}};"


def _plan(cb: dict[str, Any]) -> tuple[dict, dict, list]:
    """Return (resident_source dst->src, matmul_for accdst->cmd, ordered commit cmds)."""
    resident_source: dict[str, str] = {}
    matmul_for: dict[str, dict] = {}
    commits: list[dict] = []
    for cmd in cb.get("commands", []):
        op = cmd["opcode"]
        ops = cmd.get("operands", {})
        if op == "RES_PACK":
            resident_source[ops["dst"]] = ops["src"]
        elif op in ("MATMUL_RESIDENT", "MATMUL"):
            matmul_for[ops["dst"]] = cmd
        elif op == "COMMIT":
            commits.append(cmd)
    return resident_source, matmul_for, commits


def emit_kernel_cpp(cb: dict[str, Any], *, num_warps: int = 4) -> str:
    """Emit the full SIMT C++ kernel source for ``cb``. Raises on an unsupported command shape."""
    env = materialize_inputs(cb)
    resident_source, matmul_for, commits = _plan(cb)
    if not commits:
        raise MuonCodegenError("no COMMIT commands: Muon reference backend supports matmul kernels")

    shapes: dict[str, tuple[int, int]] = {}
    decls: list[str] = []
    for name, t in env.items():
        if len(t.shape) != 2:
            raise MuonCodegenError(f"leaf {name!r} is rank {len(t.shape)}; expected 2D")
        shapes[name] = (t.shape[0], t.shape[1])
        decls.append(_carray(name, [v for row in t.to_list() for v in row]))

    compute: list[str] = []
    outs: list[tuple[str, int, int]] = []
    for ci, commit in enumerate(commits):
        ops = commit.get("operands", {})
        attrs = commit.get("attributes", {})
        mm = matmul_for.get(ops["src"])
        if mm is None:
            raise MuonCodegenError(f"commit {ops['dst']!r} has no source matmul")
        mops = mm.get("operands", {})
        lhs, rhs = mops["lhs"], resident_source.get(mops["rhs"], mops["rhs"])
        if lhs not in shapes or rhs not in shapes:
            raise MuonCodegenError(f"matmul operands {lhs!r}/{rhs!r} not materialized/committed yet")
        m, k = shapes[lhs]
        k2, n = shapes[rhs]
        if k != k2:
            raise MuonCodegenError(f"matmul K mismatch: {lhs}{shapes[lhs]} @ {rhs}{shapes[rhs]}")
        dst = ops["dst"]
        decls.append(f"static float {_ident(dst)}[{m * n}];")
        shapes[dst] = (m, n)
        outs.append((dst, m, n))

        epi = attrs.get("epilogue", []) or []
        bias = ops.get("bias")
        epi_code = ""
        for stage in epi:
            if stage == "relu":
                epi_code += "      acc = acc < 0.0f ? 0.0f : acc;\n"
            elif stage in ("bias_add", "bias") and bias is not None:
                if bias not in shapes:
                    raise MuonCodegenError(f"bias {bias!r} not materialized")
                epi_code += f"      acc += {_ident(bias)}[c];\n"
            # requant / acc_scale / i8 are Gemmini-only; ignored for the fp32 Muon corpus
        compute.append(f"""    // commit {dst} = {lhs}[{m}x{k}] @ {rhs}[{k}x{n}]{(' ' + str(epi)) if epi else ''}
    for (uint32_t idx = tid; idx < {m * n}u; idx += nthreads) {{
      uint32_t r = idx / {n}u, c = idx % {n}u;
      float acc = 0.0f;
      for (uint32_t kk = 0; kk < {k}u; kk++) acc += {_ident(lhs)}[r * {k}u + kk] * {_ident(rhs)}[kk * {n}u + c];
{epi_code}      {_ident(dst)}[idx] = acc;
    }}
    mu_barrier(0, BLOCK_NUM_WARPS);""")

    prints = "\n".join(
        f'    mu_out_f32("{dst}", {m}, {n}, {_ident(dst)});' for dst, m, n in outs)

    return f"""{MUON_CONSOLE}
#include <mu_intrinsics.h>
#include <mu_schedule.h>
#include <stdint.h>

#define NUM_WARPS {num_warps}
#define BLOCK_NUM_WARPS MU_BLOCK_NUM_WARPS(NUM_WARPS)
extern "C" uint32_t __mu_num_warps = NUM_WARPS;

{chr(10).join(decls)}

struct KArgs {{ uint32_t pad; }};
static KArgs k_args = {{0}};

static inline void body(void* /*arg*/, uint32_t tid, uint32_t nthreads, uint32_t threadblock_id) {{
  if (threadblock_id != 0) return;   // reference: one threadblock computes the whole output
{chr(10).join(compute)}
}}

int main() {{
  mu_schedule(body, &k_args, NUM_WARPS);
  if (mu_is_print_hart()) {{
{prints}
    mu_done();
  }}
  return 0;
}}
"""

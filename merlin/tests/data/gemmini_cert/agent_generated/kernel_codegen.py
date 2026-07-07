"""Agent-generated Gemmini kernel codegen (Claude Code CLI, Opus).

Produced by merlin.targetgen.agent.kernel_slot: the agent saw the command-buffer ABI + a
Gemmini ISA reference + C0/C1 single-tile EXAMPLES (no reference outputs), and synthesized
this generate_driver. CERTIFIED bit-exact on HELD-OUT shapes it never saw (C4 multi-tile,
C4e zero-padded, C5 reuse) against the three-way gate. Round 0, first attempt. Do not edit
(regenerate via the agentic loop)."""

def generate_driver(cb: dict, *, mode: str = "explicit") -> str:
    from merlin.runtime.commandbuffer import materialize_inputs

    leaves = materialize_inputs(cb)            # {name: Tensor}; list(t.data) is row-major
    tensors = cb["tensors"]
    commands = cb["commands"]

    DIM = 16
    def ru(x):
        return ((x + DIM - 1) // DIM) * DIM

    # ---- parse command buffer -------------------------------------------------
    weight_name = None
    acc_to_act = {}                  # MATMUL dst (acc handle) -> activation (lhs)
    outputs = []                     # (act_name, out_name, relu_bool) in program order
    for cmd in commands:
        op = cmd["opcode"]
        operands = cmd.get("operands", {}) or {}
        attrs = cmd.get("attributes", {}) or {}
        if op == "RES_PACK":
            weight_name = operands["src"]
        elif op == "MATMUL_RESIDENT":
            acc_to_act[operands["dst"]] = operands["lhs"]
        elif op == "COMMIT":
            act = acc_to_act[operands["src"]]
            relu = "relu" in (attrs.get("epilogue") or [])
            outputs.append((act, operands["dst"], relu))
        elif op == "EVICT":
            pass

    K, N = tensors[weight_name]["shape"]
    Kp, Np = ru(K), ru(N)

    # ---- helpers --------------------------------------------------------------
    def fmt(vals):
        return ",".join(str(int(v)) for v in vals)

    def pad(data, r, c, rp, cp):
        out = []
        for i in range(rp):
            base = i * c
            for j in range(cp):
                out.append(int(data[base + j]) if (i < r and j < c) else 0)
        return out

    decls = []

    # weight (k x n) padded to (Kp x Np)
    wdata = list(leaves[weight_name].data)
    wpad = pad(wdata, K, N, Kp, Np)
    decls.append(
        "static const elem_t T_%s[%d] row_align(1) = {%s};"
        % (weight_name, Kp * Np, fmt(wpad))
    )

    # activations (each m x k) padded to (Mp x Kp); emit each unique tensor once
    act_emitted = set()
    act_info = {}    # name -> (m, Mp)
    for act, out, relu in outputs:
        m, k = tensors[act]["shape"]
        Mp = ru(m)
        act_info[act] = (m, Mp)
        if act not in act_emitted:
            adata = list(leaves[act].data)
            apad = pad(adata, m, k, Mp, Kp)
            decls.append(
                "static const elem_t T_%s[%d] row_align(1) = {%s};"
                % (act, Mp * Kp, fmt(apad))
            )
            act_emitted.add(act)

    # output accumulators (m x n) padded to (Mp x Np)
    out_info = {}    # out_name -> (m, n)
    out_emitted = set()
    for act, out, relu in outputs:
        m, Mp = act_info[act]
        out_info[out] = (m, N)
        if out not in out_emitted:
            decls.append(
                "static acc_t T_%s[%d] row_align_acc(1);" % (out, Mp * Np)
            )
            out_emitted.add(out)

    # ---- main body: matmul calls + prints ------------------------------------
    calls = []
    for act, out, relu in outputs:
        m, Mp = act_info[act]
        calls.append(
            "  do_matmul(T_%s, T_%s, T_%s, %d, %d, %d, %d);"
            % (act, weight_name, out, Mp, Kp, Np, 1 if relu else 0)
        )

    prints = []
    for act, out, relu in outputs:
        m, n = out_info[out]
        prints.append('  printf("OUT %s %d %d");' % (out, m, n))
        prints.append(
            "  for (int i=0;i<%d;i++) for (int j=0;j<%d;j++) "
            'printf(" %%d",(int)T_%s[i*%d+j]);' % (m, n, out, Np)
        )
        prints.append('  printf("\\n");')

    decls_s = "\n".join(decls)
    calls_s = "\n".join(calls)
    prints_s = "\n".join(prints)

    src = """#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

%s

/* C(MpxNp) = A(MpxKp) @ W(KpxNp), tiled 16x16 with K-accumulation, weight-stationary. */
static void do_matmul(const elem_t* A, const elem_t* W, acc_t* C,
                      int Mp, int Kp, int Np, int acc_relu) {
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_extended_config_st((size_t)Np * sizeof(acc_t),
                             acc_relu ? RELU : NO_ACTIVATION, ACC_SCALE_IDENTITY);

  uint32_t c_ovw = ((3u<<(ADDR_LEN-2))|(1u<<(ADDR_LEN-3))) & ~(1u<<(ADDR_LEN-2));
  uint32_t c_acc = c_ovw | (1u<<(ADDR_LEN-2));
  const uint32_t sp_w = 0, sp_a = DIM;

  int Mt = Mp/DIM, Kt = Kp/DIM, Nt = Np/DIM;
  for (int ti=0; ti<Mt; ti++) {
    for (int tj=0; tj<Nt; tj++) {
      for (int tk=0; tk<Kt; tk++) {
        gemmini_config_ld((size_t)Kp * sizeof(elem_t));
        gemmini_mvin((void*)(A + (size_t)(ti*DIM)*Kp + (size_t)(tk*DIM)), sp_a);
        gemmini_config_ld((size_t)Np * sizeof(elem_t));
        gemmini_mvin((void*)(W + (size_t)(tk*DIM)*Np + (size_t)(tj*DIM)), sp_w);
        uint32_t caddr = (tk==0) ? c_ovw : c_acc;
        gemmini_preload(sp_w, caddr);
        gemmini_compute_preloaded(sp_a, GARBAGE_ADDR);
      }
      gemmini_mvout((void*)(C + (size_t)(ti*DIM)*Np + (size_t)(tj*DIM)), c_ovw);
    }
  }
}

int main() {
  gemmini_flush(0);

  uint64_t c0 = read_cycles();
%s
  gemmini_fence();
  uint64_t c1 = read_cycles();

%s

  printf("METRIC cycles %%lu\\n", (unsigned long)(c1 - c0));
  printf("METRIC cycle_window_gemmini_region 1\\n");
  printf("DONE\\n");
  return 0;
}
""" % (decls_s, calls_s, prints_s)

    return src

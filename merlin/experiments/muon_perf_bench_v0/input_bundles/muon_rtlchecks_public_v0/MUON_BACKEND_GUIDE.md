# MUON_BACKEND_GUIDE — author a Muon SIMT backend (the 4 contract entrypoints)

You are building an **out-of-tree backend** that lowers the frozen, target-agnostic `merlin_iface`
dialect to a **Muon SIMT C++ kernel** the harness compiles with clang-muon and runs on **cyclotron**.
Read `MUON_DIGEST.md` (RTL-grounded hardware facts) first — it is your "the RTL was pre-analyzed for
you" advantage.

## Your package (`submission/`)
- `manifest.yaml` — `target: muon`, `language: python`, `integrity_exempt: false`, `entrypoints.tool: muon-opt`,
  and a `commands:` block with the 4 entrypoints (copy the argv shapes below).
- `muon-opt` — an executable exposing the 4 entrypoints. **Self-contained: no `import merlin`,
  no reference/oracle, no hardcoded outputs, no copied kernels.** You parse the interface MLIR yourself.

```yaml
# manifest.yaml
artifact_type: mlir_oot_target_backend
target: muon
package_id: my_muon_v0
language: python
authoring: {mode: agent, author: agent, generated_by_agent: true}
integrity_exempt: false
entrypoints: {tool: muon-opt}
commands:
  parse:                     {argv: ["{tool}", "--verify-diagnostics", "{input_mlir}"]}
  lower_interface_to_target: {argv: ["{tool}", "--convert-iface-to-muon", "{input_mlir}"]}
  emit_command_buffer:       {argv: ["{tool}", "--convert-iface-to-muon", "--emit-command-buffer={output_json}", "{input_mlir}"]}
  lower_target_to_llvm:      {argv: ["{tool}", "--convert-iface-to-muon", "--emit-simt-cpp", "{input_mlir}"]}
```

## The input: `capsule.interface.mlir` (merlin_iface grammar)
A tiny, fixed grammar. Example (fp32 GEMM `Y0 = X @ W`):
```mlir
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "muon", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<16x16xf32>
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x16xf32>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<16x16xf32>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %X, %W_res : (tensor<16x16xf32>, !merlin_iface.resident) -> !merlin_iface.acc<f32>
  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "f32"} : (!merlin_iface.acc<f32>) -> tensor<16x16xf32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
```
Parse: `tensor` (name/role/shape/dtype), `resident_pack` (dst←src), `matmul` (lhs, rhs, dst-acc),
`commit` (src-acc → named output, epilogue list, output_dtype). `merlin/contract/` has the full grammar +
the `command_buffer.json` schema your `--emit-command-buffer` must satisfy.

## Input materialization (so your output matches the grader, who uses the SAME rule)
Leaf inputs are deterministic from the tensor name (no RNG). For a tensor named `NAME`, shape with
`n` elements, element `k` (row-major, 0-based):
```
seed   = sum((i+1) * ord(NAME[i]) for i in range(len(NAME)))   # or 1 if that is 0
val[k] = (seed * (k+1) + k*k) % 4                                # values in 0..3 (stored as fp32)
```
Bake these exact values into your emitted kernel (host const arrays), compute the matmul, and print
the result with the console protocol below.

## The output: a self-contained SIMT C++ kernel (`--emit-simt-cpp` prints it to stdout)
Launch via `mu_schedule(callback, &args, NUM_WARPS)`; the callback is
`(void* arg, uint32_t tid_in_threadblock, uint32_t threads_per_threadblock, uint32_t threadblock_id)`.
Print results from ONE hart (`mu_is_print_hart()`), AFTER `mu_schedule` returns, using the console
helper below (paste it verbatim at the top of every emitted kernel):

```cpp
#include <vx_intrinsics.h>
extern "C" void vx_putchar(int c);
// The ELF starts on every core's hart 0 (SPMD bring-up); print from ONE hart only, else the two
// cores' vx_putchar byte streams interleave ("O2UT..."). Guard every emit with mu_is_print_hart().
static inline bool mu_is_print_hart() { return vx_core_id() == 0; }
namespace muon_console {
static inline void puts_(const char* s) { while (*s) vx_putchar(*s++); }
static inline void put_u32(unsigned v) {
  char b[12]; int n = 0;
  if (!v) { vx_putchar('0'); return; }
  while (v) { b[n++] = '0' + (v % 10u); v /= 10u; }
  while (n) vx_putchar(b[--n]);
}
static inline void put_i32(int v) { if (v < 0) { vx_putchar('-'); put_u32((unsigned)(-v)); } else put_u32((unsigned)v); }
static inline void put_f32(float f, int frac = 6) {
  if (f != f) { puts_("nan"); return; }
  if (f < 0) { vx_putchar('-'); f = -f; }
  unsigned ip = (unsigned)f; float rem = f - (float)ip;
  put_u32(ip); vx_putchar('.');
  for (int i = 0; i < frac; i++) { rem *= 10.0f; unsigned d = (unsigned)rem; vx_putchar('0' + (d % 10u)); rem -= (float)d; }
}
}  // namespace muon_console
static inline void mu_out_i32(const char* name, int rows, int cols, const int* data) {
  using namespace muon_console;
  puts_("OUT "); puts_(name); vx_putchar(' '); put_u32(rows); vx_putchar(' '); put_u32(cols);
  for (int i = 0; i < rows * cols; i++) { vx_putchar(' '); put_i32(data[i]); }
  vx_putchar('\n');
}
static inline void mu_out_f32(const char* name, int rows, int cols, const float* data) {
  using namespace muon_console;
  puts_("OUT "); puts_(name); vx_putchar(' '); put_u32(rows); vx_putchar(' '); put_u32(cols);
  for (int i = 0; i < rows * cols; i++) { vx_putchar(' '); put_f32(data[i]); }
  vx_putchar('\n');
}
static inline void mu_metric(const char* name, int v) {
  using namespace muon_console;
  puts_("METRIC "); puts_(name); vx_putchar(' '); put_i32(v); vx_putchar('\n');
}
static inline void mu_done() { muon_console::puts_("DONE\n"); }
```

Then your kernel body computes the op (use shared memory + `mu_barrier(0, BLOCK_NUM_WARPS)` to go fast
— see MUON_DIGEST for SMEM capacity / register budget / FP peak), and `main()` prints each committed
output: `mu_out_f32("Y0", rows, cols, data)` then `mu_done()`.

Available intrinsics (headers on the clang-muon sysroot): `mu_intrinsics.h` (`store_shared`,
`load32_shared`, `mu_barrier`, `mu_fence_smem`, `MU_BLOCK_NUM_WARPS`), `vx_intrinsics.h`
(`vx_thread_id`, `vx_warp_id`, `vx_core_id`).

## Verify (use it as much as you want)
```
python agent_selfcheck.py                 # redacted: pass/fail + plane + mismatch_count + cycles + %FP-peak
python agent_selfcheck.py --capsule MG00_gemm_16x16x16
```
Iterate until ALL public capsules pass, then raise %FP-peak (cycles down) toward the 32 GFLOP/s peak.

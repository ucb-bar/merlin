/* KMU entry glue — harness-owned board support, NOT compiler output.
 *
 * Vortex's KMU launches every (block, thread) coordinate of a kernel at that kernel's entry PC, and
 * it finds that PC through the `.vx_entry` stub / VXSYMTAB footer that the Vortex LLVM backend emits
 * for functions carrying `annotate("vortex.kernel")`. An image without that metadata does not launch
 * — it hangs (verified: an otherwise-correct kernel missing the annotation hangs on simx, and a
 * stock-LLVM build hangs for the same reason, because stock clang parses the attribute but has no
 * backend pass that acts on it).
 *
 * That annotation is therefore the ONE thing on the device side that requires the Vortex toolchain.
 * Rather than force the agent's compiler to depend on a forked LLVM, the harness isolates it here:
 * this file is prebuilt ONCE with the Vortex toolchain (see ../scripts/build_harness.sh) and simply
 * dispatches to the compiler-emitted body. The agent's object is then built with STOCK LLVM and
 * linked against it. Verified end to end on simx.
 *
 * Nothing about the kernel's *semantics* lives here: no work distribution (the KMU does it), no
 * identity computation (CTA CSRs), no compute.
 */
#include "merlin_vortex_abi.h"

extern "C" __attribute__((annotate("vortex.kernel"), used, retain))
void kernel_main(const merlin_vx_kernel_arg_t* arg) {
  merlin_kernel_body(arg);
}

/* The contract between the runner-owned Vortex harness and compiler-emitted device code.
 *
 * The agent's backend emits exactly one device symbol, `merlin_kernel_body`. Everything
 * Vortex-specific about *getting there* — the KMU entry annotation, startup, the linker script — is
 * harness-owned board support (see ../src/vx_entry.cpp and ../README.md).
 *
 * Thread/block identity is NOT passed in: it is read from the CTA CSRs, which the KMU populates per
 * launched coordinate. The ISA spec sheet documents the CSR map; the emitted code reads them
 * directly (a `csrr` inline-asm on stock LLVM — no Vortex header, no forked toolchain).
 */
#ifndef MERLIN_VORTEX_ABI_H
#define MERLIN_VORTEX_ABI_H

#include <stdint.h>

#define MERLIN_VX_ABI_VERSION 1
#define MERLIN_VX_MAX_ARGS    32

/* Device-visible argument block. The host harness fills this from the module's `merlin.arg_table`
 * annotation before launch: one device address per forward() operand, in table order
 * (weights, then inputs, then outputs — the order the annotation declares). */
typedef struct {
  uint32_t abi_version;                  /* == MERLIN_VX_ABI_VERSION */
  uint32_t n_args;                       /* number of valid entries in `args` */
  uint64_t args[MERLIN_VX_MAX_ARGS];     /* device addresses, in merlin.arg_table order */
} merlin_vx_kernel_arg_t;

/* THE symbol the compiler must emit. Called once per (block, thread) coordinate the KMU launches. */
#ifdef __cplusplus
extern "C"
#endif
void merlin_kernel_body(const merlin_vx_kernel_arg_t* arg);

#endif /* MERLIN_VORTEX_ABI_H */

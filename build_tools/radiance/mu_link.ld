/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Vendored from radiance-kernels/lib/linker/mu_link.ld at commit
 * 56aad6e1620c452bd131c948f72352dab0754d6e (see ./pin.txt).
 * DO NOT EDIT without bumping the pin and re-running the parity check
 * (`merlin chipyard run-radiance-muon ...` against the reference vecadd ELF).
 *
 * Sections produced by this script:
 *   .shared_mem at 0x00000000 (NOLOAD)  — kernel-local SMEM, populated at runtime
 *   .text/.data/.rodata starting at 0x10000000  — Muon instruction memory base
 *   .tohost at start-of-text (page aligned)  — HTIF mailbox for sim exit
 *
 * Memory map matches the chipyard.harness sim's `+loadmem=...` loader for
 * RadianceMuonConfig and RadianceSingleClusterConfig.
 */

OUTPUT_ARCH( "riscv" )
ENTRY(_start)

SECTIONS
{
  /* shared memory addresses start at 0x0; loader does not put stuff in there
   * so it needs to be populated at runtime. */
  .shared_mem 0 (NOLOAD) : { KEEP(*(.shared)) }

  . = 0x10000000;
  .init : { KEEP(*(.init)) }
  .text.init : { KEEP(*(.text.init)) }
  . = ALIGN(0x1000);
  .tohost : { *(.tohost) }
  . = ALIGN(0x1000);
  .text : { *(.text .text.*) }
  . = ALIGN(0x1000);
  .data : { *(.data .data.*) }
  . = ALIGN(0x1000);
  .rodata : { *(.rodata .rodata.*) }

  /* exception handling.  We don't need this, but the C++ compiler generates
   * them.  Tuck it away after .text */
  .eh_frame_hdr   : { *(.eh_frame_hdr) *(.eh_frame_entry .eh_frame_entry.*) }
  .eh_frame       : ONLY_IF_RO { KEEP (*(.eh_frame)) *(.eh_frame.*) }

  /* There's no support for TLS, edata or global_pointer currently.
   * Define them so that vx_start.S and vx_syscalls.c don't complain. */

  /* thread local storage */
  .tdata :
    {
      PROVIDE_HIDDEN (__tdata_start = .);
      *(.tdata .tdata.* .gnu.linkonce.td.*)
      PROVIDE_HIDDEN (__tdata_end = .);
    }
  PROVIDE (_edata = .);
  PROVIDE (__tdata_size = SIZEOF (.tdata));
  PROVIDE (__tcb_aligned_size = 0);
  PROVIDE (__tbss_size = 0);

  .bss : { *(.bss) }
  __global_pointer = .;

  _end = .;
}

# Bare ELF DMA arena placement

The kernel, selected Smol layer and Tiny block builders accept `--arena-base`, `--dram-base` and `--dram-bytes` as integer literals, including hexadecimal values.

Omitting `--arena-base` retains the input kernel manifest address or the layer/block default `0x100000000`; physical DRAM validation defaults to 64 GiB from `0x80000000`, matching the existing EE290 simulation memory map.

For VCU118 default DDR use `--arena-base 0x90000000 --dram-base 0x80000000 --dram-bytes 0x80000000` in the existing build command; this selects the end-exclusive physical range `[0x80000000,0x100000000)`.

The selected arena is an address choice, not a board hardware configuration or an FPGA execution claim.

## Build behavior

`tools/chipyard/block_kernels.py build` relocates the admitted raw program after validating its original input hashes, regenerates the completion wrapper, and uses the same selected base in the host payload and `.atlas_dram` linker placement.

Raw relocation changes the DMA high-half setup and temporarily adds the low-half displacement around each DMA DRAM operand; it leaves VMEM word operands unchanged and repairs Atlas instruction-word-relative branch targets.

Both raw and assembly relocation reserve x31 and require the retained 4 GiB source base. Raw relocation supports direct control flow and repairs branch targets; both paths reject unexpected DMA configuration and unsafe register use.

`tools/chipyard/smol_layer.py` validates the selected original programs before relocation, relocates and reassembles their assembly, and patches the copied runtime header and copied linker arena together; the header supplies both the host arena pointer and the adapter active-buffer address.

`--program-build` selects and hash-check older qualified programs when the current generator has advanced.

`tools/chipyard/tiny_block.py` applies per-launch x31 displacement in both default and relocated builds with the selected arena low half and DMA high-half setup; the twelve arena sections and host descriptors use the corresponding physical addresses.

`tools/chipyard/tiny_block.py` writes every activation preload byte as `0x96` and audits the final ELF; separate live activation captures remain available for numerical checks.

Each arena must be 64-byte aligned, fit unsigned 64-bit physical addressing and stay inside one 4 GiB DMA low-half window; crossing that window requires a different program strategy and is rejected.

The builders fail if any physical PT_LOAD memory extent, including its zero-filled tail, lies outside the selected DDR range or overlaps another segment, or if any nonempty allocated section, including NOLOAD/BSS/stack, lies outside DDR or its load segment.

The HTIF kernel CRT reserves its heap and boot-hart stack through symbols beyond PT_LOAD memory extents; the audit separately checks those reservations for DDR containment and overlap with load segments.

This stack audit assumes one boot hart, as in the selected EE290 workloads; it does not admit a multihart stack layout.

FireSim exports record the selected physical DDR range; the image owner must supply a matching capacity, clocks, bridges and Atlas composition.

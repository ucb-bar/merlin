// Run a DRAM-resident whole-model image on the gemmini ChipTop under GSIM.
//
// Three things this needs that the M3 flow did not, and all three now exist:
//   * a memory. axi_mem_harness backs the chip-boundary AXI port with testchipip's mm_magic_t.
//   * a way in. There is no TSI (pruned with the TestHarness), so the image is written straight into
//     the backing store and a one-word BootROM jumps to it.
//   * a way out. There is no host to service HTIF, so results are read back OUT OF DRAM rather than
//     printed; the program's output buffer is found by symbol.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include "ChipTop.h"

extern "C" void gemmini_dram_init();
extern "C" bool gemmini_dram_write(uint64_t phys, const void* src, unsigned long n);
extern "C" bool gemmini_dram_read(uint64_t phys, void* dst, unsigned long n);
extern "C" void gemmini_axi_tick(SChipTop* dut, uint8_t reset);
extern "C" void gemmini_axi_stats();

// --- minimal ELF64 program-header walk: load every PT_LOAD at its physical address -------------
struct Ehdr { uint8_t e_ident[16]; uint16_t e_type, e_machine; uint32_t e_version;
              uint64_t e_entry, e_phoff, e_shoff; uint32_t e_flags;
              uint16_t e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx; };
struct Phdr { uint32_t p_type, p_flags; uint64_t p_offset, p_vaddr, p_paddr,
              p_filesz, p_memsz, p_align; };

static bool load_elf(const char* path, uint64_t* entry) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror("open elf"); return false; }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> img(n);
  if (fread(img.data(), 1, n, f) != (size_t)n) { fclose(f); return false; }
  fclose(f);
  if (memcmp(img.data(), "\177ELF", 4) != 0) { fprintf(stderr, "not an ELF\n"); return false; }
  const Ehdr* eh = (const Ehdr*)img.data();
  *entry = eh->e_entry;
  unsigned long loaded = 0;
  for (int i = 0; i < eh->e_phnum; ++i) {
    const Phdr* ph = (const Phdr*)(img.data() + eh->e_phoff + (size_t)i * eh->e_phentsize);
    if (ph->p_type != 1 /*PT_LOAD*/ || ph->p_memsz == 0) continue;
    // filesz bytes come from the file; the rest of memsz is .bss and must be zeroed, or the
    // program starts with whatever the backing store happened to hold.
    if (ph->p_filesz && !gemmini_dram_write(ph->p_paddr, img.data() + ph->p_offset, ph->p_filesz)) {
      fprintf(stderr, "segment %d: write 0x%llx+%llu outside the backing store\n",
              i, (unsigned long long)ph->p_paddr, (unsigned long long)ph->p_filesz);
      return false;
    }
    if (ph->p_memsz > ph->p_filesz) {
      std::vector<uint8_t> z(ph->p_memsz - ph->p_filesz, 0);
      gemmini_dram_write(ph->p_paddr + ph->p_filesz, z.data(), z.size());
    }
    loaded += ph->p_memsz;
    printf("# PT_LOAD %d -> 0x%llx  filesz=%llu memsz=%llu\n", i,
           (unsigned long long)ph->p_paddr, (unsigned long long)ph->p_filesz,
           (unsigned long long)ph->p_memsz);
  }
  printf("# loaded %lu bytes, entry=0x%llx\n", loaded, (unsigned long long)*entry);
  return loaded > 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <elf> [cycles] [dump_addr dump_len]\n", argv[0]); return 2; }
  const long cycles = (argc > 2) ? atol(argv[2]) : 2000000;
  const uint64_t dump_addr = (argc > 4) ? strtoull(argv[3], nullptr, 0) : 0;
  const unsigned long dump_len = (argc > 4) ? strtoul(argv[4], nullptr, 0) : 0;

  gemmini_dram_init();
  uint64_t entry = 0;
  if (!load_elf(argv[1], &entry)) return 1;

  SChipTop* dut = new SChipTop();
  dut->set_clock_uncore(1);
  dut->set_reset_io(1);
  for (int c = 0; c < 10; c++) { gemmini_axi_tick(dut, 1); dut->step(); }
  dut->set_reset_io(0);
  dut->set_serial_tl_0$$in$$valid(0);
  dut->set_serial_tl_0$$out$$ready(1);
  dut->set_custom_boot(0);

  // Progress is reported by PC, not by a console: there is no host to service HTIF here.
  uint64_t last_pc = 0; long stuck = 0;
  for (long c = 0; c < cycles; ++c) {
    gemmini_axi_tick(dut, 0);
    dut->step();
    if ((c % 200000) == 0) {
      uint64_t pc = (uint64_t)dut->system$tile_prci_domain$element_reset_domain$rockettile$core$wb_reg_pc;
      printf("# cycle %ld pc=0x%llx\n", c, (unsigned long long)pc); fflush(stdout);
      stuck = (pc == last_pc) ? stuck + 1 : 0; last_pc = pc;
    }
  }
  gemmini_axi_stats();

  if (dump_len) {
    std::vector<uint8_t> out(dump_len);
    if (gemmini_dram_read(dump_addr, out.data(), dump_len)) {
      printf("DUMP 0x%llx %lu\n", (unsigned long long)dump_addr, dump_len);
      for (unsigned long i = 0; i < dump_len; ++i) printf("%02x", out[i]);
      printf("\n");
    } else printf("# dump address outside the backing store\n");
  }
  delete dut;
  return 0;
}

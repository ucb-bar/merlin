// Run a self-checking Gemmini ELF on the GSIM-emitted ChipTop.
//
// The program carries two ordinary, noinline marker functions.  It calls the
// pass marker only after comparing Gemmini's mvout buffer with its CPU golden,
// or the fail marker with a nonzero mismatch count.  Watching those committed
// PCs avoids both unresolved whole-model host boundaries:
//
//   * no TSI is needed to load the ELF (PT_LOAD segments go into mm_magic_t);
//   * no cache-coherent backdoor is needed to read the final result (the Rocket
//     core performs the comparison, and the harness observes its control flow).
//
// The exact same ELF remains a normal Verilator input: after either marker
// returns it reaches the stock tohost exit path there.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ChipTop.h"

extern "C" void gemmini_dram_init();
extern "C" bool gemmini_dram_write(uint64_t phys, const void* src,
                                    unsigned long n);
extern "C" void gemmini_axi_tick(SChipTop* dut, uint8_t reset);
extern "C" void gemmini_axi_stats();

namespace {

struct Ehdr {
  uint8_t e_ident[16];
  uint16_t e_type, e_machine;
  uint32_t e_version;
  uint64_t e_entry, e_phoff, e_shoff;
  uint32_t e_flags;
  uint16_t e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx;
};

struct Phdr {
  uint32_t p_type, p_flags;
  uint64_t p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align;
};

bool read_file(const char* path, std::vector<uint8_t>* bytes) {
  FILE* file = std::fopen(path, "rb");
  if (!file) {
    std::perror(path);
    return false;
  }
  if (std::fseek(file, 0, SEEK_END) != 0) return false;
  const long size = std::ftell(file);
  if (size <= 0 || std::fseek(file, 0, SEEK_SET) != 0) return false;
  bytes->resize(static_cast<size_t>(size));
  const bool ok = std::fread(bytes->data(), 1, bytes->size(), file) == bytes->size();
  std::fclose(file);
  return ok;
}

bool load_elf(const char* path, uint64_t* entry, unsigned long* loaded_bytes) {
  std::vector<uint8_t> image;
  if (!read_file(path, &image) || image.size() < sizeof(Ehdr) ||
      std::memcmp(image.data(), "\177ELF", 4) != 0) {
    std::fprintf(stderr, "%s is not a readable ELF64 image\n", path);
    return false;
  }
  const Ehdr* header = reinterpret_cast<const Ehdr*>(image.data());
  if (header->e_phentsize < sizeof(Phdr) ||
      header->e_phoff > image.size() ||
      header->e_phnum > (image.size() - header->e_phoff) / header->e_phentsize) {
    std::fprintf(stderr, "%s has an invalid program-header table\n", path);
    return false;
  }
  *entry = header->e_entry;
  *loaded_bytes = 0;
  for (unsigned i = 0; i < header->e_phnum; ++i) {
    const Phdr* ph = reinterpret_cast<const Phdr*>(
        image.data() + header->e_phoff + static_cast<size_t>(i) * header->e_phentsize);
    if (ph->p_type != 1 || ph->p_memsz == 0) continue;  // PT_LOAD
    if (ph->p_filesz > ph->p_memsz || ph->p_offset > image.size() ||
        ph->p_filesz > image.size() - ph->p_offset) {
      std::fprintf(stderr, "%s has an invalid PT_LOAD segment %u\n", path, i);
      return false;
    }
    if (ph->p_filesz &&
        !gemmini_dram_write(ph->p_paddr, image.data() + ph->p_offset, ph->p_filesz)) {
      std::fprintf(stderr, "PT_LOAD %u at 0x%llx is outside GSIM DRAM\n", i,
                   static_cast<unsigned long long>(ph->p_paddr));
      return false;
    }
    if (ph->p_memsz > ph->p_filesz) {
      std::vector<uint8_t> zero(static_cast<size_t>(ph->p_memsz - ph->p_filesz), 0);
      if (!gemmini_dram_write(ph->p_paddr + ph->p_filesz, zero.data(), zero.size())) {
        std::fprintf(stderr, "PT_LOAD %u BSS is outside GSIM DRAM\n", i);
        return false;
      }
    }
    *loaded_bytes += ph->p_memsz;
  }
  return *loaded_bytes != 0;
}

bool load_bootrom(const char* path, std::vector<uint64_t>* words) {
  std::vector<uint8_t> image;
  if (!read_file(path, &image) || image.size() > 512 * sizeof(uint64_t)) {
    std::fprintf(stderr, "%s is not a valid ChipTop bootrom image\n", path);
    return false;
  }
  image.resize((image.size() + 7) & ~size_t{7}, 0);
  words->resize(image.size() / sizeof(uint64_t));
  for (size_t i = 0; i < words->size(); ++i) {
    uint64_t word = 0;
    std::memcpy(&word, image.data() + i * sizeof(word), sizeof(word));
    (*words)[i] = word;
  }
  return !words->empty();
}

void bake_bootrom(SChipTop* dut, const std::vector<uint64_t>& words) {
  for (size_t i = 0; i < words.size(); ++i)
    dut->system$bootrom_domain$bootrom$rom[i] = words[i];
}

bool parse_u64(const char* text, uint64_t* value) {
  char* end = nullptr;
  *value = std::strtoull(text, &end, 0);
  return end != text && *end == '\0';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr,
                 "usage: %s ELF BOOTROM_BIN KERNEL_PC PASS_PC FAIL_PC [MAX_CYCLES]\n",
                 argv[0]);
    return 2;
  }
  uint64_t kernel_pc = 0, pass_pc = 0, fail_pc = 0;
  if (!parse_u64(argv[3], &kernel_pc) || !parse_u64(argv[4], &pass_pc) ||
      !parse_u64(argv[5], &fail_pc)) {
    std::fprintf(stderr, "kernel/pass/fail PCs must be integers\n");
    return 2;
  }
  const uint64_t max_cycles = argc > 6 ? std::strtoull(argv[6], nullptr, 0) : 2000000;
  if (max_cycles == 0) return 2;

  gemmini_dram_init();
  uint64_t entry = 0;
  unsigned long loaded_bytes = 0;
  if (!load_elf(argv[1], &entry, &loaded_bytes)) return 2;
  std::vector<uint64_t> bootrom;
  if (!load_bootrom(argv[2], &bootrom)) return 2;

  SChipTop* dut = new SChipTop();
  dut->set_clock_uncore(1);
  dut->set_reset_io(1);
  for (unsigned c = 0; c < 20; ++c) {
    bake_bootrom(dut, bootrom);
    gemmini_axi_tick(dut, 1);
    dut->step();
  }
  dut->set_reset_io(0);
  dut->set_serial_tl_0$$in$$valid(0);
  dut->set_serial_tl_0$$out$$ready(1);
  dut->set_custom_boot(0);

  uint64_t kernel_cycle = UINT64_MAX;
  uint64_t completion_cycle = UINT64_MAX;
  const char* status = "timeout";
  uint64_t busy_cycles = 0;
  for (uint64_t cycle = 0; cycle < max_cycles; ++cycle) {
    bake_bootrom(dut, bootrom);
    gemmini_axi_tick(dut, 0);
    dut->step();
    busy_cycles += dut->system$tile_prci_domain$element_reset_domain$rockettile$gemmini$reservation_station$_io_busy_T_4 != 0;
    if (!dut->system$tile_prci_domain$element_reset_domain$rockettile$core$wb_reg_valid)
      continue;
    const uint64_t pc = dut->system$tile_prci_domain$element_reset_domain$rockettile$core$wb_reg_pc;
    if (pc == kernel_pc && kernel_cycle == UINT64_MAX) kernel_cycle = cycle;
    if (pc == pass_pc) {
      status = "pass";
      completion_cycle = cycle;
      break;
    }
    if (pc == fail_pc) {
      status = "fail";
      completion_cycle = cycle;
      break;
    }
  }

  const bool completed = completion_cycle != UINT64_MAX;
  const bool saw_kernel = kernel_cycle != UINT64_MAX;
  const uint64_t kernel_to_verdict = completed && saw_kernel
      ? completion_cycle - kernel_cycle : 0;
  std::printf(
      "GSIM_RESULT status=%s completion=%u kernel_seen=%u entry=0x%llx "
      "completion_cycle=%llu kernel_to_verdict_cycles=%llu gemmini_busy_cycles=%llu "
      "loaded_bytes=%lu\n",
      status, completed, saw_kernel, static_cast<unsigned long long>(entry),
      static_cast<unsigned long long>(completed ? completion_cycle : max_cycles),
      static_cast<unsigned long long>(kernel_to_verdict),
      static_cast<unsigned long long>(busy_cycles), loaded_bytes);
  gemmini_axi_stats();
  delete dut;
  if (!completed) return 3;
  return std::strcmp(status, "pass") == 0 && saw_kernel ? 0 : 1;
}

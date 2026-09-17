// AXI backing store for the GSIM-emitted Gemmini ChipTop.
//
// This is deliberately a thin binding to testchipip's mm_magic_t, the same
// functional memory model used by SimDRAM.  GSIM prunes TestHarness (and hence
// SimDRAM), but it preserves ChipTop's chip-boundary AXI4 port.  Keeping the
// memory implementation in testchipip avoids inventing another AXI model here.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sys/mman.h>

#include "ChipTop.h"
#include "mm.h"

namespace {

constexpr uint64_t kDefaultBase = UINT64_C(0x80000000);
constexpr uint64_t kDefaultSize = UINT64_C(64) << 20;
constexpr uint64_t kWordBytes = 8;
constexpr uint64_t kLineBytes = 64;

std::unique_ptr<mm_magic_t> memory;
backing_data_t backing{};
uint64_t memory_base = kDefaultBase;
uint64_t memory_size = kDefaultSize;
uint64_t ar_count = 0;
uint64_t aw_count = 0;
uint64_t w_count = 0;

uint64_t env_u64(const char* name, uint64_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const uint64_t parsed = std::strtoull(value, &end, 0);
  if (end == value || *end != '\0' || parsed == 0) {
    std::fprintf(stderr, "invalid %s=%s\n", name, value);
    std::exit(2);
  }
  return parsed;
}

bool covered(uint64_t phys, unsigned long n) {
  if (!memory || phys < memory_base) return false;
  const uint64_t offset = phys - memory_base;
  return offset <= memory_size && n <= memory_size - offset;
}

}  // namespace

extern "C" void gemmini_dram_init() {
  if (memory) return;
  memory_base = env_u64("MERLIN_GSIM_DRAM_BASE", kDefaultBase);
  memory_size = env_u64("MERLIN_GSIM_DRAM_SIZE", kDefaultSize);
  void* data = mmap(nullptr, memory_size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (data == MAP_FAILED) {
    std::perror("mmap GSIM DRAM");
    std::exit(2);
  }
  backing = {static_cast<uint8_t*>(data), static_cast<size_t>(memory_size)};
  memory = std::make_unique<mm_magic_t>(memory_base, memory_size, kWordBytes,
                                        kLineBytes, backing);
}

extern "C" bool gemmini_dram_write(uint64_t phys, const void* src,
                                    unsigned long n) {
  if (!covered(phys, n)) return false;
  std::memcpy(backing.data + (phys - memory_base), src, n);
  return true;
}

extern "C" bool gemmini_dram_read(uint64_t phys, void* dst, unsigned long n) {
  if (!covered(phys, n)) return false;
  std::memcpy(dst, backing.data + (phys - memory_base), n);
  return true;
}

extern "C" void gemmini_axi_tick(SChipTop* dut, uint8_t reset) {
  if (!memory) gemmini_dram_init();

  uint64_t wdata = dut->get_axi4_mem_0$$bits$$w$$bits$$data();
  const bool ar_valid = dut->get_axi4_mem_0$$bits$$ar$$valid();
  const bool aw_valid = dut->get_axi4_mem_0$$bits$$aw$$valid();
  const bool w_valid = dut->get_axi4_mem_0$$bits$$w$$valid();
  ar_count += ar_valid && memory->ar_ready() && !reset;
  aw_count += aw_valid && memory->aw_ready() && !reset;
  w_count += w_valid && memory->w_ready() && !reset;

  memory->tick(
      reset,
      ar_valid, dut->get_axi4_mem_0$$bits$$ar$$bits$$addr(),
      dut->get_axi4_mem_0$$bits$$ar$$bits$$id(),
      dut->get_axi4_mem_0$$bits$$ar$$bits$$size(),
      dut->get_axi4_mem_0$$bits$$ar$$bits$$len(),
      aw_valid, dut->get_axi4_mem_0$$bits$$aw$$bits$$addr(),
      dut->get_axi4_mem_0$$bits$$aw$$bits$$id(),
      dut->get_axi4_mem_0$$bits$$aw$$bits$$size(),
      dut->get_axi4_mem_0$$bits$$aw$$bits$$len(),
      w_valid, dut->get_axi4_mem_0$$bits$$w$$bits$$strb(), &wdata,
      dut->get_axi4_mem_0$$bits$$w$$bits$$last(),
      dut->get_axi4_mem_0$$bits$$r$$ready(),
      dut->get_axi4_mem_0$$bits$$b$$ready());

  dut->set_axi4_mem_0$$bits$$ar$$ready(memory->ar_ready());
  dut->set_axi4_mem_0$$bits$$aw$$ready(memory->aw_ready());
  dut->set_axi4_mem_0$$bits$$w$$ready(memory->w_ready());
  dut->set_axi4_mem_0$$bits$$r$$valid(memory->r_valid());
  dut->set_axi4_mem_0$$bits$$r$$bits$$id(memory->r_id());
  dut->set_axi4_mem_0$$bits$$r$$bits$$resp(memory->r_resp());
  uint64_t rdata = 0;
  std::memcpy(&rdata, memory->r_data(), sizeof(rdata));
  dut->set_axi4_mem_0$$bits$$r$$bits$$data(rdata);
  dut->set_axi4_mem_0$$bits$$r$$bits$$last(memory->r_last());
  dut->set_axi4_mem_0$$bits$$b$$valid(memory->b_valid());
  dut->set_axi4_mem_0$$bits$$b$$bits$$id(memory->b_id());
  dut->set_axi4_mem_0$$bits$$b$$bits$$resp(memory->b_resp());
}

extern "C" void gemmini_axi_stats() {
  std::printf("GSIM_AXI ar=%llu aw=%llu w=%llu base=0x%llx size=%llu\n",
              static_cast<unsigned long long>(ar_count),
              static_cast<unsigned long long>(aw_count),
              static_cast<unsigned long long>(w_count),
              static_cast<unsigned long long>(memory_base),
              static_cast<unsigned long long>(memory_size));
}

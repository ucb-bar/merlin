#pragma once
// Production mm_t interface: the backing mapping remains shared with TSI/dumps.
#include "mm.h"
#include <memory>

class mm_fased_t final : public mm_t {
 public:
  mm_fased_t(size_t base, size_t size, size_t word, size_t line, backing_data_t& backing,
             unsigned address_bits, unsigned id_bits);
  ~mm_fased_t() override;
  bool ar_ready() override;
  bool aw_ready() override;
  bool w_ready() override;
  bool b_valid() override;
  uint64_t b_resp() override;
  uint64_t b_id() override;
  bool r_valid() override;
  uint64_t r_resp() override;
  uint64_t r_id() override;
  void* r_data() override;
  bool r_last() override;
  void tick(bool reset, bool ar_valid, uint64_t ar_addr, uint64_t ar_id,
            uint64_t ar_size, uint64_t ar_len, bool aw_valid, uint64_t aw_addr,
            uint64_t aw_id, uint64_t aw_size, uint64_t aw_len, bool w_valid,
            uint64_t w_strb, void* w_data, bool w_last, bool r_ready, bool b_ready) override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

// A seeded, response-reordering AXI4 memory model for chipyard-style simulator harnesses.
//
// WHY. testchipip's mm_magic_t answers every read in the order it was accepted, one cycle later, every
// time. A program whose correctness depends on two independent memory requests completing in issue
// order therefore passes on every simulator built on it, and fails only on hardware whose memory
// system does not keep that order (a DRAM controller with banks, an FPGA-hosted timing model). This
// model keeps everything mm_magic_t guarantees that AXI4 also guarantees, and nothing it does not:
//
//   * responses with the SAME id stay in acceptance order        (AXI4 requires it)
//   * a read burst's beats are delivered contiguously             (never interleaved with another id)
//   * read data is sampled when the request is ACCEPTED           (as mm_magic_t does)
//   * writes land when their W beat is accepted                   (as mm_magic_t does)
// and it adds exactly one freedom AXI4 allows: responses with DIFFERENT ids may complete in any order,
// each after its own latency drawn from a seeded generator. The same seed gives the same schedule, so a
// failure found under seed s reproduces under seed s.
//
// It is OFF unless asked for: a harness constructs it only when a seed is given, and otherwise builds
// the stock mm_magic_t, so a simulator linked with this file and run without the plusarg behaves as the
// stock one does.
//
// Knobs, as plusargs or, when a plusarg is absent, as environment variables (MERLIN_MEM_PERTURB_SEED,
// ..._MAX_LATENCY, ..._TAIL_PERMILLE, ..._REGION_LOG2, ..._SLOW_PERMILLE). The environment form exists
// so a caller can perturb a simulator without teaching its front end a new plusarg:
//   +mem_perturb_seed=<u64>          enables the model
//   +mem_perturb_max_latency=<n>     uniform extra latency [0,n] per response
//   +mem_perturb_tail_permille=<p>   p/1000 of responses get a further [0, 8n] (a slow bank)
//   +mem_perturb_region_log2=<k>     latency is instead a property of the ADDRESS: each 2^k-byte region
//   +mem_perturb_slow_permille=<p>   is slow (n + [0, n/8]) with probability p/1000, else fast
//                                    ([0, n/8]), fixed per seed. This is how a cache makes one operand
//                                    fast and another slow for a whole tile, which per-request noise
//                                    almost never does.
#ifndef MERLIN_MM_PERTURB_H
#define MERLIN_MM_PERTURB_H

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "mm.h"

struct mm_perturb_cfg_t {
  bool enabled = false;
  uint64_t seed = 0;
  uint64_t max_latency = 64;
  uint64_t tail_permille = 0;
  uint64_t region_log2 = 0;  // 0: per-request latency; k: per-2^k-byte-region latency class
  uint64_t slow_permille = 500;
};

// Parse the knobs out of an argv (plusargs). Unknown arguments are ignored; a malformed value is a hard
// error, since a perturbation run that silently fell back to the in-order model would certify nothing.
mm_perturb_cfg_t mm_perturb_parse(int argc, const char *const *argv);

// The stock model when `cfg.enabled` is false, the perturbing one otherwise. Logs one line to stderr
// saying which, so a console records the memory model it ran under.
mm_t *mm_perturb_make(const mm_perturb_cfg_t &cfg, size_t mem_base, size_t mem_size, size_t word_size,
                      size_t line_size, backing_data_t &dat);

class mm_perturb_t : public mm_t {
 public:
  mm_perturb_t(const mm_perturb_cfg_t &cfg, size_t mem_base, size_t mem_sz, size_t word_sz, size_t line_sz,
               backing_data_t &dat);

  bool ar_ready() override { return true; }
  bool aw_ready() override { return !store_inflight; }
  bool w_ready() override { return store_inflight; }
  bool b_valid() override { return b_cur >= 0; }
  uint64_t b_resp() override { return 0; }
  uint64_t b_id() override { return b_valid() ? bq[b_cur].id : 0; }
  bool r_valid() override { return r_cur >= 0; }
  uint64_t r_resp() override { return 0; }
  uint64_t r_id() override { return r_valid() ? rq[r_cur].id : 0; }
  void *r_data() override { return r_valid() ? (void *)&rq[r_cur].beats[r_beat][0] : (void *)data; }
  bool r_last() override { return r_valid() && r_beat + 1 == rq[r_cur].beats.size(); }

  void tick(bool reset, bool ar_valid, uint64_t ar_addr, uint64_t ar_id, uint64_t ar_size, uint64_t ar_len,
            bool aw_valid, uint64_t aw_addr, uint64_t aw_id, uint64_t aw_size, uint64_t aw_len, bool w_valid,
            uint64_t w_strb, void *w_data, bool w_last, bool r_ready, bool b_ready) override;

  // Counters a harness may print at exit: how often the model actually delivered out of order.
  uint64_t reads = 0, reads_reordered = 0, writes = 0, writes_reordered = 0;

 private:
  struct rburst_t { uint64_t id, ready_at, seq; std::vector<std::vector<char>> beats; };
  struct bresp_t { uint64_t id, ready_at, seq; };

  uint64_t draw_latency(uint64_t addr);
  uint64_t next();
  template <class T> int pick(const std::deque<T> &q);

  mm_perturb_cfg_t cfg;
  uint64_t rng, cycle = 0, seq = 0, last_r_seq = 0, last_b_seq = 0;
  std::deque<rburst_t> rq;
  std::deque<bresp_t> bq;
  int r_cur = -1, b_cur = -1;
  size_t r_beat = 0;
  bool store_inflight = false;
  uint64_t store_addr = 0, store_id = 0, store_size = 0, store_count = 0;
};

#endif

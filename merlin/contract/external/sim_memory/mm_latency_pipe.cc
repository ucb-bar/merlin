// See mm_latency_pipe.h for what this model reproduces and why.
#include "mm_latency_pipe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static bool take_u64(const char *arg, const char *key, uint64_t *out) {
  size_t n = strlen(key);
  if (strncmp(arg, key, n) != 0) return false;
  const char *v = arg + n;
  char *end = nullptr;
  unsigned long long x = strtoull(v, &end, 0);
  if (*v == '\0' || end == nullptr || *end != '\0') {
    fprintf(stderr, "[mem-latency-pipe] malformed %s%s\n", key, v);
    abort();
  }
  *out = (uint64_t)x;
  return true;
}

// One knob: the plusarg wins; otherwise the environment variable named for it.
static bool knob(int argc, const char *const *argv, const char *plus, const char *env, uint64_t *out) {
  for (int i = 1; i < argc; i++)
    if (take_u64(argv[i], plus, out)) return true;
  const char *v = getenv(env);
  if (v == nullptr) return false;
  std::string arg = std::string(plus) + v;
  return take_u64(arg.c_str(), plus, out);
}

mm_latency_pipe_cfg_t mm_latency_pipe_parse(int argc, const char *const *argv) {
  mm_latency_pipe_cfg_t cfg;
  uint64_t on = 0;
  knob(argc, argv, "+mem_latency_pipe=", "MERLIN_MEM_LATENCY_PIPE", &on);
  knob(argc, argv, "+mem_latency_pipe_read_latency=", "MERLIN_MEM_LATENCY_PIPE_READ_LATENCY", &cfg.read_latency);
  knob(argc, argv, "+mem_latency_pipe_write_latency=", "MERLIN_MEM_LATENCY_PIPE_WRITE_LATENCY", &cfg.write_latency);
  knob(argc, argv, "+mem_latency_pipe_max_reads=", "MERLIN_MEM_LATENCY_PIPE_MAX_READS", &cfg.max_reads);
  knob(argc, argv, "+mem_latency_pipe_max_writes=", "MERLIN_MEM_LATENCY_PIPE_MAX_WRITES", &cfg.max_writes);
  if (on > 1) {
    fprintf(stderr, "[mem-latency-pipe] +mem_latency_pipe takes 0 or 1, got %llu\n", (unsigned long long)on);
    abort();
  }
  // FASED's own floors: a latency of 0 would answer in the cycle of acceptance, which its releaser
  // register cannot, and a bound of 0 would never accept a request.
  if (cfg.read_latency < 1 || cfg.write_latency < 1 || cfg.max_reads < 1 || cfg.max_writes < 1) {
    fprintf(stderr, "[mem-latency-pipe] latencies and outstanding bounds must be >= 1\n");
    abort();
  }
  cfg.enabled = on == 1;
  return cfg;
}

mm_latency_pipe_t::mm_latency_pipe_t(const mm_latency_pipe_cfg_t &c, size_t mem_base, size_t mem_sz,
                                     size_t word_sz, size_t line_sz, backing_data_t &dat)
    : mm_t(mem_base, mem_sz, word_sz, line_sz, dat), cfg(c) {}

void mm_latency_pipe_t::tick(bool reset, bool ar_valid, uint64_t ar_addr, uint64_t ar_id, uint64_t ar_size,
                             uint64_t ar_len, bool aw_valid, uint64_t aw_addr, uint64_t aw_id, uint64_t aw_size,
                             uint64_t aw_len, bool w_valid, uint64_t w_strb, void *w_data, bool w_last,
                             bool r_ready, bool b_ready) {
  if (reset) {
    rq.clear(); aw_wait.clear(); w_early.clear(); bq.clear();
    pending_reads = pending_aw = pending_w = 0; r_beat = 0; cycle = 0;
    return;
  }
  // Fires are decided on the outputs presented THIS cycle, which are functions of the state left by the
  // previous tick -- nothing below changes that state until every fire is known.
  bool ar_fire = ar_valid && ar_ready();
  bool aw_fire = aw_valid && aw_ready();
  bool w_fire = w_valid && w_ready();
  bool r_fire = r_valid() && r_ready;
  bool b_fire = b_valid() && b_ready;
  ar_stall_cycles += ar_valid && !ar_fire;
  aw_stall_cycles += aw_valid && !aw_fire;
  w_stall_cycles += w_valid && !w_fire;

  if (ar_fire) {
    rburst_t b{ar_id, cycle + cfg.read_latency, {}};
    uint64_t start = (ar_addr / word_size) * word_size;
    for (uint64_t i = 0; i <= ar_len; i++) b.beats.push_back(read(start + i * word_size));
    rq.push_back(std::move(b));
    pending_reads++;
    reads++;
  }
  if (aw_fire) {
    aw_wait.push_back(aw_t{aw_addr, aw_id, 1ULL << aw_size, aw_len + 1});
    pending_aw++;
  }
  if (w_fire) {
    const char *p = (const char *)w_data;
    w_early.push_back(wbeat_t{std::vector<char>(p, p + word_size), w_strb, w_last});
    pending_w += w_last;
  }
  // Pair W beats with their AW in order (AXI4 carries no W id). A write is COMPLETE in the cycle its pair
  // completes -- the later of its AW and its last W beat -- which is when FASED's newWReq fires. At most
  // one write completes per cycle: at most one AW and one W beat arrive, and each belongs to one burst.
  while (!aw_wait.empty() && !w_early.empty()) {
    aw_t &a = aw_wait.front();
    wbeat_t &w = w_early.front();
    write(a.addr, (uint8_t *)w.data.data(), w.strb, a.size);
    a.addr += a.size;
    bool last = w.last;
    w_early.pop_front();
    if (--a.beats_left == 0) {
      if (!last) { fprintf(stderr, "[mem-latency-pipe] W burst longer than its AW\n"); abort(); }
      bq.push_back(bresp_t{a.id, cycle + cfg.write_latency});
      aw_wait.pop_front();
      writes++;
    } else if (last) {
      fprintf(stderr, "[mem-latency-pipe] W burst shorter than its AW\n");
      abort();
    }
  }
  if (b_fire) {
    bq.pop_front();
    pending_aw--;
    pending_w--;
  }
  if (r_fire && ++r_beat == rq.front().beats.size()) {
    rq.pop_front();
    r_beat = 0;
    pending_reads--;
  }
  cycle++;
}

// See mm_perturb.h for what this model guarantees and why.
#include "mm_perturb.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool take_u64(const char *arg, const char *key, uint64_t *out) {
  size_t n = strlen(key);
  if (strncmp(arg, key, n) != 0) return false;
  const char *v = arg + n;
  char *end = nullptr;
  unsigned long long x = strtoull(v, &end, 0);
  if (*v == '\0' || end == nullptr || *end != '\0') {
    fprintf(stderr, "[mem-perturb] malformed %s%s\n", key, v);
    abort();
  }
  *out = (uint64_t)x;
  return true;
}

// One knob: the plusarg wins; otherwise the environment variable of the same name, upper-cased.
static bool knob(int argc, const char *const *argv, const char *plus, const char *env, uint64_t *out) {
  for (int i = 1; i < argc; i++)
    if (take_u64(argv[i], plus, out)) return true;
  const char *v = getenv(env);
  if (v == nullptr) return false;
  std::string arg = std::string(plus) + v;
  return take_u64(arg.c_str(), plus, out);
}

mm_perturb_cfg_t mm_perturb_parse(int argc, const char *const *argv) {
  mm_perturb_cfg_t cfg;
  cfg.enabled = knob(argc, argv, "+mem_perturb_seed=", "MERLIN_MEM_PERTURB_SEED", &cfg.seed);
  knob(argc, argv, "+mem_perturb_max_latency=", "MERLIN_MEM_PERTURB_MAX_LATENCY", &cfg.max_latency);
  knob(argc, argv, "+mem_perturb_tail_permille=", "MERLIN_MEM_PERTURB_TAIL_PERMILLE", &cfg.tail_permille);
  knob(argc, argv, "+mem_perturb_region_log2=", "MERLIN_MEM_PERTURB_REGION_LOG2", &cfg.region_log2);
  knob(argc, argv, "+mem_perturb_slow_permille=", "MERLIN_MEM_PERTURB_SLOW_PERMILLE", &cfg.slow_permille);
  if (cfg.slow_permille > 1000 || cfg.region_log2 > 40) {
    fprintf(stderr, "[mem-perturb] slow_permille/region_log2 out of range\n");
    abort();
  }
  if (cfg.tail_permille > 1000) {
    fprintf(stderr, "[mem-perturb] tail_permille %llu > 1000\n", (unsigned long long)cfg.tail_permille);
    abort();
  }
  return cfg;
}

mm_t *mm_perturb_make(const mm_perturb_cfg_t &cfg, size_t mem_base, size_t mem_size, size_t word_size,
                      size_t line_size, backing_data_t &dat) {
  if (!cfg.enabled) {
    fprintf(stderr, "[mem-perturb] model=in_order (mm_magic_t)\n");
    return new mm_magic_t(mem_base, mem_size, word_size, line_size, dat);
  }
  fprintf(stderr,
          "[mem-perturb] model=perturbed seed=%llu max_latency=%llu tail_permille=%llu region_log2=%llu "
          "slow_permille=%llu\n",
          (unsigned long long)cfg.seed, (unsigned long long)cfg.max_latency,
          (unsigned long long)cfg.tail_permille, (unsigned long long)cfg.region_log2,
          (unsigned long long)cfg.slow_permille);
  return new mm_perturb_t(cfg, mem_base, mem_size, word_size, line_size, dat);
}

mm_perturb_t::mm_perturb_t(const mm_perturb_cfg_t &c, size_t mem_base, size_t mem_sz, size_t word_sz,
                           size_t line_sz, backing_data_t &dat)
    : mm_t(mem_base, mem_sz, word_sz, line_sz, dat), cfg(c), rng(c.seed ^ 0x9e3779b97f4a7c15ULL) {}

// splitmix64: small, well mixed, and identical on every host -- the schedule is a function of the seed.
uint64_t mm_perturb_t::next() {
  uint64_t z = (rng += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static uint64_t mix(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

uint64_t mm_perturb_t::draw_latency(uint64_t addr) {
  uint64_t lat;
  if (cfg.region_log2) {
    // the region's class is a pure function of (seed, region): every request to it agrees
    bool slow = mix(cfg.seed * 0x9e3779b97f4a7c15ULL ^ (addr >> cfg.region_log2)) % 1000 < cfg.slow_permille;
    lat = (slow ? cfg.max_latency : 0) + next() % (cfg.max_latency / 8 + 1);
  } else {
    lat = cfg.max_latency ? next() % (cfg.max_latency + 1) : 0;
  }
  if (cfg.tail_permille && next() % 1000 < cfg.tail_permille)
    lat += next() % (8 * cfg.max_latency + 1);
  return lat;
}

// The response to deliver next: among entries whose latency has elapsed and that are the OLDEST of
// their id (AXI4 keeps same-id order), the one that became ready first; ties go to the older request.
template <class T> int mm_perturb_t::pick(const std::deque<T> &q) {
  int best = -1;
  for (size_t i = 0; i < q.size(); i++) {
    if (q[i].ready_at > cycle) continue;
    bool blocked = false;
    for (size_t j = 0; j < i && !blocked; j++) blocked = q[j].id == q[i].id;
    if (blocked) continue;
    if (best < 0 || q[i].ready_at < q[best].ready_at ||
        (q[i].ready_at == q[best].ready_at && q[i].seq < q[best].seq))
      best = (int)i;
  }
  return best;
}

void mm_perturb_t::tick(bool reset, bool ar_valid, uint64_t ar_addr, uint64_t ar_id, uint64_t ar_size,
                        uint64_t ar_len, bool aw_valid, uint64_t aw_addr, uint64_t aw_id, uint64_t aw_size,
                        uint64_t aw_len, bool w_valid, uint64_t w_strb, void *w_data, bool w_last, bool r_ready,
                        bool b_ready) {
  if (reset) {
    rq.clear(); bq.clear(); r_cur = b_cur = -1; r_beat = 0; store_inflight = false; cycle = 0;
    return;
  }
  // Fires are decided on the outputs presented LAST cycle, exactly as mm_magic_t decides them.
  bool ar_fire = ar_valid && ar_ready();
  bool aw_fire = aw_valid && aw_ready();
  bool w_fire = w_valid && w_ready();
  bool r_fire = r_valid() && r_ready;
  bool b_fire = b_valid() && b_ready;

  if (ar_fire) {
    rburst_t b{ar_id, cycle + draw_latency(ar_addr), seq++, {}};
    uint64_t start = (ar_addr / word_size) * word_size;
    for (uint64_t i = 0; i <= ar_len; i++) b.beats.push_back(read(start + i * word_size));
    rq.push_back(std::move(b));
  }
  if (aw_fire) {
    store_addr = aw_addr; store_id = aw_id; store_count = aw_len + 1; store_size = 1ULL << aw_size;
    store_inflight = true;
  }
  if (w_fire) {
    write(store_addr, (uint8_t *)w_data, w_strb, store_size);
    store_addr += store_size;
    if (--store_count == 0) {
      store_inflight = false;
      bq.push_back(bresp_t{store_id, cycle + draw_latency(store_addr), seq++});
      if (!w_last) { fprintf(stderr, "[mem-perturb] W burst length mismatch\n"); abort(); }
    }
  }
  if (b_fire) { bq.erase(bq.begin() + b_cur); b_cur = -1; }
  if (r_fire && ++r_beat == rq[r_cur].beats.size()) { rq.erase(rq.begin() + r_cur); r_cur = -1; r_beat = 0; }

  if (r_cur < 0 && (r_cur = pick(rq)) >= 0) { reads++; reads_reordered += r_cur > 0; }
  if (b_cur < 0 && (b_cur = pick(bq)) >= 0) { writes++; writes_reordered += b_cur > 0; }
  cycle++;
}

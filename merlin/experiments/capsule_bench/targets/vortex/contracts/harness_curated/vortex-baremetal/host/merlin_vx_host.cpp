/* Generic capsule host driver for the Vortex capsule bench — runner-owned.
 *
 * Reads a launch plan (JSON-ish, emitted by the runner from the capsule + the module's
 * `merlin.arg_table` annotation), allocates one device buffer per operand, fills the inputs
 * DETERMINISTICALLY, uploads the compiler-produced `.vxbin`, launches it over the declared grid,
 * reads the outputs back, and prints the OUT / METRIC / DONE console protocol the runner parses.
 *
 * It is workload-agnostic: it knows buffer sizes and dtypes, never the operation. All semantics live
 * in the compiled kernel.
 *
 * Determinism contract: inputs come from the LCG in `fill()` below, seeded per operand. The golden
 * generator MUST reproduce it bit-for-bit --- see merlin/contract/capsules/vortex (same recurrence,
 * same element order). Do not "improve" the RNG without regenerating every golden.
 *
 *   usage: merlin_vx_host <kernel.vxbin> <plan.json>
 */
#include <vortex.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(x) do { int _r = (x); if (_r != VX_SUCCESS) { \
    std::fprintf(stderr, "ERROR %s -> %d (%s:%d)\n", #x, _r, __FILE__, __LINE__); return 1; } } while (0)

namespace {

struct Arg {
  std::string name;
  std::string role;    // weight | input | output
  std::string dtype;   // f32 | i32 | i8
  uint64_t    bytes = 0;
  uint32_t    seed = 0;
  vx_buffer_h buf = nullptr;
};

/* One 32-bit LCG (Numerical Recipes constants), reproduced exactly by the golden generator. */
inline uint32_t lcg(uint32_t& s) { s = s * 1664525u + 1013904223u; return s; }

void fill(std::vector<uint8_t>& host, const Arg& a) {
  uint32_t s = a.seed ? a.seed : 1u;
  if (a.dtype == "f32") {
    auto* p = reinterpret_cast<float*>(host.data());
    for (uint64_t i = 0; i < host.size() / 4; ++i)          // [-1, 1), 2^-24 quantised
      p[i] = static_cast<float>(static_cast<int32_t>(lcg(s) >> 8) - (1 << 23)) / static_cast<float>(1 << 23);
  } else if (a.dtype == "i8") {
    for (uint64_t i = 0; i < host.size(); ++i)
      host[i] = static_cast<uint8_t>(static_cast<int8_t>(lcg(s) >> 24));
  } else {                                                   // i32
    auto* p = reinterpret_cast<int32_t*>(host.data());
    for (uint64_t i = 0; i < host.size() / 4; ++i)
      p[i] = static_cast<int32_t>(lcg(s) >> 16) - (1 << 15);
  }
}

/* Minimal extractor for the flat plan the runner emits; avoids a JSON dependency in the harness. */
std::string field(const std::string& obj, const std::string& key) {
  auto k = obj.find("\"" + key + "\"");
  if (k == std::string::npos) return "";
  auto c = obj.find(':', k);
  if (c == std::string::npos) return "";
  auto b = obj.find_first_not_of(" \t\"", c + 1);
  auto e = obj.find_first_of(",\"}", b);
  return obj.substr(b, e - b);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s <kernel.vxbin> <plan.json>\n", argv[0]); return 2; }
  const char* kernel_file = argv[1];

  std::ifstream pf(argv[2]);
  if (!pf) { std::fprintf(stderr, "ERROR cannot open plan %s\n", argv[2]); return 2; }
  std::stringstream ss; ss << pf.rdbuf();
  const std::string plan = ss.str();

  const uint32_t grid_n = static_cast<uint32_t>(std::strtoul(field(plan, "grid").c_str(), nullptr, 10));

  /* Operands, in merlin.arg_table order — the order the kernel's arg block expects. */
  std::vector<Arg> args;
  for (size_t p = plan.find("\"name\""); p != std::string::npos; p = plan.find("\"name\"", p + 1)) {
    const std::string rec = plan.substr(p, plan.find('}', p) - p);
    Arg a;
    a.name  = field(rec, "name");
    a.role  = field(rec, "role");
    a.dtype = field(rec, "dtype");
    a.bytes = std::strtoull(field(rec, "bytes").c_str(), nullptr, 10);
    a.seed  = static_cast<uint32_t>(std::strtoul(field(rec, "seed").c_str(), nullptr, 10));
    if (a.bytes) args.push_back(a);
  }
  if (args.empty()) { std::fprintf(stderr, "ERROR plan declares no operands\n"); return 2; }

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  /* Device-side argument block: abi_version, n_args, then one address per operand
   * (must match merlin_vx_kernel_arg_t in ../include/merlin_vortex_abi.h). */
  struct { uint32_t abi_version; uint32_t n_args; uint64_t addr[32]; } kargs{};
  kargs.abi_version = 1;
  kargs.n_args = static_cast<uint32_t>(args.size());

  std::vector<std::vector<uint8_t>> host(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    Arg& a = args[i];
    const bool is_out = (a.role == "output");
    CHECK(vx_buffer_create(dev, a.bytes, is_out ? VX_MEM_WRITE : VX_MEM_READ, &a.buf));
    CHECK(vx_buffer_address(a.buf, &kargs.addr[i]));
    host[i].assign(a.bytes, 0);
    if (!is_out) {
      fill(host[i], a);
      CHECK(vx_enqueue_write(q, a.buf, 0, host[i].data(), a.bytes, 0, nullptr, nullptr));
    }
  }

  vx_module_h mod = nullptr; vx_kernel_h kern = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kern));

  uint32_t grid[1], block[1];
  CHECK(vx_device_max_occupancy_grid(dev, 1, &grid_n, grid, block));

  vx_launch_info_t li{};
  li.struct_size = sizeof(li);
  li.kernel      = kern;
  li.args_host   = &kargs;
  li.args_size   = sizeof(kargs);
  li.ndim        = 1;
  li.grid_dim[0] = grid[0];
  li.block_dim[0]= block[0];

  vx_event_h launch_ev = nullptr;
  CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &launch_ev));

  /* Read every output back, then emit the console protocol. */
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i].role != "output") continue;
    vx_event_h rd = nullptr;
    CHECK(vx_enqueue_read(q, host[i].data(), args[i].buf, 0, args[i].bytes, 1, &launch_ev, &rd));
    CHECK(vx_event_wait_value(rd, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(rd);
  }

  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i].role != "output") continue;
    std::printf("OUT %s %s %llu ", args[i].name.c_str(), args[i].dtype.c_str(),
                static_cast<unsigned long long>(args[i].bytes));
    for (uint64_t b = 0; b < args[i].bytes; ++b) std::printf("%02x", host[i][b]);   // raw bytes: bit-exact
    std::printf("\n");
  }
  /* Report the machine we actually ran on. L2 (simx) and L3 (rtlsim) are separately-built
   * simulators, and nothing stops them drifting to different geometries — which silently breaks
   * cross-tier comparability and, when the grid no longer matches, produces unwritten output
   * buffers rather than an error. The oracle asserts this line against the frozen geometry. */
  uint64_t g_clusters = 0, g_cores = 0, g_warps = 0, g_threads = 0;
  vx_dev_caps(dev, VX_CAPS_NUM_CLUSTERS, &g_clusters);
  vx_dev_caps(dev, VX_CAPS_NUM_CORES, &g_cores);
  vx_dev_caps(dev, VX_CAPS_NUM_WARPS, &g_warps);
  vx_dev_caps(dev, VX_CAPS_NUM_THREADS, &g_threads);
  std::printf("METRIC geometry clusters=%llu cores=%llu warps=%llu threads=%llu\n",
              (unsigned long long)g_clusters, (unsigned long long)g_cores,
              (unsigned long long)g_warps, (unsigned long long)g_threads);
  /* The dimensions the device was actually launched with, which are NOT simply `grid_n`:
   * vx_device_max_occupancy_grid splits the requested extent into grid x block for this machine, and
   * the split changes with core count. The kernel sees these through the CTA CSRs. */
  std::printf("METRIC launch requested=%u grid=%u block=%u\n", grid_n, grid[0], block[0]);
  std::printf("METRIC grid %u\n", grid_n);
  /* Emits the runtime's `PERF: instrs=..., cycles=..., IPC=...` line, which the oracle parses for
   * the capsule's cycle count. Diagnostic only, so a failure here must not fail the run — and the
   * line is simply absent when the runtime was built without PERF. */
  (void)vx_device_dump_perf(dev, stdout);
  std::printf("DONE\n");

  vx_event_release(launch_ev);
  for (auto& a : args) vx_buffer_release(a.buf);
  vx_kernel_release(kern);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_dev_close(dev);
  return 0;
}

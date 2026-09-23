// Board-side kernel timer for board-LOCAL MetaSchedule measurement (no tvm_rpc).
// Loads a candidate PrimFunc .so (built by LocalBuilder for rv64gcv), allocates random NDArrays per
// a manifest (one line per arg: "dtype_code dtype_bits ndim d0 d1 ..."), times the entry func over
// N calls, prints the median per-call latency in SECONDS ("LAT_SEC <x>"). The host-side custom
// MetaSchedule runner scp's the .so + manifest here, runs this, and returns the latency to the
// search — same board-local transport that fixed execution, now for measurement.
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>
#include <dlpack/dlpack.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cmath>
using namespace tvm::runtime;

int main(int argc, char** argv) {
  // argv: 1=so 2=manifest 3=entry_name 4=number 5=repeat
  if (argc < 4) { fprintf(stderr, "usage: timer so manifest entry [number] [repeat]\n"); return 1; }
  std::string so = argv[1], manifest = argv[2], entry = argv[3];
  int number = argc > 4 ? atoi(argv[4]) : 10;
  int repeat = argc > 5 ? atoi(argv[5]) : 3;

  Module mod = Module::LoadFromFile(so);
  PackedFunc f = mod.GetFunction(entry, true);
  if (f == nullptr) f = mod.GetFunction("main", true);
  if (f == nullptr) { fprintf(stderr, "no entry func %s/main\n", entry.c_str()); return 3; }

  DLDevice dev{kDLCPU, 0};
  std::vector<NDArray> args;
  std::ifstream mf(manifest); std::string line;
  while (std::getline(mf, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    int code, bits, ndim; ss >> code >> bits >> ndim;
    std::vector<int64_t> shp(ndim);
    for (int i = 0; i < ndim; i++) ss >> shp[i];
    DLDataType dt{(uint8_t)code, (uint8_t)bits, 1};
    NDArray a = NDArray::Empty(ShapeTuple(shp.begin(), shp.end()), dt, dev);
    // random-ish fill (deterministic) so the kernel does real work; content is irrelevant to latency
    int64_t n = 1; for (auto d : shp) n *= d;
    if (code == 2 && bits == 32) {  // float32
      std::vector<float> buf(n); for (int64_t i = 0; i < n; i++) buf[i] = (float)((i % 17) - 8) * 0.1f;
      a.CopyFromBytes(buf.data(), n * 4);
    } else if (code == 0 && bits == 64) {  // int64
      std::vector<int64_t> buf(n); for (int64_t i = 0; i < n; i++) buf[i] = i % 8;
      a.CopyFromBytes(buf.data(), n * 8);
    } else if (code == 0 && bits == 32) {  // int32
      std::vector<int32_t> buf(n); for (int64_t i = 0; i < n; i++) buf[i] = i % 8;
      a.CopyFromBytes(buf.data(), n * 4);
    } // else leave zero-initialized
    args.push_back(a);
  }
  // build packed args
  std::vector<TVMValue> vals(args.size()); std::vector<int> codes(args.size());
  for (size_t i = 0; i < args.size(); i++) {
    vals[i].v_handle = const_cast<DLTensor*>(args[i].operator->());
    codes[i] = kTVMDLTensorHandle;
  }
  TVMRetValue rv;
  // warmup
  f.CallPacked(TVMArgs(vals.data(), codes.data(), (int)vals.size()), &rv);
  std::vector<double> per_call;
  for (int r = 0; r < repeat; r++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < number; k++)
      f.CallPacked(TVMArgs(vals.data(), codes.data(), (int)vals.size()), &rv);
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e9 / number;
    per_call.push_back(sec);
  }
  std::sort(per_call.begin(), per_call.end());
  double median = per_call[per_call.size() / 2];
  printf("LAT_SEC %.9f\n", median);
  return 0;
}

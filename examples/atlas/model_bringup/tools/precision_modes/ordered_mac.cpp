// Exact FP8 products added in ascending K, rounding the accumulator to BF16 nearest-even after every MAC.
// FP8 operands keep each binary64 sum exact or too small to move the BF16 result, so one rounding suffices.
// Build: c++ -O3 -shared -fPIC ordered_mac.cpp -o ordered_mac.so
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" void ordered(const double *a, const double *b, double *out, int m, int k, int n) {
  std::vector<double> acc(n);
  for (int row = 0; row < m; ++row) {
    std::fill(acc.begin(), acc.end(), 0.0);
    for (int x = 0; x < k; ++x)
      for (int col = 0; col < n; ++col) {
        double sum = acc[col] + a[(size_t)row * k + x] * b[(size_t)x * n + col];
        uint64_t bits;
        std::memcpy(&bits, &sum, 8);
        bits = (bits + UINT64_C(0xfffffffffff) + ((bits >> 45) & 1)) & UINT64_C(0xffffe00000000000);
        std::memcpy(&acc[col], &bits, 8);
      }
    std::memcpy(out + (size_t)row * n, acc.data(), n * sizeof(double));
  }
}

// Reproduce the f32 width-256 AVX2 LayerNorm order in PyTorch commit
// 7f24a19c0967db04f2d5ba44c1e56eca489932ec. Diagnostic source only.
// Derived from the installed ATen/native/cpu/moments_utils.h RowwiseMoments
// and the native_layer_norm CPU affine result on the three inputs in the report.
#include <cmath>
#include <cstdint>

extern "C" void torch216_avx2_ln256(const float* x, int64_t rows,
                                     const float* gamma, const float* beta,
                                     float epsilon, float* mean_out,
                                     float* rstd_out, float* out) {
  for (int64_t row = 0; row < rows; ++row) {
    float means[8] = {}, m2s[8] = {};
    for (int chunk = 0; chunk < 2; ++chunk) {
      float chunk_means[8] = {}, chunk_m2s[8] = {};
      for (int j = 0; j < 16; ++j) {
        const float c = 1.0f / float(j + 1);
        for (int lane = 0; lane < 8; ++lane) {
          const float v = x[row * 256 + chunk * 128 + j * 8 + lane];
          const float delta = v - chunk_means[lane];
          chunk_means[lane] = std::fma(c, delta, chunk_means[lane]);
          chunk_m2s[lane] = std::fma(delta, v - chunk_means[lane], chunk_m2s[lane]);
        }
      }
      for (int lane = 0; lane < 8; ++lane) {
        const float delta = chunk_means[lane] - means[lane];
        const float m2_tmp = m2s[lane] + chunk_m2s[lane];
        const float c_delta = (chunk == 0 ? 1.0f : 0.5f) * delta;
        const float m0_delta = delta * float(chunk * 16);
        means[lane] = means[lane] + c_delta;
        m2s[lane] = std::fma(m0_delta, c_delta, m2_tmp);
      }
    }

    int count = 0;
    float mean = 0.0f, m2 = 0.0f;
    for (int lane = 0; lane < 8; ++lane) {
      const float c = 32.0f / float(count + 32);
      const float delta = means[lane] - mean;
      mean = mean + c * delta;
      // Match AddMoments' grouping: m2 += (m2_add + correction).
      m2 = m2 + (m2s[lane] + delta * delta * c * float(count));
      count += 32;
    }
    const float rstd = 1.0f / std::sqrt(m2 / 256.0f + epsilon);
    mean_out[row] = mean;
    rstd_out[row] = rstd;
    for (int j = 0; j < 256; ++j) {
      const float normalized = (x[row * 256 + j] - mean) * rstd;
      out[row * 256 + j] = std::fma(normalized, gamma[j], beta[j]);
    }
  }
}

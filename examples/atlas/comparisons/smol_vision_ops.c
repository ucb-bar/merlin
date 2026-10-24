#include "smol_vision_ops.h"
#include "smol_host_bits.h"
#include <math.h>

static int finite_word(uint16_t value) {
    return (value & UINT16_C(0x7f80)) != UINT16_C(0x7f80);
}

int smol_vision_gelu_bf16(const uint16_t *input, const uint16_t *table,
                         uint16_t *output, size_t count) {
    if (!input || !table || !output || !count) return 1;
    for (size_t i = 0; i < count; ++i)
        if (!finite_word(input[i]) || !finite_word(table[input[i]])) return 2;
    for (size_t i = 0; i < count; ++i) output[i] = table[input[i]];
    return 0;
}

static void merge_vector(unsigned additional, const float *new_mean,
                          const float *new_m2, unsigned *count,
                          float *mean, float *m2) {
    unsigned total = *count + additional;
    float fraction = (float) additional / (float) total;
    for (unsigned lane = 0; lane < 8; ++lane) {
        float delta = new_mean[lane] - mean[lane];
        float increment = fraction * delta;
        float old_delta = delta * (float) *count;
        float combined_m2 = m2[lane] + new_m2[lane];
        mean[lane] = mean[lane] + increment;
        m2[lane] = fmaf(old_delta, increment, combined_m2);
    }
    *count = total;
}

int smol_vision_layernorm_bf16(const uint16_t *input, const uint16_t *weight,
                              const uint16_t *bias, float epsilon,
                              uint16_t *output, size_t rows) {
    if (!input || !weight || !bias || !output || !rows || rows > SIZE_MAX / 768
        || !isfinite(epsilon) || epsilon <= 0) return 1;
    for (size_t i = 0; i < rows * 768; ++i) if (!finite_word(input[i])) return 2;
    for (unsigned i = 0; i < 768; ++i)
        if (!finite_word(weight[i]) || !finite_word(bias[i])) return 2;
    for (size_t row = 0; row < rows; ++row) {
        const uint16_t *source = input + row * 768;
        float means[2][8] = {{0}}, moments[2][8] = {{0}};
        unsigned count[2] = {0};
        for (unsigned chunk = 0; chunk < 3; ++chunk) {
            const unsigned vectors = 16;
            float local_mean[2][8] = {{0}}, local_m2[2][8] = {{0}};
            for (unsigned j = 0; j < vectors; ++j) {
                float reciprocal = 1.0f / (float) (j + 1);
                for (unsigned half = 0; half < 2; ++half)
                    for (unsigned lane = 0; lane < 8; ++lane) {
                        float x = bf16_to_float(source[chunk * 256 + j * 16 + half * 8 + lane]);
                        float delta = x - local_mean[half][lane];
                        local_mean[half][lane] = fmaf(delta, reciprocal, local_mean[half][lane]);
                        float delta_after = x - local_mean[half][lane];
                        local_m2[half][lane] = fmaf(delta, delta_after, local_m2[half][lane]);
                    }
            }
            merge_vector(vectors, local_mean[0], local_m2[0], &count[0], means[0], moments[0]);
            merge_vector(vectors, local_mean[1], local_m2[1], &count[0], means[0], moments[0]);
            if (chunk == 1) {
                merge_vector(count[0], means[0], moments[0], &count[1], means[1], moments[1]);
                count[0] = 0;
                for (unsigned lane = 0; lane < 8; ++lane) means[0][lane] = moments[0][lane] = 0;
            }
        }
        merge_vector(count[1], means[1], moments[1], &count[0], means[0], moments[0]);
        float mean = 0, moment = 0;
        unsigned accumulated = 0;
        for (unsigned lane = 0; lane < 8; ++lane) {
            unsigned total = accumulated + count[0];
            float fraction = (float) count[0] / (float) total;
            float delta = means[0][lane] - mean;
            mean = fmaf(fraction, delta, mean);
            float squared_fraction = delta * delta * fraction;
            moment += fmaf(squared_fraction, (float) accumulated, moments[0][lane]);
            accumulated = total;
        }
        float inverse_std = 1.0f / sqrtf(moment / 768.0f + epsilon);
        float affine_bias = -inverse_std * mean;
        for (unsigned column = 0; column < 768; ++column) {
            float normalized = fmaf(bf16_to_float(source[column]), inverse_std, affine_bias);
            float value = fmaf(normalized, bf16_to_float(weight[column]), bf16_to_float(bias[column]));
            if (!isfinite(value)) return 3;
            output[row * 768 + column] = float_to_bf16(value);
        }
    }
    return 0;
}

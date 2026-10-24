/* Bounded native-host primitives; compile with contraction/fast-math disabled.
 * These routines do not implement the complete policy, target runtime or ELF.
 */
#include "smol_host_ops.h"
#include "smol_host_bits.h"

#include <math.h>

/* Software arithmetic policy matching the pinned native CPU's eight-lane
 * cascaded FP32 sum. Lane width belongs to this explicit reference policy,
 * independent of the machine that executes this portable C implementation.
 * Derivation: PyTorch v2.10.0 SumKernel.cpp row_sum/multi_row_sum; source and
 * BSD license are pinned in the parity evidence. No SIMD ISA is required.
 */
static float norm_input(const void *input, size_t column, int input_is_bf16) {
    return input_is_bf16 ? bf16_to_float(((const uint16_t *)input)[column])
                        : ((const float *)input)[column];
}

static float sum_squares_native_cpu8(const void *input, size_t width, int input_is_bf16) {
    if (width < 8) {
        float sums[4] = {0.0f};
        size_t column = 0;
        for (; width - column >= 4; column += 4) {
            for (size_t group = 0; group < 4; ++group) {
                float value = norm_input(input, column + group, input_is_bf16);
                sums[group] += value * value;
            }
        }
        for (; column < width; ++column) {
            float value = norm_input(input, column, input_is_bf16);
            sums[0] += value * value;
        }
        for (size_t group = 1; group < 4; ++group) sums[0] += sums[group];
        return sums[0];
    }
    float partial[4][4][8] = {{{0.0f}}};
    size_t vectors = width / 8, blocks = vectors / 4;
    size_t logarithm = 0, top = blocks ? blocks - 1 : 0;
    while (top) { ++logarithm;
        top >>= 1;
    }
    size_t power = logarithm / 4;
    if (power < 4) power = 4;
    size_t step = (size_t)1 << power, mask = step - 1, block = 0;
    while (blocks - block >= step) {
        for (size_t item = 0; item < step; ++item, ++block) {
            for (size_t group = 0; group < 4; ++group) {
                for (size_t lane = 0; lane < 8; ++lane) {
                    float value = norm_input(input, (block * 4 + group) * 8 + lane, input_is_bf16);
                    partial[0][group][lane] += value * value;
                }
            }
        }
        for (size_t level = 1; level < 4; ++level) {
            for (size_t group = 0; group < 4; ++group) {
                for (size_t lane = 0; lane < 8; ++lane) {
                    partial[level][group][lane] += partial[level - 1][group][lane];
                    partial[level - 1][group][lane] = 0.0f;
                }
            }
            if (block & (mask << (level * power))) break;
        }
    }
    for (; block < blocks; ++block) {
        for (size_t group = 0; group < 4; ++group) {
            for (size_t lane = 0; lane < 8; ++lane) {
                float value = norm_input(input, (block * 4 + group) * 8 + lane, input_is_bf16);
                partial[0][group][lane] += value * value;
            }
        }
    }
    for (size_t level = 1; level < 4; ++level) {
        for (size_t group = 0; group < 4; ++group) {
            for (size_t lane = 0; lane < 8; ++lane) {
                partial[0][group][lane] += partial[level][group][lane];
            }
        }
    }
    for (size_t vector = blocks * 4; vector < vectors; ++vector) {
        for (size_t lane = 0; lane < 8; ++lane) {
            float value = norm_input(input, vector * 8 + lane, input_is_bf16);
            partial[0][0][lane] += value * value;
        }
    }
    for (size_t group = 1; group < 4; ++group) {
        for (size_t lane = 0; lane < 8; ++lane) {
            partial[0][0][lane] += partial[0][group][lane];
        }
    }
    float total = 0.0f;
    for (size_t column = vectors * 8; column < width; ++column) {
        float value = norm_input(input, column, input_is_bf16);
        total += value * value;
    }
    for (size_t lane = 0; lane < 8; ++lane) total += partial[0][0][lane];
    return total;
}

int smol_host_euler_f32(const float *state, const float *velocity, float dt,
                        float *output, size_t count) {
    if (!state || !velocity || !output || !count || !isfinite(dt)) return -1;
    for (size_t i = 0; i < count; ++i) {
        /* PyTorch executes mul and add separately on FP32 tensors. */
        float increment = dt * velocity[i];
        output[i] = state[i] + increment;
    }
    return 0;
}

int smol_host_residual_bf16(const uint16_t *left, const uint16_t *right,
                           uint16_t *output, size_t count) {
    if (!left || !right || !output || !count) return -1;
    for (size_t i = 0; i < count; ++i) {
        output[i] = float_to_bf16(bf16_to_float(left[i]) + bf16_to_float(right[i]));
    }
    return 0;
}

/* Bounded callers may omit this unused API to retain their linked symbol set. */
#ifndef SMOL_HOST_EXCLUDE_MIXED_RESIDUAL
int smol_host_residual_bf16_f32(const uint16_t *left, const float *right,
                               uint16_t *output, size_t count) {
    if (!left || !right || !output || !count) return -1;
    for (size_t i = 0; i < count; ++i) {
        output[i] = float_to_bf16(bf16_to_float(left[i]) + right[i]);
    }
    return 0;
}

#endif

int smol_host_silu_mul_bf16(const uint16_t *gate, const uint16_t *up,
                           uint16_t *output, size_t count) {
    if (!gate || !up || !output || !count) return -1;
    for (size_t i = 0; i < count; ++i) {
        float value = bf16_to_float(gate[i]);
        /* SiLU rounds to BF16 before the distinct BF16 MLP multiply. */
        uint16_t activated = float_to_bf16(value / (1.0f + expf(-value)));
        output[i] = float_to_bf16(bf16_to_float(activated) * bf16_to_float(up[i]));
    }
    return 0;
}

int smol_host_rmsnorm_bf16(const uint16_t *input, const uint16_t *weight,
                          float epsilon, uint16_t *output, size_t rows, size_t width) {
    if (!input || !weight || !output || !rows || !width || width > SIZE_MAX / rows
        || !isfinite(epsilon) || epsilon <= 0.0f) return -1;
    for (size_t row = 0; row < rows; ++row) {
        float squares = sum_squares_native_cpu8(input + row * width, width, 1);
        float inverse = 1.0f / sqrtf(squares / (float)width + epsilon);
        for (size_t column = 0; column < width; ++column) {
            uint16_t normalized = float_to_bf16(bf16_to_float(input[row * width + column]) * inverse);
            /* LlamaRMSNorm casts normalized values before the BF16 weight multiply. */
            output[row * width + column] = float_to_bf16(bf16_to_float(weight[column]) * bf16_to_float(normalized));
        }
    }
    return 0;
}

int smol_host_rmsnorm_f32_to_bf16(const float *input, const uint16_t *weight,
                                 float epsilon, uint16_t *output, size_t rows, size_t width) {
    if (!input || !weight || !output || !rows || !width || width > SIZE_MAX / rows
        || !isfinite(epsilon) || epsilon <= 0.0f) return -1;
    for (size_t row = 0; row < rows; ++row) {
        float squares = sum_squares_native_cpu8(input + row * width, width, 0);
        float inverse = 1.0f / sqrtf(squares / (float)width + epsilon);
        for (size_t column = 0; column < width; ++column) {
            /* FP32 prefix norm stays FP32 through the weight multiply; the
             * explicit query-weight dtype bridge performs the final BF16 cast. */
            float normalized = input[row * width + column] * inverse;
            output[row * width + column] = float_to_bf16(bf16_to_float(weight[column]) * normalized);
        }
    }
    return 0;
}

#include "smol_policy_raw.h"
#include "host_matrix_fp8pack.h"
#include "smol_host_bits.h"
#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
#include "smol_fusion.h"
#endif
#include <math.h>
#ifdef ATLAS_FAST_MODE
#include "smol_fast_assets.h"
#endif

#ifdef ATLAS_SMOL_LAUNCH_HOST_TIMING
extern void smol_full_report_launch_ticks(size_t index, uint64_t ticks);
static inline uint64_t host_ticks(void) {
    uint64_t value;
    __asm__ volatile ("csrr %0, 0xc00" : "=r" (value) : : "memory");
    return value;
}
#endif

static int reject(smol_policy_raw_context *w, int result) {
    w->failed = 1;
    w->result = result;
    return result;
}

static uint16_t physical(const volatile uint8_t *dram, size_t offset,
                         size_t row, size_t column, size_t columns) {
    size_t tile = (row / 32) * (columns / 32) + column / 32;
    size_t word = tile * 1024 + ((column % 32) / 16) * 512 + (row % 32) * 16 + column % 16;
    return (uint16_t) ((uint16_t) dram[offset + 2 * word]
                     | ((uint16_t) dram[offset + 2 * word + 1] << 8));
}

#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
static uint8_t physical_fp8(const volatile uint8_t *dram, size_t offset, size_t row, size_t column, size_t columns) {
    size_t tile = (row / 32) * (columns / 32) + column / 32;
    return dram[offset + tile * 1024 + (row % 32) * 32 + column % 32];
}
#endif

static float add_bias(uint16_t word, const smol_policy_call *call, size_t column) {
    float value = bf16_to_float(word);
    if (call->bias) value += call->bias_dtype == SMOL_BF16
        ? bf16_to_float(((const uint16_t *) call->bias)[column])
        : ((const float *) call->bias)[column];
    return value;
}

static int invoke(smol_policy_raw_context *w, unsigned site, unsigned layer, unsigned role,
    unsigned image, unsigned dtype, const void *input, size_t rows, size_t k,
    unsigned output_dtype, void *output, size_t n) {
    if (!w) return SMOL_RAW_INVALID;
    if (w->failed) return w->result;
    if (!w->host || w->host->active || !w->panels || !w->calls || !w->dram
        || !w->activation_copy || !w->input_copy || !w->cycle_delta || !w->max_polls
        || !input || !output || dtype > SMOL_F32 || output_dtype > SMOL_F32
        || w->call_cursor >= w->call_count || w->panel_cursor >= w->panel_count
        || w->attention_cursor > w->attention_count
        || w->attention_count > UINT32_MAX || w->panel_count > UINT32_MAX - w->attention_count
        || w->host->invocation_sequence != w->panel_cursor + w->attention_cursor
        || (w->attention_cursor < w->attention_count
            && (!w->attention_panels || w->attention_panels[w->attention_cursor].before_call <= w->call_cursor))
        || !rows || !k || !n || rows > SIZE_MAX / k || rows * k > w->input_capacity
        || w->dram_capacity > 64 * 1024 * 1024) return reject(w, SMOL_RAW_INVALID);
    const smol_policy_call *call = &w->calls[w->call_cursor];
    if (call->rows != rows || call->input_width != k || call->output_width != n
        || call->site != site || call->layer != layer || call->role != role || call->image != image
        || call->input_dtype != dtype || call->output_dtype != output_dtype
        || call->bias_dtype > SMOL_F32 || call->first_panel != w->panel_cursor
        || !call->panel_count || call->panel_count > w->panel_count - w->panel_cursor
        || call->activation_scale < -126 || call->activation_scale > 127) return reject(w, SMOL_RAW_INVALID);
#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
    int scale;
    int input_view = smol_fusion_input_view(w, input, rows, k, &scale);
    if (input_view < 0) return reject(w, SMOL_RAW_INVALID);
    unsigned fusion_operations;
    if (smol_fusion_call_operations(w, &fusion_operations)) return reject(w, SMOL_RAW_INVALID);
    if (!input_view) {
#endif
    /* One conversion and scale over the complete actual operator input. */
    for (size_t i = 0; i < rows * k; ++i) {
        float value = dtype == SMOL_BF16 ? bf16_to_float(((const uint16_t *) input)[i]) : ((const float *) input)[i];
        if (!isfinite(value)) return reject(w, SMOL_RAW_INPUT);
        w->input_copy[i] = float_to_bf16(value);
    }
#if !defined(SMOL_FUSION_RUNTIME) || !SMOL_FUSION_RUNTIME
    int scale;
#endif
    if (host_matrix_scale_bf16(w->input_copy, rows, k, &scale)) return reject(w, SMOL_RAW_INPUT);
#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
    }
#endif
    if (scale != call->activation_scale) return reject(w, SMOL_RAW_SCALE);
    size_t next_row = 0;
    for (size_t panel = 0; panel < call->panel_count; ++panel) {
#ifdef ATLAS_SMOL_LAUNCH_HOST_TIMING
        uint64_t host_before = host_ticks();
#endif
        const smol_policy_panel *p = &w->panels[w->panel_cursor];
#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
        const smol_fusion_panel *fusion;
        if (smol_fusion_panel_metadata(w, w->panel_cursor, &fusion)) return reject(w, SMOL_RAW_INVALID);
#endif
        size_t m = p->physical_rows, pk = p->physical_k, pn = p->physical_n;
#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
        /* Keep the shared loop bases in registers in fusion builds; plain builds keep their pinned code. */
        uint8_t *const dram = w->dram, *const activation = w->activation_copy;
#ifndef ATLAS_FAST_MODE
        const uint8_t *const weights = p->weights;
#endif
#define PANEL_DRAM dram
#define PANEL_ACTIVATION activation
#define PANEL_WEIGHTS weights
#else
#define PANEL_DRAM w->dram
#define PANEL_ACTIVATION w->activation_copy
#define PANEL_WEIGHTS p->weights
#endif
#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
        size_t output_element_bytes = fusion_operations & SMOL_FUSION_FP8 ? 1 : 2;
#endif
        if (!m || !pk || !pn || m % 32 || pk % 32 || pn % 32 || m > 128
            || p->call != w->call_cursor || p->row_start != next_row || p->input_width != k
            || p->output_width != n || !p->logical_rows || p->logical_rows > rows - next_row
            || m != (p->logical_rows + 31) / 32 * 32 || pk != (k + 31) / 32 * 32
            || pn != (n + 31) / 32 * 32 || m > SIZE_MAX / pk || pk > SIZE_MAX / pn
            || m > SIZE_MAX / pn / 2 || p->weight_bytes != pk * pn || !p->weights
            || p->output_offset != m * pk + p->weight_bytes + 256
#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
            || (fusion && fusion->extra_bytes
                ? p->dram_bytes != p->output_offset + output_element_bytes * m * pn + 256 + fusion->extra_bytes + 256
                : p->dram_bytes != p->output_offset + output_element_bytes * m * pn + 256)
#else
            || p->dram_bytes != p->output_offset + 2 * m * pn + 256
#endif
            || p->dram_bytes > w->dram_capacity || m * pk > w->activation_capacity
#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
            || p->store_bytes != output_element_bytes * m * pn || !p->load_bytes
#else
            || p->store_bytes != 2 * m * pn || !p->load_bytes
#endif
            || !p->program || !p->program_words || p->program_words > 8192
            || p->rescale_word >= p->program_words - 1 || p->activation_scale != scale
            || p->weight_scale < -126 || p->weight_scale > 127) return reject(w, SMOL_RAW_INVALID);
        int exponent = scale + p->weight_scale;
        if (exponent < -126 || exponent > 127
            || p->program[p->rescale_word] != (((uint32_t) (exponent + 127) << 23) | UINT32_C(0x25f)))
            return reject(w, SMOL_RAW_INVALID);
#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
        if (input_view) {
            const uint8_t *packed = input;
            for (size_t i = 0; i < m * pk; ++i) PANEL_ACTIVATION[i] = packed[next_row * pk + i];
        } else
#endif
        if (host_matrix_pack_bf16_at_scale(w->input_copy + next_row * k, p->logical_rows, k,
                m, pk, 32, 32, scale, w->activation_copy, w->activation_capacity)) return reject(w, SMOL_RAW_INPUT);
        for (size_t i = 0; i < m * pk; ++i) PANEL_DRAM[i] = PANEL_ACTIVATION[i];
#ifndef ATLAS_FAST_MODE
        for (size_t i = 0; i < p->weight_bytes; ++i) PANEL_DRAM[m * pk + i] = PANEL_WEIGHTS[i];
#endif
        for (size_t i = m * pk + p->weight_bytes; i < p->output_offset; ++i) PANEL_DRAM[i] = 0xa5;
        for (size_t i = p->output_offset; i < p->output_offset + p->store_bytes; ++i) PANEL_DRAM[i] = 0xcd;
        for (size_t i = p->output_offset + p->store_bytes; i < p->dram_bytes; ++i) PANEL_DRAM[i] = 0x5a;
#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
        if (fusion)
            for (size_t i = 0; i < fusion->extra_bytes; ++i) PANEL_DRAM[fusion->extra_offset + i] = fusion->extra[i];
#endif
#ifdef ATLAS_FAST_MODE
        const smol_fast_panel *fast = &smol_fast_panels[w->panel_cursor];
        if (fast->encoded_weight_base != (uintptr_t) p->weights || !fast->program || !fast->program_words)
            return reject(w, SMOL_RAW_INVALID);
        const uint32_t *program = fast->program;
        size_t program_words = fast->program_words;
        if (atlas_host_load_raw(w->host, program, program_words) != ATLAS_HOST_OK) return reject(w, SMOL_RAW_RUNTIME);
#else
        if (atlas_host_load_raw(w->host, p->program, p->program_words) != ATLAS_HOST_OK) return reject(w, SMOL_RAW_RUNTIME);
#endif
        w->active_buffer = (atlas_host_buffer) {(uintptr_t) w->dram, p->dram_bytes, ATLAS_HOST_DEVICE_READ_WRITE};
        uint32_t before = w->host->mmio.read32(w->host->mmio.context, w->host->map.csr_base);
#ifdef ATLAS_FAST_MODE
        w->fast_buffers[0] = w->active_buffer;
        w->fast_buffers[1] = (atlas_host_buffer) {(uintptr_t) p->weights, p->weight_bytes, ATLAS_HOST_DEVICE_READ};
        atlas_host_result result = atlas_host_start(w->host, w->fast_buffers, 2);
#else
        atlas_host_result result = atlas_host_start(w->host, &w->active_buffer, 1);
#endif
        if (result == ATLAS_HOST_OK) result = atlas_host_wait(w->host, w->max_polls);
        w->cycle_delta[w->panel_cursor + w->attention_cursor] =
            w->host->mmio.read32(w->host->mmio.context, w->host->map.csr_base) - before;
        if (result != ATLAS_HOST_OK) return reject(w, SMOL_RAW_RUNTIME);
        for (size_t i = 0; i < m * pk; ++i)
            if (((volatile uint8_t *) PANEL_DRAM)[i] != PANEL_ACTIVATION[i]) return reject(w, SMOL_RAW_INPUT);
#ifndef ATLAS_FAST_MODE
        for (size_t i = 0; i < p->weight_bytes; ++i)
            if (((volatile uint8_t *) PANEL_DRAM)[m * pk + i] != PANEL_WEIGHTS[i]) return reject(w, SMOL_RAW_INPUT);
#endif
        for (size_t i = m * pk + p->weight_bytes; i < p->output_offset; ++i)
            if (((volatile uint8_t *) PANEL_DRAM)[i] != 0xa5) return reject(w, SMOL_RAW_GUARD);
#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
        size_t guard_end = fusion && fusion->extra_bytes ? fusion->extra_offset : p->dram_bytes;
        for (size_t i = p->output_offset + p->store_bytes; i < guard_end; ++i)
            if (((volatile uint8_t *) PANEL_DRAM)[i] != 0x5a) return reject(w, SMOL_RAW_GUARD);
        if (fusion && fusion->extra_bytes) {
            for (size_t i = 0; i < fusion->extra_bytes; ++i)
                if (((volatile uint8_t *) PANEL_DRAM)[fusion->extra_offset + i] != fusion->extra[i]) return reject(w, SMOL_RAW_INPUT);
            for (size_t i = fusion->extra_offset + fusion->extra_bytes; i < p->dram_bytes; ++i)
                if (((volatile uint8_t *) PANEL_DRAM)[i] != 0x5a) return reject(w, SMOL_RAW_GUARD);
        }
#else
        for (size_t i = p->output_offset + p->store_bytes; i < p->dram_bytes; ++i)
            if (((volatile uint8_t *) PANEL_DRAM)[i] != 0x5a) return reject(w, SMOL_RAW_GUARD);
#endif
#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
        if (fusion_operations & SMOL_FUSION_FP8) {
            for (size_t row = 0; row < m; ++row)
                for (size_t column = 0; column < pn; ++column) {
                    uint8_t value = physical_fp8(PANEL_DRAM, p->output_offset, row, column, pn);
                    if ((value & 0x7f) == 0x7f
                        || ((row >= p->logical_rows || column >= n) && value)) return reject(w, SMOL_RAW_OUTPUT);
                }
            for (size_t i = 0; i < m * pn; ++i)
                ((uint8_t *) output)[next_row * pn + i] = ((volatile uint8_t *) PANEL_DRAM)[p->output_offset + i];
        } else {
#endif
        for (size_t row = 0; row < m; ++row)
            for (size_t column = 0; column < pn; ++column) {
                uint16_t value = physical(PANEL_DRAM, p->output_offset, row, column, pn);
                if ((value & 0x7f80) == 0x7f80
                    || ((row >= p->logical_rows || column >= n) && value != 0)) return reject(w, SMOL_RAW_OUTPUT);
                if (row < p->logical_rows && column < n) {
#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
                    float biased = fusion_operations & SMOL_FUSION_BIAS ? bf16_to_float(value) : add_bias(value, call, column);
#else
                    float biased = add_bias(value, call, column);
#endif
                    if (!isfinite(biased) || (output_dtype == SMOL_BF16
                        && (float_to_bf16(biased) & 0x7f80) == 0x7f80)) return reject(w, SMOL_RAW_OUTPUT);
                }
            }
        for (size_t row = 0; row < p->logical_rows; ++row)
            for (size_t column = 0; column < n; ++column) {
#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
                uint16_t word = physical(PANEL_DRAM, p->output_offset, row, column, pn);
                float value = fusion_operations & SMOL_FUSION_BIAS ? bf16_to_float(word) : add_bias(word, call, column);
#else
                float value = add_bias(physical(PANEL_DRAM, p->output_offset, row, column, pn), call, column);
#endif
                size_t index = (next_row + row) * n + column;
                if (output_dtype == SMOL_BF16) ((uint16_t *) output)[index] = float_to_bf16(value);
                else ((float *) output)[index] = value;
            }
#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
        }
#endif
        next_row += p->logical_rows;
#ifdef ATLAS_SMOL_LAUNCH_HOST_TIMING
        smol_full_report_launch_ticks(w->panel_cursor + w->attention_cursor, host_ticks() - host_before);
#endif
        ++w->panel_cursor;
#undef PANEL_DRAM
#undef PANEL_ACTIVATION
#undef PANEL_WEIGHTS
    }
    if (next_row != rows) return reject(w, SMOL_RAW_INVALID);
#if defined(SMOL_FUSION_RUNTIME) && SMOL_FUSION_RUNTIME
    if (smol_fusion_record_success(w, output, rows, n, fusion_operations)) return reject(w, SMOL_RAW_INVALID);
#endif
    ++w->call_cursor;
    return SMOL_RAW_OK;
}

int smol_policy_raw_graph(void *opaque, unsigned site, unsigned layer, unsigned role,
    unsigned dtype, const void *input, size_t rows, size_t k, unsigned output_dtype, void *output, size_t n) {
    return invoke(opaque, site, layer, role, 0, dtype, input, rows, k, output_dtype, output, n);
}

int smol_policy_raw_vision(void *opaque, unsigned layer, unsigned role,
    const uint16_t *input, size_t rows, size_t k, uint16_t *output, size_t n) {
    smol_policy_raw_context *w = opaque;
    if (!w) return SMOL_RAW_INVALID;
    return invoke(w, 7, layer, role, w->image, SMOL_BF16, input, rows, k, SMOL_BF16, output, n);
}

static uint32_t metadata_word(const smol_attention_panel *p, size_t offset) {
    const uint8_t *s = p->metadata + offset - p->descriptor_offset;
    return (uint32_t) s[0] | ((uint32_t) s[1] << 8) | ((uint32_t) s[2] << 16) | ((uint32_t) s[3] << 24);
}

static size_t attention_offset(size_t row, size_t head, size_t dim, size_t heads) {
    return ((row / 32) * heads + head) * 4096 + (dim / 16) * 1024 + (row % 32) * 32 + (dim % 16) * 2;
}

/* Both directions use individual Q/K/V tensors, with no merged-QKV stride. */
static int attention_operand(uint8_t *arena, size_t offset, const uint16_t *input,
    size_t rows, size_t heads, int verify) {
    const volatile uint8_t *observed = arena;
    size_t padded = (rows + 31) / 32 * 32;
    for (size_t row = 0; row < padded; ++row)
        for (size_t head = 0; head < heads; ++head)
            for (size_t dim = 0; dim < 64; ++dim) {
                uint16_t word = row < rows ? input[(row * heads + head) * 64 + dim] : 0;
                size_t at = offset + attention_offset(row, head, dim, heads);
                if ((word & 0x7f80) == 0x7f80) return 1;
                if (verify) {
                    if (observed[at] != (uint8_t) word || observed[at + 1] != (uint8_t) (word >> 8)) return 1;
                } else {
                    arena[at] = (uint8_t) word;
                    arena[at + 1] = (uint8_t) (word >> 8);
                }
            }
    return 0;
}

static int attention_mask(const smol_attention_panel *p, const uint8_t *mask) {
    size_t qt = (p->rows + 31) / 32, kt = (p->keys + 31) / 32;
    if ((p->expected_mask == NULL) != (mask == NULL)) return 1;
    if (mask)
        for (size_t i = 0; i < p->rows * p->keys; ++i)
            if (mask[i] > 1 || mask[i] != p->expected_mask[i]) return 1;
    for (size_t qi = 0; qi < qt; ++qi) {
        size_t visible = 0;
        for (size_t ki = 0; ki < kt; ++ki) {
            size_t desc = p->descriptor_offset + (qi * kt + ki) * 8;
            uint32_t address = metadata_word(p, desc), skip = metadata_word(p, desc + 4);
            size_t allowed = 0;
            if (skip > 1 || (skip && address)) return 1;
            size_t mask_at = 0;
            if (address) {
                if (address < 0x4000 || (address - 0x4000) % 512) return 1;
                mask_at = (size_t) (address - 0x4000) * 4;
                if (mask_at >= p->mask_bytes || p->mask_bytes - mask_at < 2048) return 1;
            }
            for (size_t r = 0; r < 32; ++r)
                for (size_t k = 0; k < 32; ++k) {
                    size_t row = qi * 32 + r, key = ki * 32 + k;
                    int yes = row < p->rows && key < p->keys && (!mask || mask[row * p->keys + key]);
                    allowed += (size_t) yes;
                    if (address) {
                        size_t at = p->mask_offset + mask_at + (k / 16) * 1024 + r * 32 + (k % 16) * 2;
                        const uint8_t *s = p->metadata + at - p->descriptor_offset;
                        uint16_t word = (uint16_t) s[0] | (uint16_t) ((uint16_t) s[1] << 8);
                        if (word != (yes ? 0 : 0x7f80)) return 1;
                    }
                }
            if ((!allowed && (!skip || address)) || (allowed == 1024 && (skip || address))
                || (allowed && allowed < 1024 && (skip || !address))) return 1;
            if (allowed) visible = ki + 1;
        }
        if (metadata_word(p, p->key_count_offset + qi * 4) != visible) return 1;
    }
    for (size_t at = qt * kt * 8; at < p->descriptor_bytes; ++at)
        if (p->metadata[at]) return 1;
    for (size_t at = qt * 4; at < p->key_count_bytes; ++at)
        if (p->metadata[p->key_count_offset - p->descriptor_offset + at]) return 1;
    return 0;
}

static int overlap(const void *a, size_t an, const void *b, size_t bn) {
    uintptr_t av = (uintptr_t) a, bv = (uintptr_t) b;
    if (an > UINTPTR_MAX - av || bn > UINTPTR_MAX - bv) return 1;
    return av < bv + bn && bv < av + an;
}

/* Admit attention's fixed scale literals separately from matmul's rescale. */
static int attention_program(const smol_attention_panel *p) {
    unsigned seen[5] = {0}, scale_seen = 0;
    int exponents[5] = {p->q_exponent, p->k_exponent, p->v_exponent, 0, 0};
    uint32_t scale = ((uint32_t) (p->q_exponent + p->k_exponent - 3 + 127) << 23) | UINT32_C(0x45f);
    if (p->program[p->program_words - 1] != UINT32_C(0x73)) return 1;
    for (size_t i = 0; i < p->program_words; ++i) {
        uint32_t word = p->program[i];
        if ((word & UINT32_C(0x707f)) == UINT32_C(0x7003)) {
            unsigned reg = (word >> 7) & 31;
            if (reg >= 5 || ++seen[reg] != 1 || word !=
                (((uint32_t) (127 - exponents[reg]) << 20) | (reg << 7) | UINT32_C(0x7003))) return 1;
        }
        if ((word & UINT32_C(0xffff)) == UINT32_C(0x45f)) {
            if (word != scale || ++scale_seen != 1) return 1;
        }
        /* vli.all and vli.row both target a BF16 register pair. */
        if ((word & UINT32_C(0xc07f)) == UINT32_C(0x5f) && (word & UINT32_C(0x80))) return 1;
    }
    return !scale_seen || !seen[0] || !seen[1] || !seen[2] || !seen[3] || !seen[4];
}

int smol_policy_raw_attention(void *opaque, unsigned kind, const uint16_t *q, const uint16_t *k,
    const uint16_t *v, const uint8_t *mask, size_t rows, size_t keys,
    size_t qheads, size_t kvheads, uint16_t *output) {
    smol_policy_raw_context *w = opaque;
    if (!w) return SMOL_RAW_INVALID;
    if (w->failed) return w->result;
#ifdef ATLAS_SMOL_LAUNCH_HOST_TIMING
    uint64_t host_before = host_ticks();
#endif
    if (!w->host || w->host->active || !w->attention_panels || !w->dram || !w->cycle_delta
        || !w->max_polls || !q || !k || !v || !output || kind > SMOL_ATTENTION_CROSS
        || w->attention_cursor >= w->attention_count || w->panel_cursor > w->panel_count
        || w->call_cursor > w->call_count || w->attention_count > UINT32_MAX
        || w->panel_count > UINT32_MAX - w->attention_count
        || w->host->invocation_sequence != w->panel_cursor + w->attention_cursor
        || rows != (kind == SMOL_ATTENTION_VISION ? 1024u : kind == SMOL_ATTENTION_PREFIX ? 177u : 50u)
        || keys != (kind == SMOL_ATTENTION_VISION ? 1024u : kind == SMOL_ATTENTION_SELF ? 227u : 177u)
        || qheads != (kind == SMOL_ATTENTION_VISION ? 12u : 15u)
        || kvheads != (kind == SMOL_ATTENTION_VISION ? 12u : 5u)) return reject(w, SMOL_RAW_INVALID);
    const smol_attention_panel *p = &w->attention_panels[w->attention_cursor];
    size_t qt = (rows + 31) / 32, kt = (keys + 31) / 32;
    size_t qb = qt * qheads * 4096, kb = kt * kvheads * 4096;
    unsigned phase_kind = p->phase < 24 ? SMOL_ATTENTION_VISION : p->phase < 40
        ? SMOL_ATTENTION_PREFIX : (p->phase % 2 ? SMOL_ATTENTION_CROSS : SMOL_ATTENTION_SELF);
    if (p->kind != kind || p->rows != rows || p->keys != keys || p->query_heads != qheads
        || p->kv_heads != kvheads || p->before_call != w->call_cursor || p->phase != w->attention_cursor
        || p->index != w->attention_cursor || phase_kind != kind || p->phase >= 200
        || !p->program || !p->metadata || !p->program_words
        || w->host->map.imem_capacity_bytes / 4 < 5
        || p->program_words > w->host->map.imem_capacity_bytes / 4 - 5
        || p->q_offset || p->k_offset != qb || p->v_offset != qb + kb
        || p->descriptor_offset != qb + 2 * kb
        || p->descriptor_bytes != (qt * kt * 8 + 31) / 32 * 32
        || p->descriptor_bytes > 16384 || p->key_count_bytes != (qt * 4 + 31) / 32 * 32
        || p->key_count_offset != p->descriptor_offset + p->descriptor_bytes
        || p->mask_offset != p->key_count_offset + p->key_count_bytes
        || p->mask_bytes > 196608 || p->mask_bytes % 2048
        || p->metadata_bytes != p->descriptor_bytes + p->key_count_bytes + p->mask_bytes
        || p->output_offset != p->descriptor_offset + p->metadata_bytes + 256
        || p->output_bytes != qb || p->store_bytes != qb
        || p->dram_bytes != p->output_offset + qb + 256
        || p->dram_bytes > w->dram_capacity || w->dram_capacity > 64 * 1024 * 1024
        || (uintptr_t) w->dram % 32 || (uintptr_t) w->dram > UINT32_MAX - p->dram_bytes
        || p->load_bytes != qb + 2 * kb + p->metadata_bytes
        || p->q_exponent < -60 || p->q_exponent > 60 || p->k_exponent < -60 || p->k_exponent > 60
        || p->v_exponent < -60 || p->v_exponent > 60
        || (kind == SMOL_ATTENTION_VISION ? mask != NULL || p->expected_mask != NULL
            : mask == NULL || p->expected_mask == NULL))
        return reject(w, SMOL_RAW_INVALID);
    size_t qlogical = rows * qheads * 128, kvlogical = keys * kvheads * 128;
    if (overlap(w->dram, p->dram_bytes, q, qlogical) || overlap(w->dram, p->dram_bytes, k, kvlogical)
        || overlap(w->dram, p->dram_bytes, v, kvlogical)
        || overlap(w->dram, p->dram_bytes, output, qlogical)
        || overlap(output, qlogical, q, qlogical) || overlap(output, qlogical, k, kvlogical)
        || overlap(output, qlogical, v, kvlogical)
        || overlap(w->dram, p->dram_bytes, p->metadata, p->metadata_bytes)
        || overlap(output, qlogical, p->metadata, p->metadata_bytes)
        || overlap(w->dram, p->dram_bytes, p->program, p->program_words * 4)
        || overlap(output, qlogical, p->program, p->program_words * 4)
        || (mask && (overlap(w->dram, p->dram_bytes, mask, rows * keys)
            || overlap(output, qlogical, mask, rows * keys)
            || overlap(w->dram, p->dram_bytes, p->expected_mask, rows * keys)
            || overlap(output, qlogical, p->expected_mask, rows * keys)))
        || attention_program(p)) return reject(w, SMOL_RAW_INVALID);
    if (attention_mask(p, mask)) return reject(w, SMOL_RAW_INPUT);
    if (attention_operand(w->dram, p->q_offset, q, rows, qheads, 0)
        || attention_operand(w->dram, p->k_offset, k, keys, kvheads, 0)
        || attention_operand(w->dram, p->v_offset, v, keys, kvheads, 0)) return reject(w, SMOL_RAW_INPUT);
    for (size_t i = 0; i < p->metadata_bytes; ++i) w->dram[p->descriptor_offset + i] = p->metadata[i];
    for (size_t i = p->output_offset - 256; i < p->output_offset; ++i) w->dram[i] = 0xa5;
    for (size_t i = 0; i < p->output_bytes; ++i) w->dram[p->output_offset + i] = 0;
    for (size_t i = p->output_offset + p->output_bytes; i < p->dram_bytes; ++i) w->dram[i] = 0x5a;
    if (atlas_host_load_raw(w->host, p->program, p->program_words) != ATLAS_HOST_OK) return reject(w, SMOL_RAW_RUNTIME);
    w->active_buffer = (atlas_host_buffer) {(uintptr_t) w->dram, p->dram_bytes, ATLAS_HOST_DEVICE_READ_WRITE};
    uint32_t before = w->host->mmio.read32(w->host->mmio.context, w->host->map.csr_base);
    atlas_host_result result = atlas_host_start(w->host, &w->active_buffer, 1);
    if (result == ATLAS_HOST_OK) result = atlas_host_wait(w->host, w->max_polls);
    w->cycle_delta[w->panel_cursor + w->attention_cursor] =
        w->host->mmio.read32(w->host->mmio.context, w->host->map.csr_base) - before;
    if (result != ATLAS_HOST_OK) return reject(w, SMOL_RAW_RUNTIME);
    if (attention_operand(w->dram, p->q_offset, q, rows, qheads, 1)
        || attention_operand(w->dram, p->k_offset, k, keys, kvheads, 1)
        || attention_operand(w->dram, p->v_offset, v, keys, kvheads, 1)) return reject(w, SMOL_RAW_INPUT);
    const volatile uint8_t *observed = w->dram;
    for (size_t i = 0; i < p->metadata_bytes; ++i)
        if (observed[p->descriptor_offset + i] != p->metadata[i]) return reject(w, SMOL_RAW_INPUT);
    for (size_t i = p->output_offset - 256; i < p->output_offset; ++i)
        if (observed[i] != 0xa5) return reject(w, SMOL_RAW_GUARD);
    for (size_t i = p->output_offset + p->output_bytes; i < p->dram_bytes; ++i)
        if (observed[i] != 0x5a) return reject(w, SMOL_RAW_GUARD);
    for (size_t row = 0; row < qt * 32; ++row)
        for (size_t head = 0; head < qheads; ++head)
            for (size_t dim = 0; dim < 64; ++dim) {
                size_t at = p->output_offset + attention_offset(row, head, dim, qheads);
                uint16_t word = (uint16_t) observed[at] | (uint16_t) ((uint16_t) observed[at + 1] << 8);
                if ((word & 0x7f80) == 0x7f80 || (row >= rows && word)) return reject(w, SMOL_RAW_OUTPUT);
            }
    for (size_t row = 0; row < rows; ++row)
        for (size_t head = 0; head < qheads; ++head)
            for (size_t dim = 0; dim < 64; ++dim) {
                size_t at = p->output_offset + attention_offset(row, head, dim, qheads);
                output[(row * qheads + head) * 64 + dim] =
                    (uint16_t) observed[at] | (uint16_t) ((uint16_t) observed[at + 1] << 8);
            }
#ifdef ATLAS_SMOL_LAUNCH_HOST_TIMING
    smol_full_report_launch_ticks(w->panel_cursor + w->attention_cursor, host_ticks() - host_before);
#endif
    ++w->attention_cursor;
    return SMOL_RAW_OK;
}

#include "smol_fusion.h"
#include "smol_policy_raw.h"

static struct {
    smol_policy_raw_context *raw;
    const smol_fusion_panel *panels;
    size_t count, last_call, last_sequence, last_attention;
    const void *last_output;
    size_t last_rows, last_width;
    unsigned configured, valid, last_success, last_image, last_layer, last_role, last_operations;
    int last_scale;
} state;

static int overlap(const void *a, size_t an, const void *b, size_t bn) {
    uintptr_t av = (uintptr_t) a, bv = (uintptr_t) b;
    if (an > UINTPTR_MAX - av || bn > UINTPTR_MAX - bv) return 1;
    return av < bv + bn && bv < av + an;
}

static int eligible(const smol_policy_call *call, unsigned operations) {
    if (!operations) return 1;
    if (operations & ~(SMOL_FUSION_BIAS | SMOL_FUSION_GELU | SMOL_FUSION_CONNECTOR | SMOL_FUSION_FP8)) return 0;
    if (call->site != 7 || call->input_dtype != SMOL_BF16 || call->output_dtype != SMOL_BF16) return 0;
    if ((operations & SMOL_FUSION_FP8) && operations != (SMOL_FUSION_BIAS | SMOL_FUSION_GELU | SMOL_FUSION_FP8)) return 0;
    if (operations & SMOL_FUSION_CONNECTOR)
        return operations == SMOL_FUSION_CONNECTOR && call->role == SMOL_VISION_CONNECTOR && call->layer == 12
            && call->rows == 64 && call->input_width == 12288 && call->output_width == 960 && !call->bias;
    if (!(operations & SMOL_FUSION_BIAS) || !call->bias || call->bias_dtype != SMOL_BF16
        || call->role > SMOL_VISION_FC2 || call->layer >= 12
        || (call->role == SMOL_VISION_PATCH && call->layer) || call->output_width % 32) return 0;
    return !(operations & SMOL_FUSION_GELU) || call->role == SMOL_VISION_FC1;
}

static int panel_valid(const smol_policy_raw_context *raw, size_t index, const smol_fusion_panel *fusion) {
    if (!raw->panels || !raw->calls || index >= raw->panel_count) return 0;
    const smol_policy_panel *p = &raw->panels[index];
    if (p->call >= raw->call_count) return 0;
    const smol_policy_call *call = &raw->calls[p->call];
    size_t m = p->physical_rows, n = p->physical_n;
    size_t element_bytes = fusion->operations & SMOL_FUSION_FP8 ? 1 : 2;
    if (!m || !n || m > SIZE_MAX / n / element_bytes || element_bytes * m * n > SIZE_MAX - 256
        || p->output_offset > SIZE_MAX - element_bytes * m * n - 256
        || p->store_bytes != element_bytes * m * n
        || (fusion->operations & SMOL_FUSION_FP8 ? fusion->output_scale < -126 || fusion->output_scale > 127 : fusion->output_scale != 0)
        || !eligible(call, fusion->operations)) return 0;
    size_t old_end = p->output_offset + element_bytes * m * n + 256;
    if (!(fusion->operations & SMOL_FUSION_BIAS))
        return !fusion->extra && !fusion->extra_offset && !fusion->extra_bytes && p->dram_bytes == old_end;
    if (n != call->output_width || n > (SIZE_MAX - 256) / 64 || !fusion->extra
        || fusion->extra_bytes != n * 64 || fusion->extra_offset != old_end
        || old_end > SIZE_MAX - 256 || fusion->extra_bytes > SIZE_MAX - old_end - 256
        || p->dram_bytes != old_end + fusion->extra_bytes + 256 || p->dram_bytes > raw->dram_capacity
        || overlap(raw->dram, raw->dram_capacity, fusion->extra, fusion->extra_bytes)) return 0;
    return 1;
}

static int bias_tiles_match(const smol_policy_call *call, const smol_fusion_panel *fusion) {
    const uint16_t *bias = call->bias;
    for (size_t column = 0; column < call->output_width; ++column)
        for (size_t row = 0; row < 32; ++row) {
            size_t word = (column / 32) * 1024 + ((column % 32) / 16) * 512 + row * 16 + column % 16;
            uint16_t value = (uint16_t) fusion->extra[2 * word] | (uint16_t) ((uint16_t) fusion->extra[2 * word + 1] << 8);
            if (value != bias[column]) return 0;
        }
    return 1;
}

static int call_valid(const smol_policy_raw_context *raw, size_t index, unsigned *operations) {
    if (!raw->calls || !raw->panels || index >= raw->call_count) return 0;
    const smol_policy_call *call = &raw->calls[index];
    if (!call->panel_count || call->first_panel >= state.count || call->panel_count > state.count - call->first_panel) return 0;
    *operations = state.panels[call->first_panel].operations;
    for (size_t i = call->first_panel; i < call->first_panel + call->panel_count; ++i)
        if (raw->panels[i].call != index || state.panels[i].operations != *operations
            || state.panels[i].output_scale != state.panels[call->first_panel].output_scale
            || !panel_valid(raw, i, &state.panels[i])) return 0;
    return 1;
}

static int fp8_consumer_valid(const smol_policy_raw_context *raw, size_t producer, int scale) {
    if (producer >= raw->call_count || producer + 1 >= raw->call_count) return 0;
    const smol_policy_call *a = &raw->calls[producer], *b = &raw->calls[producer + 1];
    if (a->site != 7 || a->role != SMOL_VISION_FC1 || b->site != 7 || b->role != SMOL_VISION_FC2
        || a->image != b->image || a->layer != b->layer || a->rows != b->rows
        || a->output_width != b->input_width || b->input_dtype != SMOL_BF16 || b->output_dtype != SMOL_BF16
        || b->activation_scale != scale || a->panel_count != b->panel_count
        || b->first_panel >= raw->panel_count || b->panel_count > raw->panel_count - b->first_panel
        || !a->rows || a->rows > SIZE_MAX / 2) return 0;
    size_t physical_rows = 0;
    for (size_t i = 0; i < a->panel_count; ++i) {
        const smol_policy_panel *p = &raw->panels[a->first_panel + i], *q = &raw->panels[b->first_panel + i];
        if (p->row_start % 32 || p->physical_rows > SIZE_MAX - physical_rows
            || q->call != producer + 1 || p->row_start != q->row_start || p->logical_rows != q->logical_rows
            || p->physical_rows != q->physical_rows || p->physical_n != q->physical_k
            || q->activation_scale != scale) return 0;
        physical_rows += p->physical_rows;
    }
    /* The public output buffer retains its original BF16 allocation. */
    return physical_rows <= 2 * a->rows;
}

int smol_fusion_configure(void *opaque, const smol_fusion_panel *panels, size_t count) {
    state.configured = opaque || panels || count;
    state.raw = opaque;
    state.panels = panels;
    state.count = count;
    state.valid = state.last_success = 0;
    state.last_operations = 0;
    state.last_scale = 0;
    if (!state.configured) return 0;
    smol_policy_raw_context *raw = opaque;
    if (!raw || !raw->host || raw->host->active || raw->failed || raw->call_cursor || raw->panel_cursor
        || raw->attention_cursor || !raw->calls || !raw->panels || !panels || !count
        || count != raw->panel_count || !raw->call_count || !raw->dram || !raw->dram_capacity
        || raw->dram_capacity > 64 * 1024 * 1024 || count > SIZE_MAX / sizeof(*panels)
        || overlap(raw->dram, raw->dram_capacity, panels, count * sizeof(*panels))) return -1;
    size_t covered = 0;
    for (size_t i = 0; i < raw->call_count; ++i) {
        unsigned operations;
        if (raw->calls[i].first_panel != covered || !call_valid(raw, i, &operations)) return -1;
        if ((operations & SMOL_FUSION_FP8) && !fp8_consumer_valid(raw, i, panels[covered].output_scale)) return -1;
        covered += raw->calls[i].panel_count;
        if (operations & SMOL_FUSION_BIAS)
            for (size_t j = raw->calls[i].first_panel; j < covered; ++j)
                if (!bias_tiles_match(&raw->calls[i], &panels[j])) return -1;
    }
    if (covered != count) return -1;
    state.valid = 1;
    return 0;
}

int smol_fusion_call_operations(void *opaque, unsigned *operations) {
    if (!operations) return -1;
    *operations = 0;
    if (!state.configured) return 0;
    state.last_success = 0;
    smol_policy_raw_context *raw = opaque;
    if (!state.valid || raw != state.raw || state.count != raw->panel_count
        || raw->failed || !call_valid(raw, raw->call_cursor, operations)) return -1;
    if ((*operations & SMOL_FUSION_FP8)
        && !fp8_consumer_valid(raw, raw->call_cursor, state.panels[raw->calls[raw->call_cursor].first_panel].output_scale)) return -1;
    return 0;
}

int smol_fusion_panel_metadata(void *opaque, size_t index, const smol_fusion_panel **panel) {
    if (!panel) return -1;
    *panel = NULL;
    if (!state.configured) return 0;
    smol_policy_raw_context *raw = opaque;
    if (!state.valid || raw != state.raw || state.count != raw->panel_count || index >= state.count
        || raw->failed || !panel_valid(raw, index, &state.panels[index])) return -1;
    *panel = &state.panels[index];
    return 0;
}

int smol_fusion_record_success(void *opaque, const void *output, size_t rows, size_t width, unsigned operations) {
    if (!state.configured) return 0;
    smol_policy_raw_context *raw = opaque;
    unsigned expected;
    if (!state.valid || raw != state.raw || raw->failed || !output || !raw->host
        || !call_valid(raw, raw->call_cursor, &expected) || operations != expected
        || rows != raw->calls[raw->call_cursor].rows || width != raw->calls[raw->call_cursor].output_width
        || raw->panel_cursor != raw->calls[raw->call_cursor].first_panel + raw->calls[raw->call_cursor].panel_count) return -1;
    const smol_policy_call *call = &raw->calls[raw->call_cursor];
    state.last_success = 1;
    state.last_call = raw->call_cursor;
    state.last_sequence = raw->host->invocation_sequence;
    state.last_attention = raw->attention_cursor;
    state.last_output = output;
    state.last_rows = rows;
    state.last_width = width;
    state.last_image = call->image;
    state.last_layer = call->layer;
    state.last_role = call->role;
    state.last_operations = operations;
    state.last_scale = state.panels[call->first_panel].output_scale;
    return 0;
}

static int completed(unsigned layer, const uint16_t *produced, size_t rows, size_t width, unsigned role, unsigned operation) {
    if (!state.configured) return 0;
    const smol_policy_raw_context *raw = state.raw;
    if (!state.valid || !state.last_success || !raw || raw->failed || !raw->host || raw->host->active
        || !raw->calls || !raw->panels || state.last_call >= raw->call_count || state.count != raw->panel_count
        || raw->call_cursor != state.last_call + 1 || raw->image != state.last_image
        || raw->host->invocation_sequence != state.last_sequence || raw->attention_cursor != state.last_attention
        || layer != state.last_layer || role != state.last_role || produced != state.last_output
        || rows != state.last_rows || width != state.last_width
        || raw->calls[state.last_call].site != 7) return -1;
    return (state.last_operations & operation) != 0;
}

int smol_fusion_vision_gelu_completed(unsigned layer, const uint16_t *produced, size_t rows, size_t width) {
    return completed(layer, produced, rows, width, SMOL_VISION_FC1, SMOL_FUSION_GELU);
}

int smol_fusion_vision_connector_scale_completed(unsigned layer, const uint16_t *produced, size_t rows, size_t width) {
    return completed(layer, produced, rows, width, SMOL_VISION_CONNECTOR, SMOL_FUSION_CONNECTOR);
}

int smol_fusion_input_view(void *opaque, const void *input, size_t rows, size_t width, int *scale) {
    if (!state.configured) return 0;
    const smol_policy_raw_context *raw = opaque;
    if (!state.valid || raw != state.raw || !raw || raw->failed) return -1;
    if (!raw->calls || !raw->panels || state.count != raw->panel_count || raw->call_cursor > raw->call_count) return -1;
    unsigned expected = 0;
    if (raw->call_cursor) {
        size_t first = raw->calls[raw->call_cursor - 1].first_panel;
        if (first >= state.count) return -1;
        expected = state.panels[first].operations & SMOL_FUSION_FP8;
    }
    if (!(state.last_operations & SMOL_FUSION_FP8) && !expected) return 0;
    if (!scale || !(state.last_operations & SMOL_FUSION_FP8) || !state.last_success || !raw->host || raw->host->active
        || raw->call_cursor != state.last_call + 1 || raw->image != state.last_image
        || raw->host->invocation_sequence != state.last_sequence || raw->attention_cursor != state.last_attention
        || input != state.last_output || rows != state.last_rows || width != state.last_width
        || !fp8_consumer_valid(raw, state.last_call, state.last_scale)
        || raw->panel_cursor != raw->calls[state.last_call + 1].first_panel) return -1;
    *scale = state.last_scale;
    return 1;
}

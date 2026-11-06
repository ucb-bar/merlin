#ifndef SMOL_FUSION_H
#define SMOL_FUSION_H
#include <stddef.h>
#include <stdint.h>

enum { SMOL_FUSION_BIAS = 1u, SMOL_FUSION_GELU = 2u, SMOL_FUSION_CONNECTOR = 4u, SMOL_FUSION_FP8 = 8u };
typedef struct {
    const uint8_t *extra;
    size_t extra_offset, extra_bytes;
    unsigned operations;
    int output_scale;
} smol_fusion_panel;

/* One serialized raw context, configured before execution. Zero means success;
 * negative means invalid configuration. (NULL,NULL,0) clears the binding.
 * The caller keeps the context, descriptors, and extra bytes alive and immutable. */
int smol_fusion_configure(void *raw, const smol_fusion_panel *panels, size_t count);

/* Query the exact just-completed vision callback in the configured image:
 * 1 means completed on device, 0 means ordinary/unconfigured, negative means
 * invalid, stale, or mismatched state and requires graph failure. */
int smol_fusion_vision_gelu_completed(unsigned layer, const uint16_t *produced, size_t rows, size_t width);
int smol_fusion_vision_connector_scale_completed(unsigned layer, const uint16_t *produced, size_t rows, size_t width);

/* Pure, repeatable query for the exact next fc2 input after FP8 fc1 completion.
 * Returns1 and its fixed decode scale, 0 for ordinary input, or negative for a
 * mismatched pending view. The returned input uses natural32x32 FP8 tiles. */
int smol_fusion_input_view(void *raw, const void *input, size_t rows, size_t width, int *scale);

/* Raw-runtime integration; these preserve the public raw context/panel ABI. */
int smol_fusion_call_operations(void *raw, unsigned *operations);
int smol_fusion_panel_metadata(void *raw, size_t index, const smol_fusion_panel **panel);
int smol_fusion_record_success(void *raw, const void *output, size_t rows, size_t width, unsigned operations);
#endif

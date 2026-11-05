#include "smol_selected_workload.h"
#define MATRIX (177 * 960)
#define KV (177 * 320)
#define HIDDEN (177 * 2560)
#define SCORES (15 * 177 * 177)
static uint16_t normalized[MATRIX], query[MATRIX], key[KV], value[KV];
static uint16_t rotary_query[MATRIX], rotary_key[KV], attention[MATRIX];
static uint16_t projected[MATRIX], residual[MATRIX], postnorm[MATRIX];
static uint16_t gate[HIDDEN], up[HIDDEN], mlp_input[HIDDEN], down[MATRIX], output[MATRIX];
static float scores[SCORES], attention_workspace[177];
static uint16_t probabilities[SCORES];
static uint8_t mask[177 * 177];
static int64_t groups[177], positions[177];
smol_block_state smol_selected_state;
smol_block_frame smol_selected_frame = {normalized, query, key, value, rotary_query, rotary_key,
 attention, projected, residual, postnorm, gate, up, mlp_input, down, output, scores,
 attention_workspace, probabilities, mask, groups, positions};
unsigned smol_output_mismatches, smol_key_mismatches, smol_value_mismatches;
static unsigned differences(const uint16_t *a, const unsigned char * expected, size_t count) {
 unsigned mismatches = 0;
 for (size_t i = 0; i < count; ++i) {
     uint16_t word = (uint16_t)(expected[2 * i] | ((uint16_t) expected[2 * i + 1] << 8));
     mismatches += a[i] != word;
 }
 return mismatches;
}
int smol_selected_run(smol_block_linear callback, void *context) {
 int result = smol_vlm_block_prototype((const uint16_t *) smol_asset_input, smol_asset_pad,
   smol_asset_attention_boundaries, &smol_parameters, &smol_selected_frame,
   &smol_selected_state, callback, context);
 if (result) return result;
 smol_output_mismatches = differences(output, smol_asset_expected_output, MATRIX);
 smol_key_mismatches = differences(rotary_key, smol_asset_expected_key_cache, KV);
 smol_value_mismatches = differences(value, smol_asset_expected_value_cache, KV);
 return (smol_output_mismatches || smol_key_mismatches || smol_value_mismatches) ? 5 : 0;
}

#ifndef SMOL_SELECTED_WORKLOAD_H
#define SMOL_SELECTED_WORKLOAD_H
#include "smol_selected_assets.h"
extern smol_block_state smol_selected_state;
extern smol_block_frame smol_selected_frame;
extern unsigned smol_output_mismatches, smol_key_mismatches, smol_value_mismatches;
int smol_selected_run(smol_block_linear callback, void *context);
#endif

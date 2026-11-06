#ifndef SMOL_FAST_ASSETS_H
#define SMOL_FAST_ASSETS_H
#include <stdint.h>
#include <stddef.h>
typedef struct { const uint32_t *program; size_t program_words; uintptr_t encoded_weight_base; } smol_fast_panel;
extern const smol_fast_panel smol_fast_panels[2715];
#endif

#ifndef SMOL_FULL_PHASE_H
#define SMOL_FULL_PHASE_H
#include <stdint.h>
/* Pointer-free continuation metadata; semantic buffers have a fixed inventory.
 * This is a reference producer ABI, not an RV64 resume admission. */
typedef struct {
    uint32_t version, next_segment, kind, ordinal, layer;
    uint32_t panel_cursor, call_cursor, sequence, error, reserved;
} smol_full_phase_cursor;
_Static_assert(sizeof(smol_full_phase_cursor) == 40, "phase cursor ABI");
#endif

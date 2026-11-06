#include "smol_attention.h"

static smol_attention_callback attention_callback;
static void *attention_context;

void smol_attention_configure(smol_attention_callback callback, void *context) {
    attention_callback = callback;
    attention_context = context;
}

int smol_attention_enabled(void) { return attention_callback != 0; }

int smol_attention_dispatch(unsigned kind, const uint16_t *q, const uint16_t *k,
    const uint16_t *v, const uint8_t *mask, size_t rows, size_t keys,
    size_t qheads, size_t kvheads, uint16_t *output) {
    if (!attention_callback) return 1;
    return attention_callback(attention_context, kind, q, k, v, mask, rows, keys, qheads, kvheads, output);
}

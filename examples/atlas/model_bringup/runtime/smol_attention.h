#ifndef SMOL_ATTENTION_H
#define SMOL_ATTENTION_H
#include <stddef.h>
#include <stdint.h>

enum { SMOL_ATTENTION_VISION, SMOL_ATTENTION_PREFIX, SMOL_ATTENTION_SELF, SMOL_ATTENTION_CROSS };
typedef int (*smol_attention_callback)(void *, unsigned, const uint16_t *, const uint16_t *,
    const uint16_t *, const uint8_t *, size_t, size_t, size_t, size_t, uint16_t *);
void smol_attention_configure(smol_attention_callback callback, void *context);
int smol_attention_enabled(void);
int smol_attention_dispatch(unsigned kind, const uint16_t *q, const uint16_t *k,
    const uint16_t *v, const uint8_t *mask, size_t rows, size_t keys,
    size_t qheads, size_t kvheads, uint16_t *output);
#endif

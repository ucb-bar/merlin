"""Prepare a private group of provider-converted words and its ordered maximum.

The provider proves that its magnitude mapping has the intended unsigned order,
the conversion's numerical contract and refusal behavior, and complete accessible
source/destination spans. This emitter owns only the reusable serial data flow.
"""

from __future__ import annotations


def emit_u16_group_preparation(count: int, *, symbol: str, conversion: str, magnitude: str) -> str:
    """Emit a call-local float-to-word preparation with explicit refusal.

    ``conversion(float, uint16_t*)`` returns zero on success. The pure
    ``magnitude(uint16_t)`` callback returns the ordered unsigned key. Inputs
    are immutable strided floats, words are private disjoint storage and the
    maximum destination cannot alias either span. A refused conversion leaves
    earlier private words written and does not publish the maximum. Callers
    must refuse before publishing any result derived from this scratch state.
    """
    if type(count) is not int or not 1 <= count <= 1048576:
        raise ValueError("word group preparation requires a bounded positive element count")
    if any(type(name) is not str or not name.isascii() or not name.isidentifier()
           for name in (symbol, conversion, magnitude)):
        raise ValueError("word group preparation requires C identifier names")
    used = {symbol, conversion, magnitude}

    def fresh(name):
        candidate = "merlin_group_" + name
        while candidate in used:
            candidate += "_"
        used.add(candidate)
        return candidate

    source, stride, words, maximum, largest, offset, word, key = map(
        fresh, ("input", "stride", "words", "maximum", "largest", "offset", "word", "key"))
    return f'''#include <stddef.h>
#include <stdint.h>
static inline int {symbol}(const float *{source}, size_t {stride}, uint16_t *{words}, uint16_t *{maximum}) {{
  uint16_t {largest} = 0;
  for (size_t {offset} = 0; {offset} < {count}u; ++{offset}) {{
    uint16_t {word};
    if ({conversion}({source}[{offset} * {stride}], &{word})) return -1;
    {words}[{offset}] = {word};
    uint16_t {key} = {magnitude}({word});
    if ({key} > {largest}) {largest} = {key};
  }}
  *{maximum} = {largest};
  return 0;
}}
'''

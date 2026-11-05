"""Render byte-preserving C boundary packing for checked tensor tiling."""

from __future__ import annotations

from merlin.semantic_compiler.ordered_mnk_tiling import LoopedMNKTiling
from merlin.semantic_compiler.streamed_mnk_tiling import StreamedMNKTiling
from merlin.semantic_compiler.tiled_softmax import RowSoftmaxTiling


def render_looped_seed_c(plan: LoopedMNKTiling, prefix: str, pad_element: bytes) -> str:
    """Pack an exact completed accumulator into row-major M/N tiles."""
    if (
        not isinstance(plan, LoopedMNKTiling)
        or type(pad_element) is not bytes
        or len(pad_element) != plan.output_element_bytes
    ):
        raise ValueError("accumulator packing needs checked output geometry and explicit pad bytes")
    if not prefix or not prefix.isascii() or not prefix.isidentifier() or not prefix[0].isalpha():
        raise ValueError("C helper prefix must begin with an ASCII letter")
    m, n, _ = plan.extents
    tm, tn, width = plan.tile_m, plan.tile_n, plan.output_element_bytes
    largest = max(plan.buffer_bytes[2], m * n * width)
    if largest > (1 << 64) - 1:
        raise ValueError("accumulator buffers exceed the supported C size domain")
    pad = ",".join(map(str, pad_element))
    return f"""#include <stddef.h>
#include <stdint.h>
_Static_assert(SIZE_MAX >= {largest}ULL, "accumulator buffers exceed host size_t");
static void {prefix}_pack_seed(uint8_t *out,const uint8_t *source) {{
  const uint8_t pad[{width}]={{{pad}}};
  for (size_t mr=0;mr<{plan.counts[0]}u;++mr)
    for (size_t nc=0;nc<{plan.counts[1]}u;++nc)
      for (size_t r=0;r<{tm}u;++r)
        for (size_t c=0;c<{tn}u;++c) {{
          size_t row=mr*{tm}u+r,col=nc*{tn}u+c;
          const uint8_t *value=row<{m}u && col<{n}u ? source+(row*{n}u+col)*{width}u : pad;
          size_t at=((mr*{plan.counts[1]}u+nc)*{tm * tn}u+r*{tn}u+c)*{width}u;
          __builtin_memcpy(out+at,value,{width}u);
        }}
}}
"""


def render_looped_mnk_c(plan: LoopedMNKTiling, prefix: str) -> str:
    """Private C11 helpers; the caller validates buffers and removes device layout.

    No arithmetic conversion occurs here. Pads come only from the qualified
    plan. Output tiles are row major; target register-pair packing is separate.
    """
    if not isinstance(plan, LoopedMNKTiling):
        raise ValueError("C packing requires a checked looped tiling plan")
    if not prefix or not prefix.isascii() or not prefix.isidentifier() or not prefix[0].isalpha():
        raise ValueError("C helper prefix must begin with an ASCII letter")
    m, n, k = plan.extents
    tm, tn, tk = plan.tile_m, plan.tile_n, plan.tile_k
    iw, ow = plan.input_element_bytes, plan.output_element_bytes
    largest = max(*plan.buffer_bytes, m * k * iw, n * k * iw, m * n * ow)
    if largest > (1 << 64) - 1:
        raise ValueError("looped buffers exceed the supported C size domain")
    mn_pad = ", ".join(str(v) for v in (plan.mn_tail_pad_element or bytes(iw)))
    k_pad = ", ".join(str(v) for v in (plan.k_head_pad_element or bytes(iw)))
    return f"""#include <stddef.h>
#include <stdint.h>
_Static_assert(SIZE_MAX >= {largest}ULL, "looped buffers exceed host size_t");

static void {prefix}_pack_rows(uint8_t *out, const uint8_t *source,
                               size_t rows, size_t tile_rows) {{
  const uint8_t row_pad[{iw}] = {{{mn_pad}}};
  const uint8_t head_pad[{iw}] = {{{k_pad}}};
  size_t written = 0;
  for (size_t block = 0; block < rows; block += tile_rows) {{
    for (size_t part = 0; part < {plan.counts[2]}u; ++part) {{
      size_t count = ({k % tk}u && part == 0) ? {k % tk}u : {tk}u;
      size_t start = part == 0 ? 0 : ({k % tk}u ? {k % tk}u + (part-1)*{tk}u : part*{tk}u);
      for (size_t r = block; r < block + tile_rows; ++r) {{
        if (r >= rows) {{
          for (size_t j = 0; j < {tk}u; ++j) {{
            __builtin_memcpy(out + written, row_pad, {iw}u); written += {iw}u;
          }}
        }} else {{
          for (size_t j = 0; j < {tk}u-count; ++j) {{
            __builtin_memcpy(out + written, head_pad, {iw}u); written += {iw}u;
          }}
          __builtin_memcpy(out + written, source + (r*{k}u+start)*{iw}u, count*{iw}u);
          written += count*{iw}u;
        }}
      }}
    }}
  }}
}}

static void {prefix}_assemble_output(uint8_t *out, const uint8_t *tiles) {{
  for (size_t row = 0; row < {m}u; ++row) {{
    for (size_t col = 0; col < {plan.counts[1]}u; ++col) {{
      size_t width = {n}u-col*{tn}u;
      if (width > {tn}u) width = {tn}u;
      size_t tile = (row/{tm}u)*{plan.counts[1]}u+col;
      size_t source = (tile*{tm}u*{tn}u + (row%{tm}u)*{tn}u)*{ow}u;
      __builtin_memcpy(out + (row*{n}u+col*{tn}u)*{ow}u, tiles + source, width*{ow}u);
    }}
  }}
}}
"""


def render_streamed_mnk_c(plan: StreamedMNKTiling, prefix: str) -> str:
    """Pack ordered panels/chunks and scatter valid cells into private output.

    The target removes lane packing before scatter. Callers validate input
    domains before launching any panel and publish the staged tensor only when
    all launches have completed successfully.
    """
    if not isinstance(plan, StreamedMNKTiling):
        raise ValueError("streamed C packing needs a checked panel plan")
    if not prefix or not prefix.isascii() or not prefix.isidentifier() or not prefix[0].isalpha():
        raise ValueError("C helper prefix must begin with an ASCII letter")
    m, n, k = plan.full.extents
    bm, bn, _ = plan.body.extents
    tm, tn, tk = plan.full.tile_m, plan.full.tile_n, plan.full.tile_k
    iw, ow = plan.full.input_element_bytes, plan.full.output_element_bytes
    largest = max(
        m * k * iw,
        n * k * iw,
        m * n * ow + sum(plan.body.buffer_bytes) + plan.carry_buffer_bytes + plan.full.tile_bytes[2],
    )
    if largest > (1 << 64) - 1:
        raise ValueError("streamed buffers exceed the supported C size domain")
    pad = ",".join(map(str, plan.pad_element))
    head = ",".join(map(str, plan.full.k_head_pad_element or bytes(iw)))
    chunked = plan.k_block_tiles is not None
    function = "pack_chunk" if chunked else "pack_block"
    extra_arguments = ",size_t first_k_tile,size_t k_tiles" if chunked else ""
    k_start = "first_k_tile" if chunked else "0"
    k_end = "first_k_tile+k_tiles" if chunked else f"{plan.full.counts[2]}u"
    return f"""#include <stddef.h>
#include <stdint.h>
_Static_assert(SIZE_MAX >= {largest}ULL, "streamed buffers exceed host size_t");
static void {prefix}_{function}(uint8_t *out,const uint8_t *source,
                                size_t source_rows,size_t first,size_t rows,size_t tile_rows{extra_arguments}) {{
  const uint8_t pad[{iw}]={{{pad}}}, head[{iw}]={{{head}}};
  size_t written=0;
  for (size_t block=0;block<rows;block+=tile_rows)
    for (size_t part={k_start};part<{k_end};++part) {{
      size_t count=({k % tk}u && part==0)?{k % tk}u:{tk}u;
      size_t start=part==0?0:({k % tk}u?{k % tk}u+(part-1)*{tk}u:part*{tk}u);
      for (size_t row=block;row<block+tile_rows;++row) {{
        if (row>=rows || first+row>=source_rows) {{
          for (size_t j=0;j<{tk}u;++j) {{
            __builtin_memcpy(out+written,pad,{iw}u); written+={iw}u;
          }}
        }} else {{
          for (size_t j=0;j<{tk}u-count;++j) {{
            __builtin_memcpy(out+written,head,{iw}u); written+={iw}u;
          }}
          __builtin_memcpy(out+written,source+((first+row)*{k}u+start)*{iw}u,count*{iw}u);
          written+=count*{iw}u;
        }}
      }}
    }}
}}
static void {prefix}_scatter_block(uint8_t *out,const uint8_t *tiles,size_t first_m,size_t first_n) {{
  for (size_t row=0;row<{bm}u && first_m+row<{m}u;++row)
    for (size_t col=0;col<{bn}u && first_n+col<{n}u;++col) {{
      size_t tile=(row/{tm}u)*{plan.body.counts[1]}u+col/{tn}u;
      size_t at=(tile*{tm * tn}u+(row%{tm}u)*{tn}u+col%{tn}u)*{ow}u;
      __builtin_memcpy(out+((first_m+row)*{n}u+first_n+col)*{ow}u,tiles+at,{ow}u);
    }}
}}
"""


def streamed_k_driver_workspace_bytes(plan: StreamedMNKTiling) -> int:
    """Private host bytes for dense input chunks, two carries and full output."""
    if not isinstance(plan, StreamedMNKTiling) or plan.k_block_tiles is None:
        raise ValueError("chunk dispatch requires an explicit ordered K plan")
    m, n, _ = plan.full.extents
    bm, bn, k = plan.body.extents
    iw, ow = plan.full.input_element_bytes, plan.full.output_element_bytes
    size = (bm + bn) * k * iw + (2 * bm * bn + m * n) * ow
    if size > (1 << 64) - 1:
        raise ValueError("chunk workspace exceeds the supported C size domain")
    return size


def render_streamed_k_driver_c(plan: StreamedMNKTiling, prefix: str) -> str:
    """Compose checked device callbacks without arithmetic on carried values.

    Phase 0 resets, phase 1 continues, phase 2 continues with the terminal
    epilogue. A single-chunk plan uses phase 0 with its epilogue included.
    Callbacks consume dense encoded A/W blocks and a dense completed output;
    the target owns physical packing, instruction loading and completion. A
    zero return means completion. On error the caller keeps every workspace
    alive until pending device work has finished or been reset.
    """
    size = streamed_k_driver_workspace_bytes(plan)
    if not prefix or not prefix.isascii() or not prefix.isidentifier() or not prefix[0].isalpha():
        raise ValueError("C helper prefix must begin with an ASCII letter")
    m, n, k = plan.full.extents
    bm, bn, bk = plan.body.extents
    first_k = plan.first_body.extents[2]
    iw, ow = plan.full.input_element_bytes, plan.full.output_element_bytes
    largest = max(size, m * k * iw, n * k * iw)
    if largest > (1 << 64) - 1:
        raise ValueError("chunk inputs exceed the supported C size domain")
    pad = ",".join(map(str, plan.pad_element))
    return f"""#include <stddef.h>
#include <stdint.h>
_Static_assert(SIZE_MAX >= {largest}ULL, "chunk buffers exceed host size_t");
#define {prefix.upper()}_CHUNK_WORKSPACE_BYTES {size}ULL
typedef int (*{prefix}_chunk_callback)(void *, unsigned, const uint8_t *,
                                      const uint8_t *, const uint8_t *, uint8_t *);
static int {prefix}_chunk_range(const void *p,size_t bytes) {{
  return p && bytes && bytes<=UINTPTR_MAX-(uintptr_t)p;
}}
static int {prefix}_chunk_overlap(const void *a,size_t an,const void *b,size_t bn) {{
  return (uintptr_t)a<(uintptr_t)b+bn && (uintptr_t)b<(uintptr_t)a+an;
}}
static void {prefix}_pack_dense_chunk(uint8_t *out,const uint8_t *source,
    size_t source_rows,size_t first_row,size_t rows,size_t first_k,size_t count) {{
  const uint8_t pad[{iw}]={{{pad}}};
  for (size_t row=0;row<rows;++row) {{
    if (first_row+row<source_rows)
      __builtin_memcpy(out+row*count*{iw}u,
          source+((first_row+row)*{k}u+first_k)*{iw}u,count*{iw}u);
    else
      for (size_t j=0;j<count;++j) __builtin_memcpy(out+(row*count+j)*{iw}u,pad,{iw}u);
  }}
}}
static int {prefix}_run_chunks(void *context,{prefix}_chunk_callback invoke,
    uint8_t *workspace,size_t bytes,const uint8_t *a,const uint8_t *w,uint8_t *out) {{
  if (!invoke || bytes<{size}ULL || !{prefix}_chunk_range(workspace,{size}ULL) ||
      !{prefix}_chunk_range(a,{m * k * iw}ULL) || !{prefix}_chunk_range(w,{n * k * iw}ULL) ||
      !{prefix}_chunk_range(out,{m * n * ow}ULL) ||
      {prefix}_chunk_overlap(workspace,{size}ULL,a,{m * k * iw}ULL) ||
      {prefix}_chunk_overlap(workspace,{size}ULL,w,{n * k * iw}ULL) ||
      {prefix}_chunk_overlap(workspace,{size}ULL,out,{m * n * ow}ULL) ||
      {prefix}_chunk_overlap(out,{m * n * ow}ULL,a,{m * k * iw}ULL) ||
      {prefix}_chunk_overlap(out,{m * n * ow}ULL,w,{n * k * iw}ULL)) return -1;
  uint8_t *pa=workspace,*pw=pa+{bm * bk * iw}ULL;
  uint8_t *partial0=pw+{bn * bk * iw}ULL,*partial1=partial0+{bm * bn * ow}ULL;
  uint8_t *staged=partial1+{bm * bn * ow}ULL;
  for (size_t mr=0;mr<{m}u;mr+={bm}u)
    for (size_t nc=0;nc<{n}u;nc+={bn}u) {{
      size_t begin=0,count={first_k}u;
      uint8_t *previous=0,*destination=partial0;
      for (size_t chunk=0;chunk<{plan.reduction_launches}u;++chunk) {{
        unsigned phase=chunk==0 ? 0u : chunk+1=={plan.reduction_launches}u ? 2u : 1u;
        {prefix}_pack_dense_chunk(pa,a,{m}u,mr,{bm}u,begin,count);
        {prefix}_pack_dense_chunk(pw,w,{n}u,nc,{bn}u,begin,count);
        int status=invoke(context,phase,pa,pw,previous,destination);
        if (status) return status;
        previous=destination; destination=destination==partial0 ? partial1 : partial0;
        begin+=count; count={bk}u;
      }}
      for (size_t row=0;row<{bm}u && mr+row<{m}u;++row) {{
        size_t columns={n}u-nc; if (columns>{bn}u) columns={bn}u;
        __builtin_memcpy(staged+((mr+row)*{n}u+nc)*{ow}u,
                         previous+row*{bn * ow}u,columns*{ow}u);
      }}
    }}
  __builtin_memcpy(out,staged,{m * n * ow}ULL);
  return 0;
}}
"""


def render_row_softmax_c(plan: RowSoftmaxTiling, prefix: str) -> str:
    """Pack a row group and scatter valid columns without target lane layout.

    The caller checks source bounds and buffer ownership. All output rows are
    staged privately until every group completes, so failed launches cannot
    publish a partial source tensor.
    """
    if not isinstance(plan, RowSoftmaxTiling):
        raise ValueError("row packing requires a checked softmax tiling plan")
    if not prefix or not prefix.isascii() or not prefix.isidentifier() or not prefix[0].isalpha():
        raise ValueError("C helper prefix must begin with an ASCII letter")
    m, n = plan.shape
    tm, tn = plan.tile_type.shape
    width = len(plan.pad_element)
    largest = max(m * n * width, tm * tn * width)
    if largest > (1 << 64) - 1:
        raise ValueError("row buffers exceed the supported C size domain")
    pad = ",".join(str(value) for value in plan.pad_element)
    return f"""#include <stddef.h>
#include <stdint.h>
_Static_assert(SIZE_MAX >= {largest}ULL, "row buffers exceed host size_t");
static void {prefix}_pack_row_tile(uint8_t *out, const uint8_t *source,
                                  size_t start_row, size_t column_tile) {{
  const uint8_t pad[{width}] = {{{pad}}};
  for (size_t row = 0; row < {tm}u; ++row) {{
    for (size_t col = 0; col < {tn}u; ++col) {{
      size_t logical_col = column_tile*{tn}u+col;
      const uint8_t *value = start_row+row < {m}u && logical_col < {n}u
          ? source+((start_row+row)*{n}u+logical_col)*{width}u : pad;
      __builtin_memcpy(out+(row*{tn}u+col)*{width}u,value,{width}u);
    }}
  }}
}}
static void {prefix}_scatter_row_tile(uint8_t *out, const uint8_t *tile,
                                     size_t start_row, size_t column_tile) {{
  size_t columns = {n}u-column_tile*{tn}u;
  if (columns > {tn}u) columns = {tn}u;
  for (size_t row = 0; row < {tm}u && start_row+row < {m}u; ++row)
    __builtin_memcpy(out+((start_row+row)*{n}u+column_tile*{tn}u)*{width}u,
                     tile+row*{tn * width}u,columns*{width}u);
}}
"""

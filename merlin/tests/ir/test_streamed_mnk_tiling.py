"""Bounded panel selection and compiled byte packing on synthetic geometries."""

import ctypes
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from merlin.compile.c_tiling import (
    render_looped_seed_c,
    render_streamed_k_driver_c,
    render_streamed_mnk_c,
    streamed_k_driver_workspace_bytes,
)
from merlin.semantic_compiler.ordered_mnk_tiling import LoopedMNKTiling, OrderedMNKTilingError
from merlin.semantic_compiler.streamed_mnk_tiling import plan_streamed_k_reduction, plan_streamed_mnk
from merlin.tests.ir.test_ordered_mnk_tiling import _source


def full_plan(m, n, k, geometry=(2, 3, 4)):
    tm, tn, tk = geometry
    return LoopedMNKTiling(
        _source(m, n, k),
        tm,
        tn,
        tk,
        "ordered-i32",
        1,
        4,
        b"\0" if m % tm or n % tn else None,
        b"\0" if k % tk else None,
    )


@pytest.mark.parametrize("shape,geometry", [((5, 7, 9), (2, 3, 4)), ((9, 3, 7), (4, 2, 2))])
def test_completed_seed_packing_preserves_bits_and_explicit_padding(tmp_path, shape, geometry):
    plan = full_plan(*shape, geometry)
    m, n, _ = shape
    tm, tn, _ = geometry
    source = render_looped_seed_c(plan, "seed", b"\xa5" * 4)
    source += "\nvoid pack(uint8_t *out,const uint8_t *in) { seed_pack_seed(out,in); }\n"
    (tmp_path / "seed.c").write_text(source)
    library = tmp_path / ("seed-" + "-".join(map(str, (*shape, *geometry))) + ".so")
    subprocess.run(
        [
            shutil.which("cc"),
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            str(tmp_path / "seed.c"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(library))
    lib.pack.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    values = (np.arange(m * n, dtype=np.uint32).reshape(m, n) + np.uint32(0x80000000)).astype("<u4")
    before = values.tobytes()
    out = ctypes.create_string_buffer(b"\x9d" * (plan.buffer_bytes[2] + 32))
    lib.pack(ctypes.addressof(out) + 16, values.ctypes.data)
    padded = np.pad(values, ((0, (-m) % tm), (0, (-n) % tn)), constant_values=0xA5A5A5A5)
    expected = padded.reshape(plan.counts[0], tm, plan.counts[1], tn).transpose(0, 2, 1, 3).tobytes()
    assert out.raw[16:-17] == expected
    assert out.raw[:16] == out.raw[-17:-1] == b"\x9d" * 16
    assert values.tobytes() == before


@pytest.mark.parametrize("shape,geometry,capacity", [((5, 7, 33), (2, 3, 4), 144), ((9, 3, 19), (4, 2, 2), 128)])
def test_compiled_chunk_dispatch_carries_values_and_publishes_atomically(tmp_path, shape, geometry, capacity):
    plan = plan_streamed_k_reduction(
        full_plan(*shape, geometry), capacity_bytes=capacity, alignment_bytes=8, pad_element=b"\0"
    )
    assert plan.reduction_launches > 1
    m, n, k = shape
    bm, bn, bk = plan.body.extents
    first_k = plan.first_body.extents[2]
    code = (
        render_streamed_k_driver_c(plan, "driver")
        + f"""
struct counters {{ unsigned calls,fail,phases; }};
static struct counters last;
static int body(void *ctx,unsigned phase,const uint8_t *a,const uint8_t *w,const uint8_t *seed,uint8_t *out) {{
  struct counters *p=ctx; ++p->calls; p->phases|=1u<<phase;
  if (p->calls==p->fail) return 37;
  if ((phase && (!seed || seed==out)) || (!phase && seed)) return 91;
  size_t count=phase ? {bk}u : {first_k}u;
  for (size_t r=0;r<{bm}u;++r)
    for (size_t c=0;c<{bn}u;++c) {{
      int32_t value=0;
      if (phase) __builtin_memcpy(&value,seed+(r*{bn}u+c)*4u,4u);
      for (size_t i=0;i<count;++i)
        value+=(int32_t)((const int8_t *)a)[r*count+i]*((const int8_t *)w)[c*count+i];
      if (phase==2 && value<0) value=0;
      __builtin_memcpy(out+(r*{bn}u+c)*4u,&value,4u);
    }}
  return 0;
}}
int run(uint8_t *workspace,size_t bytes,const uint8_t *a,const uint8_t *w,uint8_t *out,unsigned fail) {{
  last=(struct counters){{0,fail,0}};
  return driver_run_chunks(&last,body,workspace,bytes,a,w,out);
}}
unsigned calls(void) {{ return last.calls; }}
unsigned phases(void) {{ return last.phases; }}
"""
    )
    (tmp_path / "driver.c").write_text(code)
    library = tmp_path / ("driver-" + "-".join(map(str, (*shape, *geometry, capacity))) + ".so")
    subprocess.run(
        [
            shutil.which("cc"),
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            str(tmp_path / "driver.c"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(library))
    lib.run.argtypes = [ctypes.c_void_p, ctypes.c_size_t] + [ctypes.c_void_p] * 3 + [ctypes.c_uint]
    size = streamed_k_driver_workspace_bytes(plan)
    scratch = ctypes.create_string_buffer(b"\x9d" * (size + 32))
    a = (np.arange(m * k).reshape(m, k) % 3 + 1).astype(np.int8)
    w = (np.arange(n * k).reshape(n, k) % 3 + 1).astype(np.int8)
    w[:, :first_k] *= -1
    w[1::2] *= -1
    inputs_before = a.tobytes(), w.tobytes()
    out = ctypes.create_string_buffer(b"\xa7" * (m * n * 4 + 32))
    expected = np.maximum(a.astype(np.int32) @ w.astype(np.int32).T, 0).astype("<i4").tobytes()
    total = plan.launch_counts[0] * plan.launch_counts[1] * plan.reduction_launches
    for fail in (1, total, 0):
        ctypes.memset(ctypes.addressof(out), 0xA7, m * n * 4 + 32)
        status = lib.run(
            ctypes.addressof(scratch) + 16, size, a.ctypes.data, w.ctypes.data, ctypes.addressof(out) + 16, fail
        )
        assert status == (37 if fail else 0)
        assert lib.calls() == (fail or total)
        assert out.raw[16:-17] == (b"\xa7" * (m * n * 4) if fail else expected)
        assert out.raw[:16] == out.raw[-17:-1] == b"\xa7" * 16
        assert scratch.raw[:16] == scratch.raw[-17:-1] == b"\x9d" * 16
        assert inputs_before == (a.tobytes(), w.tobytes())
    assert lib.phases() == (7 if plan.reduction_launches > 2 else 5)
    assert (
        lib.run(ctypes.addressof(scratch) + 16, size - 1, a.ctypes.data, w.ctypes.data, ctypes.addressof(out) + 16, 0)
        == -1
    )
    assert lib.calls() == 0
    assert lib.run(ctypes.addressof(scratch) + 16, size, a.ctypes.data, w.ctypes.data, a.ctypes.data, 0) == -1
    assert inputs_before == (a.tobytes(), w.tobytes())


def test_panel_search_agrees_with_bounded_exhaustive_choices():
    checked = 0
    for m in range(1, 10, 2):
        for n in range(1, 12, 2):
            for k in (1, 5):
                full = full_plan(m, n, k)
                mt, nt, kt = full.counts
                a, w, out = full.tile_bytes
                for capacity in (64, 128, 256, 512):
                    feasible = []
                    for bm in range(1, mt + 1):
                        for bn in range(1, nt + 1):
                            sizes = (bm * kt * a, bn * kt * w, bm * bn * out)
                            if sum(((s + 15) // 16) * 16 for s in sizes) <= capacity:
                                calls = ((mt + bm - 1) // bm) * ((nt + bn - 1) // bn)
                                feasible.append(
                                    ((calls, calls * bm * bn, calls * kt * (bm * a + bn * w), -bm, -bn), (bm, bn))
                                )
                    if not feasible:
                        with pytest.raises(OrderedMNKTilingError):
                            plan_streamed_mnk(full, capacity_bytes=capacity, alignment_bytes=16, pad_element=b"\0")
                    else:
                        plan = plan_streamed_mnk(full, capacity_bytes=capacity, alignment_bytes=16, pad_element=b"\0")
                        if plan is None:
                            assert sum(((s + 15) // 16) * 16 for s in full.buffer_bytes) <= capacity
                        else:
                            assert plan.block_tiles == min(feasible)[1]
                            assert plan.body.extents[2] == k
                            assert plan.body.qualified_policy == full.qualified_policy
                            assert plan.footprint_bytes <= capacity
                    checked += 1
    assert checked == 240


def test_panel_search_reports_work_and_full_k_limits():
    with pytest.raises(OrderedMNKTilingError, match="work limit"):
        plan_streamed_mnk(
            full_plan(11, 14, 9),
            capacity_bytes=256,
            alignment_bytes=8,
            pad_element=b"\0",
            max_candidates=1,
        )
    with pytest.raises(OrderedMNKTilingError, match="budget"):
        plan_streamed_mnk(full_plan(2, 3, 4097), capacity_bytes=256, alignment_bytes=8, pad_element=b"\0")


@pytest.mark.parametrize("in_place", (False, True))
def test_reduction_search_matches_tiny_exhaustive_blocks_and_preserves_k_order(in_place):
    checked = 0
    for m in (1, 3, 5):
        for n in (1, 4, 7):
            for k in (1, 5, 9, 13, 17):
                full = full_plan(m, n, k)
                mt, nt, kt = full.counts
                a, w, out = full.tile_bytes
                for capacity in (32, 64, 96, 160, 256):
                    legal = []
                    for bm in range(1, mt + 1):
                        for bn in range(1, nt + 1):
                            for bk in range(1, kt + 1):
                                sizes = (bm * bk * a, bn * bk * w, bm * bn * out)
                                if not in_place:
                                    sizes += (bm * bn * out,)
                                if sum(((size + 7) // 8) * 8 for size in sizes) > capacity:
                                    continue
                                panels = ((mt + bm - 1) // bm) * ((nt + bn - 1) // bn)
                                score = (panels * ((kt + bk - 1) // bk), panels * bm * bn * kt, -bk, -bm, -bn)
                                legal.append((score, (bm, bn), bk))
                    if not legal:
                        with pytest.raises(OrderedMNKTilingError, match="budget"):
                            plan_streamed_k_reduction(
                                full,
                                capacity_bytes=capacity,
                                alignment_bytes=8,
                                pad_element=b"\0",
                                accumulator_in_place=in_place,
                            )
                    else:
                        plan = plan_streamed_k_reduction(
                            full,
                            capacity_bytes=capacity,
                            alignment_bytes=8,
                            pad_element=b"\0",
                            accumulator_in_place=in_place,
                        )
                        _, blocks, reduction = min(legal)
                        assert (plan.block_tiles, plan.k_block_tiles) == (blocks, reduction)
                        chunks = tuple(plan.reduction_chunks())
                        assert [
                            i
                            for chunk in chunks
                            for i in range(chunk["source_start"], chunk["source_start"] + chunk["source_count"])
                        ] == list(range(k))
                        assert [chunk["reset"] for chunk in chunks] == [True] + [False] * (len(chunks) - 1)
                        assert [chunk["final"] for chunk in chunks] == [False] * (len(chunks) - 1) + [True]
                        assert plan.first_body.extents[2] == chunks[0]["source_count"]
                        assert plan.first_body.k_head_pad_element == full.k_head_pad_element
                        if len(chunks) > 1:
                            assert plan.body.k_head_pad_element is None
                            assert all(chunk["source_count"] == plan.body.extents[2] for chunk in chunks[1:])
                        assert plan.footprint_bytes <= capacity
                        assert plan.record()["accumulator"]["conversion"] == "none"
                    checked += 1
    assert checked == 225
    with pytest.raises(OrderedMNKTilingError, match="work limit"):
        plan_streamed_k_reduction(
            full_plan(5, 7, 4097), capacity_bytes=128, alignment_bytes=8, pad_element=b"\0", max_candidates=1
        )


@pytest.mark.parametrize(
    "shape,geometry,capacity,width",
    [
        ((5, 7, 33), (2, 3, 4), 144, 1),
        ((1, 37, 65), (2, 3, 4), 160, 1),
        ((9, 2, 19), (4, 2, 2), 128, 1),
        ((5, 7, 33), (2, 3, 4), 192, 2),
    ],
)
def test_compiled_reduction_chunks_preserve_bytes_and_cover_outputs(tmp_path, shape, geometry, capacity, width):
    full = full_plan(*shape, geometry)
    if width == 2:
        nodes = tuple(
            replace(node, type=replace(node.type, dtype="i16")) if node.effect == "input" else node
            for node in full.source.nodes
        )
        full = replace(
            full,
            source=replace(full.source, nodes=nodes),
            input_element_bytes=2,
            mn_tail_pad_element=b"\0\0",
            k_head_pad_element=b"\0\0",
        )
    plan = plan_streamed_k_reduction(full, capacity_bytes=capacity, alignment_bytes=8, pad_element=bytes(width))
    assert plan.reduction_launches > 1
    m, n, k = shape
    bm, bn, _ = plan.body.extents
    tm, tn, tk = geometry
    source = (
        render_streamed_mnk_c(plan, "chunk")
        + f"""
void pack_a(uint8_t *out,const uint8_t *in,size_t row,size_t first,size_t count) {{
  chunk_pack_chunk(out,in,{m}u,row,{bm}u,{tm}u,first,count);
}}
void pack_w(uint8_t *out,const uint8_t *in,size_t row,size_t first,size_t count) {{
  chunk_pack_chunk(out,in,{n}u,row,{bn}u,{tn}u,first,count);
}}
void scatter(uint8_t *out,const uint8_t *in,size_t row,size_t col) {{ chunk_scatter_block(out,in,row,col); }}
"""
    )
    (tmp_path / "chunk.c").write_text(source)
    library = tmp_path / ("chunk-" + "-".join(map(str, (*shape, *geometry, capacity, width))) + ".so")
    subprocess.run(
        [
            shutil.which("cc"),
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            str(tmp_path / "chunk.c"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(library))
    for name, count in (("pack_a", 3), ("pack_w", 3), ("scatter", 2)):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_size_t] * count
    dtype = np.dtype("u1" if width == 1 else "<u2")
    a = (np.arange(m * k).reshape(m, k) % 7 + 1).astype(dtype)
    w = (np.arange(n * k).reshape(n, k) % 5 + 1).astype(dtype)
    if width == 2:
        a *= 257
        w *= 17
    before = a.tobytes(), w.tobytes()
    out = ctypes.create_string_buffer(b"\xa7" * (m * n * 4 + 32))
    for row in range(0, m, bm):
        for col in range(0, n, bn):
            accumulated = np.zeros((bm, bn), dtype="<i4")
            for chunk in plan.reduction_chunks():
                matrices = []
                for name, tensor, first, rows, tile_rows in (("pack_a", a, row, bm, tm), ("pack_w", w, col, bn, tn)):
                    count = chunk["tile_count"]
                    size = ((rows + tile_rows - 1) // tile_rows) * tile_rows * count * tk * width
                    packed = ctypes.create_string_buffer(b"\x5a" * (size + 32))
                    getattr(lib, name)(
                        ctypes.addressof(packed) + 16, tensor.ctypes.data, first, chunk["tile_start"], count
                    )
                    assert packed.raw[:16] == packed.raw[-17:-1] == b"\x5a" * 16
                    tiles = np.frombuffer(packed.raw[16 : 16 + size], dtype=dtype).reshape(-1, count, tile_rows, tk)
                    dense = tiles.transpose(0, 2, 1, 3).reshape(-1, count * tk)[:rows]
                    head = (-k) % tk if chunk["reset"] else 0
                    assert np.count_nonzero(dense[:, :head]) == 0
                    matrices.append(dense.astype(np.int32))
                accumulated += matrices[0] @ matrices[1].T
            tiles = np.zeros((plan.body.counts[0], plan.body.counts[1], tm, tn), dtype="<i4")
            for r in range(bm):
                for c in range(bn):
                    tiles[r // tm, c // tn, r % tm, c % tn] = accumulated[r, c]
            lib.scatter(ctypes.addressof(out) + 16, ctypes.create_string_buffer(tiles.tobytes()), row, col)
    assert out.raw[16 : 16 + m * n * 4] == (a.astype(np.int32) @ w.astype(np.int32).T).astype("<i4").tobytes()
    assert out.raw[:16] == out.raw[-17:-1] == b"\xa7" * 16
    assert before == (a.tobytes(), w.tobytes())


@pytest.mark.parametrize(
    "shape,geometry,capacity",
    [
        ((5, 7, 3), (2, 3, 4), 144),
        ((11, 14, 9), (2, 3, 4), 256),
        ((1, 37, 5), (2, 3, 4), 160),
        ((17, 2, 4), (4, 2, 2), 128),
    ],
)
def test_compiled_panels_cover_exact_source_outputs(tmp_path, shape, geometry, capacity):
    full = full_plan(*shape, geometry)
    plan = plan_streamed_mnk(full, capacity_bytes=capacity, alignment_bytes=8, pad_element=b"\0")
    assert plan is not None
    m, n, k = shape
    bm, bn, _ = plan.body.extents
    tm, tn, tk = geometry
    source = (
        render_streamed_mnk_c(plan, "panel")
        + f"""
void pack_a(uint8_t *out,const uint8_t *in,size_t row) {{ panel_pack_block(out,in,{m}u,row,{bm}u,{tm}u); }}
void pack_w(uint8_t *out,const uint8_t *in,size_t row) {{ panel_pack_block(out,in,{n}u,row,{bn}u,{tn}u); }}
void scatter(uint8_t *out,const uint8_t *in,size_t m,size_t n) {{ panel_scatter_block(out,in,m,n); }}
"""
    )
    (tmp_path / "panel.c").write_text(source)
    library = tmp_path / ("panel-" + "-".join(map(str, (*shape, *geometry, capacity))) + ".so")
    subprocess.run(
        [
            shutil.which("cc"),
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            str(tmp_path / "panel.c"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(library))
    for name, count in (("pack_a", 1), ("pack_w", 1), ("scatter", 2)):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_size_t] * count
    a = (np.arange(m * k, dtype=np.uint8).reshape(m, k) % 7 + 1).copy()
    w = (np.arange(n * k, dtype=np.uint8).reshape(n, k) % 5 + 1).copy()
    before = a.tobytes(), w.tobytes()
    output = ctypes.create_string_buffer(b"\xa7" * (m * n * 4 + 32))
    ab, wb, ob = plan.body.buffer_bytes
    packed = [ctypes.create_string_buffer(size) for size in (ab, wb)]

    def decode(payload, rows, tile_rows):
        cells = np.frombuffer(payload, dtype=np.uint8).reshape(-1, full.counts[2], tile_rows, tk)
        dense = cells.transpose(0, 2, 1, 3).reshape(-1, full.counts[2] * tk)
        # The short K tile is head-padded, preserving source K order.
        return dense[:rows, (-k) % tk : (-k) % tk + k].astype(np.int32)

    for row in range(0, m, bm):
        for col in range(0, n, bn):
            lib.pack_a(packed[0], a.ctypes.data, row)
            lib.pack_w(packed[1], w.ctypes.data, col)
            aa = decode(packed[0].raw, bm, tm)
            ww = decode(packed[1].raw, bn, tn)
            expected_panel = aa @ ww.T
            tiles = np.zeros((plan.body.counts[0], plan.body.counts[1], tm, tn), dtype="<i4")
            for r in range(bm):
                for c in range(bn):
                    tiles[r // tm, c // tn, r % tm, c % tn] = expected_panel[r, c]
            physical = ctypes.create_string_buffer(tiles.tobytes())
            assert tiles.nbytes == ob
            lib.scatter(ctypes.addressof(output) + 16, physical, row, col)
    assert output.raw[16 : 16 + m * n * 4] == (a.astype(np.int32) @ w.astype(np.int32).T).astype("<i4").tobytes()
    assert output.raw[:16] == output.raw[-17:-1] == b"\xa7" * 16
    assert (a.tobytes(), w.tobytes()) == before
    with pytest.raises(OrderedMNKTilingError):
        replace(plan, capacity_bytes=1)
    with pytest.raises(OrderedMNKTilingError):
        replace(plan, pad_element=b"")

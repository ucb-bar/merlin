"""Partition ordered matrix tiles into reusable panels under a byte budget."""

from dataclasses import dataclass, replace

from .ordered_mnk_tiling import LoopedMNKTiling, OrderedMNKTilingError


def _ceil(value, width):
    return (value + width - 1) // width


@dataclass(frozen=True)
class StreamedMNKTiling:
    """Fixed device bodies and counted host loops over disjoint M/N panels.

    By default each launch retains the complete K reduction. An explicit K
    block adds an accumulator carry obligation between launches. The caller
    supplies legal padding bytes; no additional K identity is inferred here.
    """

    full: LoopedMNKTiling
    block_tiles: tuple[int, int]
    capacity_bytes: int
    alignment_bytes: int
    pad_element: bytes
    k_block_tiles: int | None = None
    accumulator_in_place: bool = False

    def __post_init__(self):
        if (
            not isinstance(self.full, LoopedMNKTiling)
            or type(self.block_tiles) is not tuple
            or len(self.block_tiles) != 2
            or any(
                type(n) is not int or not 1 <= n <= limit
                for n, limit in zip(self.block_tiles, self.full.counts[:2], strict=True)
            )
            or type(self.capacity_bytes) is not int
            or self.capacity_bytes <= 0
            or type(self.alignment_bytes) is not int
            or self.alignment_bytes <= 0
            or type(self.pad_element) is not bytes
            or len(self.pad_element) != self.full.input_element_bytes
            or (
                self.k_block_tiles is not None
                and (type(self.k_block_tiles) is not int or not 1 <= self.k_block_tiles <= self.full.counts[2])
            )
            or type(self.accumulator_in_place) is not bool
            or (self.accumulator_in_place and self.k_block_tiles is None)
        ):
            raise OrderedMNKTilingError("streamed panels need checked geometry, byte budget and pad encoding")
        if self.footprint_bytes > self.capacity_bytes:
            raise OrderedMNKTilingError("streamed panel exceeds its aligned device budget")

    @property
    def body(self):
        k = self.full.extents[2] if self.reduction_launches == 1 else self.k_block_tiles * self.full.tile_k
        return self._body(k)

    def _body(self, k):
        m, n, _ = self.full.extents
        bm = min(m, self.block_tiles[0] * self.full.tile_m)
        bn = min(n, self.block_tiles[1] * self.full.tile_n)
        a, w, out = self.full.source.nodes
        source = replace(
            self.full.source,
            nodes=(
                replace(a, type=replace(a.type, shape=(bm, k))),
                replace(w, type=replace(w.type, shape=(bn, k))),
                replace(out, type=replace(out.type, shape=(bm, bn))),
            ),
        )
        return replace(
            self.full,
            source=source,
            mn_tail_pad_element=self.pad_element if bm % self.full.tile_m or bn % self.full.tile_n else None,
            k_head_pad_element=self.full.k_head_pad_element if k % self.full.tile_k else None,
        )

    @property
    def reduction_launches(self):
        return _ceil(self.full.counts[2], self.k_block_tiles) if self.k_block_tiles is not None else 1

    @property
    def first_body(self):
        first_tiles = (self.full.counts[2] - 1) % (self.k_block_tiles or self.full.counts[2]) + 1
        first_k = self.full.extents[2] - (self.full.counts[2] - first_tiles) * self.full.tile_k
        return self._body(first_k)

    def reduction_chunks(self):
        """Yield source-element ranges, preserving the original head pad only.

        A chunk consumes the previous completed accumulator without rounding or
        conversion. Its storage remains live until the next chunk finishes.
        The target must implement that seed/readout interface explicitly.
        """
        kt = self.full.counts[2]
        block = self.k_block_tiles or kt
        first = (kt - 1) % block + 1
        head = (-self.full.extents[2]) % self.full.tile_k
        for index in range(self.reduction_launches):
            start = 0 if index == 0 else first + (index - 1) * block
            count = first if index == 0 else block
            begin = max(0, start * self.full.tile_k - head)
            end = (start + count) * self.full.tile_k - head
            yield {
                "tile_start": start,
                "tile_count": count,
                "source_start": begin,
                "source_count": end - begin,
                "reset": index == 0,
                "final": index + 1 == self.reduction_launches,
            }

    @property
    def footprint_bytes(self):
        sizes = (*self.body.buffer_bytes, self.carry_buffer_bytes)
        return sum(_ceil(size, self.alignment_bytes) * self.alignment_bytes for size in sizes)

    @property
    def carry_buffer_bytes(self):
        # Separate pure input/output buffers are the default carry ABI. An
        # explicitly qualified in-place route can reuse the completed output.
        return self.body.buffer_bytes[2] if self.k_block_tiles is not None and not self.accumulator_in_place else 0

    @property
    def launch_counts(self):
        return tuple(_ceil(n, block) for n, block in zip(self.full.counts[:2], self.block_tiles, strict=True))

    def record(self):
        chunked = self.k_block_tiles is not None
        record = {
            "schema": "merlin.streamed_mnk_tiling.v2" if chunked else "merlin.streamed_mnk_tiling.v1",
            "source_request_digest": self.full.source.digest(),
            "body": self.body.record(),
            "block_tiles": list(self.block_tiles),
            "launch_counts": list(self.launch_counts) + ([self.reduction_launches] if chunked else []),
            "launch_order": ["m", "n", "k"] if chunked else ["m", "n"],
            "k_reduction": "ordered_across_launches" if self.reduction_launches > 1 else "complete_in_each_launch",
            "capacity_bytes": self.capacity_bytes,
            "alignment_bytes": self.alignment_bytes,
            "pad_element_hex": self.pad_element.hex(),
            "footprint_bytes": self.footprint_bytes,
        }
        if self.k_block_tiles is not None:
            record.update(
                k_block_tiles=self.k_block_tiles,
                reduction_launches=self.reduction_launches,
                first_body=self.first_body.record(),
                accumulator={
                    "storage": "output_buffer" if self.accumulator_in_place else "separate_input_buffer",
                    "read_before_write": self.accumulator_in_place,
                    "dtype": self.full.source.nodes[-1].type.dtype,
                    "numerical_policy": self.full.qualified_policy,
                    "conversion": "none",
                    "initialization": "reset_first_chunk",
                },
                epilogue="after_last_chunk",
            )
        return record


def plan_streamed_mnk(full, *, capacity_bytes, alignment_bytes, pad_element, max_candidates=65536):
    """Keep fitting programs unchanged; otherwise minimize launches then padding.

    Search is finite and deterministic. Candidates are ordered by launch count,
    padded output tiles, transferred operand bytes, then decreasing M/N blocks.
    A K panel that cannot fit at the minimum output geometry is a resource
    failure; this transform never splits or reassociates its reduction.
    """
    minimal = StreamedMNKTiling(full, (1, 1), capacity_bytes, alignment_bytes, pad_element)
    if type(max_candidates) is not int or max_candidates <= 0:
        raise OrderedMNKTilingError("streamed panel search needs a positive work budget")
    if sum(_ceil(n, alignment_bytes) * alignment_bytes for n in full.buffer_bytes) <= capacity_bytes:
        return None
    mt, nt, kt = full.counts
    a, w, out = full.tile_bytes

    def footprint(m, n):
        return sum(_ceil(v, alignment_bytes) * alignment_bytes for v in (m * kt * a, n * kt * w, m * n * out))

    best, work = None, 0
    for m in range(1, mt + 1):
        if footprint(m, 1) > capacity_bytes:
            break
        for n in range(1, nt + 1):
            if footprint(m, n) > capacity_bytes:
                break
            work += 1
            if work > max_candidates:
                raise OrderedMNKTilingError("streamed panel search work limit exceeded")
            calls = _ceil(mt, m) * _ceil(nt, n)
            score = (calls, calls * m * n, calls * kt * (m * a + n * w), -m, -n)
            if best is None or score < best[0]:
                best = score, (m, n)
    assert best is not None  # The checked minimum panel is feasible.
    return replace(minimal, block_tiles=best[1])


def plan_streamed_k_reduction(
    full,
    *,
    capacity_bytes,
    alignment_bytes,
    pad_element,
    accumulator_in_place=False,
    max_candidates=65536,
):
    """Choose M/N/K blocks for a target with an exact accumulator carry route.

    This does not authorize splitting a reduction on targets without that
    route. The first chunk may be shorter; every later chunk has the same
    complete-tile geometry. No additional K padding is introduced. The work
    budget counts footprint queries, including infeasible placements.
    """
    minimal = StreamedMNKTiling(full, (1, 1), capacity_bytes, alignment_bytes, pad_element, 1, accumulator_in_place)
    if type(max_candidates) is not int or max_candidates <= 0:
        raise OrderedMNKTilingError("streamed reduction search needs a positive work budget")
    mt, nt, kt = full.counts
    a, w, out = full.tile_bytes
    work = 0

    def feasible(m, n, k):
        nonlocal work
        work += 1
        if work > max_candidates:
            raise OrderedMNKTilingError("streamed reduction search work limit exceeded")
        sizes = (m * k * a, n * k * w, m * n * out)
        if not accumulator_in_place:
            sizes += (m * n * out,)
        return sum(_ceil(v, alignment_bytes) * alignment_bytes for v in sizes) <= capacity_bytes

    best = None
    for m in range(1, mt + 1):
        if not feasible(m, 1, 1):
            break
        for n in range(1, nt + 1):
            if not feasible(m, n, 1):
                break
            low, high = 1, kt
            while low < high:
                middle = (low + high + 1) // 2
                if feasible(m, n, middle):
                    low = middle
                else:
                    high = middle - 1
            panels = _ceil(mt, m) * _ceil(nt, n)
            score = (panels * _ceil(kt, low), panels * m * n * kt, -low, -m, -n)
            if best is None or score < best[0]:
                best = score, (m, n), low
    assert best is not None
    return replace(minimal, block_tiles=best[1], k_block_tiles=best[2])

"""DMA / stream / buffer analysis — the data-movement search-space inputs.

From the P9 per-region memory envelope (:mod:`.memory_envelope`) this identifies the structural
data-movement *streams* each region implies (weight read, activation read, output write, and the
scale / KV / intermediate streams that a flat dequantized capture does not expose) and the minimum
*buffering* each region would need to overlap movement with compute. Every stream and buffer is a
structural search-space candidate — it does **not** claim a bandwidth, a buffer is sufficient for a
deadline, or a speedup. Bandwidth feasibility needs an explicit design YAML (absent here).

Streams whose bytes the capture cannot expose (scale sideband, KV/prefix, intermediate writeback)
are emitted with ``bytes = unavailable`` and the reason, never an invented size.
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.dse_guidance.design_envelope import CAPACITY_FORMATS, E_DERIVED, E_IR, E_NA

# P9-c candidate DMA abstraction vocabulary (the verifier checks membership).
ALLOWED_DMA_ABSTRACTIONS = {"multi_stream_dma_descriptor", "prefetch_descriptor",
                            "prefetch_weight_once", "double_buffered_activation_tile",
                            "scale_sideband_stream", "output_commit_stream", "kv_stream_handle",
                            "packed_weight_store", "activation_ring_buffer"}


@dataclass
class DmaStream:
    region: str
    stream: str                       # weight | activation_input | output | scale_sideband | ...
    bytes: object                     # int or "unavailable"
    direction: str                    # read | write | unavailable
    reuse_count: object
    prefetchable: str                 # yes | no | unknown
    can_overlap_compute: str          # yes | no | unknown
    sideband: bool
    candidate_abstraction: str
    evidence: str


def region_streams(rm) -> list[DmaStream]:
    """Structural data-movement streams for one region (from its memory envelope)."""
    return [
        DmaStream(rm.region, "weight", rm.weight_bytes, "read", rm.reuse_factor,
                  prefetchable="yes", can_overlap_compute="yes", sideband=False,
                  candidate_abstraction="prefetch_weight_once", evidence=E_IR),
        DmaStream(rm.region, "activation_input", rm.activation_input_bytes, "read", 1,
                  prefetchable="unknown", can_overlap_compute="yes", sideband=False,
                  candidate_abstraction="double_buffered_activation_tile", evidence=E_IR),
        DmaStream(rm.region, "output", rm.output_bytes, "write", 1,
                  prefetchable="no", can_overlap_compute="yes", sideband=False,
                  candidate_abstraction="output_commit_stream", evidence=E_IR),
        DmaStream(rm.region, "scale_sideband", "unavailable", "read", "unavailable",
                  prefetchable="unknown", can_overlap_compute="unknown", sideband=True,
                  candidate_abstraction="scale_sideband_stream", evidence=E_NA),
        DmaStream(rm.region, "kv_prefix", "unavailable", "unavailable", "unavailable",
                  prefetchable="unknown", can_overlap_compute="unknown", sideband=False,
                  candidate_abstraction="kv_stream_handle", evidence=E_NA),
        DmaStream(rm.region, "intermediate_writeback", "unavailable", "unavailable", "unavailable",
                  prefetchable="unknown", can_overlap_compute="unknown", sideband=False,
                  candidate_abstraction="activation_ring_buffer", evidence=E_NA),
        DmaStream(rm.region, "command_descriptor", "unavailable", "read", rm.reuse_factor,
                  prefetchable="unknown", can_overlap_compute="unknown", sideband=True,
                  candidate_abstraction="multi_stream_dma_descriptor", evidence=E_NA),
    ]


def all_streams(region_mem) -> list[DmaStream]:
    out = []
    for rm in region_mem:
        out.extend(region_streams(rm))
    return out


@dataclass
class BufferRequirement:
    region: str
    min_input_buffer_count: int
    min_output_buffer_count: int
    double_buffering_needed: str      # yes | no | unknown
    producer_consumer_queue_candidate: str
    input_buffer_bytes: int           # one activation-input tile (captured dtype)
    output_buffer_bytes: int
    resident_weight_bytes: int
    buffer_bytes_by_dtype: dict       # input+output tile scaled to candidate dtypes
    evidence: str


def buffer_requirements(region_mem) -> list[BufferRequirement]:
    """Minimum structural buffering per region to overlap movement with compute (no deadline claim).

    Double-buffer rule: a stream that is read/written every invocation while the next invocation's
    movement could proceed needs >= 2 buffers; the resident weight needs 1 (loaded once, reused)."""
    from merlin.dse_guidance.design_envelope import ELEMENT_BYTES
    out = []
    for rm in region_mem:
        tile = rm.activation_input_bytes + rm.output_bytes
        # captured element width to convert the tile byte count into element count for dtype scaling
        elem = 4.0
        n_elem = tile / elem
        out.append(BufferRequirement(
            region=rm.region,
            min_input_buffer_count=2,          # double-buffer the streamed activation tile
            min_output_buffer_count=2,         # double-buffer the committed output tile
            double_buffering_needed="yes",     # to overlap DMA with compute (structural)
            producer_consumer_queue_candidate="yes",
            input_buffer_bytes=rm.activation_input_bytes, output_buffer_bytes=rm.output_bytes,
            resident_weight_bytes=rm.weight_bytes,
            buffer_bytes_by_dtype={f: int(n_elem * ELEMENT_BYTES[f]) for f in CAPACITY_FORMATS},
            evidence=E_DERIVED))
    return out


# --------------------------------------------------------------------------- emitters

def dma_stream_csv(stream_by_workload: dict) -> str:
    from merlin.dse_guidance.corpus import _csv
    rows = []
    for wl, streams in stream_by_workload.items():
        for s in streams:
            rows.append({
                "workload": wl, "region": s.region, "stream": s.stream, "bytes": s.bytes,
                "direction": s.direction, "reuse_count": s.reuse_count,
                "prefetchable": s.prefetchable, "can_overlap_compute": s.can_overlap_compute,
                "sideband": s.sideband, "candidate_abstraction": s.candidate_abstraction,
                "evidence": s.evidence})
    return _csv(rows, ["workload", "region", "stream", "bytes", "direction", "reuse_count",
                       "prefetchable", "can_overlap_compute", "sideband", "candidate_abstraction",
                       "evidence"])


def buffer_requirement_csv(buf_by_workload: dict) -> str:
    from merlin.dse_guidance.corpus import _csv
    rows = []
    for wl, bufs in buf_by_workload.items():
        for b in bufs:
            rows.append({
                "workload": wl, "region": b.region,
                "min_input_buffer_count": b.min_input_buffer_count,
                "min_output_buffer_count": b.min_output_buffer_count,
                "double_buffering_needed": b.double_buffering_needed,
                "producer_consumer_queue_candidate": b.producer_consumer_queue_candidate,
                "input_buffer_bytes": b.input_buffer_bytes,
                "output_buffer_bytes": b.output_buffer_bytes,
                "resident_weight_bytes": b.resident_weight_bytes,
                "buffer_bytes_int8": b.buffer_bytes_by_dtype.get("int8"),
                "buffer_bytes_bf16": b.buffer_bytes_by_dtype.get("bf16"), "evidence": b.evidence})
    return _csv(rows, ["workload", "region", "min_input_buffer_count", "min_output_buffer_count",
                       "double_buffering_needed", "producer_consumer_queue_candidate",
                       "input_buffer_bytes", "output_buffer_bytes", "resident_weight_bytes",
                       "buffer_bytes_int8", "buffer_bytes_bf16", "evidence"])


def dma_pressure_report_md(stream_by_workload: dict, region_mem_by_workload: dict) -> str:
    from collections import Counter
    L = ["# DMA / stream pressure report\n",
         "> Structural data-movement streams each region implies, and which might justify a separate "
         "DMA/channel abstraction. **No bandwidth/speedup is claimed** (no bandwidth feasibility) — needs a "
         "explicit design YAML. Streams a flat dequantized capture cannot size (scale sideband, KV, "
         "intermediate) are `unavailable`.\n"]
    # which streams carry bytes vs unavailable
    L.append("## Streams per region (bytes-known vs unavailable)\n")
    L.append("| stream | bytes-known | direction | candidate abstraction | prefetchable | overlap |")
    L.append("|---|---|---|---|---|---|")
    seen = {}
    for streams in stream_by_workload.values():
        for s in streams:
            if s.stream not in seen:
                seen[s.stream] = s
    for s in seen.values():
        known = "yes" if s.bytes != "unavailable" else "unavailable"
        L.append(f"| {s.stream} | {known} | {s.direction} | {s.candidate_abstraction} | "
                 f"{s.prefetchable} | {s.can_overlap_compute} |")
    L.append("")
    # how many independent byte-carrying streams per workload
    L.append("## Independent byte-carrying streams per workload\n")
    L.append("| workload | regions | byte-carrying streams (weight/act/output) |")
    L.append("|---|---|---|")
    for wl, mems in region_mem_by_workload.items():
        L.append(f"| {wl} | {len(mems)} | {3 * len(mems)} |")
    L.append("")
    L.append("## Findings\n")
    L.append("- **Three byte-carrying streams per region** (weight read, activation read, output "
             "write) structurally suggest a `multi_stream_dma_descriptor` with independent channels.")
    L.append("- **The weight stream is prefetchable and reused** (`prefetch_weight_once`); the "
             "activation stream structurally suggests a `double_buffered_activation_tile`.")
    L.append("- **Scale-sideband, KV, and intermediate streams are `unavailable`** — the capture is "
             "dequantized (scales erased), attention is lowered (no KV), and fused intermediates are "
             "not materialized. They are named, not invented.")
    L.append("\n## Missing for real bandwidth feasibility\n")
    L.append("- per-stream bandwidth and a target memory hierarchy (a design YAML) — absent here; "
             "**no bandwidth, channel count, or overlap feasibility is claimed.**\n")
    return "\n".join(L)

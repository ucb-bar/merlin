# Compose tiled load addressing with panel completion

`TileLoadAddressLaw` states source-byte and tiled local-band addressing. It has
no synchronous flag. `propose_tile_load_coalescing` preserves exact mappings and
returns an unadmitted proposal; the old `TileLoadLaw` and synchronous optimizer
remain separate and reject an address-only law.

`admit_panel_tile_loads` consumes an original checked panel result plus the
actual proposed wide calls. A target callback decodes the final source pitch,
local block stride, rows and columns. `expand_tile_load_mapping` exposes every
local band, including checking the band's encoded operands against the final
command. The consumer joins these effects to original dense source operands and
runs the existing source, residency, accumulator continuation and epoch hazard
verifiers on that expansion. Every load requires a complete mapping. Dynamic
width is accepted only for an unchanged one-band tail; unknown wide dynamic
widths and oversized checking expansions refuse.

The checked result emits the original wide calls and revalidates their complete
mapping digest around SCF materialization. It records physical load commands and
expanded logical tile bands separately. It preserves the original full drains,
issue budgets, source identity and independent panel execution premises. An
addressing proposal neither grants overlap nor enlarges physical queue capacity.

The bridge forwards the original typed readout, bias and readout qualification
premises into the unchanged source/epoch checks. Original initializer and store
commands retain their separate drained epochs; only operand loads coalesce.
Original boundary identity remains live before and after checking/emission.
Target callbacks and numerical/physical execution remain ordinary independent
qualification obligations. No target or timing qualification follows from this
API or its synthetic tests.

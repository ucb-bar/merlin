# Atlas SmolVLA hybrid schedule status

The artifact now has an executable schedule constructor, but it intentionally
fails closed for whole-model execution. `build_hybrid_schedule.py` reconstructs
the live capture inventory, emits 6,104 deterministic capture-order events, and
records every accelerator, host, layout, and quantization boundary. It does not
promote structural partitions to executable partitions.

The earlier 2,033 “layout bridge” count needed refinement. These are candidates,
not aliases: 1,675 `view`/`unsqueeze` regions are proven metadata-only from
equal element counts and reshape/cast operations; 246 `expand` regions require a
strided/zero-stride descriptor that the current contiguous ABI lacks; and 112
`copy` regions require materialization. Independently, 2,430 regions require
host semantics. The existing bounded bridge implements 22 of those semantic
host regions, leaving 2,408 unresolved.

The schedule covers all 391 structural accelerator partitions, with explicit
host-to-device quantization, device-to-host dequantization, and 88
device-to-device requantization boundaries. Only the three previously qualified
partitions are executable, so 388 partitions remain blocked on capture-specific
calibration/numeric qualification. There are 1,250 conversion events in total;
only the 12 around qualified partitions are marked executable.

An inclusive-lifetime, 32-byte-aligned first-fit activation allocator assigns
all 391 partition outputs. Its deterministic symbolic arena peaks at 29,884,416
bytes, versus 550,404,032 bytes if every output had dedicated storage, saving
520,519,616 bytes. This is address planning, not proof of a physical Atlas DRAM
capacity or runtime speed.

The builder also executes the real host implementation for the bounded
`atlas_p0243 -> atlas_p0244` chain. It replays all 27 bridge regions and obtains
the exact saved p0244 activation hash
`23457ea06c994031379dcfc4491e866b1f9e5558a2ad4c5ad5a310366145bcce`.
The adjacent device steps are tied to their retained assertion-clean RTL
receipts; they are not rerun. This validates a two-partition hybrid chain, not
SmolVLA end to end.

The exact blockers are therefore measurable: 2,408 missing host semantic
implementations, 388 unqualified accelerator partitions, and 358 unrealized
layout bridges. Full schedule and compact summary are in
`whole_capture_plan/hybrid_schedule.json` and
`whole_capture_plan/hybrid_schedule_summary.json`.

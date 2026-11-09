---
title: "Zephyr model memory and image sizing"
kind: design
status: current
owner: core
last_verified: 2026-11-09
related: [runtime]
code_refs: [src/merlin/compiler/build/zephyr/memory.py, src/merlin/compiler/build/zephyr/config.py, src/merlin/runtime/backends/zephyr_model.py]
---

# Zephyr model memory and image sizing

`compiler/build/zephyr/memory.py` owns `ram_for_weights`, `memory_plan` and
`require_external_arena`; `compiler/build/zephyr/config.py` owns image
configuration. `runtime/backends/zephyr_model.py:build_app` consumes those
results while constructing the image.

## `ram_for_weights`

RAM-region size to hold the weights blob (linked into .data) plus an activation
arena (the leftover, claimed by ARENA_SIZE=-1). Headroom scales with the model
(30% of weights + 128 MiB), and the result is rounded up to 16 MiB. The minimum
is the selected board's `default_ram_bytes`, supplied by the caller; 256 MiB is
not a universal compiler default. Earlier FireSim measurements motivated keeping
a small image at its port's default region rather than enlarging every image.

TWO ESTIMATES OF THE SAME QUANTITY, and the measured one wins.

``activation_bytes`` (from :func:`mlir_query.activation_peak_bytes`) is LIVE bytes inferred from the
MLIR. It exists because a flat 128 MB of headroom does not always cover a model's working set:
whisper_tiny's encoder attention peaks at 210 MB live where the weights-scaled formula left it a
163 MB arena, i.e. a provisioning failure on a board with enough physical DRAM.

``allocation_bytes_total`` (from :func:`allocation_bytes`) is what the EMITTED IR actually asks the
heap for, summed. It is the better number wherever it exists, in both directions:

* It is an UPPER bound on live bytes -- every live byte was allocated -- so it cannot under-provision
  the way a liveness estimate can when bufferization inserts copies the estimate never saw.
* It prices only what comes out of the arena. The MLIR estimate prices tensors that never become heap
  allocations at all, and it can therefore be enormously PESSIMISTIC: measured on a 12-layer Gemma
  section, the estimate said 2827 MB while the emitted IR's allocations total 1533 MB -- an estimate
  larger than the sum of every allocation the program makes, which is impossible for real arena
  demand. Sizing from it asked for a 4416 MB region on a 4096 MB design and refused to build a model
  that fits in 3136 MB.

So when the measured total is available it REPLACES the estimate rather than joining it in the max.
That is deliberately not monotone with the older formula: a model whose estimate was pessimistic gets
a smaller region, which is the whole point. The 128 MB slack stays on top of whichever number is used,
because neither covers allocator fragmentation and because an allocation whose size is computed rather
than literal is not in the measured total (:func:`allocation_bytes` reports how many).

A shortfall is not silent, which is what makes this safe to tighten: the image links a malloc guard
that prints ``FAIL alloc bytes=<n> op=<id>`` and terminates in the protocol, so a returned log names
the allocation that did not fit.

## Selected board layout

`memory_plan` checks the selected board's default region, link limit and physical
DRAM capacity before choosing linked or external weights. An external layout
requires the board's explicit low-memory region and tail reservation; exceeding
the link limit cannot invent another SoC's map. `require_external_arena` checks
the actual linked ELF segments leave enough low RAM for the activation demand.
An override must fit the weights and declared board limits. These are image
construction checks, not evidence that Zephyr or the selected hardware ran.

## `build_app`

Lower the model, generate the Zephyr app, and build ``zephyr.elf``.

``device`` reaches preparation, but active device calls refuse: this builder has no
device-object/final-link admission consumer. Host-only and inert routes retain their path.

``backend``: ``"rvv"`` (vector tile) or ``"scalar"`` (scalar tile). The
scalar build is the portable FireSim-safe path; the vector build targets the vector
tile (worker on ``rvv_hart``). Returns ``{elf, app_dir, backend, **c_runtime_info}``.

``rvv_schedule`` overrides the default ``RVV_TRANSFORM_SCHEDULE`` (the seam through which
an isolated RVV target-package supplies its own transform schedule); ``cflags_override``
replaces ``_cflags(backend)``. Both default ``None`` -> the build is byte-identical to the
shipping codegen, so the global flow is never perturbed by the package machinery.

``n_harts > 1`` lowers an outer OpenMP-parallel loop under the RVV schedule
(``parallel_harts``) and links the Zephyr shim's COOP worker pool, one per hart.
``n_harts=1`` (default) retains the single-worker image.

``iters``/``warmup`` build the SUSTAINED-inference image (per-iteration cycle metrics
over a reused arena); the defaults are the single-shot behavior.

UART boards require ``sdk_dir``; ``runtime.sdk_facts`` derives the UART address and
baud-divisor clock rates from its headers rather than assuming them.

``completion_metric_prefix`` is an optional terminal cycle marker included in the image build hash.

``masked_contraction_effects`` forwards an explicit nontrapping,
unobserved-floating-flags contract to closed-mask scalar scheduling. It
requires separate feature/schedule selection; the default adds no permission.

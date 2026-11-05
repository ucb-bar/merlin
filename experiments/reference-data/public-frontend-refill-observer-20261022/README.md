# Public instruction-cache refill observation

This operator diagnostic uses the unchanged public frontend discriminator ELF
and selected RTL objects. Only the emulator's read-only observer and trace
transport differ. It is not an author input or an activated timing model.

The completed run took 21.738 seconds, checked 4,608 output values and 576 guard
values, and preserved all twelve original cycle measurements. The full trace has
26,689 events, no drops, and a normal completion footer. Earlier admission and
file-transport failures remain in this directory, including the paid physical
run whose trace could not be returned.

## Independent ICache analysis

`icache-analysis-index.json` pins the parser and complete attribution. Parsing
the retained trace found 112 complete local request/response sequences, no
unmatched requests, and no framing errors. Event sequences are contiguous and
sample times agree with the pinned 1,000 ps reference period and 500 ps sampling
offset.

| Scope condition | Scopes | Cycles each | Refills inside each scope | First / final response |
|---|---:|---:|---:|---:|
| Instruction cache invalidated | 6 | 177 | 3 | 6 / 9 cycles |
| Warm | 6 | 126 | 0 | Not applicable |

Both instruction layouts produce the same result. Each in-window refill returns
four beats, spaced one cycle apart. The observed writeback-marker spans match
the guest counter measurements in this run.

These are request-to-response durations, **not additive CPU stall costs**. The
51-cycle cold/warm difference is not automatically attributable to a particular
cache stage. Retained directory and bank events lack the transaction identities
needed for a proven join to these local ICache requests. Timing coincidence is
insufficient to claim an L2-hit service measurement.

This evidence constrains the public SoC model's instruction-fetch behavior. It
does not qualify that model, fit a coefficient, alter its active selection, or
establish a compiler performance improvement. Further analysis can reuse these
events without another physical run.

`retention-index.json` inventories original evidence; `icache-analysis-index.json`
inventories the later independent analysis. `analyze_icache.py` is the archived
bounded analysis script; its input is an uncompressed copy of `events.jsonl.gz`
and its `--out` destination must be outside this reference directory.

# Current public WS residuals and FIFO admission correction

The previous 521-cycle queued-drain and 753-cycle readout estimates are obsolete.
Fresh Spike replays against the **same retained public gSIM ELF, embedded inputs,
counter scopes and outputs** give the following results. There were no new gSIM
or FireSim runs, no private reference labels, and no fitted coefficients.

| Instrumented interval | Selected model | FIFO candidate | gSIM |
|---|---:|---:|---:|
| Queued submission | 88 | 88 | 83 |
| Queued drain | 146 | 129 | 130 |
| Queued readout | 1473 | 1473 | 1521 |
| Intercompute-fenced submission | 663 | 663 | 683 |
| Intercompute-fenced drain | 4 | 4 | 3 |
| Intercompute-fenced readout | 1473 | 1473 | 1603 |

The one-cycle fenced-drain discrepancy is a tiny diagnostic interval, not a
complete-case relative-error qualification failure to optimize in isolation.
All 4096 checked values in the instrumented probe match. The physical engine
uses the retained `mm_magic_t` memory configuration, not the newly integrated
native FASED adapter and not established deployed FireSim timing equivalence.

| Complete public call | Selected model | FIFO candidate | gSIM | Candidate error |
|---|---:|---:|---:|---:|
| Unfenced | 1717 | 1700 | 1740 | −2.30% |
| Intercompute-fenced | 2153 | 2153 | 2245 | −4.10% |

Both models rank the unfenced variant ahead. The selected model predicts a
436-cycle improvement versus 505 measured; the candidate predicts 453.
The candidate **worsens the unfenced total error** (previously −1.32%) while
correcting the drain component. This demonstrates cancellation with the
remaining readout underprediction. These developmental cases are not independent
held-out cases, and neither result qualifies a 4% model.

## What was corrected

The queued trace has 17-cycle operand-feed spacing until the seventh tile,
where it jumps to 34 cycles. The CPU already submitted that command in time.
`StagedObserver::pair_preload` refused paired execution whenever either FIFO
entry was unavailable at the earliest CPU admission, even when both would become
available before the already-reserved feed. The fix previews both admissions,
allows their legal wait, and still requires both issue beats to complete before
the feed. It never shifts that feed or releases data dependencies earlier.

The sanitizer test suite retains late-arrival, capacity, weight-bank and output
hazard refusals. A new eight-tile regression passes the candidate and fails the
old source. Eight isolated replays retain exact output and scope identity.
`candidate-build.json` pins the new provider and all source dependencies.
Only the provider library identity changes in its copied model; prices remain
unchanged. This is **not an active runtime selection**.

The original drain interval spans modeled times 18496–18642. Its fence waits
143 cycles for already-submitted device work. These times are modeled scheduling
events, not physical counters or CPU utilization. `baseline-timeline.json`
retains the relevant controller and accumulator reservations.

## Remaining structural gap

Both readout arms have identical observed modeled features and the same
1473-cycle prediction, but gSIM differs by 82 cycles. A global compute or preload
price cannot express this state-dependent difference. The selected Rocket RTL
conditions probe admission on metadata arbitration, outstanding releases and
acknowledgments, grants, CPU pipeline activity and progress state. The current
private-cache service formula is a no-backpressure floor; it does not reproduce
that full stateful arbitration or inclusive-cache/TileLink traffic.

The next structural work is a shared CPU/DMA probe–release–ack admission timeline,
with source-derived service and queue limits. Reuse the existing public clean/
dirty ownership probe and obtain matching handshake evidence before changing
service costs. An arm-specific additive constant would hide the missing state.
No new physical witness was needed to locate the FIFO defect.

`proposed-supported-domain.json` describes a **blocked proposal**, not an executable
calibration profile. It requires independent source-disjoint complete calls,
known initial state and exact runtime/configuration identity, with every held-out
case at or below 4% error. Releasing the current compute/preload fit before the
readout structural review remains unjustified. The active admission denial and
rehearsal runtime are unchanged.

## Reproduction and identity

`baseline-windows.json`, `baseline-complete-calls.json`, and
`candidate-comparison.json` preserve exact command arguments with machine roots
redacted, plus ELF/input/window, model, observer, report and physical-label hashes.
The raw commands, scripts, reports and provider are retained in the task's
`out/artifacts/public-ws-current-residuals-20261023` and
`out/artifacts/ws-pair-admission-candidate-20261023`. All subprocesses used CPUs
29–30, outside the rehearsal reservation. `redaction.json` identifies the original
raw receipt bytes. `SHA256SUMS.json` covers curated files.

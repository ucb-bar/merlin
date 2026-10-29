# Experiment 2: compose a new convolution optimization from Fresh6

**Question:** Do the reusable composition APIs added after Fresh6 help an agent
extend a working compiler faster, with fewer errors and transferable performance?

Both agents start with the complete, identical graded Fresh6 source, original
library, accepted/rejected work, rules and handoff. Preserve the seed's existing
functionality; do not ask either author to reconstruct already working passes.
Fresh6 already has dense retention, coalescing, ready-readout and schedule plans.
The old single-M retention guard task is only a harness smoke test.

## Same task, one additional capability

Extend the convolution compiler to reuse rolling input rows and array weights
across capacity-bounded output groups. Preserve every original output, bias,
readout, required completion and a legal fallback. Select by source geometry,
layout and resource predicates; never by a capsule or model name.

- **Control:** Fresh6's original library and unrestricted backend editing. The
  author implements the new composition with inherited primitives or own code.
- **Treatment:** the same seed and freedom, plus the newer checked composition
  APIs and their generic contracts. No completed backend patch is supplied.

The actual additions are `plan_rolling_convolution_output_group`, its emitted
K/N/M array-reuse consumer, grouped rolling-window SCF transport, capacity-derived
RHS phases and compatible wide readout composition. Fresh6 has some individual
primitives, but its convolution invocation owner cannot consume this composition.
The treatment must build the integration and choose reusable rules itself.

Both arms receive identical relevant raw reference files, trusted expert targets,
hardware/ISA facts, numerical contracts, feedback and budgets. CCA is off in both.
Existing scheduling tools remain common. The frozen oracle, selected parameters,
winning pass sequence and hidden cases stay outside author inputs. This measures
incremental library assistance; it does not measure removal of all libraries.

## Small panel and generalization

Use `inputs.py --profile conv-family-v2`. The source is SAME NHWC convolution:
3×3, stride/dilation 1, one zero-padding element on every side, signed-INT8
operands, INT32 bias and saturating INT8 output with effective identity scaling.
There is no pooling or activation. Input and output both have the stated H×H
spatial extent; padding belongs to the operation, with no prepared host halo.

- Public development: B4/H28/C128→128 anchor, B2/H28/C128→128 and
  B4/H21/C128→128. All three receive correctness and performance feedback.
- Frozen batch sweep: B1, B6, B8.
- Frozen spatial sweep: H14, H32, H35, H42.
- Frozen input-channel sweep: Cin64, Cin96, Cin192, Cin256.
- Frozen output-channel sweep: Cout64, Cout96, Cout192, Cout256.

Each of the 15 hidden shapes changes one axis relative to B4/H28/C128→128.
`same_conv_case` updates the typed source through the existing window and
interface owners. Inputs use ordinary intake/admission/staging. Hidden source,
metadata and checks stay operator-only until compiler freeze. The older
`conv-composition` profile is preserved for its historical valid-convolution
artifacts and is not the new expert-matched panel.

The expert function handles one B1/H28/C128→128 image with the same zero-padding
and readout contract. Compare one logical B4 compiler call against four unchanged
expert calls inside one common completed timing region; permit the compiler to
optimize batching. Use the same logical arrays, preparation/cache sequence,
weight/bias state and readback boundary, and disclose both callee counts. Exclude
preparation, checking and printing consistently. This is a wrapper requirement,
not a claim that a fresh matched physical measurement already exists.

The exact expert binary also supports batch-only cases through repeated B1
calls. It is not a resized expert for other H/Cin/Cout shapes. Software/FSM and
operator-oracle comparisons require exact shape, numerical and wrapper support;
otherwise leave the comparison blank with its reason.

## Lean execution and acceptance

Use the shared bounded seeded runner, three public capsules and one evaluator per
arm; no new experiment harness. Allocate five cores to the experiment: four
shared cores with an equal 200% CPU quota per arm, plus an optional fifth core
for preparation. Give each arm 30 minutes, with a separate compact evaluation
allowance. Reuse inherited
numerics only for identical executed programs under an equivalent judge. Check
new emissions and affected convolution paths rather than rerunning all Fresh6.

Freeze one final compiler, rules and settings before hidden evaluation. Reopen
actual emitted effects to check ring lifetime, full-K RHS placement, fresh/stay
reuse, store mapping and completion before private reuse. Use the common model to
shortlist; compare matched FireSim completed-call samples after freezing, retaining
first/warm order. Report failures, refusals, regressions and unsupported baselines.
Record author wall/CPU time, provider tokens, candidate count, invalid proposals,
time to a qualified mechanism and final performance on every panel shape.

The operator oracle is a separate capability target, not an author solution.
Both untouched Fresh6 and the current six-module composition overlay have now
compiled the SAME anchor and matched all 401,408 original expected outputs on
the functional accelerator target. Neither has new physical timing. Historical
C128 measurements of 2.093M versus Jack's 2.122M used explicit halo preparation
and different B4/B1 wrappers; retain their original scope instead of rebinding
them to this new source or rebuilt binary.

The unchanged expert, untouched Fresh6 and new SAME oracle now each match all
401,408 anchor outputs using byte-identical ordinary harnesses. The expert's
original function bytes are unchanged. This qualifies the mathematical and
wrapper correspondence; matched physical cache preparation and timing follow.

The earlier byte-identical-ELF reuse proof and Fresh6 FireSim samples
11,580,402 / 11,570,506 / 11,589,649 belong to the old valid-input30 study.
Keep them as historical evidence, not the baseline for this changed source.
No unrelated library files are updated or added. After freeze, compare the
author's actual grouping, reuse, phases and transfers with the frozen oracle
policy on each supported shape; report different legal choices and measured
costs rather than requiring the same constants or pass sequence.

No author or FPGA campaign is launched by preparation. A single matched pair is
a pilot; repeat promising comparisons before drawing statistical conclusions.

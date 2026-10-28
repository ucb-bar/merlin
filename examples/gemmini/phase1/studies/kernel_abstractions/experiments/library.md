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

Use `inputs.py --profile conv-composition`. The source contract is valid NHWC,
3×3, stride/dilation 1, no padding, signed-INT8 operands, original INT32 bias and
saturating INT8 output with effective identity scaling. No pooling or activation.
Spatial dimensions below are **output** dimensions; input dimensions are two
larger on each axis.

| Split | Batch | Output | Cin → Cout | Change |
|---|---:|---|---|---|
| Author feedback | 4 | 28×28 | 128 → 128 | Anchor |
| Hidden after freeze | 8 | 28×28 | 128 → 128 | Batch scaling |
| Hidden after freeze | 4 | 32×32 | 128 → 128 | Spatial scaling |
| Hidden after freeze | 4 | 28×28 | 128 → 64 | Channel scaling |

`resize_conv` changes the weight, bias, output and readout coefficient extent
along with Cout. Inputs use the ordinary intake/admission/staging path. The
held-out descriptor and exact checks are operator-only. Tails, stride changes
and a capacity-crossing Cout256 case belong to a later stress panel.

## Lean execution and acceptance

Use the shared bounded seeded runner, one train capsule and one evaluator per
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
Historical C128 measurements reach 2.093M cycles against Jack's 2.122M. They used
explicit halo preparation outside the kernel ROI and different B4/B1 wrappers;
they do not automatically qualify a newly rebuilt consumer or generated inputs.
A current real-target consumer must be frozen and numerically checked first.
The public generated-effect toy example alone is insufficient target evidence.

This qualification is complete: the scoped six-file overlay rebuilds the exact
ELF that passed all 401,408 outputs, so that numerical result was reused. No
unrelated library files are updated or added. Retained exact-source Fresh6
evidence also checks all outputs and records FireSim samples
11,580,402 / 11,570,506 / 11,589,649 cycles. Preserve its original ROI/cache and
judge identity; it is not a newly timed result or an automatic current certificate.
The historical oracle timing remains a separate binary/measurement record.

No author or FPGA campaign is launched by preparation. A single matched pair is
a pilot; repeat promising comparisons before drawing statistical conclusions.

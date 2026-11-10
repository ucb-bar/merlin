# Voyager regression inputs

This is a finite set of pinned public-compiler outputs used by the compact test
suite. It is not a workload corpus or an input default for compiler generation.
The reader under test is `merlin.evaluation.frameworks.voyager_ir`; host operation
references live in `merlin.evaluation.frameworks.voyager_schedule`.

The linear and convolution folders hold Voyager model JSON plus the manifest
recording the producing compiler revision, hardware configuration and quantization
flags. They exercise output-column grids, operand slots, guarded loads, split-K
chains, convolution halos and host layout changes. The residual-block folders also
hold scale records; the basic-block case includes a small independent input/output
archive. `host_ops_golden/` records public Voyager operation results for rounding,
saturation, pooling and bfloat16 conversion cases.

Keep these exact manifests and independent outputs together. They test different
reader obligations; do not replace them with one generated happy-path sample.
Public source provenance and the recorded compiler revision govern each fixture.
A result from another revision is a new observation, not a repair to the old one.

The original export scripts were study inputs and are no longer active source
commands. Historical regeneration requires the scripts and upstream environment
from the recorded source revision. New exploratory exports belong under ignored
`out/`; review a small permanent regression only when it protects a distinct core
obligation. Preserve the repository-wide limit of 150 collected test cases.

Do not expose independent golden answers to an evaluated compiler candidate.
Generated experiment corpora, captures and compiler snapshots never belong in
this directory.

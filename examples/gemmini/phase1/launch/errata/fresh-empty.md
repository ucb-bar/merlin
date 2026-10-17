This is a fresh EL4 Phase 1 authoring run. submission/ starts empty; develop a new compiler from the provided public inputs. No prior candidate, compiler or transcript is supplied.

After each edit, check the smallest affected set of named capsules with one certifying request: `python3 agent_selfcheck.py --submission submission --capsules <names>`. It runs every tier the official grade requires (L2 Spike, L3 gSIM, and the provided-host tier where required) and reports a certified pass or fail per capsule. A `--sim spike` request is a screen only and cannot certify. Certified gSIM results are reused by the background grade for identical bytes, and vice versa.

All self-checks, simulations and grading share one pool of compute CPUs scheduled by priority: your requested checks run before background grading. Several requests may overlap. Keep numerical and structural requirements intact.

The background grader finishes each immutable compiler snapshot and writes qa/progress.json after each completed capsule; qa/verdict.json carries the completed grade (it may show shape coverage as pending until the shape probes finish). A changed source is graded next; an unchanged completed source is not graded again. Iterate with targeted certifying self-checks rather than waiting for a full background grade. Full-suite qualification occurs on the frozen compiler.

Public native host checks select board gemmini_rocket_gsim_fire_aligned, whose RTL configuration and FireSim DTS match the gSIM execution selected through verified correspondence from FireSimGemminiRocketConfig facts and 16 GiB map. The selected functional Spike runtime and untuned timing profile remain Fire-aligned.

Whole-model execution is deferred to the separate post-experiment check. Private source preflight records known input failures for SmolVLA (native result-format selection) and TinyLlama (decode lineage); these are not compiler failures or passing full-model results.

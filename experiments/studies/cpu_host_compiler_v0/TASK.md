# Build a reusable CPU-host compiler

Implement an out-of-tree MLIR CPU backend for the supplied target contract and generic capsule interface.
The backend must provide scalar, RVV, and deterministic multicore lowering; it must handle dynamic vector
tails and may use a fixed VLEN specialization only behind the supplied runtime proof.

Improve only against the public training and validation capsules.  Do not inspect or infer paper-model
artifacts, checkpoints, captures, profiles, or results.  Do not dispatch on workload or model names.

Knobs, flags, heuristic choices, and pass pipelines should be searched deterministically.  Author a new
pass only when the existing transformations cannot express a general optimization.  New passes must state
their legality conditions, change emitted code, preserve the scalar reference semantics, and clear the same
cross-family validation and noise-margin gates as every other candidate.

When the arm provides `policy/beam_search.py`, use that runner and its frozen
`policy/optimization_space.yaml`; agent judgment must not select a winner. Run it with
`/usr/bin/python3 -B`, output at workspace-local `scratch/search_work` outside `submission/`, and
evaluator argument array `/usr/bin/python3 -B policy/trusted_evaluator.py`. `scratch/` is the only
sanctioned location for transient build and search state. After convergence, copy only `search_record.json` and
`selected_policy.json` into `submission/search/`. The staged shim
receives only a candidate policy and the deterministic public capsule sample. A driver-owned broker first
screens every legal extension on two train capsules from each of all six generic families, with trusted
Spike correctness, cycle counts, and emitted-code SHA-256 digests. Only the deterministic width-one top
survivor receives exactly six balanced parent/child K1 timings per capsule, using one controller-private
post-freeze train and validation shape from each of all six generic families. Validation is consulted only for
promotion. Copy the final `selected_policy.json` byte-for-byte
to the manifest-declared submission policy. Do not add a held-out path, edit the search space, replace the
trusted evaluator, or change compiler source after search begins; the driver replays and seals the search.

The deliverable is a self-contained compiler package plus a machine-readable policy.  It is complete only
when it passes the portable RVV grader and the K1 silicon promotion gate without fallback.

## Public pre-campaign runtime diagnostic

A controller-owned qualification on the public `runtime_parallel/static_partition` family found that
creating and joining the worker team inside every `merlin_capsule_run` invocation overwhelms useful work
for small regions.  Treat parallel-region granularity as a general compiler problem: retain a serial path
when the compiler can prove that dispatch overhead is not amortized, and use multicore only when the
public shape/reuse/target facts justify it.  A multicore artifact is not an optimization merely because it
creates the requested threads.

This experiment's deliberately narrow per-capsule ABI requires each multicore invocation to create and
join its exact worker team so the trusted L3 harness can replay every submitted shard independently and
prove physical work, affinity, disjoint coverage, and worker dependence.  A persistent worker pool is a
separate session-runtime optimization and is neither implementable nor scored through this ABI.  Do not
create one in the submitted kernel.  This boundary lets all four arms receive the same observable rules;
the continuous-inference study evaluates the compiler inside a separate persistent model session.

Any compiler remedy must be selected from generic operation, shape, reuse, and target facts; it may not
key on a capsule identity or paper model. The grader remains the authority for physical hart use, exact
coverage, numerical correctness, and speed.

The package and per-capsule executable ABI are frozen in `SUBMISSION_CONTRACT.md`. Implement that
interface exactly; the same isolated grader is used for all arms and does not interpret prose in the final
answer as an artifact.

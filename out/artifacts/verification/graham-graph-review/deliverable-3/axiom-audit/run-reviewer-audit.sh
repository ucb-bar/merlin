#!/usr/bin/env bash
# Graham D3: run ReviewerAudit.lean over each row's compiled environment.
# Requires build-and-audit.sh and build-inhabited.sh to have produced build/lib*.
set -uo pipefail
here=/home/grammy/blast/merlin-pass-verification/out/artifacts/verification/graham-graph-review/deliverable-3/axiom-audit
T=/home/grammy/.elan/toolchains/leanprover--lean4---v4.35.0/bin
B="$here/build"
run() { name=$1; shift; lp=$1; shift
    (cd "$here" && PATH="$T:$PATH" LEAN_PATH="$lp" "$T/lean" --run ReviewerAudit.lean "$@") \
        > "$B/logs/reviewer-audit-$name.stdout" 2> "$B/logs/reviewer-audit-$name.stderr"
    echo "$name rc=$?"; }
run scalar "$B/lib" GraphStateScalarInstance
run pipeline "$B/lib" GraphStatePipelineInstance
run integer "$B/lib" GraphStateIntegerInstance Merlin.GraphStateIntegerCountermodels Merlin.GraphStateIntegerModels
run integer-query-only "$B/lib" GraphStateIntegerInstance
run shape "$B/lib" Obligation
run memory-scf "$B/lib" Merlin.GraphLoopsMemorySCFBounds Merlin.GraphLoopsMemorySCFWitness Merlin.GraphLoopsReuseCountermodels
run declared "$B/lib" Check Merlin.GraphRegionsDeclaredCountermodels
run pipeline-symbolic-provider "$B/lib-pipeline-symbolic-provider:$B/lib" GraphStatePipelineInstance
run integer-provider "$B/lib-integer-provider:$B/lib" GraphStateIntegerInstance

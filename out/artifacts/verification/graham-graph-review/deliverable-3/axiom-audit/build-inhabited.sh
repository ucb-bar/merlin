#!/usr/bin/env bash
# Graham D3: compile the two retained inhabited-provider queries (same module names as
# the main queries, so each gets its own object directory) and print their axioms.
# Read-only with respect to all sources; outputs stay under this directory.
set -uo pipefail
here=/home/grammy/blast/merlin-pass-verification/out/artifacts/verification/graham-graph-review/deliverable-3/axiom-audit
lean=/home/grammy/.elan/toolchains/leanprover--lean4---v4.35.0/bin/lean
logs="$here/build/logs"
status=0
while read -r label module path; do
    lib="$here/build/lib-$label"
    mkdir -p "$lib"
    (cd "$(dirname "$path")" && LEAN_PATH="$here/build/lib" "$lean" -j1 "$(basename "$path")" -o "$lib/$module.olean") \
        > "$logs/query-$label.stdout" 2> "$logs/query-$label.stderr"
    rc=$?; printf '%s\t%s\n' "query:$label" "$rc" >> "$logs/modules.tsv"; [ $rc -eq 0 ] || status=1
done < "$here/queries-inhabited.txt"
run_print() {
    (cd "$here/print-inhabited" && LEAN_PATH="$here/build/lib-$2:$here/build/lib" "$lean" -j1 "$1.lean") \
        > "$logs/print-$1.stdout" 2> "$logs/print-$1.stderr"
    rc=$?; printf '%s\t%s\n' "print:$1" "$rc" >> "$logs/modules.tsv"; [ $rc -eq 0 ] || status=1
}
run_print PrintPipelineInhabited pipeline-symbolic-provider
run_print PrintIntegerInhabited integer-provider
exit $status

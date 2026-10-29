#!/usr/bin/env bash
# Graham D3: fresh, read-only local compile of the transitive Lean closure and
# independent `#print axioms` on final theorems. Sources are read in place; every
# object/log is written under this directory. No toolchain, source, receipt or
# historical object is modified. This is NOT an original-bound replay.
set -uo pipefail
repo=/home/grammy/blast/merlin-pass-verification
here="$repo/out/artifacts/verification/graham-graph-review/deliverable-3/axiom-audit"
lean=/home/grammy/.elan/toolchains/leanprover--lean4---v4.35.0/bin/lean
lib="$here/build/lib"
logs="$here/build/logs"
mkdir -p "$lib" "$logs"
export LEAN_PATH="$lib"
"$lean" --version > "$logs/lean-version.txt"
status=0
while read -r module path; do
    root="$repo/src/merlin/verify/lean"
    case "$module" in MerlinVerify.*) root="$repo/merlin/lean" ;; esac
    rel="${module//./\/}"
    mkdir -p "$(dirname "$lib/$rel")"
    start=$(date +%s)
    (cd "$root" && "$lean" -j1 "$rel.lean" -o "$lib/$rel.olean") > "$logs/$module.stdout" 2> "$logs/$module.stderr"
    rc=$?
    printf '%s\t%s\t%s\n' "$module" "$rc" "$(( $(date +%s) - start ))" >> "$logs/modules.tsv"
    [ $rc -eq 0 ] || status=1
done < "$here/build-order.txt"
# Generated queries (read in place, compiled to this directory).
while read -r name path; do
    start=$(date +%s)
    (cd "$(dirname "$path")" && "$lean" -j1 "$(basename "$path")" -o "$lib/$name.olean") > "$logs/query-$name.stdout" 2> "$logs/query-$name.stderr"
    rc=$?
    printf '%s\t%s\t%s\n' "query:$name" "$rc" "$(( $(date +%s) - start ))" >> "$logs/modules.tsv"
    [ $rc -eq 0 ] || status=1
done < "$here/queries.txt"
# Reviewer-authored #print axioms files.
for file in "$here"/print/*.lean; do
    name=$(basename "$file" .lean)
    (cd "$here/print" && "$lean" -j1 "$name.lean") > "$logs/print-$name.stdout" 2> "$logs/print-$name.stderr"
    rc=$?
    printf '%s\t%s\n' "print:$name" "$rc" >> "$logs/modules.tsv"
    [ $rc -eq 0 ] || status=1
done
# Reviewer-authored independent witnesses.
for file in "$here"/../witness/*.lean; do
    name=$(basename "$file" .lean)
    (cd "$here/../witness" && "$lean" -j1 "$name.lean") > "$logs/witness-$name.stdout" 2> "$logs/witness-$name.stderr"
    rc=$?
    printf '%s\t%s\n' "witness:$name" "$rc" >> "$logs/modules.tsv"
    [ $rc -eq 0 ] || status=1
done
exit $status

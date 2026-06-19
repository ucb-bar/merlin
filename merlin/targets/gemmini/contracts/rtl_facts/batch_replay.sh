#!/usr/bin/env bash
# Batch the arc replay across a capsule run tree: for each capsule, reconstruct the RoCC stream,
# build the harness, run on the isolated @Gemmini arc model, report bit-exact vs golden.
set -u
ROOT=/scratch/agustin/projects/oscar-merlin
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python3
OUT="$ROOT/merlin/targets/gemmini/contracts/rtl_facts"
RUNS="${1:-$ROOT/runs/capsule_bench_v1/runs/gemmini-capsule-bench}"
printf "%-32s %-14s %-9s %-7s\n" CAPSULE result cycles unkop
pass=0; tot=0
for d in "$RUNS"/*/; do
  cap=$(basename "$d"); T="$d/generated/instruction_trace.json"
  [ -f "$T" ] || continue
  CAP=$(find "$ROOT/bench_contract/capsules" -type d -name "$cap" | head -1)/capsule.yaml
  [ -f "$CAP" ] || { printf "%-32s %-14s\n" "$cap" "no-capsule"; continue; }
  tot=$((tot+1))
  if ! "$PY" -m merlin.targetgen.rtl.gen_rocc_replay "$CAP" "$T" --out "$OUT/r.json" >/dev/null 2>/tmp/g.e; then
    printf "%-32s %-14s\n" "$cap" "GEN-ERR"; continue; fi
  meta=$("$PY" "$ROOT/merlin/python/merlin/targetgen/rtl/replay_json_to_h.py" "$OUT/r.json" "$OUT/replay_active.h" 2>&1)
  unk=$(echo "$meta" | grep -oE 'unknown_operands=[0-9]+' | cut -d= -f2)
  if ! clang -O2 -w -I"$OUT" "$OUT/gemmini_arc_replay.c" "$OUT/gemmini.o" -o "$OUT/rbin" 2>/tmp/cc.e; then
    printf "%-32s %-14s\n" "$cap" "CC-ERR"; continue; fi
  out=$(timeout 120 "$OUT/rbin" 2>&1 | tail -1)
  cyc=$(echo "$out" | grep -oE 'cycles=[0-9]+' | cut -d= -f2)
  if echo "$out" | grep -q 'BIT-EXACT PASS'; then verd=PASS; pass=$((pass+1));
  elif echo "$out" | grep -q MISMATCH; then verd=MISMATCH;
  else verd="TO/ERR"; fi
  printf "%-32s %-14s %-9s %-7s\n" "$cap" "$verd" "${cyc:-?}" "${unk:-0}"
done
echo "----"
echo "BIT-EXACT: $pass / $tot capsules"

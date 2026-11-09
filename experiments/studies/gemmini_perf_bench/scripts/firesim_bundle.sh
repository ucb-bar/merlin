#!/bin/bash
# Bundled FireSim runner: flash the FPGA bitstream ONCE, then boot+run many bare-metal ELFs
# back-to-back in the same held session. The cycle-accurate sim is seconds/ELF; the ~6min cost is the
# bitstream flash + driver build in `infrasetup`, so doing it once amortizes it across the whole batch.
# Runs AS a single firesim-queue job (holds the FPGA lock for the bundle). Fail-open: a hung/crashed ELF
# is recorded (no METRIC cycles) and we self-heal with kill+infrasetup before the next one.
#
# Args: $1 = manifest (TSV: "<label>\t<elf_path>" per line), $2 = outdir, $3 = per-run timeout secs,
#       $4 = per-bundle config_runtime.yaml (workload_name must be merlin-perfbench.json).
set +e
MANIFEST="$1"; OUTDIR="$2"; RUNTIMEOUT="${3:-300}"; CFG="$4"
CFGARG=""; [ -n "$CFG" ] && CFGARG="-c $CFG"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_SHLVL CONDA_EXE _CE_M _CE_CONDA
export PATH=/path/to/miniforge3/condabin:$PATH
source /path/to/chipyard/env.sh
cd /path/to/chipyard/sims/firesim || exit 2
source ./sourceme-manager.sh --skip-ssh-setup
cd deploy || exit 2

SLOT=/path/to/FIRESIM_RUNS_DIR/sim_slot_0
BOOT="$SLOT/merlin-perfbench0-merlin-perfbench.elf"   # TSI loader's view (prefixed)
WLBIN="workloads/merlin-perfbench/merlin-perfbench.elf"
mkdir -p "$OUTDIR" "$SLOT"

echo "=== BUNDLE: kill + infrasetup (flash bitstream ONCE) cfg=$CFG ==="
firesim $CFGARG kill >/dev/null 2>&1 </dev/null
firesim $CFGARG infrasetup </dev/null
echo "=== infrasetup done; looping ELFs ==="

# Read the manifest on FD 3 — `firesim` reads stdin, which would otherwise consume the loop's input
# and stop after the first ELF.
n=0; ok=0
while IFS=$'\t' read -r -u 3 label elf; do
  [ -z "$label" ] && continue
  n=$((n+1))
  if [ ! -f "$elf" ]; then echo "  SKIP $label (no elf: $elf)"; continue; fi
  cp "$elf" "$BOOT" 2>/dev/null; cp "$elf" "$WLBIN" 2>/dev/null; chmod 755 "$BOOT" "$WLBIN" 2>/dev/null
  : > "$SLOT/uartlog"
  t0=$SECONDS
  timeout "$RUNTIMEOUT" firesim $CFGARG runworkload >/dev/null 2>&1 </dev/null
  rc=$?
  cp "$SLOT/uartlog" "$OUTDIR/$label.uartlog" 2>/dev/null
  if grep -q "METRIC cycles" "$OUTDIR/$label.uartlog" 2>/dev/null; then
    ok=$((ok+1))
    echo "  OK   $label  $(grep -oE 'METRIC cycles [0-9]+' "$OUTDIR/$label.uartlog" | head -1)  ($((SECONDS-t0))s rc=$rc)"
  else
    echo "  FAIL $label  rc=$rc no-METRIC ($((SECONDS-t0))s) — self-heal kill+infrasetup"
    firesim $CFGARG kill >/dev/null 2>&1 </dev/null
    firesim $CFGARG infrasetup >/dev/null 2>&1 </dev/null
  fi
done 3< "$MANIFEST"

firesim $CFGARG kill >/dev/null 2>&1 </dev/null
echo "=== BUNDLE DONE: $ok/$n produced cycles ==="

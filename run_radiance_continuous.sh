#!/bin/bash
# Radiance, CONTINUOUS: produce a working compiler out of the experiment itself.
#
# Differences from run_ab_v11 / run_baseline_v12, and why each one is here:
#
#   * --schedule continuous. The round COUNT stops being a terminator. v12 arm-4 reached its ceiling in
#     round 0 and then spent two more rounds and 37.9M tokens going nowhere, while other runs were cut at
#     the cap mid-fix. Neither stop said anything about the submission. The run now ends on convergence,
#     on a plateau, or on the declared wall budget.
#
#   * ONE arm (merlin_rtlchecks). This is not an A/B — the goal is a working compiler, not a tooling
#     delta. arm-4 is the strongest configuration measured (27 earned / 33 headline on v11 and v12).
#
#   * 38 op capsules, up from 36: RX0/RX1 are int8 movement, which radiance declares no datapath for
#     (movement is fp32/fp16/bf16 only) while int8 stays in its declared dtypes — so they are GRADED and
#     unservable by the SIMT cluster, which is what forces the compiler off that path.
#
#   * L3 promotion is per-capsule and fires on EVERY verdict (tier_promote, both brokers + the round
#     grade), so a capsule's cert tier is enqueued the moment its L2 passes. Nothing waits for a round
#     boundary. GSIM is the L3 engine (~58x verilator), which is what makes that affordable.
#
#   * M0_small_llama carries gate.after_op_pass_fraction 0.8 — the whole-model capstone is scheduled only
#     once 80% of the op capsules pass, so it costs nothing until the compiler has earned it.
set -uo pipefail
cd /scratch/agustin/projects/oscar-merlin/.claude/worktrees/codex-radiance-smoke
set -a; . ./.env; set +a
export PYTHONPATH=$PWD/merlin/python:$PWD/merlin/experiments/capsule_bench/harness
export MERLIN_TARGET_EXPERIMENT=$PWD/merlin/experiments/capsule_bench/targets/radiance/target_experiment.yaml
# TMPDIR on /scratch: / is full, and an empty TMPDIR silently breaks whole-model builds.
export TMPDIR=/scratch/agustin/tmp/codex_runs
mkdir -p "$TMPDIR"
export TMP=$TMPDIR TEMP=$TMPDIR
# GSIM as the L3 cert engine — without this, default_adapters falls back to Verilator and the promoted
# cert jobs cost ~45 min/capsule instead of seconds. Measured: a bare re-grade took 60+ min on Verilator
# versus 1.5 min with the RTL cert dropped.
export MERLIN_MUON_GSIM_EMU=/scratch/agustin/projects/gsim/build/radiance_gsim/emu_radiance_gsimconfig
# 8M, up from 2M. MEASURED: RX1_movement_i8_vector reaches L3 pass at 8M having timed out at 2M, so
# several capsules reported L3 "unavailable" were budget-starved, not defective -- and the failure message
# blamed the cycle cap, which reads as a hang. Raising this was UNSAFE until the console stopped being
# buffered in RAM (the parent reached 72.67 GB at a 12M cap and the host's OOM killer took out a live
# 10-hour run); the console now spools to disk, so the cap is a real knob. Not raised further: a
# 16x128x128 tile did not complete even at 12M, so non-termination there is not ruled out and a bigger
# budget would only spend longer failing.
export MERLIN_MUON_GSIM_MAXCYCLES=8000000
# MERLIN_MLC_DIR is NOT in .env, and without it isa_encoding_for() returns None, _model_for() raises,
# and EVERY fork-free compile fails closed — the run then grades only the capsules that need no oracle.
# Measured: a first attempt sat flat at 6/39 (the six MX fixtures) for 101 minutes. codegen_smoke now
# refuses to launch in that state, but setting it here is the actual fix.
export MERLIN_MLC_DIR=${MERLIN_MLC_DIR:-/scratch2/agustin/mvp-lhwir/modeling}

echo "head:    $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"
echo "corpus:  $(ls merlin/contract/capsules/radiance/model_slices | wc -l) model_slices, \
$(ls merlin/contract/capsules/radiance/isa | wc -l) isa"
echo "gsim:    $(test -x "$MERLIN_MUON_GSIM_EMU" && echo present || echo MISSING)"
echo "holdout: $(grep -c 'cat: hidden' merlin/contract/capsules/profiles/radiance.yaml) hidden specs in the tracked profile (want 0)"
echo "start:   $(date -u +%Y%m%dT%H%M%SZ)"

CHIA_PY=/scratch/agustin/projects/oscar-merlin/build/chia-venv/bin/python
$CHIA_PY merlin/experiments/capsule_bench/harness/chia_ab_batch.py \
    --tag radiance_continuous_v15 --arms merlin_rtlchecks --repeats 1 \
    --driver codex --model gpt-5.6-sol --effort high \
    --codex-slots 1 \
    --schedule continuous --max-wall-s 36000 \
    --max-rounds 40 --round-timeout 14400 \
    --skip-hidden --sandbox bwrap "$@"
echo "exit:  $?"
echo "end:   $(date -u +%Y%m%dT%H%M%SZ)"

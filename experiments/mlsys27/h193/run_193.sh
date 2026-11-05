#!/bin/bash
# run_193.sh SUBMISSION_DIR [LABEL] [P]   : the authored backend (a frozen Phase 1/2 submission dir with manifest.yaml) on ALL 193 core-ATen overloads, canonical case each.
#   e.g.  ./run_193.sh /scratch2/loren/f2-cycles-backend/snap/frozen-202347Z/joint_frozen_submission frozen-202347Z 12
#   DEVICE57 (exercised.py ALL57) -> int8/integer op-capsule form, must retire RoCC instructions with exact output; HOST136 -> corpus_193 canonical f32 capture, must run with ZERO accelerator opcodes.
#   Writes RESULT-193-LABEL.md and result-193-LABEL.csv (HEADLINE=1 also RESULT-193.md/.csv).  Runs on merlin-compute-1 cores 0-23, nice 10.  ELFs are kept on compute-1 (path in CSV column elf) for later F2 queueing.
#   FULL=1 ./run_193.sh ... : additionally apply the paper's full 'supported' definition to every DEVICE57 case (grader host_compute veto + trace_check, in the FIN6yn neutral-release namespace, compute-1 cores 0-23 nice 10) -> extra CSV columns + 'FULL definition' tables in the RESULT md. HOST136 keep zero-opcodes+correct output (checks n/a).
#   CPUS=<taskset list> (default 0-23) selects the compute-1 cores for both stages.
#   VERIFICATION INSTRUMENT: do not copy any of this directory into an author-visible input.
set -uo pipefail
export SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-/home/loren/.ssh/agent/sock}
PKG=$(readlink -f "$1"); LABEL=${2:-$(basename $(dirname $PKG))}; P=${3:-12}
export H=${H193_H:-${H:-/scratch2/loren/backend-verify-author}}
export H193_H=$H R=${R:-/scratch/loren/backend-verify-author} H193_TAG=${H193_TAG:-m27}
SSH=${H193_SSH:-ssh -F /scratch2/loren/aws/ssh/config -o BatchMode=yes merlin-compute-1}
[ -f $PKG/manifest.yaml ] || { echo "no manifest.yaml in $PKG"; exit 2; }
PID=$(cd $PKG && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | sort | xargs sha256sum | sha256sum | cut -c1-16)
echo "label=$LABEL package-id=$PID compiler.py sha256=$(sha256sum $PKG/compiler.py | cut -c1-12)"
[ -f $H/canon193.json ] || python3 $H/tools/select_193.py
cd $H && tar cf - tools canon193 canon193.json forms93 controls | $SSH "mkdir -p $R && cd $R && tar xf -"
# content-addressed package dir: written once, never modified (PYTHONDONTWRITEBYTECODE is set when it runs)
PK=$R/pkgs/g-$LABEL-$PID
$SSH "test -f $PK/.complete" || { tar cf - --exclude=__pycache__ -C $(dirname $PKG) $(basename $PKG) | $SSH "mkdir -p $PK && cd $PK && tar xf - && touch .complete"; }
$SSH "CPUS=${CPUS:-0-23} bash $R/tools/drive193.sh $R $LABEL $PK/$(basename $PKG) $P"
if [ -n "${FULL:-}" ]; then
  ls $H/forms93/*/capsule_meta.json >/dev/null 2>&1 || { echo "forms93 capsule_meta.json missing: run tools/gen_expected.py on compute-1 with the opcaps env first"; exit 3; }
  $SSH "R=$R H193_TAG=$H193_TAG CPUS=${CPUS:-0-23} bash $R/tools/fullcheck_drive.sh $LABEL ${FULLW:-8}"
fi
rm -rf $H/runs193/$LABEL; mkdir -p $H/runs193/$LABEL; $SSH "cd $R/runs193/$LABEL && tar cf - --exclude=prog.elf --exclude=kernel.o --exclude=kernel.ll ." | tar xf - -C $H/runs193/$LABEL
echo "$PID" > $H/runs193/$LABEL/PACKAGE_ID
python3 $H/tools/aggregate193.py $LABEL $PID

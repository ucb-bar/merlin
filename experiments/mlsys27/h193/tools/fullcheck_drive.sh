#!/bin/bash
# fullcheck_drive.sh LABEL [W]  (compute-1): FULL=1 stage. Runs fullcheck.py over runs193/LABEL inside the fin6yn neutral-release namespace (copy of p12-unsealed ns.sh + a bind of this dir). cores 0-23, nice 10.
L=$1; W=${2:-8}; B=${R:-/scratch/loren/backend-verify-author}
export R=$B
CPUS=${CPUS:-0-23}; NC=$(python3 -c "import sys;n=0
for p in sys.argv[1].split(','):
 a=p.split('-'); n+=int(a[-1])-int(a[0])+1
print(n)" "$CPUS")
export P12_CODEX_CPUS=${P12_CODEX_CPUS:-0} P12_TAG=${H193_TAG:-m27} P12_CPUS=$CPUS P12_NICE=10 P12_POOL=0-$((NC-1)) P12_RUNS=$B/fullenv-$L P12_GRADE_WORKERS=1
bash $B/tools/ns_full.sh bash -c "source ${P12:-/scratch/loren/p12-unsealed}/env.sh && export MERLIN_COMPUTE_CPUS=$CPUS && cd $B && \$PY -B tools/fullcheck.py runs193/$L forms93 $W" > $B/runs193/$L/fullcheck.log 2>&1
tail -3 $B/runs193/$L/fullcheck.log

# usage: launch-tuning.sh kernel|resnet JOBS DEADLINE_HOURS
. /work/exp/tvmfs/env.sh
K=$1; J=$2; H=$3; cd $EXP
S=$WORK/$K-study; [ $K = kernel ] && S=$WORK/kernel-study2
nohup $PY $COMPARISON/run_firesim_study.py --study $S/study.json --toolchain f2/toolchain.json --pool f2/pool.json \
  --output-dir work/tune-$K --campaign-kind $K --target-clock-hz 60000000 --candidate-bootstrap-seconds 120 \
  --candidate-timeout-ceiling-seconds 600 --deadline-epoch $(python3 -c "import time;print(time.time()+$H*3600)") \
  --jobs $J --ledger f2/ledger-$K.json --execute > logs/tune-$K.log 2>&1 &
echo "launched $K pid $!"

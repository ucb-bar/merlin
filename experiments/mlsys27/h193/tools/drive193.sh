#!/bin/bash
# drive193.sh R LABEL PKG P (runs on compute-1): for the 193 overloads: canonical f32 capture (c_NNN; policy host_f32 for HOST136, exact for DEVICE57 where it is only a secondary column)
# and, for DEVICE57 with an integer form, the form (f_NNN, exact).  cores 0-23, nice 10.  CPUS env overrides the core list.
R=$1; L=$2; PKG=$3; P=$4; PY=${H193_PY:-/scratch/loren/core-aten-snap/venvs/merlin-wtC/bin/python3}
cd $R && rm -rf runs193/$L && mkdir -p runs193/$L
for c in ctl-movement ctl-movement-neg; do taskset -c ${CPUS:-0-23} nice -n 10 $PY -B tools/run_case.py --pkg $PKG --case controls/$c --out runs193/$L/ctl_$c --policy exact > runs193/$L/ctl_$c.log 2>&1; done
$PY -B -c '
import json
for s in json.load(open("canon193.json")):
    pol = "host_f32" if s["klass"]=="HOST136" else "exact"
    print("canon193/%s c_%s %s" % (s["n"], s["n"], pol))
    if s["form_testable"]: print("forms93/%s f_%s exact" % (s["form_slug"], s["n"]))' > runs193/$L/list.txt
xargs -P $P -L1 bash -c "taskset -c ${CPUS:-0-23} nice -n 10 $PY -B tools/run_case.py --pkg $PKG --case \$0 --out runs193/$L/\$1 --policy \$2 > runs193/$L/\$1.log 2>&1" < runs193/$L/list.txt
echo FINISHED > runs193/$L/DONE

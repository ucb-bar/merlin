"""Build original baseline and device-extension fixtures without changing inputs."""
import argparse
from operator_contracts import fixtures as baseline_fixtures
from verify_device_ops import fixtures as device_fixtures
from verify_tensorir import prepare

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',required=True,type=__import__('pathlib').Path)
parser.add_argument('--adapter-receipt',required=True,type=__import__('pathlib').Path)
parser.add_argument('--riscv-gcc',required=True,type=__import__('pathlib').Path)
args=parser.parse_args()
cases=list(baseline_fixtures())+device_fixtures()
assert len({case[0] for case in cases})==len(cases)
args.cases=[case[0] for case in cases]
args.timing_repeats,args.rtol,args.atol,args.libm=0,0.,0.,True
prepare(args,cases=cases,additional_bindings=[__file__])

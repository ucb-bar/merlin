export EXP=/work/exp/tvmfs
export WORK=$EXP/work
export MERLIN=$EXP/merlin
export TVM_SOURCE=$EXP/tvm
export TVM_BUILD=$EXP/build/tvm
export PY=/work/exp/tvm-el4-prelim/.venv/bin/python
export RISCV=/work/shared/chipyard/.conda-env/riscv-tools
export RISCV_GCC=$RISCV/bin/riscv64-unknown-elf-gcc
export ADAPTER_RECEIPT=$WORK/baselines/tvm-gemmini/adapter-gemmini-8c3f9923/receipt.json
export COMPARISON=$MERLIN/examples/gemmini/comparisons/tvm
export TVM_LIBRARY_PATH=$TVM_BUILD
export PYTHONPATH=$MERLIN/src:$TVM_SOURCE/python:$TVM_SOURCE/apps/gemmini:$COMPARISON
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TVM_NUM_THREADS=1
export PATH=$RISCV/bin:$PATH

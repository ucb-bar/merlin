# K1 static OpenMP runtime (`libomp.a`)

`libomp.a` here is the LLVM OpenMP runtime cross-built **static** for the SpacemiT K1 board
(riscv64-unknown-linux-gnu, `rv64gcv`, `lp64d`, glibc). It provides the `__kmpc_*` symbols the
**multicore lowering** emits (`merlin.llvmlower.pipeline._parallel_pipeline`:
`convert-linalg-to-parallel-loops` → `convert-scf-to-openmp` → `convert-openmp-to-llvm`).
Linking it lets the model's parallel loops fan across the board's 8 cores
(`OMP_NUM_THREADS=8`, set by `rvvgen.k1.run_on_k1`).

It is referenced **only** by the gated `parallel=True` build path in `rvvgen/k1.build_k1_binary`;
the default fp32/int8/RVV flows never link it.

## Symbols (proof the lowering needs exactly these)
`__kmpc_fork_call`, `__kmpc_global_thread_num`, `__kmpc_for_static_init_8u`,
`__kmpc_for_static_fini`, `__kmpc_barrier` (+ `omp_get_*`). Verify:
`llvm-nm libomp.a | grep kmpc_fork`.

## Rebuild recipe
Built from `third_party/llvm-project` (the `runtimes/` superproject — the legacy standalone
`openmp/runtime` configure was removed in this LLVM). Use the **SpacemiT clang-19** toolchain
(it has a complete builtin-header resource dir + glibc sysroot; the repo clang-23 install lacks
`lib/clang/23/include`). `libomp.a` is just relocatable riscv64 objects, so it links fine with
the clang-23 model object in the final merlin build.

```bash
SM=/path/to/merlin-iree/build_tools/riscv-tools-spacemit/spacemit-toolchain-linux-glibc-x86_64-v1.1.2
cmake -G Ninja -S third_party/llvm-project/runtimes -B /path/to/tmp/k1_libomp_build \
  -DLLVM_ENABLE_RUNTIMES=openmp \
  -DCMAKE_C_COMPILER=$SM/bin/clang -DCMAKE_CXX_COMPILER=$SM/bin/clang++ \
  -DCMAKE_C_COMPILER_TARGET=riscv64-unknown-linux-gnu \
  -DCMAKE_CXX_COMPILER_TARGET=riscv64-unknown-linux-gnu \
  -DCMAKE_SYSROOT=$SM/sysroot \
  -DCMAKE_C_FLAGS="-march=rv64gcv -mabi=lp64d" -DCMAKE_CXX_FLAGS="-march=rv64gcv -mabi=lp64d" \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
  -DCMAKE_C_COMPILER_WORKS=1 -DCMAKE_CXX_COMPILER_WORKS=1 \
  -DLIBOMP_ENABLE_SHARED=OFF -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
  -DLIBOMP_OMPT_SUPPORT=OFF -DLIBOMP_USE_HWLOC=OFF
ninja -C /path/to/tmp/k1_libomp_build omp
cp /path/to/tmp/k1_libomp_build/openmp/runtime/src/libomp.a .
```

Final link (handled by `k1.build_k1_binary`): `... libomp.a -lstdc++ -ldl -lm -lpthread`.

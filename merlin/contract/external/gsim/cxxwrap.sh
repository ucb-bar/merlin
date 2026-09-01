#!/bin/bash
# clang-23 wrapper for building GSIM: silence the clang-23-only gcc-install-dir warning (harmless;
# GSIM builds with -Werror) and pin the gcc toolchain so libstdc++ headers resolve deterministically.
exec /scratch/agustin/projects/oscar-merlin/.claude/worktrees/gemmini-eviction/third_party/llvm-install/bin/clang++ \
  --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 \
  -Wno-gcc-install-dir-libstdcxx -Wno-error=gcc-install-dir-libstdcxx "$@"

#!/bin/bash
# clang-23 wrapper for building GSIM: silence the clang-23-only gcc-install-dir warning (harmless;
# GSIM builds with -Werror) and pin the gcc toolchain so libstdc++ headers resolve deterministically.
#
# The compiler is resolved from THIS SCRIPT's location, not from an absolute path: the wrapper used
# to name one developer's checkout (and, worse, one worktree inside it), so every other clone -- and
# this clone once that worktree is removed -- invoked a compiler that was not there. Override with
# MERLIN_GSIM_CXX when building against a toolchain outside the tree.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
cxx="${MERLIN_GSIM_CXX:-$root/third_party/llvm-install/bin/clang++}"
if [ ! -x "$cxx" ]; then
  echo "cxxwrap: no clang++ at $cxx (build third_party/llvm-install, or set MERLIN_GSIM_CXX)" >&2
  exit 127
fi
exec "$cxx" \
  --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 \
  -Wno-gcc-install-dir-libstdcxx -Wno-error=gcc-install-dir-libstdcxx "$@"

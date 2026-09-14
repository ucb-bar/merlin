#!/usr/bin/env bash
# Plane B (Voyager's own hardware): run the accelerator release's own flow -- codegen, SystemC
# fast-sim, Catapult HLS -> Verilog, VCS RTL simulation -- with the environment and the disclosed
# deviations recorded in merlin/experiments/voyager_h2h/PLANE_B.md.
#
#   plane_b_regression.sh env                      print the environment it runs under
#   plane_b_regression.sh fast-sim <network>       Voyager's run_regression.py --sims fast-systemc
#   plane_b_regression.sh rtl                      make rtl (Catapult HLS, all blocks)
#   plane_b_regression.sh rtl-sim <network>        Voyager's run_regression.py --sims rtl
#
# Configuration comes from the environment with the paper's 256-MAC point as default:
# DATATYPE=INT8 IC_DIMENSION=16 OC_DIMENSION=16, default 1024-element buffers, TECHNOLOGY=generic
# (Catapult's shipped nangate-45nm_beh + ccs_sample_mem), CLOCK_PERIOD=1 (ns; 1 GHz as in the paper).
# MAX_TILES must cover every L2 tile of every layer, or the release's regression extrapolates.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
ext="$repo/out/build/external"
work="$ext/voyager-accelerator-work"            # git worktree of voyager_accelerator @ e3a725db
env_prefix="$repo/out/build/voyager-accel-env"  # mamba env: py3.10, torch 2.6, protoc 29.3
catapult="${CATAPULT_ROOT:-/ecad/tools/mentor/catapult/2023.1_1/Mgc_home}"
vcs_home="${VCS_HOME:-/ecad/tools/synopsys/vcs/current}"
shim_dir="${VOYAGER_HEADER_SHIM:-$ext/voyager-acmath-shim}"   # holds ac_math/ac_gelu_pwl.h only

setup_env() {
  export PATH="$env_prefix/bin:$catapult/bin:$vcs_home/bin:$PATH"
  export CONDA_PREFIX="$env_prefix" CATAPULT_ROOT="$catapult" VCS_HOME="$vcs_home"
  export VG_GNU_PACKAGE="$vcs_home/gnu/linux64"
  unset VCS_ARCH_OVERRIDE   # the site default forces the 32-bit VCS, which cannot load libelf
  export PYTHONPATH="$ext/voyager-interstellar-shim"   # public interstellar, nothing else
  export TMPDIR="${TMPDIR:-/scratch/agustin/tmp}" HF_HOME="${HF_HOME:-/scratch/agustin/tmp/hf-home}"
  export DATATYPE="${DATATYPE:-INT8}" IC_DIMENSION="${IC_DIMENSION:-16}"
  export OC_DIMENSION="${OC_DIMENSION:-16}" TECHNOLOGY="${TECHNOLOGY:-generic}"
  export CLOCK_PERIOD="${CLOCK_PERIOD:-1}"
  # The release disagrees with itself about an unset MAX_TILES: the C++ harness then simulates EVERY
  # L2 tile (test/common/Utils.h get_tile_count), while run_regression.py assumes 1 simulated tile and
  # scales runtime by full_tiles/1 -- over-counting any multi-tile layer. Setting it above every
  # layer's tile count makes both sides agree on "all tiles, no extrapolation".
  export MAX_TILES="${MAX_TILES:-1000000}"
  # Deviation 1+2: host multiarch headers; the one ac_math header Catapult 2023.1 lacks, searched last.
  export BASE_FLAGS="-idirafter $shim_dir -isystem /usr/include/x86_64-linux-gnu"
  # Deviation 3: host multiarch C runtime start files for Catapult's bundled gcc.
  export LDFLAGS="-B/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu"
  # As upstream env.sh: the env's newer libstdc++ first, Catapult's SystemC after it.
  export LD_LIBRARY_PATH="$env_prefix/lib:$catapult/shared/lib/Linux/gcc-10.3.0-64:${LD_LIBRARY_PATH:-}"
  export PROJECT_ROOT="$work" CODEGEN_DIR=test/compiler
  if [ -z "${HF_TOKEN:-}" ] && [ -f "$repo/.env" ]; then
    # Gated ImageNet calibration (user-approved); read the value without echoing it.
    HF_TOKEN="$(command grep -m1 -e '^HF_TOKEN=' -e '^HUGGING_FACE_HUB_TOKEN=' "$repo/.env" \
                | cut -d= -f2- || true)"
    [ -n "$HF_TOKEN" ] && export HF_TOKEN
  fi
  # Deviation 4: Catapult's C++ front end only searches the design's lib/ plus its own defaults, not
  # the host's multiarch include dir, and <linux/errno.h> includes <asm/errno.h>. A symlink in the
  # work tree's lib/ (never the pinned checkout) supplies exactly that directory.
  if [ ! -e "$work/lib/asm" ]; then
    ln -s /usr/include/x86_64-linux-gnu/asm "$work/lib/asm"
  fi
  mkdir -p "$shim_dir/ac_math" "$TMPDIR" "$HF_HOME"
  if [ ! -f "$shim_dir/ac_math/ac_gelu_pwl.h" ]; then
    cp "$ext/hlslibs-ac_math/include/ac_math/ac_gelu_pwl.h" "$shim_dir/ac_math/"
  fi
  # Deviation 2, for Catapult's front end too: it does not see BASE_FLAGS, only the design's lib/.
  # lib/ac_math holds that ONE header, so every other ac_math include still resolves to Catapult's.
  mkdir -p "$work/lib/ac_math"
  if [ ! -f "$work/lib/ac_math/ac_gelu_pwl.h" ]; then
    cp "$shim_dir/ac_math/ac_gelu_pwl.h" "$work/lib/ac_math/"
  fi
}

cmd="${1:-env}"; shift || true
setup_env
cd "$work"
case "$cmd" in
  env)
    for v in DATATYPE IC_DIMENSION OC_DIMENSION TECHNOLOGY CLOCK_PERIOD CATAPULT_ROOT VCS_HOME \
             CONDA_PREFIX PYTHONPATH BASE_FLAGS LDFLAGS LD_LIBRARY_PATH PROJECT_ROOT MAX_TILES; do
      printf '%s=%s\n' "$v" "${!v:-}"
    done
    printf 'HF_TOKEN=%s\n' "$([ -n "${HF_TOKEN:-}" ] && echo set || echo unset)" ;;
  codegen)
    net="${1:?network}"
    NETWORK="$net" make network-proto ;;
  fast-sim)
    net="${1:?network}"
    python run_regression.py --models "$net" --sims fast-systemc --num_processes "${NPROC:-8}" \
      --uniquify_layers --skip_layers ;;
  rtl)
    make rtl ;;
  block)
    make "${1:?block name, e.g. ProcessingElement}" ;;
  rtl-sim)
    net="${1:?network}"
    python run_regression.py --models "$net" --sims rtl --num_processes "${NPROC:-8}" \
      --uniquify_layers --skip_layers --keep_build ;;
  *)
    echo "unknown command $cmd" >&2; exit 2 ;;
esac

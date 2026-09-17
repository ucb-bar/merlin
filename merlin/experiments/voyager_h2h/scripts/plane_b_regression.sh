#!/usr/bin/env bash
# Plane B (Voyager's own hardware): run the accelerator release's own flow -- codegen, SystemC
# fast-sim, Catapult HLS -> Verilog, VCS RTL simulation -- with the environment and the disclosed
# deviations recorded in merlin/experiments/voyager_h2h/PLANE_B.md.
#
#   plane_b_regression.sh env                      print the environment it runs under
#   plane_b_regression.sh fast-sim <network>       Voyager's run_regression.py --sims fast-systemc
#   plane_b_regression.sh rtl                      make rtl (Catapult HLS, all blocks)
#   plane_b_regression.sh rtl-sim <network>        Voyager's run_regression.py --sims rtl
#   plane_b_regression.sh rtl-layer <net> <layer> <out> [ucli.tcl]
#                                                  one layer of the existing SCVerify build
#   plane_b_regression.sh rtl-trace <net> <layer> <out> <nets-file>
#                                                  rtl-layer + an event log of every value change
#                                                  of the listed vld/rdy nets, reduced to
#                                                  per-channel handshake counts
#   plane_b_regression.sh systemc-layer <net> <layer> <out>
#                                                  Voyager's pre-HLS SystemC model (Makefile `sim`,
#                                                  CONNECTIONS_ACCURATE_SIM) on one layer
#
# VOYAGER_WORK selects another tree than the flow's work tree (e.g. a patched A/B copy).
# Configuration comes from the environment with the paper's 256-MAC point as default:
# DATATYPE=INT8 IC_DIMENSION=16 OC_DIMENSION=16, default 1024-element buffers, TECHNOLOGY=generic
# (Catapult's shipped nangate-45nm_beh + ccs_sample_mem), CLOCK_PERIOD=5 (ns; see setup_env).
# MAX_TILES must cover every L2 tile of every layer, or the release's regression extrapolates.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
ext="$repo/out/build/external"
work="${VOYAGER_WORK:-$ext/voyager-accelerator-work}"  # git worktree of voyager_accelerator @ e3a725db
env_prefix="$repo/out/build/voyager-accel-env"  # mamba env: py3.10, torch 2.6, protoc 29.3
catapult="${CATAPULT_ROOT:-/ecad/tools/mentor/catapult/2023.1_1/Mgc_home}"
vcs_home="${VCS_HOME:-/ecad/tools/synopsys/vcs/current}"
shim_dir="${VOYAGER_HEADER_SHIM:-$ext/voyager-acmath-shim}"   # holds ac_math/ac_gelu_pwl.h only

setup_env() {
  export PATH="$env_prefix/bin:$catapult/bin:$vcs_home/bin:$PATH"
  export CONDA_PREFIX="$env_prefix" CATAPULT_ROOT="$catapult" VCS_HOME="$vcs_home"
  # SCVerify's generated wave script sources $MGC_HOME/pkgs/sif/userware/En_na/flows/vcs_funcs.tcl.
  # The site default points MGC_HOME at Calibre, and ucli then aborts the run at that source line
  # (CLE-10) before the layer starts, so name Catapult's own Mgc_home.
  export MGC_HOME="$catapult"
  # VCS's SystemC 2.3.3 front end accepts only g++ 7.3 / 9.2 / 9.5; the gcc VCS V-2023.12 bundles is
  # 13.2. Upstream documents the VCS GNU package S-2021.09 (gcc 9), which this host has inside its
  # S-2021.09 VCS install: gcc 9.2.0 + binutils 2.33.1 via its own source_me script.
  # Exported so VCS does not substitute its bundled gcc 13. The package's source_me script is not
  # sourced: the wrapper below names the gcc 9.2 driver directly, and that script would only put the
  # package's binutils 2.33.1 first on PATH.
  export VG_GNU_PACKAGE="${VG_GNU_PACKAGE_OVERRIDE:-/ecad/tools/synopsys/vcs/S-2021.09-SP1-1/gnu/linux}"
  # Catapult's make and HLS still use Catapult's own g++ (the Makefile names it explicitly).
  unset VCS_ARCH_OVERRIDE   # the site default forces the 32-bit VCS, which cannot load libelf
  export PYTHONPATH="$ext/voyager-interstellar-shim"   # public interstellar, nothing else
  # Whole-model capture needs a LARGE scratch filesystem; set TMPDIR to one before running.
  export TMPDIR="${TMPDIR:-/tmp}" HF_HOME="${HF_HOME:-$TMPDIR/hf-home}"
  export DATATYPE="${DATATYPE:-INT8}" IC_DIMENSION="${IC_DIMENSION:-16}"
  export OC_DIMENSION="${OC_DIMENSION:-16}" TECHNOLOGY="${TECHNOLOGY:-generic}"
  # 5 ns, not the paper's 1 ns: on the generic nangate-45nm library VectorAccumulator's bf16 add
  # recurrence does not schedule at 1 ns (SCHD-30), and the release's own README example uses a 40 nm
  # library at CLOCK_PERIOD=5.0. RTL "Total Runtime" is in ns, so cycles = runtime / CLOCK_PERIOD.
  export CLOCK_PERIOD="${CLOCK_PERIOD:-5}"
  # The release disagrees with itself about an unset MAX_TILES: the C++ harness then simulates EVERY
  # L2 tile (test/common/Utils.h get_tile_count), while run_regression.py assumes 1 simulated tile and
  # scales runtime by full_tiles/1 -- over-counting any multi-tile layer. Setting it above every
  # layer's tile count makes both sides agree on "all tiles, no extrapolation".
  export MAX_TILES="${MAX_TILES:-1000000}"
  # Deviation 1+2: host multiarch headers; the one ac_math header Catapult 2023.1 lacks, searched last.
  export BASE_FLAGS="-idirafter $shim_dir -isystem /usr/include/x86_64-linux-gnu"
  # Deviation 3: host multiarch C runtime start files for Catapult's bundled gcc. The env's lib dir
  # goes ahead of the multiarch dir: VCS's link takes LDFLAGS before its own -L, and the host's
  # protobuf 3.21 (libprotobuf.so.32) would otherwise shadow the env's 29.3 the harness is built with.
  export LDFLAGS="-B/usr/lib/x86_64-linux-gnu -L$env_prefix/lib -L/usr/lib/x86_64-linux-gnu"
  # Pin VCS's SystemC compiler to the gcc 9.2 package through Catapult's own ccs_vcs.mk hooks. The
  # package's own xbin/ wrapper cannot be used as is: it forces the package's binutils 2.33.1 ahead
  # of any user flag, and that ld rejects this host's glibc 2.39 shared objects (their .relr.dyn
  # section postdates it), so simv does not link. VCS also puts its own bundled binutils 2.33.1
  # first on PATH for the link. This wrapper runs the same gcc 9.2 driver the package's does, with
  # the package's Ubuntu -B for the multiarch crt files, the host's as/ld (binutils 2.42) forced
  # first on PATH (gcc finds its own cc1plus/collect2 through its install prefix, as/ld through
  # PATH), and deviation 1's multiarch include dir searched after every system directory. vlogan
  # spawns its own syscan with an empty -cflags, so the compiler is the only place that include
  # reaches every compile.
  g9="$VG_GNU_PACKAGE/gcc-9.2.0_64-shared/bin"
  g9wrap="$ext/voyager-gcc9-multiarch"
  mkdir -p "$g9wrap"
  for tool in gcc g++; do
    printf '#!/bin/sh\nunset GCC_EXEC_PREFIX CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH\nPATH=/usr/bin:$PATH; export PATH\nexec %s -B/usr/lib/x86_64-linux-gnu "$@" -idirafter /usr/include/x86_64-linux-gnu\n' \
      "$g9/$tool" > "$g9wrap/$tool"
    chmod +x "$g9wrap/$tool"
  done
  export VCS_EXEC_VLOGAN="vlogan -cpp $g9wrap/g++ -cc $g9wrap/gcc"
  # Deviation 9: the RTL testbench compiles Voyager's SystemC design too (Catapult's sysc_sim.h
  # includes src/Accelerator.h), and that source needs the Connections generation it was written
  # against: src/Tieoff.h assigns a T to Out<T>::dat (DIRECT_PORT, the pre-HLS default since 2.2.0)
  # and Catapult 2023.1's own Connections 1.3.0/1.4 Fifo writes a literal 0 into a struct message
  # (fixed in 1.5.0). The testbench's syscan compiles therefore see public Connections 2.2.0 first;
  # HLS keeps Catapult's own copy. Pinned by commit, fail closed otherwise.
  conn_dir="$ext/hlslibs-matchlib_connections"
  conn_pin=6a3003b85c251c88dd2f02881bb98ce71f9aa42b   # hlslibs/matchlib_connections tag 2.2.0
  if [ "$(git -C "$conn_dir" rev-parse HEAD 2>/dev/null)" != "$conn_pin" ]; then
    echo "need $conn_dir at $conn_pin (git clone https://github.com/hlslibs/matchlib_connections;" \
         "git checkout 2.2.0)" >&2
    exit 2
  fi
  export VCS_EXEC_SYSCAN="syscan -cpp $g9wrap/g++ -cc $g9wrap/gcc -cflags -I$conn_dir/include"
  export VCS_EXEC_VCS="vcs -cpp $g9wrap/g++ -cc $g9wrap/gcc"
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
  env-sh)
    # Quoted exports for `source <(plane_b_regression.sh env-sh)`; never includes HF_TOKEN.
    for v in PATH CONDA_PREFIX CATAPULT_ROOT VCS_HOME VG_GNU_PACKAGE PYTHONPATH TMPDIR HF_HOME \
             DATATYPE IC_DIMENSION OC_DIMENSION TECHNOLOGY CLOCK_PERIOD MAX_TILES BASE_FLAGS \
             LDFLAGS VCS_EXEC_VLOGAN VCS_EXEC_SYSCAN VCS_EXEC_VCS LD_LIBRARY_PATH PROJECT_ROOT \
             CODEGEN_DIR; do
      printf 'export %s=%q\n' "$v" "${!v:-}"
    done
    echo "unset VCS_ARCH_OVERRIDE" ;;
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
  rtl-layer|rtl-trace|systemc-layer)
    # Single-layer runs, used to trace where the per-layer 2x comes from (PLANE_B.md, "Where the 2x
    # comes from"). Paths must be absolute: this script has already changed into the work tree.
    net="${1:?network}"; layer="${2:?layer}"; out="${3:?absolute output dir}"
    case "$out" in /*) ;; *) echo "output dir must be absolute" >&2; exit 2 ;; esac
    # Same defaults as run_regression.py set_default_env_vars / get_build_folder.
    export INPUT_BUFFER_SIZE="${INPUT_BUFFER_SIZE:-1024}" WEIGHT_BUFFER_SIZE="${WEIGHT_BUFFER_SIZE:-1024}"
    export ACCUM_BUFFER_SIZE="${ACCUM_BUFFER_SIZE:-1024}"
    export DOUBLE_BUFFERED_ACCUM_BUFFER="${DOUBLE_BUFFERED_ACCUM_BUFFER:-false}"
    export SUPPORT_MVM="${SUPPORT_MVM:-false}" SUPPORT_SPMM="${SUPPORT_SPMM:-false}"
    bdir="build/${DATATYPE}_${IC_DIMENSION}x${OC_DIMENSION}_${INPUT_BUFFER_SIZE}x${WEIGHT_BUFFER_SIZE}x${ACCUM_BUFFER_SIZE}_${DOUBLE_BUFFERED_ACCUM_BUFFER}_${SUPPORT_MVM}_${SUPPORT_SPMM}"
    export NETWORK="$net" TESTS="$layer" SIMS="gold,accelerator"
    mkdir -p "$out"
    if [ "$cmd" = systemc-layer ]; then
      # The Makefile's `sim` TestRunner: pre-HLS SystemC, CONNECTIONS_ACCURATE_SIM (DIRECT_PORT).
      # Deviation 9 applies here as well: against Catapult 2023.1's Connections, src/Tieoff.h does
      # not compile (MARSHALL_PORT), so the pinned public 2.2.0 goes first on the include path.
      export BASE_FLAGS="-I$conn_dir/include $BASE_FLAGS"
      make -j"${NPROC:-4}" "$bdir/cc/TestRunner" > "$out/build.log" 2>&1
      "./$bdir/cc/TestRunner" > "$out/run.log" 2>&1 || true
    else
      sol="$bdir/Catapult/$TECHNOLOGY/clock_$CLOCK_PERIOD"
      tcl="${4:-}"
      if [ "$cmd" = rtl-trace ]; then
        nets="${4:?file of RTL nets (vld/rdy), one hierarchical name per line}"
        tcl="$out/change_log.tcl"
        # ucli: log every value change of each net with the simulator's own time (event-driven, so
        # no sampling phase is assumed). ucli here rejects `dump -type VCD`, hence a text log.
        cat > "$tcl" <<'EOF'
global env
set ::f [open $env(VG_TRACE) w]
set fh [open $env(VG_SIGS) r]
foreach l [split [read $fh] "\n"] {
  set l [string trim $l]
  if {$l ne ""} {
    if {[catch {stop -change $l -continue -command "puts \$::f \"\[senv time\] $l \[get $l\]\""} e]} {
      puts "CHG-SKIP $l: $e"
    }
  }
}
close $fh
run
close $::f
quit
EOF
        export VG_SIGS="$nets" VG_TRACE="$out/changes.txt"
      fi
      [ -n "$tcl" ] || tcl="./Accelerator/Accelerator.v1/scverify/concat_sim_rtl_v_vcs/scverify_vcs_wave.tcl"
      ( cd "$sol" && LD_PRELOAD="$env_prefix/lib/libstdc++.so.6" \
          SYNOPSYS_SIM_SETUP=./Accelerator/Accelerator.v1/scverify/concat_sim_rtl_v_vcs/synopsys_sim.setup \
          ./Accelerator/Accelerator.v1/scverify/concat_sim_rtl_v_vcs/sc_main -systemcrun +vcs+lic+wait \
          -verilogrun -cm assert -ucli -ucli2Proc -i "$tcl" -l "$out/vcs_sim.log" ) > "$out/run.log" 2>&1 || true
      if [ "$cmd" = rtl-trace ]; then
        # Per channel (<name>_vld with a matching <name>_rdy): transfers (vld && rdy at a rising edge),
        # cycles with vld high, cycles with rdy high, over the layer's Started..Finished window.
        "$env_prefix/bin/python" - "$out/changes.txt" "$out/run.log" "$CLOCK_PERIOD" > "$out/handshakes.txt" <<'EOF'
import bisect, collections, sys
changes, log, per = sys.argv[1], sys.argv[2], float(sys.argv[3])
t0 = t1 = None
for line in open(log):
    if "Accelerator Layer" in line and line.split()[1:2] == ["ns"]:
        if "Started" in line: t0 = float(line.split()[0])
        if "Finished" in line: t1 = float(line.split()[0])
if t0 is None or t1 is None:
    sys.exit("layer window not found in " + log)
ch = collections.defaultdict(list)
for line in open(changes):
    p = line.split()
    if len(p) >= 4 and p[1] == "ps":
        ch[p[2]].append((int(p[0]), p[3]))
def high(sig, t):
    ev = ch[sig]; i = bisect.bisect_left(ev, (t, "")) - 1
    return i >= 0 and ev[i][1].endswith("1")
edges = [round(k * per * 1000) for k in range(int(t0 // per) + 1, int(t1 // per) + 1)]
print(f"window {t0:.0f}..{t1:.0f} ns, {len(edges)} rising edges")
for v in sorted(n for n in ch if n.endswith("_vld") and n[:-3] + "rdy" in ch):
    r = v[:-3] + "rdy"
    x = sum(1 for e in edges if high(v, e) and high(r, e))
    vh = sum(1 for e in edges if high(v, e)); rh = sum(1 for e in edges if high(r, e))
    if vh:
        print(f"xfers={x:7d} vld_hi={vh:7d} rdy_hi={rh:7d}  {v[:-4]}")
EOF
        gzip -f "$out/changes.txt"
      fi
    fi
    command grep -E "ideal runtime|Total Runtime|Error count" "$out/run.log" || true ;;
  *)
    echo "unknown command $cmd" >&2; exit 2 ;;
esac

# Merlin Build Targets

## Core Commands

Bootstrap once:
```bash
conda env create -f env_linux.yml
conda activate merlin-dev
```

Sync environment in active conda env:
```bash
python tools/setup.py env
```

Optional toolchain setup:
```bash
python tools/setup.py toolchain
```

## `tools/build.py` Options

- `--target`: `host`, `spacemit`, `firesim`
- `--config`: `debug`, `release`, `asan`, `trace`, `perf`
- `--with-plugin`: enable Merlin compiler plugin (`iree_compiler_plugin.cmake`, `iree_runtime_plugin.cmake`)
- `--clean`: delete computed build dir before configure/build
- `--cmake-target <name>`: build a specific target (default is `install`)
- `--verbose`: pass verbose mode to CMake build

## Target Requirements

### `host`

- No cross-toolchain variable required.
- Typical command:
```bash
python tools/build.py --target host --config debug
```

### `spacemit`

- Requires a RISC-V toolchain root.
- Preferred: set `RISCV_TOOLCHAIN_ROOT`.
- Script fallback: `${REPO_ROOT}/riscv-tools-spacemit/spacemit-toolchain-linux-glibc-x86_64-v1.1.2` if present.
- Typical command:
```bash
python tools/build.py --target spacemit --config debug
```

### `firesim`

- Uses `scripts/riscv_firesim/riscv_firesim.toolchain.cmake`.
- Defaults `RISCV_TOOLCHAIN_ROOT` to `${REPO_ROOT}/riscv-tools-iree/toolchain/clang/linux/RISCV` when unset.
- Typical command:
```bash
python tools/build.py --target firesim --config debug
```

## Config Notes

- `debug`: assertions on, debug flags.
- `release`: `RelWithDebInfo`.
- `perf`: `Release` with reduced tracing/cpuinfo overhead.
- `asan`: debug + address sanitizer flags.
- `trace`: runtime/compiler tracing enabled.

## Plugin Builds

Enable plugin explicitly:
```bash
python tools/build.py --target host --config debug --with-plugin
```

## Output Paths

Build structure is:

`build/<variant>/<target>/<config>/iree-<target>-<variant>-<version>/`

- `<variant>`: `vanilla` or `merlin`
- install prefix: `<build_dir>/install`

## Troubleshooting

- `cmake: error while loading shared libraries ...`: fix host runtime/library environment first.
- `SpacemiT toolchain not found`: set `RISCV_TOOLCHAIN_ROOT` or run `python tools/setup.py toolchain`.
- Wrong env packages: rerun `python tools/setup.py env` inside `merlin-dev`.

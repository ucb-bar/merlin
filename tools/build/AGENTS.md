# `tools/build/` — agent guide

## Mental model

`tools/build/cli.py` is the registered shim for `./merlin build`. It owns
cmake configure + ninja-build orchestration across the merlin build
profiles (vanilla, full-plugin, spacemit, firesim, gemmini, zephyr,
qrb5165, …). Profiles compose with `--target` to produce
`build/<target>-<variant>-<config>/`.

For the module-by-module map, read `__init__.py`. The dict
`presets.PROFILE_PRESETS` is the public surface every preset key in there
is referenced by `./merlin build --profile <name>` somewhere in the wild.

## Pitfalls

- **Profile names are the public API.** Renaming a key in
  `presets.PROFILE_PRESETS` silently breaks every dev-blog command and
  any `./merlin build --profile <x>` someone has scripted. Add new
  presets, don't rename existing ones.
- **Build-dir naming has special cases.** `qnn-compiler` →
  `host-merlin-release-qrb`, `qrb5165 + plugin_runtime` →
  `<target>-runtime-<config>`, `zephyr-task` → `<target>-task-<config>`,
  else `<target>-<variant>-<config>`. Mirror these in `mcp/build.py`'s
  `_build_dir_for_profile` if you change them.
- **Default config is `release`.** Per AGENTS.md Golden Rule #2 — debug
  builds balloon to 150 GB+ and have hit disk-full failures historically.
  Don't silently switch.
- **`build_check_freshness` MCP depends on this package's source-root
  list.** If you add a new dependency directory (say `extensions/`), add
  it to `mcp/build.py:_SOURCE_ROOTS` too, or rebuilds will go stale
  undetected.

## Cross-references

- Consumes: `models/<target>.yaml`, `build_tools/<target>/` toolchain
  configs, `third_party/iree_bar/` source.
- Produces: build trees used by every other subcommand
  (`./merlin compile` reads `host-vanilla-release` / `host-merlin-release`
  for `iree-compile`; `./merlin spike` / `sim` need their corresponding
  trees).
- MCP wrapper: `tools/mcp_servers/build.py` exposes `build_list_profiles`,
  `build_status`, `build_check_freshness`, `build_profile`. The freshness
  check is the canonical pre-flight before any compile/run/verify
  invocation (Golden Rule #6).

## Update triggers

Re-read this file and update it in the same turn if you edit:

- `tools/build/cli.py` (new flag / changed dispatch) — refresh the
  module map; touch `tools/mcp_servers/build.py` if the CLI surface changed.
- `tools/build/presets.py:PROFILE_PRESETS` (new / renamed preset) —
  update the Pitfalls section and check `tools/mcp_servers/build.py:_build_dir_for_profile`.
- `tools/build/cmake.py` / `packaging.py` / `radiance.py` — usually
  internal; AGENTS.md update only if cross-refs change.

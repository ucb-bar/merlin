# build_tools/

Build-side support: cross toolchains, packaging, hardware recipes, and patch
management. Most users never touch these directly — `./merlin build`,
`./merlin chipyard`, and `./merlin patches` invoke them as needed.

- `SpacemiT/`, `firesim/` — RISC-V cross-toolchain bootstrap scripts
  (`setup_toolchain.sh`) and toolchain cmake files. Run before
  `./merlin build --profile spacemit` / `--profile firesim` on a fresh
  machine.
- `riscv-tools-spacemit/`, `riscv-tools-iree/` — toolchain output trees
  (untracked; populated by the setup scripts above).
- `docker/` — release-build container (`linux-builder.Dockerfile`,
  `build_release.sh`, `in_container_release.sh`).
- `hardware/` — Chipyard recipes (`*.yaml`) describing how each hardware
  target maps to a Chipyard config; consumed by `./merlin chipyard`.
- `patches/` — series files and helpers for the downstream IREE patches
  carried in `third_party/iree_bar`. Verified by `./merlin patches verify`.

A future cleanup may split this folder into `toolchains/`, `containers/`,
`hardware/`, and `patches/`. New code should land in the most-specific
subdir; do not add unrelated concerns to the top of `build_tools/`.

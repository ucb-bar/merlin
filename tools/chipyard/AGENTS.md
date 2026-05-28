# `tools/chipyard/` — agent guide

## Mental model

`tools/chipyard/cli.py` dispatches `./merlin chipyard <action>` across 15
subactions split into 8 topic modules: config (chipyard root persistence),
recipe (yaml hardware recipes from `build_tools/hardware/`), git_ops
(checkout state machine + validate + submodule backup/restore),
bare_metal (VCS/Verilator builds + ELF run), firesim (deploy pipeline),
zephyr (workload staging + run), radiance (Muon kernel runner), status
(info/status/firemarshal).

For the per-action → module map, read `__init__.py`. Action discovery
goes through `cli.py:setup_parser`; if your new action isn't registered
there it isn't reachable from `./merlin chipyard`.

## Pitfalls

- **`cmd_checkout` is a git state machine.** It backs up untracked files
  in chipyard's submodules before switching state, then restores after.
  All four helpers (`_find_submodule_path`, `_backup_untracked`,
  `_restore_untracked`, `_git`) live in `git_ops.py` and must stay
  together — splitting them invites silent data loss on checkout.
- **`require_chipyard_root()` is the auth check** every command needs.
  Forgetting it lets a command run against a missing or wrong chipyard
  tree. Always call it first inside a new `cmd_*` handler.
- **Recipe yaml schema is loose**: scripts read `firesim`, `submodules`,
  `branch`, `sha` blocks defensively. When you add a new recipe field,
  update both `recipe.py:load_recipe` and every consumer that reads it
  (grep across `chipyard/*.py`).
- **`cmd_run_zephyr` chains `cmd_stage_zephyr_workload`** — the two are
  tightly coupled. If you split zephyr into more files, keep them
  together.
- **`build_firemarshal` is a 1-line shell wrapper** — don't add logic
  here; if it grows, hoist to its own module.

## Cross-references

- Consumes: `build_tools/hardware/*.yaml` recipes,
  `.chipyard_config.json` for the path persistence, FireSim deploy infra.
- Produces: bare-metal sim binaries, FireSim deploy configs, Zephyr
  workload overlays, hwdb registrations.
- No MCP wrapper today — chipyard interactions are interactive enough
  (validate + checkout do file-system mutations with user confirmation)
  that structured returns wouldn't add much over `--help` + Bash.

## Update triggers

Re-read this file and update it in the same turn if you edit:

- `tools/chipyard/cli.py` (new subaction / changed dispatch) — refresh
  module map.
- `tools/chipyard/{config,recipe,git_ops,bare_metal,firesim,zephyr,radiance,status}.py` —
  if you split or fold a module, update the per-action table.
- Recipe YAML schema under `build_tools/hardware/` — update
  `recipe.py:load_recipe` and refresh Pitfalls if a new field is added.

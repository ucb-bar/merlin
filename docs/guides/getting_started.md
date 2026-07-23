---
title: Getting started — the setup and prerequisites reference
kind: guide
status: current
owner: core
last_verified: 2026-07-22
related: [reproducibility, architecture, repo_structure, integrations, model2mlir, rvv_e2e,
          beam_search, gemmini_experiment, zephyr, dse, dse_guidance, targetgen, adding_a_target]
code_refs:
  - pyproject.toml
  - .env.example
  - .gitmodules
  - build_tools/scripts/check_repro_env.py
  - merlin/python/merlin/common/paths.py
  - merlin/python/merlin/common/artifacts.py
---

# Getting started

This is the single, consolidated **setup + prerequisites reference** for the whole repo: clone → a
working environment → a first run. Every workflow guide links back here for the shared base and then
lists only its own workflow-specific extras. For *what* the repo is read
[Architecture](../reference/architecture.md); for *where things live* read
[Repository structure](../reference/repo_structure.md); for the by-intent workflow index read the
[reproducibility master guide](reproducibility.md).

**Read this first, then jump to your workflow.** The core SDK and the full test suite install and run
with nothing but `uv` — no submodules, no external toolchains. Every board / simulator / model-lowering
/ agentic feature is an **opt-in** that resolves an external tool through a `MERLIN_*` environment
variable and **degrades fail-closed** when unset (its tests skip, its runner records `not_run` — never
a fabricated pass). So set up only the pieces for the workflow you actually run; §4 is the map.

## 1. Base install — needed for every workflow

The project uses [uv](https://docs.astral.sh/uv/) (the in-repo `.venv` is Python 3.13; the package
supports 3.10+). A plain clone is enough — **no `--recursive`**; the multi-GB `third_party/`
submodules are opt-in per task (§5).

```bash
git clone https://github.com/ucb-bar/merlin.git
cd merlin
uv sync --all-extras                                    # .venv + merlin (editable) + every extra below
uv run python build_tools/scripts/check_structure.py    # verify the tree/docs invariants hold
.venv/bin/python -m pytest merlin/tests                 # run the suite (plain `python` is not on PATH)
```

`.venv/bin/python` is the driver interpreter for everything in the docs (plain `python` is not on
PATH). Without `uv`: `pip install -e '.[dev,xdsl,targetgen,kernels-ast,kernels-exo,kernels-parquet,kernels-plots]'`.

The RTL-grounded targetgen flow additionally reuses the sibling **`mlc`** package (§5). It lives at a
machine-specific path (`$MERLIN_MLC_DIR`), so it is not a `uv sync` dependency — install it editable
into the same `.venv` once you have the checkout: `uv pip install -e "$MERLIN_MLC_DIR"` (else
`.venv/bin/python -m pip install -e "$MERLIN_MLC_DIR"`). Without it, `import mlc` fails and the
mlc-derived RTL facts degrade honestly (the rest of the SDK is unaffected).

**Optional-dependency extras** (`[project.optional-dependencies]` in `pyproject.toml`; `--all-extras`
installs them all):

| extra | pulls in | needed for |
|---|---|---|
| `dev` | pytest, pytest-timeout, pyyaml, pydantic, rich | the test suite (always) |
| `xdsl` | xdsl≥0.68 | the xDSL prototyping plane / dialect descent |
| `targetgen` | jsonschema | targetgen contract validation (fail-closed schema checks) |
| `board` | *(no PyPI deps)* | documented install target for the board/sim/RTL flows — the actual tools are external (§5) |
| `kernels-ast` | tree-sitter, tree-sitter-c | AST structural kernel mining |
| `kernels-exo` | exo-lang | Exo compile-to-C ingest in kernel mining |
| `kernels-parquet` | pyarrow | columnar kernel-feature table |
| `kernels-plots` | matplotlib | kernel-mining / DSE evaluation plots |

## 2. The `.env` file — how external tools are found

External, machine-specific tool locations live **outside** the repo and are passed by environment
variable; the repo finds *itself* via `merlin.common.paths.repo_root()`. Copy the template and set the
paths for the workflows you run:

```bash
cp .env.example .env        # .env is gitignored and never committed
```

Resolution order for every key (via `merlin.common.paths.env` / `ext_path`): a real **process env
var wins**, then **`.env`**, then a built-in **default**. `.env.example` is the authoritative,
commented list of every variable and what it is for — treat it as ground truth alongside this doc.

> Two chipyard conventions, both real: most tools read **`MERLIN_CHIPYARD`** directly (~25 sites:
> cost model, ceiling drivers, the Gemmini backend), while the `ext_path()` consumers read
> **`MERLIN_EXT_CHIPYARD`**. They do **not** fall back to each other — point **both** at the same
> chipyard checkout.

## 3. Verify what will actually run

Before starting real runs, ask each capability's own availability guard what is runnable here:

```bash
.venv/bin/python build_tools/scripts/check_repro_env.py            # human table (always exit 0)
.venv/bin/python build_tools/scripts/check_repro_env.py --json     # machine-readable
.venv/bin/python build_tools/scripts/check_repro_env.py --require spike_rv64gcv,k1_board   # exit 1 if any unavailable
```

It probes each capability (`available()` guards), lists the three isolated interpreters (`.venv`,
`out/build/chia-venv`, `$MERLIN_M2M_VENV`) and the `out/` roots, and for anything `unavailable` tells
you exactly **which env var to set**. It never runs a workload and has no side effects. The
capabilities it reports (key → what it needs):

| capability key | what it enables | env vars it checks |
|---|---|---|
| `xdsl` | xDSL prototyping plane | pip extra `.[xdsl]` |
| `llvm_m2m_toolchain` | model2MLIR + clang-23 whole-model lowering | `MERLIN_M2M_DIR`, `MERLIN_M2M_VENV`, `MERLIN_CLANG`, `MERLIN_IREE_BIN` |
| `spike_rv64gcv` | spike rv64gcv RVV oracle | `MERLIN_CHIPYARD`, `MERLIN_SPIKE`, `MERLIN_RISCV_GCC` |
| `saturn_vec` | Saturn-vectors RVV spike cert | `MERLIN_CHIPYARD`, `MERLIN_SPIKE` |
| `k1_board` | SpacemiT K1 board (real cycles) | `MERLIN_K1_HOST`, `MERLIN_K1_SSH_KEY`, `MERLIN_K1_TOOLCHAIN` |
| `gemmini_spike` / `gemmini_verilator` | Gemmini functional (L2) / RTL cycle (L3) | `MERLIN_GEMMINI_SPIKE` / `MERLIN_GEMMINI_VERILATOR`, `MERLIN_CHIPYARD` |
| `gemmini_vcs` / `firesim` | Gemmini VCS (L4) / FireSim (L5) | `MERLIN_GEMMINI_SIMV` / `MERLIN_EXT_FIRESIM_QUEUE`, `FIRESIM_ROOT` |
| `zephyr_spike` | Zephyr SW whole-model build_app path | `ZEPHYR_BASE`, `MERLIN_ZEPHYR_SW`, `ZEPHYR_SDK_INSTALL_DIR`, `MERLIN_CHIPYARD` |
| `circt_firtool` | CIRCT firtool + FileCheck (RTL checks) | `firtool` / `FileCheck` on PATH, `MERLIN_CHIPYARD` |
| `chia` | chia agentic-loop framework | a `uv venv` at `out/build/chia-venv` |
| `llm_api` | Anthropic API for real agentic runs | `ANTHROPIC_API_KEY`, `MERLIN_LLM_MODEL` |

## 4. Prerequisites by workflow

Each row is: what you need **beyond §1** to run that workflow, and the honest fallback when a piece is
not available on a fresh machine. "board-gated" items are called out in §5.

| Workflow / guide | Required beyond base install | Optional / fallback |
|---|---|---|
| Kernel mining ([kernel_mining](kernel_mining.md)) | external kernel-source repos by `MERLIN_<SOURCE>_REPO` (XNNPACK, OpenBLAS, Exo, Triton, Autocomp); extras `.[kernels-exo,kernels-parquet,kernels-plots]` | LLM escalation needs `ANTHROPIC_API_KEY` (deterministic outputs stand alone without it) |
| DSE / design-pressure ([dse](dse.md), [design_pressure](design_pressure.md), [dse_guidance](dse_guidance.md)) | base install only — runs on committed captures/fixtures | measured evidence needs an `aet` run; cycle-exact calibration needs the sim toolchain (see [dse_guidance](dse_guidance.md)) |
| model2MLIR capture ([model2mlir](model2mlir.md)) | `MERLIN_M2M_DIR` + the m2m capture venv (`MERLIN_M2M_VENV`); model repos | — (a full smolVLA capture is RAM-heavy; committed bundles ingest without re-capture) |
| RVV end-to-end ([rvv_e2e](rvv_e2e.md)) | `llvm_m2m_toolchain` (m2m + clang-23) + capture bundles + `spike_rv64gcv` | K1 board is **optional** — spike rv64gcv is the bit-exact fallback |
| Beam search ([beam_search](beam_search.md)) | frozen `hand_v0` baseline (in-tree) + expert objdump fixtures + a K1 board **or** spike | `spike_rv64gcv` substitutes for the physical K1 (correctness/cycles, no wall-clock); `chia` only for Ray fan-out |
| Gemmini experiment ([gemmini_experiment](gemmini_experiment.md)) | `bwrap` on PATH + the sim toolchain (`MERLIN_CHIPYARD` → spike/verilator) + `ANTHROPIC_API_KEY` for real agentic runs + `out/build/chia-venv` for fan-out | mock LLM fallback runs without a key (no real agentic run); VCS/FireSim rungs skip if absent |
| Zephyr / FireSim / spike ([zephyr](zephyr.md)) | `ZEPHYR_BASE`, `MERLIN_ZEPHYR_SW`, `ZEPHYR_SDK_INSTALL_DIR`, `MERLIN_CHIPYARD` | spike substitutes for 2-tile FireSim |
| Target generation ([targetgen](targetgen.md), [adding_a_target](adding_a_target.md)) | base install + extra `.[targetgen]` (jsonschema) | RTL-grounded targets additionally use `circt_firtool` and the sibling `mlc` package (editable-installed from `MERLIN_MLC_DIR`, §5) |
| External baselines ([integrations](integrations.md)) | the relevant framework repo/build + its venv by `MERLIN_*` var (§5) | each arm skips independently when its var is unset |
| Publish a champion (reproducibility §8) | base install; a local `git init --bare` remote | a real GitHub push is human-gated (never automatic) |

## 5. External dependencies — where each one comes from

Everything here is **out of band** (not a PyPI dep, not vendored). Set the matching `.env` key.

**model2MLIR (`m2m`) — the model frontend.** Clone
[`ucb-bar/model2MLIR`](https://github.com/ucb-bar/model2MLIR) and run
`build_tools/scripts/setup_model2mlir.sh` (creates m2m's own torch + torch-mlir venv and a dedicated
capture venv). Point `MERLIN_M2M_DIR` at the checkout; the lowering path runs inside
`MERLIN_M2M_VENV` (defaults to `$MERLIN_M2M_DIR/.venv`). See [model2MLIR frontend](model2mlir.md).
Quantization happens **in m2m, not Merlin** — int8 (W8A8) is the only measured-working format; fp8 /
int4 are a documented plan.

**mlc — the model-ladder compiler (RTL frontend for targetgen).** The CIRCT+xDSL frontend Merlin
reuses to derive ISA/geometry/capacity facts from accelerator RTL (`targetgen.rtl.mlc_bridge`). Point
`MERLIN_MLC_DIR` at the checkout and **pip-install its Python editable into `.venv`**
(`uv pip install -e "$MERLIN_MLC_DIR"`) — this replaces the old runtime `sys.path` shim, so `import
mlc` is a normal import. `MERLIN_MLC_DIR` still locates mlc's **non-Python assets**: the prebuilt
`circt-opt` binary (`third_party/circt/build/bin`), the cached `runs/circt-arc/<target>/outputs`
(HW-dialect `*_hw.mlir` + `discovered_roles.json`), and schemas. Unset or not installed => the
mlc-derived RTL facts report honest-unavailable rather than crashing.

**LLVM / MLIR 23 + clang-23.** The whole-model path needs a standalone LLVM/MLIR-23 install
(`mlir-translate`) and `clang-23`. Point `MERLIN_MLIR_INSTALL` / `MERLIN_MLIR_TRANSLATE` and
`MERLIN_CLANG` at them, or set `MERLIN_IREE_BIN` at the IREE-Merlin build that ships clang-23. In-repo
there is a prebuilt `third_party/llvm-install/` and the `third_party/llvm-project` submodule (the LLVM
monorepo — the dominant clone cost, init only if you build it yourself:
`git submodule update --init --depth 1 third_party/llvm-project`).

**spike + RISC-V toolchain (chipyard).** The rv64gcv RVV oracle and the RISC-V cross toolchain come
from a [chipyard](https://github.com/ucb-bar/chipyard) checkout (`spike`,
`riscv64-unknown-elf-gcc` under `.conda-env/riscv-tools/bin`, `sims/verilator`). Set **both**
`MERLIN_CHIPYARD` and `MERLIN_EXT_CHIPYARD` to it (or point `MERLIN_SPIKE` / `MERLIN_RISCV_GCC`
directly). spike is the **primary bit-exact RVV verifier** and the fresh-machine substitute for the
physical K1 board.

**SpacemiT K1 board — physical hardware, NOT fresh-machine reproducible.** Real on-silicon cycle
numbers need the physical K1 development board (a DHCP-leased board on a private Wi-Fi segment); it
cannot be provisioned by a `git clone`. Set `MERLIN_K1_HOST=root@<board-ip>` and
`MERLIN_K1_SSH_KEY=/path/to/key` when you have one. **Fallback for everyone without the board:**
`spike rv64gcv` gives bit-exact RVV correctness (and cycle counts under the simulator); only the
real-wall-clock speedup claims require the board, and those steps record `not_run` when it is absent.
Board-SSH note: the campus path filters inbound `:22` to that segment, so the board also listens on
**2222** and `.env` sets `MERLIN_K1_SSH_PORT=2222` (honored across all ssh/scp by `rvvgen/k1.py`); a
board that pings but hangs on `:22` is not down — use 2222.

**Gemmini / Saturn / Muon simulators.** Gemmini functional (spike, L2) and RTL-cycle (verilator, L3)
sims come from chipyard; set `MERLIN_GEMMINI_SPIKE` / `MERLIN_GEMMINI_VERILATOR` (+
`MERLIN_GEMMINI_HARNESS_DIR`). **VCS (L4, `MERLIN_GEMMINI_SIMV`) needs a Synopsys VCS license** and
**FireSim (L5, `MERLIN_EXT_FIRESIM_QUEUE` + `FIRESIM_ROOT`) needs an FPGA/manager** — neither is
fresh-machine reproducible; both fail-closed to `not_run`, and spike+verilator cover
functional+cycle-accurate certification without them. Saturn (`MERLIN_SATURN_*`) and Muon
(`MERLIN_MUON_*`, `MERLIN_RADIANCE_KERNELS`) sims are analogous opt-ins.

**Zephyr SDK.** The Zephyr whole-model build path needs the Zephyr SW workspace and SDK 0.17.0:
`MERLIN_ZEPHYR_SW`, `ZEPHYR_BASE` (the zephyr tree), `ZEPHYR_SDK_INSTALL_DIR`, plus `MERLIN_CHIPYARD`
for spike. See [Zephyr backend](zephyr.md).

**chia (agentic loops).** The Ray-based agentic fan-out runs under an **isolated** venv at
`out/build/chia-venv` (never the main `.venv`):
`uv venv out/build/chia-venv --python 3.13 && uv pip install --python out/build/chia-venv -e /path/to/chia -e .`.

**Anthropic API key (agentic loops).** The kernel-mining beam proposer and the Gemmini QA-loop arms
call the Anthropic API via `merlin.common.llm.complete`. Set `ANTHROPIC_API_KEY` (+ optionally
`MERLIN_LLM_MODEL`) for real agentic runs; **unset ⇒ `complete()` returns `None` and the caller uses
its deterministic mock fallback** — tests still pass, but no real agentic run happens.

**External baseline frameworks** (cross-framework K1 comparison — [integrations](integrations.md)).
Each arm is an adapter over an external checkout/build, passed by `MERLIN_*` var, and skips
independently when unset: `MERLIN_XNNPACK_REPO`, `MERLIN_OPENBLAS_REPO`, `MERLIN_TVM_LIBRARY_PATH`
(+ the `MERLIN_TVM_*` family), `MERLIN_BUDDY_BUILD` / `MERLIN_BUDDY_LLVM_BUILD`, `MERLIN_GGML_BUILD`,
`MERLIN_ET_VENV` (ExecuTorch). The baseline submodules under `third_party/baselines/` are initialized
only for the arms you run (`tvm` / `executorch` recurse into their own submodules).

## 6. The CLI surface

Every workflow is a console-script (`uv sync` / `pip install -e` installs them); each is a thin
entrypoint over a `merlin.*` module. The full table is the generated [CLI reference](../reference/cli.md).
Run any with `--help`. The main entry points, by workstream:

| Workstream | CLI | Guide |
|---|---|---|
| Kernel mining | `merlin-rvv-mine`, `kernel-index`, `kernel-bench`, `merlin-rvv-beam` | [Kernel mining](kernel_mining.md), [Beam search](beam_search.md) |
| Design-pressure & DSE | `merlin-design-pressure`, `merlin-dse`, `merlin-dse-guidance` | [DSE](dse.md), [DSE guidance](dse_guidance.md) |
| Target generation | `merlin-targetgen`, `merlin-target-publish`, `merlin-compare` | [Target generation](targetgen.md) |

## 7. Where output goes

**Never** hand-build an output path. Generated output lives under a single `out/` root with exactly
three subdirs — `out/runs/`, `out/artifacts/`, `out/build/` — created via `merlin.common.artifacts`
(`start_run`, `new_product`, `cache_dir`, `recaptures_dir`). A PreToolUse hook blocks writes
elsewhere. See `CLAUDE.md` "Generated-output convention" and the `artifact-layout` skill.
Point-in-time reports live under `out/artifacts/`, **not** in `docs/`.

## 8. Conventions before you commit

- **Shared working tree** — do not switch branches; commit on the current branch, small verified
  batches (`CLAUDE.md`).
- **Tests** go in `merlin/tests/<bucket>/test_*.py` (`test-layout` skill).
- **Docs** carry front-matter and are indexed by the hub; run
  `build_tools/scripts/gen_docs_index.py` after adding one. See `docs/AGENT.md`.
- Re-run `check_structure.py` after any structural change.
</content>
</invoke>

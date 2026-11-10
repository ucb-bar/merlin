# Repository tools

Start with the [user guide](../docs/README.md) to compile or run a program.
These tools maintain the repository and prepare its supported workflows.

| Directory | Purpose |
| --- | --- |
| `checks/` | CI and local checks, their shared parsers, and reviewed debt ledgers |
| `docs/` | Generate optional CLI, schema and module references under `out/` |
| `packaging/` | Package build support, author pass-library generation and installed qualification |
| `deploy/` | Build delivery packages, derive host views and inspect model builds |
| `hooks/` | Optional Git hooks and their installer |
| `toolchains/` | Compiler environment selections and cross-toolchain/OpenMP build recipes |
| `dependencies/` | Exact backend, support and external source pins |
| `chipyard/` | Authored SoC configurations used by the current target examples |
| `firesim/` | Read-only environment checks and existing link/build support; infrastructure remains external |

`package_resources.json` selects the public package payload.
`logical_components.json` records implementation ownership.
`generalization_debt.txt` records unresolved coverage obligations.
These are reviewed source records, not generated reports.

For a local source check:

```sh
python build_tools/checks/check_structure.py
python build_tools/checks/check_test_budget.py
```

To install the optional hooks for this clone:

```sh
python build_tools/hooks/install_git_hooks.py
```

Read [CI workflows](../.github/workflows/README.md) for release/install checks
and qualification with explicitly selected generated inputs. Historical paper
figures, captured tables and one-off campaigns are available from their original
Git revisions; they are not part of the maintained tool surface.

Corpus preparation lives in `src/merlin/development/testgen/`, numerical references
in `src/merlin/evaluation/`, and expert disassembly capture in `src/merlin/mining/`.
Target-specific preparation stays with its example.

Repository checks cover source policy, packaging, documentation and maintained
interfaces. They do not certify a generated corpus or a device run. Checks tied
to the retired tracked corpora and study plans have been removed from CI and the
hooks; qualify explicitly selected generated inputs as described in the CI guide.

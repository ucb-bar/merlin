# AGENT.md — merlin/contract

The **experiment ABI**: the repo-independent, versioned contract an out-of-tree target-backend
package is built against (see `README.md`). Hand-authored, frozen source-of-record — consumed by
`merlin.targetgen` runners + tests via `merlin.targetgen.contract.contract_dir()`.

- Contract specs: `*_contract.yaml`, `command_buffer_abi.yaml`, `scoring.yaml`, `integrity_policy.md`,
  `interface_grammar.md`, `VERSION`.
- `schemas/` — fail-closed JSON-Schema validators. `examples/` — frozen golden inputs (g0–g2).
- `capsules/` — the gemmini reference corpus (regenerable via `capsules/generate_corpus.py`, but
  tracked as a frozen test suite). A root `bench_contract` symlink was retired; refs use `merlin/contract`.

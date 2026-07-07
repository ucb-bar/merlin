---
title: Generated target repositories
kind: reference
status: current
owner: targetgen
last_verified: 2026-07-07
related: [targetgen, adding_a_target]
code_refs: [merlin/python/merlin/targetgen/generate]
---

# Generated target repositories

TargetGen generates a `merlin-target-<name>/` repository per target. Only the reference
`toy_npu` (and a minimal `example_vector`) live in-tree; serious target dialects are
generated into external repos.

## Layout

```
merlin-target-<target>/
├── README.md, AGENT.md, pyproject.toml, CMakeLists.txt
├── contracts/        # the five synchronized plans (+ source manifest)
│   ├── target_contract.yaml
│   ├── dialect_plan.yaml
│   ├── runtime_adapter_plan.yaml
│   ├── zephyr_plan.yaml
│   ├── llvm_extension_plan.yaml
│   └── target_source_manifest.yaml
├── docs/             # evidence_report.md, validation_report.md, evidence_index.yaml, stubs
├── xdsl/             # xDSL prototype dialect + smoke test
├── include/, lib/, tools/   # MLIR/C++ dialect scaffold (placeholder)
├── runtime/          # Merlin runtime ADAPTER (adapter.py, simulator semantics, encodings)
├── zephyr/           # Zephyr runtime-backend module
├── llvm/             # LLVM extension plan + out-of-tree placeholders
├── examples/, tests/
```

Every directory carries an `AGENT.md` (top-level dirs get rich ones; nested dirs are
backfilled with a generic one).

## The five synchronized plans

| Plan | Schema | Says |
| ---- | ------ | ---- |
| `target_contract` | target_contract | what the hardware/runtime exposes |
| `dialect_plan` | dialect_plan | which dialect ops/types/lowerings to scaffold |
| `runtime_adapter_plan` | runtime_adapter_plan | how the target implements the Merlin runtime ABI |
| `zephyr_plan` | zephyr_plan | the Zephyr backend scaffold to generate |
| `llvm_extension_plan` | llvm_extension_plan | whether/how LLVM changes are needed |

## Confidence and review

For `toy_npu` the plans are concrete (`confidence: high`, `requires_human_review: false`).
For real targets (gemmini/saturn/radiance) the plans are conservative skeletons seeded from
keyword-detected concepts, flagged `confidence: low|medium` and `requires_human_review:
true`. **TargetGen does not claim its synthesized artifacts are correct.**

## Validation

```
python build_tools/scripts/check_generated_target.py <generated-repo>
# or
python -m merlin.targetgen.cli inspect --target <generated-repo>
```

Checks the five plans validate against the schemas, `docs/evidence_report.md` exists, the
per-layer directories exist, and every directory has an `AGENT.md`.

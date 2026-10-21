# Additive mask runtime candidate

This packet contains an inactive interpretation and acceptance-to-execution proofs for the
actual SmolVLA mask-selection cut. Start with [the semantic review](../runtime-candidate-review.md).
The runtime interpretation and claim/lock admission require human review. No activation or
full-model successful-execution claim is included.

## Formal entry points

| Declaration | Meaning |
|---|---|
| `FrontendMaskWhole.check_sound` | Accepted actual cut executes successfully with exact true count and initialized output, for every input in its declared domain |
| `FrontendMaskProjection.check_sound` | The checked cut is exactly the selected slice of the actual exported function; it inherits the cut execution theorem |
| `FrontendBoolSimulationWith.check_sound_candidate` | Unchanged actual bool-repair checker implies exact execution against the approved original-program f32 repair policy under the candidate interpretation |
| `FrontendBoolCandidateChain.compose` | A suffix proved under that same interpretation composes with the repair policy; a frozen-interpretation receipt cannot discharge the suffix premise |

`statements.json` contains the elaborated public statements, for review rather than active
admission. Component proofs establish initialized sizes and conditions, bounds, compact/scatter
invariants, structural loop termination and representable final counter advancement. The
whole-cut proof executes the actual count-provider operations, allocation, both loops, reshape
chains and return through the existing Lean evaluator.

The cut domain is canonical row-major, initialized `tensor<1x32x32xi1>`,
`tensor<1x1024xi64>` values and `tensor<1x1024xi64>` seed, with index64 and an iteration limit
of at least 1024. The second loop scatters compact values into selected positions and retains
the seed elsewhere. The all-false case has a zero-sized compact tensor and no compact reads.
The full model's establishment of this domain is unproved. Other full-model obligations have
not been exhaustively inventoried.

## Reproduction

Use Lean and Mathlib from `merlin/lean/{lean-toolchain,lake-manifest.json}`, the authoritative
compiler Python environment, LLVM runner utilities, and a separate CPU oracle environment
containing exactly `torch==2.10.0+cpu` and `numpy==2.5.4`.

From the checkout root, set `PATH` to include elan and set `PYTHONPATH=src`,
`MERLIN_LEAN_REQUIRED=1`, `MERLIN_COMPILER_PYTHON`, `MERLIN_M2M_PYTHON`,
`MERLIN_TORCH_PYTHON`, `MERLIN_MLIR_INSTALL`, `MERLIN_LLVM_LLC` and `MERLIN_CLANG`
for the installed tools. Then:

```sh
(cd merlin/lean && lake build MerlinVerify merlin-lean merlin-lean-audit)
python -m pytest -q merlin/tests/ir/test_lean_frontend_mask_runtime.py
python out/artifacts/verification/merlin-passes/frontend/runtime-candidate/qualify.py
python build_tools/scripts/check_lean_guard.py
python build_tools/scripts/check_structure.py
```

`qualify.py` regenerates full before/after MLIR and authoritative JSON exports, compares their
hashes with the retained original boundary, runs the actual Lean checkers, and records native
and pinned PyTorch execution evidence. Large raw exports/results live in ignored `work/` and
`../actual/`; their digests and reproduction procedure are retained. The native runner is the
torch-mlir ExecutionEngine, with the exact pipeline, wheel version and runner-library digest
recorded. This tests the interpretation's fidelity; it does not prove LLVM implementations.

`generate_template.py` transcribes an authoritative isolated-cut JSON export into raw Lean AST
data. It contributes no proof facts. The maintained `FrontendMaskTemplate` factors that same
AST into named fragments. A regenerated raw template is separately accepted by the Lean
whole-cut checker during qualification of this packet.

The earlier f32 IEEE edge-case tests remain in `test_bool_cast_poison.py` and
`test_lean_spec_frontend_torch.py`. They include NaNs, infinities, signed zeros, fractions,
subnormals and ordinary values. Universal f32 comparison correctness is proved using Lean's
IEEE model; native samples are separate execution evidence. No f16/bf16/f64 proof is added.

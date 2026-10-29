# Ledger review corrections

This review applies to `feat/pass-verif-integration` after `6a6ffacb2`.
Frozen semantics, existing theorem pins, approval flags and producer behavior
are unchanged. Prior receipts retain their original ledger/source cut.

## Review disposition

| Finding | Resolution |
| --- | --- |
| Half-extension injectivity | Added `MerlinVerify.Rules.Bf16F32AccFusion.fused_step_on_image`, which needs no injectivity hypothesis and quantifies over all half-precision bit patterns. The historical pinned theorem is unchanged. |
| Fusion premise wording | Driver and Python receipts now cite the stronger theorem. Program correspondence still needs one operand format, uniform fusion and no other contract-flagged f32 multiply/add operations outside contract steps. Source license counts do not prove target fusion behavior. |
| Missing projection/effect entries | Added T7 for Python/xDSL correspondence and A10 for the existing selected primitive/ISA/effect hypotheses, including all-index `Models`. Family mappings cite both IDs where applicable. No new Lean axiom or approval was introduced. |
| Nine projection-field spellings | Preserved existing report schemas and documented a common T7 mapping. The AST inventory records 56 dictionary/keyword emission sites across the nine spellings. Free-text premises are outside that count. |
| F12 status | Marked the production defect corrected on both routes while retaining the original counterexample and failed result. Per-channel i64 formal source support remains uninstalled. |
| DequantWidth dispatcher | Corrected the ledger to identify installed per-tensor i64 dispatch through `Spec.Dequant.perTensorAt`, separately from the uninstalled per-channel i64 extension. |
| Old K/initializer gaps | Early component sections now point to the existing checked K-bound, source-derived bound, integer destination and initialized abs/max bridges. They distinguish component premises from obligations discharged by later checkers. |
| Undefined S1 and ambiguous files | Removed S5's dangling S1 reference, documented that S1–S3 are undefined, preserved S4–S6 identifiers, and qualified every `.lean` file reference with its repository tree. All literal Lean file paths resolve. |

The fusion proof uses `Function.invFun_eq` on a value already in the conversion's
image. After packing and unpacking a selected preimage, applying the conversion
again gives the original converted operand. Recovering the original half bits
is unnecessary. Tests instantiate a conversion that maps every input to zero,
prove that it is not injective, and prove the fused-step law for every operand
pair and arbitrary FMA, independently for f16 and bf16. This removes the bad
injectivity justification without assuming a non-NaN domain or a backend shift.
It remains a local parametric-model result, not a concrete IEEE/backend proof or
independent approval of the draft f32-accumulation policy.

The T7 inventory does not rename legacy report keys or pretend that those reports
already serialize literal T7/A10 IDs. It supplies a machine-readable mapping to
the ledger entries. A canonical report-field migration is a separate versioned
interface change and is not applied here.

## Qualification and freshness

The rebuilt checker succeeds. The fusion import audit checks 1,580 theorems with
no violations; the new lemma uses only the three permitted standard axioms.
All 235 existing pinned statements and frozen file hashes still match.
The numerical-domain, MACC contract and compiler-gate suite passes 58 tests;
three retained-capture replay tests skip because their historical artifacts are
absent. The two new F12 public-compiler tests pass. The separate legacy F12 run
passes its i32 folder regression and skips the missing native execution runner.
Full commands, results, source hashes and explicit skips are retained in
[results.json](results.json). The projection inventory is
[projection-inventory.json](projection-inventory.json).

Fresh generation and independent-process replay cover all six Q/DQ actual
before/after boundaries across both routes. The multiply-to-add mutation still
refuses. These are strict lowering-refinement checks; quantizer definedness
remains explicitly abstracted in these receipts. They do not rerun the older
pinned PT2 capture-producer campaign.

Both actual host-fusion compiler boundary pairs from the previous handoff were
replayed in fresh processes against the new ledger. The MACC regression suite
includes fresh live delivery/replay for bf16 and f16 and then deliberately
mutates its input files to test stale-evidence refusal. Those retained test
receipts are negative-test evidence, not currently replayable boundaries.
Separate `macc-bf16-v2/` and `macc-f16-v2/` generation/replay directories retain
untampered boundaries and fresh independent-process replays. Other historical loop, buffer and configuration receipts were not
regenerated and must not be described as current-ledger qualification.

F12 now also has two public-compiler regressions. Both xDSL and textual routes
execute the original signed-byte `1`, i64 zero point `16777217`, f32 scale `1`
witness and produce `-16777216`. The old float-first expression gives
`-16777215`. This is native regression evidence, not a per-channel i64 Lean
source theorem. The older execution-engine test remains explicitly skipped
because `libmlir_c_runner_utils.so` is missing. The existing i32 folder regression
passes separately.

## Reproduce

Use new output directories to preserve existing evidence. The tested setup uses
Lean 4.35.0 and the installed compiler interpreter from the earlier handoffs.

```sh
export MERLIN_LEAN_REQUIRED=1
export MERLIN_LEAN_SDK_BIN="$PWD/out/build/lean/toolchain/lean-4.35.0-linux/bin"
export MERLIN_COMPILER_PYTHON="$PWD/out/build/lean/mlir-venv/bin/python"
export MERLIN_CLANG="$MERLIN_LEAN_SDK_BIN/clang"
export C_INCLUDE_PATH="$MERLIN_LEAN_SDK_BIN/../include/clang"
export PATH="$MERLIN_LEAN_SDK_BIN:$PATH"
mkdir -p out/build/ledger-review-replay

(cd merlin/lean && lake build MerlinVerify merlin-lean merlin-lean-audit)
(cd merlin/lean && lake exe merlin-lean-audit MerlinVerify.Rules.Bf16F32AccFusion)
.venv/bin/python - <<'PY'
from build_tools.scripts.check_lean_guard import check_lock, pinned_statements
statements = pinned_statements()
assert statements is not None and check_lock(False, statements)
print(len(statements), "unchanged pinned statements")
PY

.venv/bin/python -m pytest -q -ra --tb=short \
  -o tmp_path_retention_policy=all \
  merlin/tests/ir/test_lean_numerical_domains.py \
  merlin/tests/ir/test_lean_bf16_f32acc.py \
  merlin/tests/ir/test_compiler_macc.py \
  --basetemp=out/build/ledger-review-replay/macc

.venv/bin/python -m pytest -q -ra --tb=short \
  merlin/tests/ir/test_lean_w8a8_dequant_width.py -k public_compiler \
  --basetemp=out/build/ledger-review-replay/f12-native

env -u MERLIN_LEAN_REQUIRED .venv/bin/python -m pytest -q -ra --tb=short \
  merlin/tests/ir/test_lean_w8a8_dequant_width.py::test_actual_i64_channel_lowering_matches_integer_promotion \
  merlin/tests/ir/test_lean_dequant_scalar.py::test_per_channel_large_zero_point_regression \
  --basetemp=out/build/ledger-review-replay/f12-legacy

.venv/bin/python build_tools/scripts/qualify_qdq_lowering.py generate \
  --output out/artifacts/verification/ledger-review-qdq-replay
.venv/bin/python build_tools/scripts/qualify_qdq_lowering.py replay \
  --output out/artifacts/verification/ledger-review-qdq-replay
.venv/bin/python build_tools/scripts/qualify_qdq_lowering.py mutate \
  --output out/artifacts/verification/ledger-review-qdq-replay
```

The host-fusion replay API is documented in the
[prior handoff](../host-graph-fusion-20261029/README.md). Fresh replay outputs here
are `host-fusion-0/` and `host-fusion-1/`. The original receipt bytes are untouched.

Generate durable MACC receipts independently of the mutation tests with the
same exported tool environment and a new output root.

```sh
.venv/bin/python - <<'PY'
import os
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / "merlin/tests/ir"))
from lean_bf16_f32acc_cases import bmm, run_pass
from merlin.verify.compiler_macc import check_boundary, deliver

sdk = Path(os.environ["MERLIN_LEAN_SDK_BIN"])
compiler = Path(os.environ["MERLIN_COMPILER_PYTHON"])
replay_script = '''import sys
from pathlib import Path
from merlin.verify.compiler_macc import replay, deliver, validate_completion
c = replay(Path(sys.argv[1]), Path(sys.argv[2]),
           sdk_bin=Path(sys.argv[3]), compiler_python=Path(sys.argv[4]))
d = validate_completion(c)
assert deliver(c) == Path(d["after"]["path"]).read_text()
assert d["fusion_premises"]["theorem"].endswith(".fused_step_on_image")
assert d["policy_reviewed"] is False
'''
for fmt in ("bf16", "f16"):
    work = root / "out/artifacts/verification/ledger-review-macc-replay" / fmt
    work.mkdir(parents=True, exist_ok=False)
    before = bmm(fmt, batch=1, m=2, n=3, depth=5)
    after, count = run_pass(before)
    assert count == 1
    paths = (work / "before.mlir", work / "after.mlir")
    paths[0].write_text(before)
    paths[1].write_text(after)
    c = check_boundary(*paths, work / "proof", sdk_bin=sdk, compiler_python=compiler)
    assert deliver(c) == after
    subprocess.run([sys.executable, "-c", replay_script, str(c.report),
                    str(work / "replay"), str(sdk), str(compiler)], check=True)
PY
```

## Preserved failed attempts and remaining limits

`fusion-regressions-v1.log` retains a test setup error from a missing parent
output directory. `fusion-regressions-v2.log` retains a generated test-query
binder-name error; that failed elaboration reported `sorryAx` and was never
accepted. The implementation lemma itself compiled on its first build. The
corrected test proves both noninjectivity and the unconditional step law.
The first standalone MACC generation was interrupted by a server restart before
producing a receipt. Its partial `macc-bf16/` directory and empty
`macc-durable-replay.log` are retained; qualification uses the separate `-v2` run.

The main suite's retained-artifact skips and the separate missing-runner skip
remain explicit in the results. Whole-repository structure/dependency checks
retain their baseline findings; no guard was relaxed. This review does not
approve numerical specifications, broaden compiler source domains, establish
whole-capture definedness or prove upstream MLIR/LLVM/native correspondence.

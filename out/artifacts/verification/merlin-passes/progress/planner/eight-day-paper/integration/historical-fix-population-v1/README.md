# Historical fix replay population: registered, execution pending

Pinned local ancestry ends at 229f42c627854c49476a101cb393d0b15044fab8. The collector
records all nonmerge subjects beginning `fix(` or `fix:` in three current source-file
histories and two explicit legacy histories before the architecture migration.
Git's automatic rename following stops at that large migration; the legacy segments
are therefore declared explicitly. No fetch occurred. This is a file-level superset,
not a claim that these paths exhaust every possible predecessor of the passes.
The exact commands, stdout/stderr and source hashes are retained. No sample was drawn.

Four candidates are retained, with source AST triage:

| Commit | Recorded fix | Relation to the current supported set |
| --- | --- | --- |
| 3d281b414 | Narrow per-channel dequantization arithmetic order | Changes lower_quant_ext and helpers; existing F12 counterexample is relevant. Current qualified subset is per-tensor, so rejecting the unsupported fixed version does not demonstrate historical bug detection. |
| b652f91ed | Live handoffs and optional rewrite recording | Changes preprocessing orchestration; no listed pass entry body changes. Dependency effects still need review. |
| 704565a7d | Rename dequant provenance tag | Changes lower_quant_ext; provenance classification is separate from arithmetic preservation. |
| 1381d5e47 | Extend f32 accumulation to fp16 matmul | Changes lower_bf16_matmul_f32acc, outside the three primary implementation tracks and outside blanket bf16-only admission. |

No qualifying fix-subject commits were found in the declared W8A8 or TorchAO source
segments. That is an inventory result, not proof that these passes had no bugs.
Function AST comparison omits positions but does not establish semantic equivalence,
helper independence or replay eligibility. population.json keeps every candidate;
none was silently removed after looking at outcomes.

No historical compiler was executed by this inventory. There is no detection rate,
no historical replay success and no new pass-correctness claim. A future replay must
record the exact parent/fixed source, dependencies, inputs and supported-variant
classification, and compare both valid before/after outputs without counting blanket
unsupported refusal as a successful bug catch. Existing numerical counterexamples
and modern mutation tests remain separate evidence.

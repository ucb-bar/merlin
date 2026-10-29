# Portable source-plan reproduction and review

The completed local batch used only Python stdlib planning code. It read SOURCE/QUERY/METADATA refs as data, with no copied provider/controller/Lean execution. Actual validation matched487 files;10 tiny controls and491 same-file relocation associations passed. No old Q/W test suite or archive campaign ran. All compile/audit/qualification/replay work isUNEXECUTED.

For a transferred packet, independently review/pin INPUTS.json, PLAN.json and the two utility sources, then arrange its payload relative paths. The plain source-only commands are:

```sh
python3 -I -S -B build_plan.py --packet /chosen/packet --output rebuilt-plan.json
python3 -I -S -B verify_plan.py --packet /chosen/packet --plan rebuilt-plan.json
```

These example command lines areUNEXECUTED here; the validated code paths ran on the same current utility bytes in validate_data_batch.py. The relocation check ran the relocated verifier entry point from an unrelated CWD with its own copied builder module. It used hard links to identical files to avoid a large duplicate payload; it does not require hard links on another host.

Builder output is a source/data plan. A verifier integrity match can coexist with a clearlyUNAVAILABLE profile/case if input metadata/source is missing; inspect unavailable lists and status fields, not only an exit code. This actual cut has0 unavailable selected profiles/cases. A changed source/query pin, conflicting module, missing source/file, cycle, unknown header, support-root swap or plan-root swap is explicitly observed in BOUNDED_CONTROLS.json. Controls are new tiny DATA fixtures and do not count as semantic tests.

The independent closed header decoder handles UTF8, nested/line comments, optional prelude, one literal ASCII dotted module import per line and explicit recognized body starts. Unknown/ambiguous headers refuse rather than guess. It is a conservative header decoder, not a Lean body parser, theorem checker or actual imported-object report. Default Init and explicit Init.Data.List.Lemmas/Lean.Elab.Tactic.Omega are external NAMES ONLY. No external Std/SDK source or object is read. Final declaration excerpts are descriptive source text; the exact kernel name/type/axiom audits remain required.

PROFILE_INDEX.json links six separate plans. BASELINE_PLAN_V1.json and plans/W-whole-v1.json stay preserved; W_ENTRY_SUCCESSOR_PLAN_V1.json names the fixed entries successor separately. Every model/domain/root roster is retained independently. Source union compilation does not establish a union theorem. The profile/case output directories are requests keyed by source-selection digest, not existing old workspace authority or permission to create a runtime context.

Read BUILD_AND_AUDIT_ROSTER.md before any later build/audit/qualification design. Root alone publishes/integrates; F alone handles runtime investigation. A NEW independently reviewed official SDK context is a separate missing input. Never reuse old absolute pins, old project objects, a basic Init success, historical failures or saved reports as fresh proof authority. No native command, --deps, guard, loader, installation or network action is authorized by this runbook.

Known exclusions:4 optional historical stdout/stderr text artifacts were not read because the assembly allowlist selected source/query/metadata extensions; their supplied pins remain in the fixed manifest/source inventory. They are not missing required DAG inputs and establish no new failure. Raw actual buffers for all selected cases are retained when bound in pinned report metadata; source-derived models/declarations remain conditional. Optional external API/portability and unspecified schedule comparisons remain null/UNKNOWN.

import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
join=json.loads((OUT/'REGISTERED_COVERAGE.json').read_text());ledger=json.loads((OUT/'inputs/ledger/CLAIM_LEDGER.json').read_text());obligations=json.loads((OUT/'OBLIGATIONS.json').read_text())
limits={
 'shape_view_normalization':'Historical possible successful descriptor/world outcomes with trusted target matching; new actual-after bridge unqualified.',
 'exact_integer_simplification':'Canonical word entry outcomes, including modeled error/world; local canonical-output contracts; no numerical/poison/native fidelity.',
 'scalar_redundancy_elimination':'Typed total-pure CSE/DCE with retained effects; common finite abstract control. Static no-tail restrictions apply to the longer chains.',
 'scalar_loop_invariant_motion':'Total speculation and scoped captures; common finite abstract control/effect contracts. The composed SCF domain is narrower.',
 'ordered_loop_unrolling':'Selected chain: static nonnegative positive-stride no-tail bulk. Inline/tail/general nested variants do not inherit that composition.',
 'ordered_loop_segmentation':'Static signed-control/carry bounds in scalar chain; memory chain uses possible public successes under a different relation.',
 'buffer_reuse':'Initialized private epochs and possible public successes; total/private/unobservable allocation. Finite-capacity failure preservation excluded.',
 'sequential_dependency_schedule':'No family in the selected claim ledger. Development/local routes do not establish general full-entry schedule correctness.'}
lines=['# Paper appendix: registered-kind coverage','', 'The literal producer registry and actual pipeline AST have the same eight keys. These are eight registered transformation kinds, not eight proved passes. Scope tags below are contracts, not aliases or additional transformations. This table joins the frozen claim ledger; it does not extend its domains or approve a specification.','', '| Registered kind | Actual scope tags | Selected claim/evidence families | Relation/domain limit | New complete gate | Graham variant partitions |','| --- | --- | --- | --- | --- | --- |']
for r in join['rows']:
 ids=', '.join(f['row_id'] for f in r['selected_claim_families']) or '**Unmapped in selected ledger**'
 gate='Two-kind source route; held/unqualified' if r['new_graph_gate_supported_kind'] else 'Unsupported by new route (source-only determination)'
 lines.append('| '+' | '.join([r['kind'],', '.join(r['registry_scope_tags']['value']),ids,limits[r['kind']],gate,str(r['graham_variant_count'])])+' |')
lines+=['', 'All exact roots, domain/observation relations, evidence selectors and statuses are in [REGISTERED_COVERAGE.json](REGISTERED_COVERAGE.json). Historical engineering qualification belongs to its exact original family; corrected admission/gate/actual-after source versions inherit none. Global native permissions are held. Parsing/projection, primitive/effect/control/descriptor/allocation/frame correspondence and human review remain explicit.','', 'Shared family counts are shown once below. Reusing a family under several registered kinds does not multiply its tests, receipts or unique inputs. Null metrics remain unknown.','', '| Family census | Original tests | Accepted CHECKs | Issued original receipts | Distinct raw tuples | Positive replay records |','| --- | --- | --- | --- | --- | --- |']
for r in ledger['rows']:
 def number(k):
  v=r['counts'][k]['value'];return '?' if v is None else str(v)
 lines.append('| '+' | '.join([r['row_id'],number('pytest_original_tests'),number('accepted_check_records_original'),number('issued_receipts_original'),number('distinct_raw_input_tuples_original'),number('positive_replay_records')])+' |')
lines+=['', 'Shape’s14 accepted CHECK records cover8 pairs; memory’s17 cover8 triples in each historical suite. Their78/71 test totals are not receipt counts. Integer’s17 receipts cover16 tuples. Declared-call v1 is a separate supplemental route, not a registry key:46 is its original fixed test roster and2 records form its positive replay association, not a fresh46-test suite. New candidate and scalar-family unknown counts are not zeros.','', 'Graham’s pinned inventory partitions34 registered variants and57 broader mechanisms. Its broader categories are:','', '| Broader category | Inventory rows |','| --- | --- |']
for cat,n in join['denominator_reconciliation']['category_counts'].items():
 if cat!='registered_common_transformation':lines.append('| '+cat+' | '+str(n)+' |')
lines+=['', 'The combined91 inventory rows include helpers, candidate transforms, extension/target/emission boundaries and a selector/admission mechanism. They are neither91 passes nor a proved-pass denominator. The57 broader rows remain explicitly outside this fixed registry join; their full row IDs/status boundaries are retained in the JSON and original CSV. No coverage percentage is defined.','', 'Required next evidence/reviews and owners are in [OBLIGATIONS.md](OBLIGATIONS.md). The pending service-profile harness is reported separately; this table alone does not establish ordinary service enforcement.']
(OUT/'APPENDIX_TABLE.md').write_text('\n'.join(lines)+'\n')
lines=['# Missing obligations: owner and acceptance check','', 'This list records scoped review/acceptance obligations. It does not authorize new features, native execution or broader qualification. Preserve existing failures/counterexamples and use separately identified source cuts.','', '| Obligation | Scope | Owner | Remaining requirement | Test or review needed |','| --- | --- | --- | --- | --- |']
for r in obligations['items']:lines.append('| '+' | '.join(r[k] for k in ['id','scope','owner','missing_obligation','required_test_or_review'])+' |')
lines+=['', 'Structured list: [OBLIGATIONS.json](OBLIGATIONS.json). Source/claim join and raw references: [REGISTERED_COVERAGE.json](REGISTERED_COVERAGE.json), [INPUTS.json](INPUTS.json), [LEDGER_INPUTS.json](LEDGER_INPUTS.json).']
(OUT/'OBLIGATIONS.md').write_text('\n'.join(lines)+'\n')
print({'appendix_registered_rows':len(join['rows']),'obligations':len(obligations['items'])})

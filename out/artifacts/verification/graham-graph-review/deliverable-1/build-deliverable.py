"""Generate the reviewed audit tables. Does not modify compiler/proof sources."""
import ast
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
INDEX = json.loads((OUT / 'source-index.json').read_text())
PRODUCER = Path(INDEX['producer'])
VERIFY = Path(INDEX['verification'])
OLD = OUT.parent / 'registry-eight'
FILES = {f['path']: f for f in INDEX['source_files']}
REFERENCES = {}

def ref(path, symbol=None, root='producer'):
    if not path.startswith(('src/', 'merlin/', 'out/')):
        path = 'src/merlin/' + path
    base = PRODUCER if root == 'producer' else VERIFY
    actual = base / path
    assert actual.is_file(), actual
    line = 1
    if symbol:
        if path.endswith('.py'):
            nodes = [n for n in ast.walk(ast.parse(actual.read_text())) if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.name == symbol]
            assert len(nodes) == 1, (path, symbol, len(nodes))
            line = nodes[0].lineno
        else:
            matches = [i for i,s in enumerate(actual.read_text().splitlines(), 1) if 'theorem '+symbol+' ' in s or 'theorem '+symbol+'\n' in s]
            assert len(matches) == 1, (path, symbol, matches)
            line = matches[0]
    key = f'{root}:{path}'
    REFERENCES[key] = {'root':root, 'path':path, 'sha256':hashlib.sha256(actual.read_bytes()).hexdigest()}
    return f'{key}:{line}' + (f'#{symbol}' if symbol else '')

def test_ref(name):
    paths = list((PRODUCER/'merlin/tests').rglob(f'test_{name}.py'))
    return '; '.join(ref(str(p.relative_to(PRODUCER))) for p in paths) if paths else 'No matching named test source located; no changed artifact assigned'

with (OLD/'coverage.csv').open() as f:
    old_rows = list(csv.DictReader(f))
extra = ['claim_boundary', 'support_status', 'evidence_status', 'related_lean_only', 'classification_notes']
FIELDS = list(old_rows[0]) + extra
rows = []
for original in old_rows:
    row = dict(original)
    # Every row is re-audited at the current pin; audit-deliverable rechecks the reviewed hashes.
    row['verification_commit'] = INDEX['verification_revision']
    # Preserve prior detailed assessments; make relative evidence paths unambiguous.
    for field in ('local_positive','local_projection_results','local_refusal_artifacts'):
        for token in ('producer-examples/', 'projection-probes.json', 'projection-probes/'):
            row[field] = row[field].replace(token, '../registry-eight/'+token)
    row['claim_boundary'] = 'actual common-pipeline before/after MLIR; scope/domain as recorded'
    row['support_status'] = 'conditional_subset_only; consult graph_domain and local_projection_results'
    if row['kind'] in ('shape_view_normalization','exact_integer_simplification'):
        row['support_status'] = 'missing_whole_boundary_route'
    refused = ('scalar-internal-call','licm-dynamic-stride','segment-nested','unroll-nested','reuse-wide','reuse-rank-zero','reuse-memory-space','reuse-dynamic-access','schedule-opaque')
    if row['row_id'] in refused:
        row['support_status'] = 'unsupported_by_inspected_graph_projection'
    row['evidence_status'] = 'fresh_changed_producer_bytes_retained; projection_is_not_kernel_acceptance; historical_status_in_qualification_fields'
    row['related_lean_only'] = 'Do not substitute other semantic frameworks without a proved bridge'
    row['classification_notes'] = 'Carried forward from registry-eight; no extra registry kind'
    if row['qualification_status'].startswith('retained raw conditional'):
        row['qualification_status']='Family has a retained raw CSE/DCE-to-LICM worked chain; this fresh row is not a newly qualified instance. Match variant-specific originals before inheriting qualification; wider cases remain reported-only.'
    gap_map={'G01':'D1-G01','G02':'D1-G02','G03':'D1-G05','G04':'D1-G04','G05':'D1-G06','G06':'D1-G03','G07':'D1-G08','G08':'D1-G07'}
    row['gap_ids']=';'.join(gap_map[g] for g in row['gap_ids'].split(';'))
    rows.append(row)

def add(id, family, variant, source, symbol, checker, checker_symbol, domain, assumptions, refusals, test=None, category='broader_transformation', selection='Explicit caller selection; outside TRANSFORM_SCOPES', boundary='actual selected schedule/source before and after', related='No complete-boundary Lean route identified in this audit'):
    source_ref = ref(source,symbol)
    check_ref = ref(checker,checker_symbol) if checker else 'No separate validator entry established; source guards only'
    test = test_ref(test) if test else 'No separate retained changed positive located; source route inspected'
    row = {f:'' for f in FIELDS}
    row.update(row_id=id,kind=family,variant=variant,inventory_category=category,
        producer_commit=INDEX['producer_revision'],verification_commit=INDEX['verification_revision'],
        producer_source=source_ref,selection=selection,producer_domain=domain,
        structural_validator=check_ref,legacy_validator_entry='Not a registered verify_pipeline_change branch',
        legacy_projection='not assigned',legacy_lean_roots='not assigned',legacy_relation='not assigned',legacy_qualification='not established for this route',
        graph_validator_entry='No complete-boundary Lean validator identified for this exact variant',
        graph_projection='No complete-boundary Lean projection assigned',graph_lean_roots='not established for this exact changed boundary',
        graph_relation='Python/SMT/structural checks described by validator; no Lean semantic theorem inferred',
        graph_domain='not established for a Lean whole-boundary claim',assumptions=assumptions,
        whole_boundary_status='whole_boundary_Lean_coverage_not_established',
        local_positive='No serialized original-bound Lean-positive boundary retained by this audit; see behavioral test source separately',
        local_projection_results='not run',local_refusal_artifacts='No new Lean-refusal artifact assigned',
        retained_positive='Test/source reference only, not a retained qualified boundary: '+test,refusal_evidence=refusals+'; '+test,
        qualification_reference='none identified for this exact route',qualification_status='not_established',
        replay_status='no original-bound Lean replay established',new_qualification='none',
        composition_status='missing_semantic_bridge; ordinary Python checks do not establish a Lean pipeline composition',
        host_integration='Shared Python contract/checker path; no new graph-broker admission inferred',
        independent_review='Source/route assessment only; target facts, operation interpretation and projection premise approval remain open',
        gap_ids='D1-G02;D1-G03;D1-G05;D1-G07',claim_boundary=boundary,
        support_status='producer_implemented_Lean_whole_boundary_not_established',evidence_status='source_and_test_definition; selected behavioral suites executed in behavioral-tests/suite.json',
        related_lean_only=related,classification_notes='Meaningful implementation form; parameter combinations are not separately counted')
    if category in ('helper_plan','analysis_admission','selector','extension_point','target_implementation_boundary','emission_boundary'):
        row['whole_boundary_status']='not_an_additional_proved_graph_rewrite'
        row['support_status']='support_mechanism_not_rewrite_denominator'
    rows.append(row)
    return row

S='compile/scheduling/'
T='transforms/'
L='llvmlower/'

add('schedule-orchestrator','schedule_pipeline','admission, candidate selection, storage realization, completion, emission, final checking',S+'schedule_pipeline.py','compile_schedule',T+'pipeline_state.py','validate_pass_order',
    'Typed Kernel; selected contracts; bounded repairs; ordered optional stages',
    'Target effect and byte decoders; genuine current source/contract custody; complete emitted semantics',
    'Bad stage order, stale state, changed actual encoding, unsafe cut, unsupported loops refused','schedule_pipeline','selector',
    boundary='Call path and stage accounting, not a new semantic rewrite',related='Individual graph leaves do not automatically compose through PipelineSession')
for id,var,func,domain in [
 ('dense-block','bounded contraction tiling and operand/loop-order choices','schedule_contraction','Finite matmul; explicit derived store geometry; all knob-selected orders checked'),
 ('conv-block','convolution gather schedule','schedule_convolution','Finite convolution windows and explicit zero path; no pooling'),
 ('conv-tiled','bounded convolution panel tiling','schedule_tiled_convolution','Finite panelized convolution; explicit resource capacities')]:
    add(id,'block_scheduling',var,S+'block_schedule.py',func,S+'block_schedule.py','check_residency',domain,'Physical geometry, operation semantics and zero path supplied by target; residency check is not a numerical Lean proof','Invalid residency, overlapping overwrites, missing capacity/unsupported pooling refuse','block_schedule')
add('dense-compact','dense_scheduling','counted full-K rectangular schedule with tails/operand retention',S+'compact_dense.py','schedule_compact_dense',S+'compact_verification.py','verify_compact_dense','Zero-seeded integer dense source; bounded index arithmetic; full ordered K','Physical address/effect adapter; dense arithmetic/readout contract; target instruction fidelity','Unknown semantics, bad bounds, missing contract, checker-budget exhaustion cannot accept unchecked product','compact_dense')
add('dense-source','dense_scheduling','typed source and optional readout-chain realization',S+'dense_source.py','schedule_source_dense',S+'compact_verification.py','verify_compact_dense','Admitted current typed contraction/readout; compatible pointer ABI','Source reader and physical adapter correctness; source-to-counted binding','Nonzero/unsupported seed, unsupported scalar body/alias contract declined','source_dense')
add('dense-group','dense_scheduling','shared-LHS full-K multi-output counted groups',S+'compact_dense_group.py','schedule_compact_dense_group',S+'compact_dense_group_check.py','verify_compact_dense_group','Separate RHS/output objects; admitted full-K members and LHS reuse law','Physical noninterference and exact original DenseLHSReuseLaw; implementation correspondence','Wrong member products, changed source, bad retention or physical overlap refused','compact_dense_group')
for id,func,meaning in [
 ('dense-resident','plan_dense_resident_panel','resident full-K operand panel'),('dense-stationary','plan_dense_stationary_panel','stationary operand panel'),('dense-output','plan_dense_output_panel','output panel capacity'),('dense-joint','plan_dense_joint_panel','joint operand/output panels')]:
    add(id,'dense_capacity_plans',meaning,S+'dense_resident_panel.py',func,None,None,'Explicit bank-separated capacities and full-K sizes','Target must assign private physical storage, emit and validate actual schedule','Infeasible sizes/ownership do not establish a legal physical realization',category='helper_plan',boundary='Capacity/cost proposal; no emitted boundary and no standalone graph rewrite')
add('dense-rhs','dense_capacity_plans','complete resident RHS product proposal',S+'dense_rhs_residency.py','plan_dense_rhs_residency',None,None,'Explicit private capacity and dense dimensions','Actual ownership, placement and emitted-address checking required','Infeasible capacity declined',category='helper_plan')
add('shared-input','dense_scheduling','shared-input finite block schedule',S+'shared_input_schedule.py','schedule_shared_input',S+'shared_input_verification.py','verify_shared_input','Closed compatible member group; retained shared operand','Exact common source identity, complete physical effects and target capability','Changed product, clobbered resident input and bad ownership refused','shared_input_schedule')
for id,func,var in [('staged-load','eliminate_redundant_rhs_loads','delete redundant live RHS reloads'),('staged-order','order_output_tiles_n_before_m','reorder complete output tiles N before M')]:
    add(id,'staged_dense',var,S+'staged_dense.py',func,S+'staged_dense_verification.py','verify_staged_dense','Explicit finite descriptor/consume stream; whole tile zero/reduction/readout/fence epochs','Immutable/disjoint external buffers; assigned target rows; actual emitted effects must match','Unknown effect, incomplete tile, descriptor/dataflow mismatch and absent ownership refused','staged_dense_tile_order' if id=='staged-order' else 'staged_dense')
add('staged-stripe','staged_dense','place RHS panels in private full-K stripe',S+'staged_rhs_residency.py','place_rhs_stripe',S+'staged_dense_verification.py','verify_staged_dense','Already assigned expanded operand rows; nonstationary bounded finite source','New target ownership/placement contract; capacity alone is not assignment','Other placements, pending/unmatched descriptors, bad source panels refused','staged_rhs_residency',selection='Must compose through rewrite_staged_dense; no new default roster admission')
add('command-compact','counted_scheduling','compress affine finite calls into counted loops',S+'command_compaction.py','compact_command_stream',S+'command_compaction.py','verify_command_compaction','Finite affine primitive stream and bounded counted index expressions','Call meaning and concrete loop/index execution correspondence','Changed expansion, invalid integer domain and ineligible calls refused','command_compaction',related='GraphLoops.checked_execution proves checked counted expansion, but exact command_compaction route binding/qualification not established here')
add('panel-pack','transfer_scheduling','affine matrix-to-panel packing and rectangular aggregation',S+'panel_transfers.py','plan_packed_panels',S+'panel_transfers.py','check_panel_transfers','Explicit immutable affine source, packing layout and transfer limits','Actual source storage bounds, alias ownership, target copy implementation','Invalid range/coverage/limits refused','panel_transfers')
add('tile-coalesce','transfer_scheduling','contiguous columns into strided local tile rows',S+'tile_load_coalescing.py','TileLoadCoalescing',None,None,'Positive bounded rectangular geometry, capacity and block stride','Actual transfer caller implements returned commands/location map','Invalid geometry/overlap/capacity rejected',category='helper_plan',boundary='Geometric command proposal, not proof of emitted transfer semantics')
add('conv-gather','transfer_scheduling','constant-stride convolution gather runs/resident-panel plan',S+'affine_gather_runs.py','emit_convolution_gather_runs',None,None,'Valid flattened windows split at row boundaries','Source indexing and actual target transfer semantics','Unsupported window geometry cannot justify a transfer','affine_gather_runs','helper_plan')
add('config-elide','configuration','remove complete redundant persistent setup calls',S+'configuration.py','eliminate_redundant_configuration',S+'configuration.py','verify_configuration_rewrite','Straight-line calls; pure complete setters; first write retained; unknown operations barriers','Exact total pure-setter law with no hidden pulse/fault; other retained calls may be partial in new theorem','First setter, partial multi-write redundancy, clobber crossing, changed actual effects refused','configuration_elimination')
add('config-ir','configuration','apply configuration cleanup to declared instruction MLIR',S+'configuration_ir.py','eliminate_configuration_ir',S+'configuration_ir.py','verify_configuration_ir','Actual declared instruction MLIR and original configuration selection','MLIR-to-Kernel bridge and concrete setter/effect correspondence','Changed actual attributes/effects/order, unbound declarations refused','configuration_ir',related='GraphState.Elision covers a Kernel boundary; an xDSL↔Kernel bridge is still required for this wrapper')
for id,var in [('effect-setter','redundant setter omission proposal'),('effect-barrier','unprotected redundant barrier omission proposal')]:
    add(id,'effect_cleanup',var,S+'effect_cleanup.py','cleanup_indices',S+'effect_cleanup.py','verify_cleanup','Closed ordered Effect records; only original index deletion; unknowns reset state and pending work','Target exact effect projection and barrier/completion semantics; caller must realize and validate actual calls','Protected or first barrier, pending activity, changed retained order or unknown state cannot be omitted','effect_cleanup',category='candidate_transformation',boundary='Retained-index proposal; concrete target rewriting is outside this helper',related='GraphState.Elision setter theorem is not a theorem for barrier deletion or this separate helper route')
add('config-hoist','configuration','hoist persistent setup prefix from nonempty loop',S+'configuration_loops.py','hoist_loop_configuration',S+'configuration_loops.py','verify_loop_configuration_rewrite','Producer: ordered pure-write constant prefix; body preserves fields; new leaf narrower one setup/flat loop','Exact idempotent overwrite semantics, complete effects, total sequential actions','Zero-trip speculation, pulses, pointer setup, clobbers, changed loop/context refused','configuration_loops')
add('private-copy','transfer_storage','forward private initialized exact copy into original source storage',S+'copy_elimination.py','eliminate_private_copies',S+'temporary_storage.py','verify_temporary_storage','Complete one-copy initialization; readonly destination; source unchanged through uses','Physical shared backing/alias facts; synchronous exact copy; longer lifetime must fit final allocator','Partial initialization, intervening writes/unknown effects rejected; allocation may fall back','private_copy_elimination',selection='allocate_temporary_storage tries forwarding then independently validates final allocation/falls back',related='ResourceGeometry/lifetime lemmas do not prove full byte-dataflow forwarding')
add('copy-coalesce','transfer_storage','merge adjacent exact synchronous copy requests',S+'copy_coalescing.py','coalesce_private_copies',S+'copy_coalescing_check.py','verify_coalesced_copies','Original CopyMergeLaw; bounded adjacent actual calls; explicit maximum transfer/alias windows','Exact-copy alone does not imply fault or nonmemory-observation equivalence; merge law independently justified','Union alias, observers, absent merge law, changed request or bound refused','copy_coalescing')
add('prefetch','transfer_scheduling','move one outstanding private input-copy launch earlier',S+'temporary_storage.py','prefetch_private_copies',S+'temporary_storage.py','verify_copy_prefetch','Explicit prefetch admission; waits retained; bounded launch motion','Qualified concurrency pair, complete effects/completion and private ownership','Hazard, unknown state effect, unsupported concurrency or changed waits refused','temporary_storage')
add('copy-pipeline','transfer_scheduling','overlap multiple launches using existing private slots',S+'copy_pipeline.py','pipeline_private_copies',S+'copy_pipeline.py','verify_copy_pipeline','Original queue/pair admission; fixed bounded slots; exact launch membership and waits','Complete physical accesses, target queue semantics and overlappable-pair truth','Over-limit queue, unqualified pair, state/descriptor crossing, nonmember launch and alias hazard refused','copy_pipeline',related='GraphLoops.Capacity checks bounded traces; not a proof of general async scheduling or actual target overlap')
add('dead-write','transfer_storage','delete killed complete private payload writes',S+'dead_writes.py','eliminate_dead_writes',S+'dead_write_check.py','verify_dead_writes','Already placed calls, full private payload overwrite before observation; original WriteDeletionLaw','No hidden nonmemory effects/fault distinction; fixed physical assignment','Unknown/asynchronous effects, partial kill, observer and missing deletion law retain/refuse','dead_payload_writes')
add('storage-place','storage','allocate and rewrite actual temporary addresses',S+'temporary_storage.py','allocate_temporary_storage',S+'temporary_storage.py','verify_temporary_storage','Actual selected effects, explicit banks/reservations/windows, bounded repair/spill policy','Target access/effect decoder and private ownership; final dataflow checked separately','Out-of-range/overlap, uninitialized read, altered effects and infeasible candidates refused','temporary_storage',related='ResourceGeometry: fit/separation/epochs only; not complete numerical/dataflow preservation')
add('device-spill','storage','insert exact device spill/reload then reallocate',S+'device_spills.py','allocate_with_spills',S+'spill_conformance.py','verify_device_spills','Selected exact spill classes and bounded insertion/repair budget','Spill primitive bit preservation, external backing ownership, final physical effects','Wrong read/write payload, unknown spill class, bad interference and capacity refused','device_spills')
add('ssa-allocation','storage','resource assignment annotation for sequential SSA',T+'resource_allocation.py','emit_resource_allocation',T+'resource_allocation.py','validate_allocated_ir','Explicit representation/extent/bank and instruction constraints; finite sequential graph','Actual graph/lifetime projection, time-offset meaning, physical bank facts','Changed graph/addresses, illegal overlap/range/epoch refused','resource_allocation',boundary='Resource annotation, not instruction semantics or ordinary buffer_reuse',related='Merlin.ResourceGeometry fit/disjoint/overwrite/read-until-use lemmas only')
add('physical-fusion','fusion','adjacent synchronous integer elementwise physical composition',S+'fusion.py','plan_fusion',S+'fusion.py','verify_fusion_step','Private unobserved intermediate; admitted scalar graphs and physical FusionCapability','Concrete fused call computes local composition; suppressed transfers unobservable/no-fault; alias/effects complete','Escaping/intervening observers, unsupported scalar body, wrong actual composition refused','physical_fusion',related='GraphRegionsFusion does not automatically cover every FusionCapability or general plan_fusion route')
add('pointwise-realize','fusion','source-checked pointwise calls and final decoded dataflow',S+'pointwise_source.py','compile_source_pointwise',S+'pointwise_source.py','compile_source_pointwise','Bounded contiguous synchronous integer scalar source; original physical bindings','Source and final byte decoders, primitive semantics and complete effects','Unknown arithmetic, init reads, shifted external aliases, asynchronous arithmetic refused','pointwise_source',boundary='Typed source → selected calls → final decoded image; Python correspondence, no assigned whole-route Lean proof')
add('pointwise-slot','extension','bounded PointwiseScheduleRewrite callback sequence',S+'pointwise_source.py','PointwiseScheduleRewrite',S+'pointwise_source.py','compile_source_pointwise','At most eight named selected callbacks; checked actual final source/dataflow','Each callback needs its own current owner/implementation; name is not a proof','Unknown/changed code or invalid output cannot inherit another callback guarantee',category='extension_point',boundary='Extension API, not an enumerated set of implemented transformations')
add('post-storage-slot','extension','post-allocation deletion callback and original-host DSE wrapper',S+'schedule_pipeline.py','PostStorageRewrite',S+'schedule_pipeline.py','compile_schedule','Fixed allocation; delete-only original call subsequence; actual verifier callbacks','Independent original callback/source custody; all storage/conformance checks retained','Changed retained call/order/placement and stale callback refused','schedule_pipeline','extension_point')
add('region-plan','connected_regions','ordered connected source plan with all external live-outs', 'compile/connected_regions.py','plan_connected_region','compile/connected_regions.py','revalidate','Producer supports admitted pointwise or zero-seeded contraction source; graph leaf only narrower pointwise integer subset','Complete SSA use projection and unread initializer meaning; physical residency not implied','Lost external consumer/live-out, bad order/identity/type/lifetime and unknown intervening op refused','connected_regions','helper_plan',boundary='Source MLIR unchanged; region representation/interface is checked')
add('region-realize','connected_regions','sequential pointwise or mixed dense region launches','compile/connected_regions.py','compile_connected_region','compile/connected_regions.py','_compile_connected','Explicit windows, stage compilers and synchronous continuation; mixed dense needs explicit ABI','Runtime completion/persistence and concrete stage implementation contracts','Unbound continuation/window, aliased dense inputs, unsupported mixed ABI refused','connected_regions',related='GraphRegions plan theorem does not qualify mixed dense runtime compilation')
add('multioutput-fuse','fusion','private producer and two live consumer outputs',S+'multioutput_fusion.py','compile_source_multioutput',S+'multioutput_fusion_check.py','verify_multioutput_fusion','Closed integer pointwise fan-out; producer private; both consumer outputs retained','MultiOutputCompositionLaw; physical alias/effects; suppressed transfers no-fault/unobserved; concrete native local implementation premise','Escaped producer, wrong second output/SSA/body/type, altered downstream observer refused','multioutput_fusion')
add('recompute','source_graph','bounded permitted producer recomputation in consumers',T+'recomputation.py','plan_source_recomputation',T+'recomputation_check.py','verify_source_recomputation','Explicit original permissions; bounded closed integer source and coordinate semantics; private clone','Total pure arithmetic, coordinate/layout reconstruction and original permission truth; no target/cost benefit implied','Unknown full-module control/coordinate semantics, overflow budgets, changed outside source refused','recomputation',boundary='Actual cloned module candidate; function name plan does not make this analysis-only')
add('row-batch','source_graph','batch common-RHS integer source dots',T+'dense_row_batching.py','plan_source_dense_row_batching',T+'dense_row_batching_check.py','verify_dense_row_batching','Bounded static exact integer contractions with shared RHS and supported views','Tensor coordinate/algebra correspondence; no physical retention or profitability claim','Wrong RHS/coordinates, intervening use, altered exterior and budget violations refused','dense_row_batching',boundary='Changed cloned tensor module; separate from dense capacity plan')
add('row-normalize','source_graph','complete-row replay of normalization islands',T+'row_normalization.py','fuse_row_normalizations',T+'row_normalization_check.py','verify_row_normalizations','Closed selected row islands; original max/subtract/exp/sum/divide operations retained','Exact source scalar/numerical contract, row coordinates and unchanged surrounding module; not a new numeric approximation','Unsupported captures/reductions/coordinates or altered exterior refuse','row_normalization',related='Not shape_view_normalization; no automatic link to legacy W8A8/TinyLlama proofs')
add('row-softmax','row_scheduling','complete-domain counted row softmax/masking',S+'row_softmax.py','compile_row_softmax',S+'row_softmax_verification.py','verify_row_softmax','Selected complete rows and explicit numerics/mask/resources','Chosen exp/division/rounding/reduction policy and concrete primitive fidelity','Partial row, incompatible policy, overflow/resource/coverage error refused','row_softmax')
add('host-quant','host_source','bounded round/clamp/conversion legalization',L+'automatic_host.py','optimize_host_quantization',L+'quant_round_check.py','verify_quantization_rewrite','Canonical pure host tensor scalar arithmetic; selected original numerical policy','Floating/environment and source-expression interpretation; no new quantization model','Unknown effects, flags, control flow or code growth declined/refused','automatic_host')
add('host-round','host_source','recognize bounded residual round expression',L+'automatic_rounding.py','rewrite_rounding',L+'quant_round_check.py','verify_quantization_rewrite','Closed bounded round input; supported container metadata; unused old intermediates','Finite range/rounding law and exact source expression reconstruction','Changed external/captured meaning, live intermediates, unbounded input refused','automatic_rounding')
for id,var,func,check in [('destination-slice','pointwise destination into padded tensor','rewrite_module','verify_destination_rewrite'),('destination-permute','pointwise destination into permuted tensor','rewrite_permuted_outputs','verify_permutation_rewrite')]:
    add(id,'host_source',var,L+'insert_slice_destination.py',func,L+'destination_check.py',check,'Exact pointwise destination coordinates and supported exterior slabs','No hidden alias/observation; full writes and surrounding source wiring preserved','Wrong permutation, exterior write, alias or unsupported body refused','insert_slice_destination')
add('im2col','host_source','bounded convolution panels and prepacked im2col layout',L+'im2col_pack.py','rewrite_bounded_match',L+'im2col_pack.py','verify_bounded_match','Selected bounded source windows/panel arms; immutable source','Packing ownership/footprint, padding/index semantics and final host LLVM correspondence','Changed source/arm/index/layout/budget refused','bounded_im2col')
add('im2col-table','host_source','table-selected panel packing with optional parallel panels and tails',L+'im2col_pack.py','rewrite_module',None,None,'Eligible filled-accumulator source; preselected per-operation block table; ordinary or tail panel route','Packing, tail and parallel ownership/index correspondence; bounded helper checker cannot be assumed for this different entry','Unpriced geometry, narrow panel width and unfilled accumulator declined','im2col_pack')
add('conv-normalize','host_source','normalize selected convolution source before dense discovery',L+'convolution_routes.py','normalize_convolution_source',L+'im2col_pack.py','verify_bounded_match','Explicit selected supported source route','Underlying rewriting/packing and source-reader correspondence; no automatic graph proof','Unmatched source retains baseline','convolution_routes')
add('weight-project','abi_storage','remove unused immutable weight entry arguments',L+'entry_weight_projection.py','apply_entry_weight_projection',L+'entry_weight_projection.py','recheck_argument_table','Explicit original generated entry ownership/table','Runtime argument-order/delivery and immutable payload correspondence','Stale source/table or used/wrong-owned argument refused','entry_weight_projection',boundary='Entry ABI projection, not an eight-kind graph normalization')
add('weight-blob','abi_storage','compact selected lazy weight byte spans/runtime rows',L+'blob_weights.py','compact',None,None,'Generated immutable weight layout and selected retained arguments','Exact byte layout/alignment and runtime ABI delivery','Malformed layout/selection cannot establish correct executable payload','blob_weights',boundary='Host storage/ABI mechanism; no assigned whole-program Lean proof')
add('final-segment','encoded_program','split final encoded program only at safe closed boundaries',S+'program_limits.py','finalize_program',S+'program_limits.py','finalize_program','Actual decoded bytes, instruction/address/branch limits and explicit continuation','Decoder fidelity, completion, storage persistence and continuation semantics','Bad instruction boundary/branch/address, live volatile state or unsafe completion cut refused','program_limits',boundary='Final encoded segments, distinct from MLIR ordered_loop_segmentation')
add('periodic-admit','counted_effects','check counted materialization, drained periods and pending capacity',S+'periodic_effects.py','check_drained_period',S+'periodic_effects.py','check_drained_period','Finite declared counted calls; complete concrete effects, windows, epochs and capacity','Target effect decoding and queue/ownership facts; no arbitrary concurrent implementation equivalence','Pending overlimit, undrained end, bad index expansion/epoch/window refused','periodic_effects','analysis_admission',boundary='Complete counted trace admission; not an optimizing before/after graph pass')
add('target-adapters','target_boundary','physical effects, rewrite/emit, decoder and runtime callbacks',S+'schedule_pipeline.py','ScheduleEncoding',S+'schedule_pipeline.py','_emission_effects','Caller-selected concrete target implementation','Truth of software/RTL facts, native instructions, encoding/decoder and runtime correspondence','A matching identity or self-consistent lying decoder is not independent execution qualification','schedule_pipeline','target_implementation_boundary',boundary='Shared interface only; Gemmini/Atlas/OOT concrete implementations are separate obligations')
add('llvm-emission','backend_boundary','emit checked index program through LLVM adapter',S+'llvm_program.py','emit_index_program',None,None,'Caller LLVM builder and target primitive lowering callback','LLVM/index lowering, ABI and target instruction implementation','Unknown index/primitive lowering requires rejection','llvm_program','emission_boundary',boundary='Code emission; no additional proved graph rewrite')

# Add precise newer proof routes without upgrading their qualification.
by_id = {r['row_id']:r for r in rows}
def route(id, validator, vsym, projection, psym, lean, theorem, relation, domain, qualification, packet, composition):
    r=by_id[id]
    r.update(graph_validator_entry=ref('verify/'+validator,vsym,'verification'),graph_projection=ref('verify/'+projection,psym,'verification'),
        graph_lean_roots=ref('verify/lean/Merlin/'+lean,theorem,'verification'),graph_relation=relation,graph_domain=domain,
        whole_boundary_status='conditional_complete_modeled_boundary; actual operation/projection premises remain external',
        support_status='conditional_subset_only',qualification_status=qualification,
        qualification_reference=ref('out/artifacts/verification/'+packet,root='verification'),
        evidence_status='proof/validator sources and report locally present; named raw historical qualification bundle not located',
        replay_status='not freshly reproduced; report is not independent raw-replay inspection',composition_status=composition,
        related_lean_only='Assigned theorem applies only to the explicitly recorded relation/domain',gap_ids='D1-G02;D1-G03;D1-G04;D1-G05;D1-G07')
route('config-elide','lean_graph_state_elision.py','check_boundary','lean_graph_state_elision_capture.py','capture',
    'GraphStateElision.lean','check_sound','Merlin.GraphState.Elision.Preserves: equal complete success or error plus terminal state for arbitrary initial state; context_preserved extends prefix/suffix',
    'Straight-line exact complete call omissions; total pure setters with literal interpretation; retained calls admitted partial',
    'unqualified_SIGSEGV_zero_qualified_originals_or_replays','graph-state-elision/active/README.md','Within-model context theorem exists; hoist→cleanup refused because cleanup is straight-line; no xDSL bridge/broker claim')
route('config-hoist','lean_graph_state_configuration.py','check_boundary','lean_graph_state_configuration_capture.py','capture',
    'GraphStateConfiguration.lean','check_sound','Equal complete modeled total-state program execution under exact overwrite and frame laws',
    'Narrow flat nonempty loop, one constant setup, preserved complete surrounding calls; producer ordered prefixes are broader',
    'reported_only_raw_bundle_unavailable','graph-state/review/sealed-v2/README.md','Context preservation inside total state semantics; not automatically compatible with partial elision/scalar outcomes')
route('region-plan','lean_graph_regions.py','verify_plan','lean_graph_regions_projection.py','project',
    'GraphRegions.lean','checkBoundary_sound','before=after; accepted plan decodes exact ordered source nodes and complete live-outs; substituted decoded plan execution unchanged',
    'One single-block static positive integer pointwise function; closed pure bodies; excludes actual producer contraction variants',
    'reported_only_v4_raw_bundle_unavailable','graph-regions/README.md','Logical region/context theorem; no mixed dense/native runtime proof')
route('multioutput-fuse','lean_graph_regions_fusion.py','verify_fusion','lean_graph_regions_projection.py','project',
    'GraphRegionsFusion.lean','checkRealization_sound','Complete returned-observer equality for logical fused body in typed total Action semantics; checkRealization_native additionally assumes local native implements contract',
    'Supported private-producer/two-output integer pointwise subset; source MLIR retained while Kernel changes',
    'reported_only_v4_raw_bundle_unavailable','graph-regions/README.md','Whole projected observer context supported; native local implementation and source-operational correspondence remain unproved')
by_id['multioutput-fuse']['graph_lean_roots'] += '; '+ref('verify/lean/Merlin/GraphRegionsFusion.lean','checkRealization_native','verification')
by_id['multioutput-fuse']['retained_positive']='Report records actual before Kernel eb49b2f7… and after 4a3a3a77… with downstream observer; raw capture not present locally. '+test_ref('multioutput_fusion')
by_id['region-plan']['whole_boundary_status']='complete_plan_correspondence; no source MLIR rewrite or native implementation theorem'
r=by_id['ssa-allocation']; r.update(graph_validator_entry=ref('verify/lean_resource_geometry.py','project_resource_geometry'),graph_projection=ref('verify/lean_resource_geometry.py','lean_resource_query'),graph_lean_roots=ref('verify/lean/Merlin/ResourceGeometry.lean',root='verification'),graph_relation='Finite fit/alignment, disjoint ranges, live epochs and read-before-overwrite obligations; generated merlin_transform conjunction',whole_boundary_status='geometry_only_not_complete_program_semantics',support_status='local_obligation_only',gap_ids='D1-G01;D1-G02;D1-G05;D1-G07')
r=by_id['periodic-admit']; r.update(graph_validator_entry=ref('verify/lean_graph_loops_capacity.py','verify_capacity_files','verification'),graph_projection=ref('verify/lean_graph_loops_capacity_capture.py','project','verification'),graph_lean_roots=ref('verify/lean/Merlin/GraphLoopsCapacity.lean','check_sound','verification'),graph_relation='Checked complete counted materialization and every-event pending bound; storage/epoch trace soundness; not a transformation equivalence theorem',graph_domain='Explicit supported finite counted trace, positive pending limit, complete admitted event stream, empty final pending set',qualification_status='reported_only_raw_bundle_unavailable',qualification_reference=ref('out/artifacts/verification/graph-loops/capacity-mode-v2/README.md',root='verification'),evidence_status='source/report only; raw v2 capacity qualification not located',gap_ids='D1-G02;D1-G03;D1-G07')
by_id['row-batch']['evidence_status']='Source checked; native CPU pair executed in behavioral-tests/suite.xml with baseline/candidate products, nine outputs, carried state and guard checks retained. No Lean qualification.'
by_id['row-batch']['local_positive']='behavioral-tests/row-batching-native/baseline.mlir -> candidate.mlir; actual native pair.json plus compiled objects/images and six output archives retained; initialized i16 inputs, ordered wrapping i32 reductions, nine outputs and three state-carrying calls per arm'
by_id['row-batch']['retained_positive']='behavioral-tests/native-evidence.json binds 61 copied files and records 80 checked owner/product hashes; ordinary native CPU test only, not a Lean-positive boundary'
by_id['row-batch']['local_refusal_artifacts']='No Lean-refusal artifact assigned; behavioral refusals are the producer tests named in refusal_evidence'

# Sealed graph packets. Each is engineering evidence from its implementation
# owner; local presence counts only how much of its own seal is retained in this checkout.
HISTORICAL_ROOT='/home/jeremy/Projects/merlin/'
def presence(manifest):
    data=json.loads((VERIFY/'out/artifacts/verification'/manifest).read_text())
    entries=data['files'] if isinstance(data,dict) else data
    if isinstance(entries,dict): entries=list(entries.values())
    counts={'entries':0,'matched':0,'missing':0,'mismatched':0}
    for entry in {e['path']:e for e in entries}.values():
        path=VERIFY/entry['path'].removeprefix(HISTORICAL_ROOT)
        counts['entries']+=1
        if not path.is_file(): counts['missing']+=1
        elif hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']: counts['matched']+=1
        else: counts['mismatched']+=1
    return {'manifest':'verification:out/artifacts/verification/'+manifest,**counts}
PACKETS={
    'shape':presence('graph-shape-normalization/standalone-mode-v1/EVIDENCE_SEAL.json'),
    'integer':presence('graph-state-integer/review/sealed-v2/MANIFEST.json'),
    'pipeline':presence('graph-state-pipeline/review/sealed-v1/MANIFEST.json'),
    'memory_scf':presence('graph-memory-scf/memory-scf-mode-v1/EVIDENCE_INDEX.json'),
    'declared':presence('graph-regions-declared/qualification-v1/SEAL.json')}
def held(name):
    p=PACKETS[name]
    return f"{p['matched']}/{p['entries']} sealed evidence entries present locally ({p['missing']} missing, {p['mismatched']} mismatched)"
probes={c['case']:c for c in json.loads((OUT/'projection-probes-sealed.json').read_text())['cases']}
def probed(id, route):
    p=probes[id]
    return f"; sealed {route} projection at {INDEX['verification_revision'][:9]}: {p['status']} (projection-probes-sealed/{id}/{route}/RESULT.json)"
LIVE_GATE='; old-broker live gate still admits this kind on its legacy local statement (D1-G09)'

shape_roots='; '.join(ref('verify/lean/Merlin/GraphLoopsShapeModel.lean',t,'verification') for t in ('checked_complete','checked_memory','checked_context'))
for id in ('shape-cast-generic','shape-dim'):
    r=by_id[id]
    r.update(graph_validator_entry=ref('verify/lean_graph_loops_shape.py','verify_shape_files','verification'),
        graph_projection=ref('verify/lean_graph_loops_shape_capture.py','export','verification')+' -> '+ref('verify/lean_graph_loops_shape_projection.py','project_exports','verification'),
        graph_lean_roots=shape_roots,
        graph_relation='Equal possible complete outcomes (returned Runtime values and entire World; MemoryWorld heap/public specialization) for all typed entry values and common interpretations; relates the false/true readings of ONE projected blueprint module; categorized error/divergence equality not claimed',
        graph_domain='Static ranked integer memrefs, identity layout, default space, element widths 8/16/32/64 plus i1 scalars; signed32/64 index; identical casts preserve the entire descriptor; dynamic views, unknown layouts/encodings and unsupported region/control forms refuse the pair; internal calls by finite call depth',
        whole_boundary_status='conditional_complete_modeled_boundary; actual before/after-to-blueprint binding is a trusted Python full-tree matcher; Lean checks sourceBinding/targetBinding only as nonempty strings',
        support_status='conditional_subset_only; consult graph_domain and local_projection_results',
        qualification_reference=ref('out/artifacts/verification/graph-shape-normalization/standalone-mode-v1/SEAL.json',root='verification'),
        qualification_status='engineering_qualified_standalone_v1 per SEAL.json (78 originals, one 78-test fresh replay, 35 main/41 witness std3 roots); this fresh row is not one of its originals; '+held('shape'),
        replay_status='fresh replay recorded on the producing machine; not reproduced here',
        composition_status='standalone only; shape->reuse->segment adapter is kernel-checked development, not qualified; paper-evidence/COMPOSITION_REFUSAL.json retains the historical join refusal',
        gap_ids='D1-G09;D1-G02;D1-G05;D1-G03;D1-G04')
    r['local_projection_results']+=probed(id,'shape')
    r['retained_positive']+='; sealed worked pair: verification:out/artifacts/verification/graph-shape-normalization/paper-evidence/before.mlir -> after.mlir with CHECKER_OUTPUT.json and WRONG_BUFFER_COUNTEREXAMPLE.json'
    r['host_integration']+=LIVE_GATE

integer_roots='; '.join(ref('verify/lean/Merlin/GraphStateIntegerCode.lean',t,'verification') for t in ('Relation.sound','programs_sound'))+'; generated Merlin.GraphStateIntegerInstance.complete/prefix_actual/exactMiddle per sealed-v2 STATEMENTS.json'
for r in rows:
    if r['kind']!='exact_integer_simplification': continue
    r.update(graph_validator_entry=ref('verify/lean_graph_state_integer.py','check_boundary','verification'),
        graph_projection=ref('verify/lean_graph_state_integer_capture.py','projection','verification')+' -> '+ref('verify/lean_graph_state_scalar_capture.py','project','verification'),
        graph_lean_roots=integer_roots,
        graph_relation='Equal complete Outcome: returned vector and persistent world, or the same modeled error and terminal world, on canonical word inputs; private SSA stacks unobserved; under ExactModel (total deterministic state-free word primitives) and LocalTyping (canonical arity-correct opaque successes)',
        graph_domain='Unflagged signless scalar integers, widths 1..256 (257 refused); structured SSA/SCF with positive constant stride; external nonnested declared calls only, internal calls refused; i1 literal is -1 so i1 does not exercise +1 neutral rules; composed suffix keeps the sealed static no-tail bulk domain',
        whole_boundary_status='conditional_complete_modeled_boundary; Python parser/projection and identity transport remain trusted',
        support_status='conditional_subset_only; consult graph_domain and local_projection_results',
        qualification_reference=ref('out/artifacts/verification/graph-state-integer/review/sealed-v2/README.md',root='verification'),
        qualification_status='engineering_qualified_v2 (75 tests, 17 original receipts 14 changed/3 identity, one fresh batch of 17); v1 stays an UNQUALIFIED diagnostic; static host-status limitation in planner integer-v2-host-status-limitation.json; this fresh row is not one of its originals; '+held('integer'),
        replay_status='fresh batch recorded on the producing machine; not reproduced here',
        composition_status='integer -> CSE/DCE -> LICM -> unroll -> segment six-buffer chain with exactMiddle under one interpretation, within the sealed suffix domain only',
        gap_ids='D1-G09;D1-G02;D1-G05;D1-G03;D1-G06')
    r['local_projection_results']=r['local_projection_results'].replace('not run','legacy not run')+probed(r['row_id'],'integer')
    r['host_integration']+=LIVE_GATE

pipeline_ref=ref('out/artifacts/verification/graph-state-pipeline/review/sealed-v1/README.md',root='verification')
pipeline_roots=ref('verify/lean/Merlin/GraphStatePipelineProgram.lean','checkEntries_sound','verification')+'; generated Merlin.GraphStatePipelineInstance.complete'
PIPELINE=('four-pass CSE/DCE->LICM->unroll->segment chain engineering-qualified ('+pipeline_ref+'; '+pipeline_roots+'; 64 tests, 8 all-four-changed originals, one 8-case fresh replay; '+held('pipeline')+
    '): equal returns/world or error/terminal world in the static nonnegative no-tail bulk domain (factors 2..4, tiles 2..16, 1..4 carries, two inputs, region-free body, signed32/64)')
outside={'scalar-internal-call':'internal calls refused','licm-empty':'empty iteration','licm-dynamic':'dynamic bounds','licm-nested':'nested source loop',
    'licm-dynamic-stride':'runtime stride','unroll-inline_empty':'inline mode','unroll-inline_many':'inline mode','unroll-bulk_tail':'tail mode',
    'unroll-nested':'nested source loop','segment-empty':'empty iteration','segment-nested':'nested carried source loop'}
for id in ('scalar-cse','scalar-dce','licm-nonempty','unroll-bulk_exact','segment-nonempty'):
    by_id[id]['composition_status']=PIPELINE+'; this row\'s own fresh example is a single-pass boundary, not one of its chains'
for id,reason in outside.items():
    by_id[id]['composition_status']='outside the sealed four-pass domain ('+reason+'); no composed bridge for this variant'

MEMORY=('reuse -> two-initializer segmentation engineering-qualified ('+ref('out/artifacts/verification/graph-memory-scf/memory-scf-mode-v1/README.md',root='verification')+'; '+
    '; '.join(ref('verify/lean/Merlin/GraphLoopsMemorySCFModel.lean',t,'verification') for t in ('checked_chain','checked_complete'))+
    '; 71 tests, one 71-test fresh replay, 61/63 std3 roots; '+held('memory_scf')+
    '): equal possible complete public outcomes in one direct allocation-owner block, rank1/rank2 static contiguous default-space i8..i64, complete 0..extent step1 no-carry initializers each segmented; total/private/unobservable allocation premise retained')
by_id['reuse-epochs']['composition_status']=MEMORY
by_id['reuse-epochs']['qualification_status']='historical leaf qualification reported in README only; MemorySCF composition packet qualified, see composition_status'
for id in ('segment-nonempty','segment-empty'):
    by_id[id]['composition_status']+='; also '+MEMORY
for id,reason in (('reuse-wide','i128 refused'),('reuse-rank-zero','rank zero refused'),('reuse-memory-space','nondefault space refused'),('reuse-dynamic-access','dynamic later access refused')):
    by_id[id]['composition_status']='outside the sealed MemorySCF domain ('+reason+'); no composed bridge for this variant'

SCHEDULE='; GraphStateSchedule registered-route checker and its composition with the sealed integer/scalar/loop chain are active development (planner graphstate-sequential-assignment-v1.json); nothing qualified'
for id in ('schedule-pure','schedule-multiple','schedule-opaque'):
    by_id[id]['composition_status']+=SCHEDULE

r=by_id['multioutput-fuse']
r['graph_lean_roots']+='; '+'; '.join(ref('verify/lean/Merlin/GraphRegionsDeclaredBridge.lean',t,'verification') for t in ('derived_local_contract','checked_calls_preserve_observers'))
r['qualification_status']=('GraphRegions v4 reported_only (raw bundle unavailable); declared-call qualification-v1 engineering-qualified (46 cases, one fresh batch, two actual typed boundaries; '+held('declared')+
    '; its two cited original EVIDENCE.json receipts under qualification-v1-originals are not in this checkout); stronger ABI/full-after-effect wrapper bound-qualification-v1 PARTIAL/UNQUALIFIED after a pre-oracle host-reference gate, cause UNKNOWN')
r['qualification_reference']+='; '+ref('out/artifacts/verification/graph-regions-declared/qualification-v1/SEAL.json',root='verification')+'; '+ref('out/artifacts/verification/graph-regions-declared/bound-qualification-v1/STATUS.json',root='verification')
r['composition_status']+='; declared-call v1 derives the local region agreement from per-instruction/packing/ABI/Action contracts, which remain premises'

def write_csv(path,data,fields=None):
    with (OUT/path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(data[0]));writer.writeheader();writer.writerows(data)
write_csv('coverage.csv',rows,FIELDS)

# File-level reconciliation: every source in the bounded denominator receives a role.
# Mixed-role files deliberately keep both roles; a planner name is not decisive.
inventory=[]
role_overrides={
    'transforms/pipeline.py':'pass_selector_and_host_orchestrator',
    'transforms/pipeline_policy.py':'pass_selector',
    'transforms/pipeline_state.py':'analysis_custody_and_ordering',
    'transforms/validation.py':'structural_validator',
    'transforms/scalar_witness.py':'witness_and_structural_validator',
    'compile/fusion_discovery.py':'source_selector',
    'compile/connected_runtime.py':'runtime_contract_and_execution_boundary',
    'compile/scheduling/direct_source.py':'source_selector',
    'compile/scheduling/dead_write_pipeline.py':'wrapper_for_dead-write_and_schedule-orchestrator',
    'llvmlower/quant_round.py':'implementation_helper_for_host-quant',
}
for path,f in FILES.items():
    linked=[r for r in rows if ('producer:'+path+':') in r['producer_source']]
    categories=sorted({r['inventory_category'] for r in linked})
    stem=Path(path).stem
    if path.removeprefix('src/merlin/') in role_overrides:
        categories=[role_overrides[path.removeprefix('src/merlin/')]]
    if not categories:
        if stem=='__init__' or stem in ('staged_dense_transforms','export','library_sources'):
            categories=['export_or_source_membership']
        elif any(x in stem for x in ('check','verification','conformance','admission')) or '/verify/' in path:
            categories=['validator_or_admission']
        elif any(x in stem for x in ('contracts','contract','binding')):
            categories=['contract_or_binding']
        elif any(x in stem for x in ('automatic','source_model','source_dense','source_rows','source_pointwise','routing','full_build','host_provider','spike_model')):
            categories=['selector_or_runtime_integration']
        elif any(x in stem for x in ('analysis','analyses','groups','dense_ir','allocation_types','window_transfer','integer_domain','tokens','derive','dense_planning','allocate','convolution_output')):
            categories=['analysis_or_helper']
        elif stem in ('c_runtime','cpu_softmax'):
            categories=['backend_or_runtime_implementation']
        else:
            categories=['support_helper_or_transport']
    inventory.append({'source':ref(path),'sha256':f['sha256'],'classification':';'.join(categories),
        'coverage_rows':';'.join(r['row_id'] for r in linked) or 'supporting source; not a separate transformation row',
        'reference_document_code_ref':str(path in INDEX['code_refs']).lower(),
        'purpose':f['module_doc'].split('\n\n')[0].replace('\n',' '),
        'note':'Role within explicitly bounded inventory; not a transitive implementation or target qualification claim'})
write_csv('source-inventory.csv',inventory)

section_rows={
76:('registry eight; schedule-orchestrator','Contracts/checking infrastructure, not one theorem per PassSpec'),
94:('target-adapters; llvm-emission','Typed input, facts, analysis and realization are distinct interfaces'),
141:('target-adapters; pointwise-realize','Ownership/effect truth remains target obligation'),
168:('schedule-orchestrator','Freshness/order guards do not prove semantics'),
313:('source-inventory: invocation_record, compilation_recipe, cpu_pool','Timing/provenance support; not graph rewrite'),
355:('weight-project; weight-blob','Host storage and argument ABI; not Lean graph acceptance'),
440:('source-inventory: export, library_sources','V6 default, separate V7/V8 memberships; source membership is not pass count or import authority'),
502:('host-quant; host-round','Current original numeric policy; no assumed W8A8 bridge'),
538:('destination-slice; destination-permute','Actual destination changes checked independently'),
661:('dense-compact; dense-resident; dense-stationary; dense-output; dense-joint; dense-rhs','Capacity proposals separated from realized schedules'),
729:('dense-source; row-batch','Source recognition separated from schedule generation'),
804:('im2col; im2col-table; conv-normalize; conv-block; conv-tiled; conv-gather; panel-pack','Packing/source/transfer routes distinct; no implicit target execution proof'),
936:('dense-source; target-adapters','Selected software/RTL facts constrain route, do not prove interpretation'),
1059:('schedule-orchestrator; post-storage-slot','Candidate stages may be nonmutating until storage commits; final byte segmentation separate'),
1145:('config-elide; config-ir; config-hoist; effect-setter; effect-barrier','Hoisting, deletion and helper barrier cleanup use distinct relations/domains; elision qualification failed'),
1269:('storage-place; ssa-allocation','Physical allocation and resource annotation distinct from registry buffer reuse'),
1313:('storage-place; target-adapters; prefetch; copy-pipeline; periodic-admit','Resource/effect facts, async motion and trace admission separated'),
1505:('private-copy; copy-coalesce; dead-write','Forwarding, merging and write deletion have different original laws'),
1545:('pointwise-realize; region-plan; region-realize; multioutput-fuse','Logical region theorem is narrower than all physical realization'),
1649:('pointwise-slot; post-storage-slot; recompute','Callback slots are not an exhaustive set of algorithms'),
1792:('physical-fusion; multioutput-fuse','General adjacent fusion does not inherit scoped fan-out theorem'),
1870:('physical-fusion; target-adapters','Native contract remains independently justified'),
2014:('device-spill','Exact spill/reload and complete dataflow checker'),
2052:('device-spill; target-adapters','Original spill primitive/geometry truth remains a premise'),
2164:('final-segment','Actual decoded-byte checks and safe segmentation; not MLIR loop segmentation'),
2203:('source-inventory: model_check/program_admission, targetgen/contract/elf_admission','Linked-image checking, not a new graph rewrite'),
2296:('final-segment; target-adapters','Limits, continuation and decoder are selected original contracts'),
2429:('row-softmax','Complete row schedule and explicit numerical policy'),
2515:('row-softmax; target-adapters','No independently approved physical/numerical theorem inferred'),
2635:('source-inventory: source_rows, automatic_rows, source_model','Discovery/selection wrappers, distinct from selected actual rewrite'),
2771:('row-normalize','Row-island scheduling is distinct from shape normalization and numeric redesign'),
2853:('all broader rows; scope limits in coverage-review.md','Whole-model/global placement and backend implementations remain outside a universal graph claim'),
2875:('staged-load; staged-order; staged-stripe; shared-input','Finite staged effects; descriptor issue is not completion')}
docrows=[]
for sec in INDEX['document_sections']:
    mapped,note=section_rows[sec['line']]
    docrows.append({'document':'producer:docs/reference/shared_compiler_passes.md:'+str(sec['line']), 'section':sec['title'],'coverage_or_support':mapped,'assessment':note})
write_csv('document-reconciliation.csv',docrows)
(OUT/'sealed-packet-presence.json').write_text(json.dumps(PACKETS,indent=2)+'\n')
(OUT/'reviewed-references.json').write_text(json.dumps(list(REFERENCES.values()),indent=2)+'\n')
print(json.dumps({'coverage_rows':len(rows),'registry_rows':len(old_rows),'broader_rows':len(rows)-len(old_rows),'source_files':len(inventory),'document_sections':len(docrows),'code_refs':len(INDEX['code_refs']),'checked_reference_files':len(REFERENCES)},indent=2))

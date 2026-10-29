"""Assemble the source-reviewed eight-kind matrix; performs no proof checking."""
from pathlib import Path
import ast
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parent
snapshot=json.loads((ROOT/'inventory.json').read_text())
PRIMARY=Path(snapshot['verification_root'])
PRODUCER=Path(snapshot['producer_root'])
PROBES=json.loads((ROOT/'projection-probes.json').read_text())['cases']
refs=set()


def ref(side,path,symbol=None):
    location=(PRODUCER if side=='producer' else PRIMARY)/path
    assert location.is_file(),location
    refs.add((side,path))
    line=None
    if symbol and location.suffix=='.py':
        tree=ast.parse(location.read_text())
        matches=[n.lineno for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name==symbol]
        assert matches,(path,symbol)
        line=matches[0]
    elif symbol:
        matches=[i for i,s in enumerate(location.read_text().splitlines(),1) if symbol in s]
        assert matches,(path,symbol)
        line=matches[0]
    return side+':'+path+((':'+str(line)+'#'+symbol) if symbol else '')


def p(path,symbol=None): return ref('producer','src/merlin/'+path,symbol)
def v(path,symbol=None): return ref('verification','src/merlin/verify/'+path,symbol)
def lean(file,symbol): return v('lean/Merlin/'+file+'.lean',symbol)
def report(path): return ref('verification','out/artifacts/verification/'+path)
def test(side,file,symbol=None): return ref(side,'merlin/tests/ir/'+file,symbol)

legacy_entry=p('verify/lean_checker.py','verify_pipeline_change')
legacy_host=p('verify/lean_checker.py','accepted_pipeline_receipt')
common_validation=p('transforms/validation.py','validate_change')
old_obligations=p('verify/lean_obligations.py','extended_obligations')
foundation=p('verify/lean/Merlin/Foundation.lean')
base={
 'producer_commit':snapshot['producer_revision'], 'verification_commit':snapshot['verification_revision'],
 'inventory_category':'registered_common_transformation',
 'legacy_validator_entry':legacy_entry,
 'legacy_qualification':'not_reproduced; matching selected Foundation/LeanMLIR toolchain manifest not established locally',
 'host_integration':'apply_common_pipeline requires fresh old-broker completion unless plan_only=True; newer graph leaf not automatically accepted by '+legacy_host,
 'independent_review':'source/route review in this audit; operation-by-operation mapping and semantic premise approval still open',
 'new_qualification':'none; local producer and projection probes only',
}
groups={}

groups['shape_view_normalization']={
 'producer_source':p('transforms/normalization.py','normalize_shapes'),
 'selection':'PipelineOptions.normalize_shapes; enabled by default; stage 0',
 'producer_domain':'identical source/destination memref type; or ranked memref dimension with constant in-range axis and statically nonnegative selected extent',
 'structural_validator':common_validation+' reexecutes normalize_shapes and compares complete resulting IR and witnesses',
 'legacy_projection':legacy_entry+' ShapeWitness branch',
 'legacy_lean_roots':'generated merlin_transform; identity/rfl only',
 'legacy_relation':'local identity or literal Nat reflexivity; no modeled memref descriptor/observer preservation theorem',
 'graph_validator_entry':'not located for this registered normalization route',
 'graph_projection':'not established', 'graph_lean_roots':'not established',
 'graph_relation':'not established for complete changed boundary',
 'graph_domain':'no independently qualified graph mode identified in inspected checkout',
 'assumptions':'trusted xDSL parsing; standard memref cast/dimension meaning and valid descriptors; host reconstruction binds the actual rewrite; metadata and surrounding context require justification',
 'whole_boundary_status':'local_obligation_only; complete memory/context preservation not established',
 'retained_positive':'no original-bound Lean qualification packet located for this registered route; local actual changed example supplied separately',
 'qualification_status':'missing_whole_boundary_evidence', 'replay_status':'not_established',
 'composition_status':'no established graph composition edge for this route',
 'refusal_evidence':test('producer','test_shared_transforms.py')+'; dynamic/invalid axes and nonidentical casts are declined by implementation; declines are not semantic mutation detections',
 'gap_ids':'G01;G02;G03;G06',
}
groups['exact_integer_simplification']={
 'producer_source':p('transforms/normalization.py','simplify_integers'),
 'selection':'PipelineOptions.simplify_integers; enabled by default; stage 0',
 'producer_domain':'scalar signless IntegerType add/sub/mul with no overflow flags; constants reduced modulo declared width; no reassociation; index/tensor/float excluded',
 'structural_validator':common_validation+' reexecutes simplify_integers and compares complete resulting IR and witnesses',
 'legacy_projection':legacy_entry+' IntegerWitness branch',
 'legacy_lean_roots':'generated merlin_transform; Merlin.bvAdd/bvSub/bvMul with decide or simp; '+foundation,
 'legacy_relation':'per-rewrite BitVec equality; surrounding SSA/effects/context correspondence remains trusted Python',
 'graph_validator_entry':'no whole-boundary extension established for this registered simplification route',
 'graph_projection':'no demonstrated bridge to the new scalar graph relation',
 'graph_lean_roots':'none assigned; MerlinVerify arithmetic lemmas and Scalar.check_sound are not silently substituted',
 'graph_relation':'not established for complete changed boundary',
 'graph_domain':'producer width set has no explicit 256 upper bound; legacy scalar exporter and new pure-node guard are narrower',
 'assumptions':'wrapping integer primitive fidelity; trusted parser/reconstruction; complete SSA replacement and context justification',
 'whole_boundary_status':'local_obligation_only',
 'retained_positive':'no original-bound complete graph qualification packet located for this registered route',
 'qualification_status':'missing_whole_boundary_evidence','replay_status':'not_established',
 'composition_status':'no established common graph composition edge',
 'refusal_evidence':test('producer','test_shared_transforms.py','test_overflow_flags_are_not_folded_and_index_overflow_refuses')+'; flagged/index/float/tensor operations are declined, not qualified positive changes',
 'gap_ids':'G01;G02;G03;G06',
}
scalar_common={
 'producer_domain':'closed pure constants/add/sub/mul/sext/zext/trunc; signless widths 1..256; no regions/successors/attributes/unmodeled properties/overflow flags; effects empty',
 'structural_validator':p('transforms/scalar_witness.py','validate_scalar_change')+' replays admitted producer against actual whole IR',
 'legacy_projection':old_obligations,
 'legacy_relation':'generic total pure functional identity; host replay supplies actual SSA/context association',
 'graph_validator_entry':v('lean_graph_state_scalar.py','check_boundary')+'; chains: '+v('lean_graph_state_scalar.py','check_chain'),
 'graph_projection':v('lean_graph_state_scalar_capture.py','project')+' -> '+v('lean_graph_state_scalar.py','emit_query'),
 'graph_lean_roots':lean('GraphStateScalar','theorem check_sound')+'; '+lean('GraphStateScalarSound','theorem normalize_sound')+'; '+lean('GraphStateScalar','theorem check_operands_initialized'),
 'graph_relation':'Merlin.GraphState.Scalar.Preserves: equality of full returned vector/world on success or error/terminal world on failure; private SSA stacks excluded',
 'graph_domain':'complete one-block functions; nested one-block scf.for with positive constant stride; dynamic/negative bounds and empty iteration sequences admitted under common finite-control interpretation; external declarations allowed; internal calls/CFG/unknown regions refused',
 'assumptions':'deterministic total world-free pure primitives; common partial opaque descriptor/operand interpretation and successful result arity; common finite control/index interpretation; trusted parser/complete Python projection; primitive fidelity and poison freedom unproved',
 'whole_boundary_status':'conditional_complete_modeled_function_outcomes; all functions/interfaces bound in trusted projection, no internal-call composition',
 'retained_positive':report('merlin-passes/progress/planner/graph-first-20261025/worked-example/original/checked/report.json')+' plus actual before/middle/after and exact query in same packet',
 'qualification_status':'retained raw conditional CSE/DCE-to-LICM chain inspected; copy-manifest integrity passed; wider scalar cases reported by packet, not all raw cases copied',
 'replay_status':'retained original-bound replay/report.json inspected; not rerun in this audit; compiled historical objects omitted',
 'composition_status':'CSE/DCE -> LICM: exact middle and common registry in scalar chain; four-pass scalar/unroll/segment operational bridge remains unqualified',
 'refusal_evidence':test('verification','test_lean_graph_state_scalar_mutations.py')+'; same-typed wrong operand, changed/dropped observer, poison metadata, effect/carry order, local calls/CFG/unknown regions; source tests inspected, mutation campaign not rerun',
 'gap_ids':'G02;G03;G04;G06',
}
groups['scalar_redundancy_elimination']={**scalar_common,
 'producer_source':p('transforms/pure_scalar.py','eliminate_scalar_redundancy'),
 'selection':'PipelineOptions.eliminate_redundancy; default False; stage 0',
 'legacy_lean_roots':'Merlin.eliminateDuplicate; Merlin.eliminateUnused; '+foundation,
}
groups['scalar_loop_invariant_motion']={**scalar_common,
 'producer_source':p('transforms/pure_scalar.py','hoist_scalar_invariants'),
 'selection':'PipelineOptions.hoist_invariants; default False; stage 0',
 'legacy_lean_roots':'Merlin.hoistInvariant; '+foundation,
}
loop_common={
 'assumptions':'trusted parser/full-tree exporter/canonical SSA correspondence; same complete partial body interpretation and identical context meaning; explicitly selected signed index width; body safety/native meaning not derived',
 'graph_relation':'equivalence of complete relational executions, exact original-body IV trace and final carried/state value; contextual substitution; no separate first-error/terminal-world or divergence equivalence',
 'whole_boundary_status':'conditional_complete_modeled_execution_relation_for_one_supported_boundary; raw-SCF operational bridge open',
 'qualification_status':'qualification reported in retained README; referenced raw seals/queries/results absent from named local packet directory',
 'replay_status':'historical fresh-process replay reported only; not locally inspectable/reproduced for this packet',
 'retained_positive':'actual example names and source generators retained; raw original qualified before/after and CHECK files not present at cited packet roots',
 'gap_ids':'G02;G03;G04;G05;G06',
}
groups['ordered_loop_segmentation']={**loop_common,
 'producer_source':p('transforms/loops.py','segment_loops'),
 'selection':'PipelineOptions.tile_iterations by original nesting depth; empty tuple disables; tile 1 unchanged; stage 2',
 'producer_domain':'static nonnegative index bounds with upper>=lower and positive step/tile; generated signed arithmetic bounded; supports multiple/nested selected loops and preserves cloned body/carry order',
 'structural_validator':p('transforms/validation.py','_LoopRecognizer')+' via '+common_validation,
 'legacy_projection':legacy_entry+' LoopWitness branch; bounds/tile/count + generic fold/index laws',
 'legacy_lean_roots':'Merlin.tileIndex; Merlin.tileTail; Merlin.orderedSegments; '+foundation,
 'legacy_relation':'arithmetic partition and generic chunk fold equality; actual control/body/carry/context link trusted',
 'graph_validator_entry':v('lean_graph_loops_segment.py','verify_segment_files'),
 'graph_projection':v('lean_graph_loops_segment_project.py','project')+' -> '+v('lean_graph_loops_segment_projection.py','project_trees')+' -> '+v('lean_graph_loops_segment.py','obligation'),
 'graph_lean_roots':lean('GraphLoopsSegmentModel','theorem checked_complete_trace')+'; '+lean('GraphLoopsSegmentModel','theorem checked_context')+'; Merlin.GraphLoopsSegment.Model.check_sound; checked_carry_state',
 'graph_domain':'exactly one changed source loop per complete module; static nonnegative controls; positive step; tile>1; signed32/64; region-free original body; no unknown loop metadata; arbitrary carry arity and identical surrounding context',
 'composition_status':'supported same-SCF no-tail bulk-unroll -> segment bridge; arbitrary nested/multiple or memory/scalar operational composition not established',
 'refusal_evidence':test('verification','test_lean_graph_loops_segment.py','test_valid_ir_mutations_refuse_complete_actual_boundary')+'; wrong bounds/strides/carries/yields/observer/context; local nested-projection refusal retained',
 'qualification_reference':report('graph-loops/segment-mode-v1/README.md'),
}
groups['ordered_loop_unrolling']={**loop_common,
 'producer_source':p('transforms/unrolling.py','unroll_ordered_loops')+'; '+p('transforms/unrolling.py','_unroll_one'),
 'selection':'PipelineOptions.unroll_factors by original depth; empty tuple disables; factor1 unchanged; explicit growth budget; stage2',
 'producer_domain':'static nonnegative index bounds, positive step; factors>1 changed; index_bits 2..128; unknown loop metadata declined; whole inline/empty or grouped bulk with optional tail; nested/multiple selected loops allowed',
 'structural_validator':p('transforms/unrolling.py','validate_unrolled_change')+' via '+common_validation,
 'legacy_projection':old_obligations+' count/groups/remainder and generic ordered fold/range laws',
 'legacy_lean_roots':'Merlin.rangeSplit; Merlin.tileIndex; Merlin.orderedSegments; '+foundation,
 'legacy_relation':'arithmetic partition and generic body-fold equality; actual emitted SSA/effects/context association trusted',
 'graph_validator_entry':v('lean_graph_loops_unroll.py','verify_unroll_files'),
 'graph_projection':v('lean_graph_loops_segment_project.py','project')+' -> '+v('lean_graph_loops_unroll_projection.py','project_trees')+' -> '+v('lean_graph_loops_unroll.py','obligation'),
 'graph_lean_roots':lean('GraphLoopsUnrollModel','theorem checked_complete_trace')+'; '+lean('GraphLoopsUnrollModel','theorem checked_context')+'; '+lean('GraphLoopsUnrollComposition','theorem checked_composed'),
 'graph_domain':'one complete changed outer source loop, factor>1, static nonnegative controls and signed32/64; region-free body/standard yields/no unknown control metadata; inline and bulk/tail preserve aliases/carries/context',
 'composition_status':'only compatible no-tail bulk unroll -> segmentation under grouped original-body interpretation; inline disappearance/tail/multiple loops and four-pass scalar composition not established',
 'refusal_evidence':test('verification','test_lean_graph_loops_unroll.py','test_type_valid_actual_mutations_refused')+'; wrong lane/IV/carry/tail/effect/observer; budget and overflow distinct producer refusals; nested local projection refusal retained',
 'qualification_reference':report('graph-loops-unroll/unroll-mode-v1/README.md'),
}
groups['buffer_reuse']={
 'producer_source':p('transforms/planning.py','reuse_storage')+' using '+p('transforms/planning.py','plan_storage'),
 'selection':'PipelineOptions.reuse_buffers; enabled by default; stage1 after storage_analysis',
 'producer_domain':'identical memref type/alignment per slot; static positive extents/default contiguous layout (rank0 also accepted); power-of-two byte-addressable integer width and alignment; full rectangular initializer before reads; direct frees/nonoverlap; no aliases/escapes; processes allocation blocks separately; no memory-space or later-static-index guard',
 'structural_validator':common_validation+' freshly recomputes storage plan and reexecutes reuse_storage against whole IR',
 'legacy_projection':legacy_entry+' BufferReuseWitness interval pairs',
 'legacy_lean_roots':'Merlin.disjointLifetime; '+foundation,
 'legacy_relation':'same-slot lifetime intervals do not overlap; no memory values/initializer/allocator/context preservation theorem',
 'graph_validator_entry':v('lean_graph_loops_reuse.py','verify_reuse_files'),
 'graph_projection':v('lean_graph_loops_reuse_capture.py','export')+' -> '+v('lean_graph_loops_reuse_projection.py','project_exports')+' -> '+v('lean_graph_loops_reuse.py','obligation'),
 'graph_lean_roots':lean('GraphLoopsReuseModel','theorem checked_complete_observations')+'; '+lean('GraphLoopsReuseModel','theorem checked_context')+'; Merlin.GraphLoopsReuse.execution_forward/execution_backward; Model.checked_initializer_coverage',
 'graph_relation':'equality of sets of possible complete public outcomes under live initialized epoch/slot simulation; not categorized failure/divergence equality and not equality of private freed cells/pointers',
 'graph_domain':'one direct allocation-owner block; nonempty rank/static positive contiguous default-space i8/i16/i32/i64; full rectangular initializers; static valid later accesses; signed32/64; identical type/alignment reused groups',
 'assumptions':'TOTAL PRIVATE UNOBSERVABLE allocation/free; same admitted partial retained primitives/public context; raw rectangular initializer meaning and pure prefix/effect facts trusted; parser/exporter/SSA mapping trusted',
 'whole_boundary_status':'conditional_complete_possible_public_outcomes; allocator/resource claims explicitly excluded',
 'retained_positive':'source generator two_epochs etc retained; referenced qualified raw actual cases and seals absent from local reuse packet',
 'qualification_status':'historical qualification reported in README only; current MemorySCF composition leaves remain unqualified',
 'qualification_reference':report('graph-buffer-reuse/reuse-mode-v1/README.md'),
 'replay_status':'historical replay reported only; not locally inspectable/reproduced',
 'composition_status':'actual reuse -> two-initializer segmentation join reported refused by old segment decoder; new MemorySCF work has no qualified triple in this snapshot',
 'refusal_evidence':test('verification','test_lean_graph_loops_reuse.py','test_valid_ir_mutations_and_unsupported_aliases_refuse')+'; omitted initializer, wrong buffer/free, views/escapes, invalid allocator; local i128 refusal; documented actual 8->16 byte finite-capacity counterexample',
 'gap_ids':'G02;G03;G04;G05;G06;G07',
}
groups['sequential_dependency_schedule']={
 'producer_source':p('transforms/planning.py','apply_order')+' using '+p('transforms/planning.py','plan_dependencies'),
 'selection':'PipelineOptions.dependency_orders; empty tuple disables; each selected block order separately recorded; stage4',
 'producer_domain':'permutation of all selected block ops; complete nested operand dependency walk; ordered chain of every effectful/unknown operation; terminator remains after predecessors; supports multiple functions and effectful context',
 'structural_validator':common_validation+' reconstructs current dependencies and actual full permutation',
 'legacy_projection':p('verify/lean_export.py','export_scalar')+' -> '+p('verify/lean_export.py','equivalence_query'),
 'legacy_lean_roots':'generated merlin_transform over all actual scalar arguments; Foundation BitVec semantics',
 'legacy_relation':'complete returned-value equality for one defined single-block scalar function; this is stronger than a local arithmetic lemma but narrower than general scheduling',
 'graph_validator_entry':v('lean_graph_state.py','check_boundary'),
 'graph_projection':v('lean_graph_state_capture.py','capture')+' -> '+p('verify/lean_state_effects.py','derive_effect_schedule')+' / '+p('verify/lean_state_projection.py','derive_effect_schedule')+' -> '+v('lean_graph_state.py','emit_query'),
 'graph_lean_roots':lean('GraphState','theorem check_sound')+'; '+lean('GraphStateContract','def Preserves')+'; Merlin.StateEffects.preserveExchanges',
 'graph_relation':'complete modeled state equality for the exact action permutation in arbitrary sequential prefix/suffix contexts; total deterministic read-local write-framed action domain',
 'graph_domain':'complete selected single-block kernel entry and unchanged module context; exact operation membership, types/SSA, full effect cells; known standard meanings or independently selected complete effect contract; unknown effects/partial/asynchronous actions outside admitted model',
 'assumptions':'actual effects are total/deterministic/read-local/write-framed/sequentially complete; resource/alias/observer declarations truthful; standard memory/index/allocation premises where used; trusted parser/whole-module correspondence/effect projection',
 'whole_boundary_status':'conditional_complete_model_state_equality for supported action domain; old scalar route only complete scalar return equality',
 'retained_positive':'sealed-v2 README reports four actual apply_order originals; raw originals/queries/replays absent from local named packet; local actual examples and projection models supplied',
 'qualification_status':'historical StateEffects schedule qualification reported only; must match each actual variant to originals before inheriting status',
 'qualification_reference':report('graph-state/review/sealed-v2/README.md'),
 'replay_status':'historical originals/fresh replay reported only; not locally inspectable/reproduced',
 'composition_status':'adjacent exchanges composed inside GraphState; not a composition theorem for this route with scalar/SCF/reuse semantics',
 'refusal_evidence':test('producer','test_shared_transforms.py','test_dependency_schedule_preserves_effects_and_unknown_operations')+'; invalid permutation/effect inversion rejected structurally; old exporter rejects multiple functions/opaque context; newer default contract rejects unknown observer in local probe',
 'gap_ids':'G02;G03;G04;G05;G06',
}

variants={
 'shape-cast-generic':'identical memref cast removal', 'shape-dim':'static dimension materialization',
 'scalar-cse':'common subexpression elimination (closed pure primitive set)',
 'scalar-dce':'unused leaf and transitive dead-chain elimination',
 'scalar-internal-call':'eligible scalar rewrite in context containing a locally defined call',
 'licm-nonempty':'nonempty loop invariant chain motion', 'licm-empty':'empty-loop total speculation',
 'licm-dynamic':'dynamic-bound invariant motion with positive constant stride', 'licm-nested':'nested lexical invariant motion',
 'licm-dynamic-stride':'producer invariant motion with runtime stride; graph projection requires positive constant stride',
 'segment-nonempty':'single-loop ordered segmentation with carries/nonunit step/tail',
 'segment-empty':'empty-loop segmentation', 'segment-nested':'nested/multiple selected boundaries (wider producer domain)',
 'unroll-inline_empty':'empty inlining with result forwarding', 'unroll-inline_many':'nonempty full inlining (single/multiple iterations; aliases/carries)',
 'unroll-bulk_exact':'grouped bulk with no remainder', 'unroll-bulk_tail':'grouped bulk plus ordered remainder loop',
 'unroll-nested':'nested/multiple selected boundaries (wider producer domain)',
 'reuse-epochs':'identical-type initialized epoch groups (ranks/epochs/slots/alignment parameterized)',
 'reuse-wide':'producer-only wider physical integer domain represented by i128',
 'reuse-rank-zero':'producer-only rank-zero scalar memref reuse',
 'reuse-memory-space':'producer-only nondefault memory-space reuse',
 'reuse-dynamic-access':'producer-only dynamic later private access after complete initialization',
 'schedule-pure':'one pure scalar entry function', 'schedule-multiple':'selected entry with additional untouched functions',
 'schedule-opaque':'pure motion around retained undeclared opaque observer',
}
for op in ('add','sub','mul'): variants['integer-fold-'+op]='constant wrapping '+op
for rule in ('add-left','add-right','sub-right','mul-left','mul-right'): variants['integer-neutral-'+rule]='neutral operand '+rule

rows=[]
for case_id,description in variants.items():
    local=ROOT/'producer-examples'/case_id
    capture=json.loads((local/'capture.json').read_text())
    assert capture['status']=='actual_changed_producer_output',case_id
    kind=capture['kind']
    group=groups[kind]
    row={**base,**group,'row_id':case_id,'kind':kind,'variant':description,
         'local_positive':str(local.relative_to(ROOT))+'/before.mlir -> after.mlir; capture.json binds bytes; explicit plan_only=True; no Lean acceptance',
         'local_projection_results':'; '.join(r['route']+': '+r['status']+(' ('+r['reason']+')' if 'reason' in r else '') for r in PROBES if r['case']==case_id) or 'no new whole-boundary projection selected',
         'local_refusal_artifacts':'; '.join('projection-probes/'+case_id+'/'+r['route']+'.json' for r in PROBES if r['case']==case_id and r['status']=='projection_refused') or 'none; source-level refusal references supplied',
    }
    if case_id=='shape-cast-generic': row['legacy_relation']='generated forall (alpha : Type) (value : alpha), id value = value; proves generic identity, not actual memref semantics/context'
    if case_id=='shape-dim': row['legacy_relation']='generated (extent : Nat) = extent by rfl; retrieving actual static dimension/descriptor semantics remains entirely outside this proposition'
    if case_id in ('segment-nested','unroll-nested','reuse-wide','reuse-rank-zero','reuse-memory-space','reuse-dynamic-access','licm-dynamic-stride','scalar-internal-call'):
        row['whole_boundary_status']='unsupported_by_retained_graph_mode; producer applies rewrite; legacy local obligations do not close gap'
        row['qualification_status']='no graph qualification for this wider variant; do not inherit narrower packet status'
        row['replay_status']='not_established_for_variant'
        row['retained_positive']='no qualified whole-boundary original located for this wider variant; local actual changed source/result supplied separately'
    if case_id=='schedule-opaque':
        row['whole_boundary_status']='unsupported_by_both_inspected_default_routes; admitted external effect contract would define a different reviewed domain'
        row['qualification_status']='not_established_for_unadmitted_opaque_variant'
        row['replay_status']='not_established_for_variant'
    if case_id=='schedule-multiple':
        row['legacy_relation']='unsupported: old exporter requires exactly one defined function; new state-effects projection binds unchanged surrounding functions and emits a model locally'
        row['qualification_status']='projection capability observed; no matching retained raw qualified multiple-function original located'
    rows.append(row)

assert {r['kind'] for r in rows}==set(snapshot['registry'])
columns=['row_id','kind','variant','inventory_category','producer_commit','verification_commit','producer_source','selection','producer_domain','structural_validator','legacy_validator_entry','legacy_projection','legacy_lean_roots','legacy_relation','legacy_qualification','graph_validator_entry','graph_projection','graph_lean_roots','graph_relation','graph_domain','assumptions','whole_boundary_status','local_positive','local_projection_results','local_refusal_artifacts','retained_positive','refusal_evidence','qualification_reference','qualification_status','replay_status','new_qualification','composition_status','host_integration','independent_review','gap_ids']
with (ROOT/'coverage.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=columns)
    writer.writeheader()
    writer.writerows({key:row.get(key,'not_applicable') for key in columns} for row in rows)

for side,path in [('producer','src/merlin/transforms/certificates.py'),('producer','src/merlin/transforms/pipeline.py'),('producer','src/merlin/transforms/pipeline_policy.py'),('producer','src/merlin/compile/scheduling/schedule_pipeline.py'),('producer','docs/reference/shared_compiler_passes.md'),('verification','out/artifacts/verification/graph-memory-scf/active/README.md'),('verification','out/artifacts/verification/graph-state-pipeline/active/README.md')]:
    ref(side,path)
inventory=[]
for side,path in sorted(refs):
    file=(PRODUCER if side=='producer' else PRIMARY)/path
    inventory.append({'source':side,'path':path,'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
(ROOT/'reviewed-source-identities.json').write_text(json.dumps({'producer_commit':snapshot['producer_revision'],'verification_commit':snapshot['verification_revision'],'files':inventory},indent=2)+'\n')
print(json.dumps({'registered_kinds':len({r['kind'] for r in rows}),'variant_rows':len(rows),'referenced_source_files':len(inventory),'new_qualification_issued':False}))

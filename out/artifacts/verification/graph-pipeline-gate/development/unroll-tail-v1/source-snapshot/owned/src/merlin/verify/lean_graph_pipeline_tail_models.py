"""Closed actual-syntax structural success and first-fault witness candidates.

No Python evaluator supplies semantic authority. No source-success premise is
accepted. Witness Lean is UNCOMPILED and the proposed outcomes are UNOBSERVED.
"""
import json
from dataclasses import asdict
from .lean_graph_pipeline_gate_projection import project_lean_sources
from .lean_graph_pipeline_tail import query_roots


def witness(prepared):
    pair=json.loads(prepared.projection)
    chosen=[(i,f) for i,f in enumerate(pair['functions']) if not f['declaration'] and any('loop' in r for r in f['before'])]
    if len(chosen)!=1:
        raise ValueError('one actual loop entry for this concrete witness')
    index,f=chosen[0]
    loop=next(r['loop'] for r in f['before'] if 'loop' in r)
    facts=json.loads(prepared.producer_diagnostics)
    if len(facts)!=1 or [facts[0][k] for k in ('lower','upper','step','factor','remainder')]!=[0,5,1,2,1]:
        raise ValueError('this witness instantiates the retained actual trip5/factor2 case')
    if loop['carries']!=[{'bits':8}] or len(loop['body'])!=2 or 'effect' not in loop['body'][0]:
        raise ValueError('concrete one-call/one-carry source body required')
    identity=loop['body'][0]['effect'];base=f'Merlin.GraphStatePipelineTailInstance.F{index}'
    if pair['registry'][identity]['inputs']!=[{'index':True},{'bits':8}]:
        raise ValueError('actual index/value observer signature required')
    q='import Merlin.GraphStatePipelineTailWitness\nimport Merlin.GraphStateIntegerOperations\n'+prepared.query
    q+='''namespace Merlin.GraphUnrollTailWitness
open Merlin.GraphState.Scalar Merlin.GraphState.Pipeline
open Merlin.GraphStatePipelineTailInstance
abbrev wordPrimitive := Merlin.GraphState.Integer.primitive
'''
    q+=f'''def chosen (failAt : Option Int) : External Nat Int registry where
  run := fun identity arguments world =>
    if identity = {identity} ∧ failAt = some (arguments[0]?.getD 0) then .error (17,world+1)
    else .ok ((signature registry identity).outputs.map
      (fun _ => if identity = {identity} then arguments[1]?.getD 0 else arguments.headD 0),world+1)
  arity := by
    intro identity arguments world values final success
    split at success
    · cases success
    · cases success; simp only [List.length_map]
def body (failAt : Option Int) (i : Int) (values : List Int) (world : Nat) : Outcome Int Nat Int :=
  evalProgram (M wordPrimitive (chosen failAt)) {base}.B2
    (i::values++{base}.E2 wordPrimitive 3 5) world
theorem body_actual (failAt : Option Int) (i : Int) (world : Nat) :
    body failAt i [3] world =
      if failAt = some i then .error (.external 17,world+1) else .ok ([3],world+1) := by
  simp [body,{base}.B2,{base}.E2,M,indexModel,indexEffect,controls,controlAt,
    chosen,registry,signature,opaqueStep,evalProgram,reads]
theorem source_success : evalProgram (M wordPrimitive (chosen none)) {base}.S2 [3,5] 7 = .ok ([3],13) := by
  have run := Tail.constant_carry_run 1 [3] (body none) rfl
    ((progression 0 1 5).map Int.ofNat)
    (fun i w _ => by simp only [body_actual,reduceCtorEq,↓reduceIte]) 7
  have counted : indexIterations 0 5 1 = (progression 0 1 5).map Int.ofNat :=
    indexIterations_count (by decide) (by decide)
  rw [{base}.form2,counted]
  change (loopEval 1 (body none) ((progression 0 1 5).map Int.ofNat) [3] 7).bind _ = _
  rw [run,List.length_map,Tail.progression_length]
  rfl
'''
    for label,ordinal in (('bulk_failure',1),('tail_failure',4)):
        q+=f'''theorem source_{label} :
    evalProgram (M wordPrimitive (chosen (some {ordinal}))) {base}.S2 [3,5] 7 =
      .error (.external 17,{8+ordinal}) := by
  have prefix : ∀ i w, i ∈ ((progression 0 1 {ordinal}).map Int.ofNat) →
      body (some {ordinal}) i [3] w = .ok ([3],w+1) := by
    intro i w member
    obtain ⟨n,inside,rfl⟩ := List.mem_map.mp member
    obtain ⟨k,bound,value⟩ := Tail.progression_member inside
    have different : (some ({ordinal} : Int)) ≠ some (Int.ofNat n) := by
      intro same
      have equal := Option.some.inj same
      omega
    simp only [body_actual,different,↓reduceIte]
  have failed := Tail.fault_after_prefix 1 [3] (body (some {ordinal})) rfl
    ((progression 0 1 {ordinal}).map Int.ofNat)
    ((progression {ordinal+1} 1 {4-ordinal}).map Int.ofNat) {ordinal} prefix (.external 17)
    (fun w => by simp only [body_actual,↓reduceIte]) 7
  have split : indexIterations 0 5 1 =
      (progression 0 1 {ordinal}).map Int.ofNat ++
        ({ordinal} : Int)::(progression {ordinal+1} 1 {4-ordinal}).map Int.ofNat := by
    rw [indexIterations_count (show 0 < (1 : Nat) by decide)
      (show Merlin.GraphLoopsUnroll.CountCorrect 0 5 1 ({ordinal}+{5-ordinal}) by decide),
      Tail.progression_append,List.map_append]
    rfl
  rw [{base}.form2,split]
  change (loopEval 1 (body (some {ordinal})) _ [3] 7).bind _ = _
  rw [failed,List.length_map,Tail.progression_length]
  rfl
'''
    for label,mode,result in (('success','none','.ok ([3],13)'),
                              ('bulk_failure','(some 1)','.error (.external 17,9)'),
                              ('tail_failure','(some 4)','.error (.external 17,12)')):
        q+=f'''theorem target_{label} : evalProgram (M wordPrimitive (chosen {mode})) {base}.S3 [3,5] 7 = {result} := by
  rw [← {base}.actual wordPrimitive (chosen {mode}) [3,5] 7 rfl]
  exact source_{label}
'''
        for side,stage in (('source',2),('target',3)):
            q+=f'''theorem {side}_{label}_initialized :
    Initialized (M wordPrimitive (chosen {mode})) {base}.S{stage} [3,5] 7 :=
  (check_operands_initialized (show check registry {base}.inputs {base}.outputs {base}.S{stage} {base}.S{stage} = true by decide)
    (M wordPrimitive (chosen {mode})) [3,5] 7 rfl).1
'''
    q+='end Merlin.GraphUnrollTailWitness\n'
    roots=tuple(dict.fromkeys((*prepared.roots,'Merlin.GraphState.Pipeline.Tail.constant_carry_run',
        'Merlin.GraphState.Pipeline.Tail.fault_after_prefix',*query_roots(q))))
    return q,roots,{'query_status':'UNCOMPILED','outcome':'UNOBSERVED','source_success_assumed':False,
        'initialized_is_not_poison_or_native_fidelity':True,'model_approved':False,
        'success_proposal':{'arguments':[3,5],'initial_world':7,'result':[3],'final_world':13},
        'bulk_failure_proposal':{'iv':1,'fault':17,'terminal_world':9},
        'tail_failure_proposal':{'iv':4,'fault':17,'terminal_world':12},
        'meaning':'actual step echoes i8 carry and advances world; optional failure keyed to actual IV; actual post-loop observer echoes and advances world',
        'no_trip_sized_literal_loop_evaluation':True,
        'additional_sources':[asdict(r) for r in project_lean_sources((
            'Merlin.GraphStatePipelineTailWitness','Merlin.GraphStateIntegerOperations'))]}

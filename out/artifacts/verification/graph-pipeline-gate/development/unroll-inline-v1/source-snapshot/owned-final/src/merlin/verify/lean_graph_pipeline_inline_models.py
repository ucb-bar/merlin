"""Bounded actual-program nonzero success/error/initialized-read candidates.

The selected inline profile has count <= factor <= 4. Literal evaluation below
is only for these bounded actual examples, never an unbounded loop proof.
"""
import json
from dataclasses import asdict
from .lean_graph_pipeline_gate_projection import project_lean_sources
from .lean_graph_pipeline_tail import query_roots


def witness(prepared):
    pair=json.loads(prepared.projection);facts=json.loads(prepared.field_facts)
    candidates=[(i,f) for i,f in enumerate(pair['functions']) if not f['declaration'] and any('loop' in r for r in f['before'])]
    if len(candidates)!=1 or len(facts)!=1:raise ValueError('one actual inline witness entry required')
    fi,f=candidates[0];fact=facts[0];count=fact['count'];lower,_,stride=fact['source_controls']
    if not 0<=count<=4 or f['inputs']!=[{'bits':8},{'bits':8}] or f['outputs']!=[{'bits':8}]:
        raise ValueError('bounded i8 witness profile required')
    loop=next(r['loop'] for r in f['before'] if 'loop' in r)
    effects=[r['effect'] for r in loop['body'] if 'effect' in r]
    observers=[r['effect'] for r in f['before'] if 'effect' in r and pair['registry'][r['effect']]['inputs']==[{'bits':8},{'bits':8}]]
    if len(effects)!=1 or len(observers)!=1:raise ValueError('one step and one final observer required')
    step,observer=effects[0],observers[0]
    base=f'Merlin.GraphStatePipelineInlineInstance.F{fi}'
    q='import Merlin.GraphStateIntegerOperations\n'+prepared.query
    q+='''namespace Merlin.GraphInlineWitness
open Merlin.GraphState.Scalar Merlin.GraphState.Pipeline
open Merlin.GraphStatePipelineInlineInstance
abbrev wordPrimitive := Merlin.GraphState.Integer.primitive
'''
    q+=f'''def chosen (failLane : Option Int) (failObserver : Bool) (failEveryBody : Bool) : External Nat Int registry where
  run := fun identity arguments world =>
    if (identity = {step} ∧ (failEveryBody = true ∨ failLane = some (arguments[0]?.getD 0))) ∨
        (identity = {observer} ∧ failObserver = true) then .error (17,world+1)
    else .ok ((signature registry identity).outputs.map
      (fun _ => if identity = {step} then arguments[1]?.getD 0 else arguments.headD 0),world+1)
  arity := by
    intro identity arguments world values final success
    split at success
    · cases success
    · cases success; simp only [List.length_map]
'''
    cases=[('success','none false false',f'.ok ([{3+2*count}],{8+count})'),
           ('observer_failure','none true false',f'.error (.external 17,{8+count})')]
    if count==0:
        cases.append(('empty_body_fault_skipped','none false true','.ok ([3],8)'))
    else:
        cases.extend([('first_body_failure',f'(some {lower}) false false','.error (.external 17,8)'),
                      ('last_body_failure',f'(some {lower+(count-1)*stride}) false false',f'.error (.external 17,{7+count})')])
    for name,mode,result in cases:
        q+=f'theorem source_{name} : evalProgram (M wordPrimitive (chosen {mode})) {base}.S0 [3,2] 7 = {result} := rfl\n'
        q+=f'''theorem target_{name} : evalProgram (M wordPrimitive (chosen {mode})) {base}.S1 [3,2] 7 = {result} := by
  rw [← {base}.actual wordPrimitive (chosen {mode}) [3,2] 7 rfl]
  exact source_{name}
'''
        for side,stage in (('source',0),('target',1)):
            q+=f'''theorem {side}_{name}_initialized :
    Initialized (M wordPrimitive (chosen {mode})) {base}.S{stage} [3,2] 7 :=
  (check_operands_initialized (show check registry {base}.inputs {base}.outputs {base}.S{stage} {base}.S{stage} = true by decide)
    (M wordPrimitive (chosen {mode})) [3,2] 7 rfl).1
'''
    q+='end Merlin.GraphInlineWitness\n'
    roots=tuple(dict.fromkeys((*prepared.roots,*query_roots(q))))
    return q,roots,{'query_status':'UNCOMPILED','outcome':'UNOBSERVED','source_success_assumed':False,
        'model_approved':False,'native_or_poison_claim':False,'inputs':[3,2],'initial_world':7,
        'count':count,'case_outcomes':[{'name':n,'provider':m,'proposed_outcome':r} for n,m,r in cases],
        'model':'existing wrapping word primitive; actual step echoes incremented carry, advances world; final observer echoes/advances or fails',
        'evaluation_bound':'count <= selected factor <= 4; no unbounded trip-sized witness evaluation',
        'additional_sources':[asdict(r) for r in project_lean_sources(('Merlin.GraphStateIntegerOperations',))]}


def empty_countermodel(projection):
    """Actual valid erroneous empty-body execution; source-only countermodel."""
    from .lean_graph_state_scalar import program,types
    rows=[f for f in projection['functions'] if not f['declaration'] and any('loop' in r for r in f['before'])]
    if len(rows)!=1:raise ValueError('one original empty source entry required')
    f=rows[0];loop=next(r['loop'] for r in f['before'] if 'loop' in r)
    step=next(r['effect'] for r in loop['body'] if 'effect' in r)
    registry='['+','.join('{inputs := '+types(r['inputs'])+',outputs := '+types(r['outputs'])+'}' for r in projection['registry'])+']'
    q='import Merlin.GraphStateIntegerOperations\nimport Merlin.GraphStatePipelineControl\nnamespace Merlin.GraphInlineEmptyCountermodel\nopen Merlin.GraphState.Scalar Merlin.GraphState.Pipeline\n'
    q+=f'def registry : List Signature := {registry}\ndef source : Program := {program(f["before"])}\ndef target : Program := {program(f["after"])}\n'
    # The actual empty capture is lb=ub=3. In this witness all actual index
    # constants are3 or2. Derive their values from the registry, not a fake loop.
    constants={i:r['index_constant'] for i,r in enumerate(projection['registry']) if r['index_constant'] is not None}
    q+='def constantAt (identity : Nat) : Option Int :=\n'
    for i,value in constants.items():q+=f'  if identity = {i} then some {value} else\n'
    q+='  none\n'
    q+=f'''def model : Model Int Nat Int registry where
  default := 0
  primitive := Merlin.GraphState.Integer.primitive
  index := id
  iterations := indexIterations
  effect := fun identity arguments world =>
    if identity = {step} then .error (17,world+1)
    else .ok ((signature registry identity).outputs.map
      (fun _ => (constantAt identity).getD (arguments.headD 0)),
      if (constantAt identity).isSome then world else world+1)
  arity := by
    intro identity arguments world values final success
    split at success
    · cases success
    · cases success; simp only [List.length_map]
theorem source_skips_body : evalProgram model source [3,2] 7 = .ok ([3],8) := rfl
theorem target_executes_body : evalProgram model target [3,2] 7 = .error (.external 17,8) := rfl
theorem observations_differ : evalProgram model source [3,2] 7 ≠ evalProgram model target [3,2] 7 := by
  rw [source_skips_body,target_executes_body]
  intro equal
  cases equal
end Merlin.GraphInlineEmptyCountermodel
'''
    return q,query_roots(q)

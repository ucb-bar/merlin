"""Actual-syntax success/error/initialized-read candidates, never authority."""
import json
from dataclasses import asdict
from .lean_graph_pipeline_gate_projection import project_lean_sources


def witness(prepared):
    p=json.loads(prepared.projection)
    bits={'bits':8}
    functions=[(i,f) for i,f in enumerate(p['functions']) if not f['declaration']
               and f['inputs']==[bits,bits] and f['outputs']==[bits]]
    if len(functions)!=1 or any(r['inputs']!=[bits] or r['outputs']!=[bits] for r in p['registry']):
        raise ValueError("concrete i8 two-observer witness domain required")
    index,f=functions[0]
    effects=[r['effect'] for r in f['before'] if 'effect' in r]
    if len(effects)!=2 or effects[0]==effects[1] or any('loop' in r for r in f['before']):
        raise ValueError("two distinct ordered opaque observers and region-free source required")
    query='import Merlin.GraphStateIntegerOperations\n'+prepared.query
    query+='''namespace Merlin.GraphScheduleWitness
open Merlin.GraphState.Scalar
abbrev registry := Merlin.GraphStateScalarInstance.registry
def chosen (failAt : Option Nat) : Model Int Nat Int registry where
  default := 0
  primitive := Merlin.GraphState.Integer.primitive
  index := id
  iterations := fun _ _ _ => []
  effect := fun identity arguments world =>
    if failAt = some identity then .error (17,world+1)
    else .ok ((signature registry identity).outputs.map (fun _ => arguments.headD 0),world+1)
  arity := by
    intro identity arguments world values final success
    split at success
    · cases success
    · cases success; simp only [List.length_map]
'''
    roots=list(prepared.roots)
    base=f'Merlin.GraphStateScalarInstance.F{index}'
    cases=(('success','[3,5]','none','.ok ([21],9)'),
           ('success_other','[4,5]','none','.ok ([28],9)'),
           ('first_failure','[3,5]',f'(some {effects[0]})','.error (.external 17,8)'),
           ('second_failure','[3,5]',f'(some {effects[1]})','.error (.external 17,9)'))
    for label,args,mode,result in cases:
        # Literal source execution must reduce from ACTUAL Program terms. These
        # are proposed outcomes, not a source-success/equality hypothesis.
        query+=f'theorem source_{label} : evalProgram (chosen {mode}) {base}.before {args} 7 = {result} := rfl\n'
        query+=f'theorem target_{label} : evalProgram (chosen {mode}) {base}.after {args} 7 = {result} := by\n'
        query+=f'  rw [← {base}.actual Int Nat Int (chosen {mode}) {args} 7 rfl]\n  exact source_{label}\n'
        for side,raw,n in (('source','before',1),('target','after',2)):
            query+=f'theorem {side}_{label}_initialized : Initialized (chosen {mode}) {base}.{raw} {args} 7 :=\n'
            query+=f'  ({base}.operands_initialized (chosen {mode}) {args} 7 rfl).{n}\n'
        roots += ['Merlin.GraphScheduleWitness.'+name for name in (
            'source_'+label,'target_'+label,'source_'+label+'_initialized','target_'+label+'_initialized')]
    query+='end Merlin.GraphScheduleWitness\n'
    return query,tuple(roots),{'state':'UNCOMPILED','outcome':'UNOBSERVED','model_approved':False,
        'success_proposals':[{'input':[3,5],'output':[21],'world':9},{'input':[4,5],'output':[28],'world':9}],
        'failure_proposals':[{'observer':0,'error':17,'world':8},{'observer':1,'error':17,'world':9}],
        'source_success_assumed':False,'opaque_order':effects,
        'meaning':'existing exact wrapping primitive; each actual unary observer echoes operand and increments world or fails',
        'initialized_is_not_poison_or_native_fidelity':True,
        'additional_sources':[asdict(r) for r in project_lean_sources(('Merlin.GraphStateIntegerOperations',))]}


def observer_countermodel(projection):
    """Concrete observable difference for the retained valid bad permutation.

    This does not assert an accepted schedule. Both programs are actual parsed
    terms; the SAME frozen evaluator computes their distinct failure worlds.
    """
    from .lean_graph_pipeline_gate_emit import program,types
    bits={'bits':8}
    rows=[f for f in projection['functions'] if not f['declaration'] and f['inputs']==[bits,bits] and f['outputs']==[bits]]
    if len(rows)!=1:raise ValueError('one actual two-input countermodel entry required')
    f=rows[0];effects=[r['effect'] for r in f['before'] if 'effect' in r]
    if len(effects)!=2 or effects[0]==effects[1]:raise ValueError('two distinct observer descriptors required')
    registry='['+','.join('{inputs := '+types(r['inputs'])+',outputs := '+types(r['outputs'])+'}' for r in projection['registry'])+']'
    text='import Merlin.GraphStateIntegerOperations\nnamespace Merlin.GraphScheduleObserverCountermodel\nopen Merlin.GraphState.Scalar\n'
    text+='def registry : List Signature := '+registry+'\n'
    text+='def before : Program := '+program(f['before'])+'\n'
    text+='def after : Program := '+program(f['after'])+'\n'
    text+=f'''def chosen : Model Int Nat Int registry where
  default := 0
  primitive := Merlin.GraphState.Integer.primitive
  index := id
  iterations := fun _ _ _ => []
  effect := fun identity arguments world =>
    if identity = {effects[0]} then .error (17,world+1)
    else .ok ((signature registry identity).outputs.map (fun _ => arguments.headD 0),world+1)
  arity := by
    intro identity arguments world values final success
    split at success
    · cases success
    · cases success; simp only [List.length_map]
theorem source_failure : evalProgram chosen before [3,5] 7 = .error (.external 17,8) := rfl
theorem reordered_failure : evalProgram chosen after [3,5] 7 = .error (.external 17,9) := rfl
theorem terminal_world_differs : evalProgram chosen before [3,5] 7 ≠ evalProgram chosen after [3,5] 7 := by
  rw [source_failure,reordered_failure]
  decide
end Merlin.GraphScheduleObserverCountermodel
'''
    return text,tuple('Merlin.GraphScheduleObserverCountermodel.'+n for n in ('source_failure','reordered_failure','terminal_world_differs'))

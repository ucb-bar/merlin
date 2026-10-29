from dataclasses import replace
import pytest
from merlin.sched.ir import Call,Const,Var,Kernel,Loop,evaluate
from merlin.compile.scheduling.configuration import ConfigurationAdapter,ConfigurationEffects
from merlin.compile.scheduling.configuration_loop_cleanup import eliminate_counted_configuration,verify_counted_configuration_rewrite

def setmode(value):return Call('set',(('value',Const(value)),))
def effects(call):
    if call.instr=='set':return ConfigurationEffects(writes=(('mode',call.arg('value').value),),ordered=True,pure_write=True)
    if call.instr=='read':return ConfigurationEffects(ordered=True)
    if call.instr=='drain':return ConfigurationEffects(ordered=True)
    return None
A=ConfigurationAdapter(effects,'neutral-registry','test-functional-only')

def run(kernel):
    state=-1;seen=[];setters=0
    def body(items,env):
        nonlocal state,setters
        for call in items:
            if type(call)is Loop:
                for index in range(call.extent):body(call.body,{**env,call.var:index})
            elif call.instr=='set':state=call.arg('value').value;setters+=1
            elif call.instr=='read':seen.append((state,evaluate(call.arg('index'),env)))
            elif call.instr=='unknown':state=-1
    body(kernel.body,{})
    return seen,state,setters

def test_intact_nested_loop_indices_and_complete_observations():
    k=Kernel('nested',(),(setmode(7),Loop('i',17,(setmode(7),Call('read',(('index',Var('i')),)),Call('drain',()),setmode(7))),))
    r=eliminate_counted_configuration(k,A)
    assert run(k)[:2]==run(r.kernel)[:2]
    assert run(k)[2]==35 and run(r.kernel)[2]==1
    assert r.report['removed_dynamic_calls']==34

def test_entry_state_not_assumed_for_later_iterations_when_body_changes_it():
    k=Kernel('clobber',(),(setmode(7),Loop('i',3,(setmode(7),Call('read',(('index',Var('i')),)),setmode(9))),))
    r=eliminate_counted_configuration(k,A)
    assert run(k)==run(r.kernel)
    illegal=replace(k,body=(k.body[0],replace(k.body[1],body=k.body[1].body[1:])))
    with pytest.raises(ValueError,match='lacks independently established'):verify_counted_configuration_rewrite(k,illegal,A)

def test_unknown_barrier_requires_new_setter_and_data_calls_cannot_disappear():
    k=Kernel('barrier',(),(setmode(7),Call('unknown',()),setmode(7),Call('read',(('index',Const(0)),))))
    r=eliminate_counted_configuration(k,A)
    assert run(k)==run(r.kernel)
    with pytest.raises(ValueError,match='nonconfiguration'):verify_counted_configuration_rewrite(k,replace(k,body=k.body[:-1]),A)

def test_huge_counted_loop_stays_counted_without_execution():
    k=Kernel('large',(),(setmode(7),Loop('i',10**12,(setmode(7),Call('read',(('index',Var('i')),)))),))
    r=eliminate_counted_configuration(k,A)
    assert type(r.kernel.body[1])is Loop and r.report['removed_dynamic_calls']==10**12

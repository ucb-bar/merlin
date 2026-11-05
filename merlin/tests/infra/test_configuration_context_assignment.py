import pytest
from merlin.compile.scheduling.configuration_contexts import ConfigurationRequest,assign_configuration_contexts

def request(name,pitch):
    return ConfigurationRequest(name,(("stride",pitch),("scale",17),("width",8)))

def test_shares_equal_complete_requests_without_burning_contexts():
    p=assign_configuration_contexts((request("left",512),request("right",512)),(7,11))
    assert p.context("left")==p.context("right")==7
    assert len(p.states)==1

def test_assigns_independent_states_with_original_field_values():
    p=assign_configuration_contexts((request("left",33),request("right",19),request("bias",0)),(9,5,2))
    assert p.bindings==(("left",9),("right",5),("bias",2))
    assert dict(p.states[1][1])["stride"]==19

def test_refuses_insufficient_contexts_or_varying_static_consumer():
    assert assign_configuration_contexts((request("a",3),request("b",5)),(0,))is None
    assert assign_configuration_contexts((request("a",3),request("a",5)),(0,1))is None

def test_complete_state_not_only_stride_controls_identity():
    x=ConfigurationRequest("b",(("stride",33),("scale",18),("width",8)))
    p=assign_configuration_contexts((request("a",33),x),(0,1))
    assert p.context("a")!=p.context("b")

@pytest.mark.parametrize("contexts",[(1,1),(-1,),(False,),()])
def test_rejects_ambiguous_or_unprovided_contexts(contexts):
    with pytest.raises(ValueError):assign_configuration_contexts((request("a",3),),contexts)

def test_budgets_and_repeated_source_requests():
    p=assign_configuration_contexts((request("a",3),request("a",3)),(0,1))
    assert len(p.bindings)==len(p.states)==1
    with pytest.raises(ValueError):assign_configuration_contexts((request("a",3),),(0,),max_fields=2)
    with pytest.raises(ValueError):assign_configuration_contexts((request("a",3),request("b",4)),(0,1),max_requests=1)

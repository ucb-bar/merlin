"""Private representations retain public types, ordering and ownership guards."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr, bf16

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects.lowering.private_transport import (
    PrivateTensorTransport,
    private_call_edges,
    prune_unobserved_transport_views,
    rewrite_private_transports,
)


def module(*, barrier="", live=False, repeated=False, index=0):
    arguments = "%v,%v" if repeated else "%v,%b" if index == 0 else "%b,%v"
    returns = "(tensor<2x16xf32>,tensor<1x2x16xf32>)" if live else "tensor<1x2x16xf32>"
    values = "%p,%r" if live else "%r"
    result = parse_mlir_text(f"""module {{
      func.func private @effect()
      func.func private @produce(tensor<2x16xf32>,tensor<2x16xf32>)->tensor<2x16xf32>
      func.func private @consume(tensor<1x2x16xf32>,tensor<1x2x16xf32>,tensor<1x2x16xf32>)
        ->tensor<1x2x16xf32>
      func.func @forward(%a:tensor<2x16xf32>,%b:tensor<1x2x16xf32>)->{returns} {{
        %e=tensor.empty():tensor<2x16xf32>
        %p=func.call @produce(%a,%e):(tensor<2x16xf32>,tensor<2x16xf32>)->tensor<2x16xf32>
        {barrier}
        %f=tensor.collapse_shape %p [[0,1]]:tensor<2x16xf32> into tensor<32xf32>
        %v=tensor.expand_shape %f [[0,1,2]] output_shape [1,2,16]
          :tensor<32xf32> into tensor<1x2x16xf32>
        %d=tensor.empty():tensor<1x2x16xf32>
        %r=func.call @consume({arguments},%d)
          :(tensor<1x2x16xf32>,tensor<1x2x16xf32>,tensor<1x2x16xf32>)->tensor<1x2x16xf32>
        func.return {values}:{returns.removeprefix("(").removesuffix(")")}
      }}
    }}""")
    for function in result.body.block.ops:
        if function.name == "func.func" and not function.body.blocks and function.sym_name.data != "effect":
            count = len(function.function_type.inputs)
            function.properties["arg_attrs"] = ArrayAttr(
                [
                    DictionaryAttr({"bufferization.access": StringAttr("write" if i == count - 1 else "read")})
                    for i in range(count)
                ]
            )
    return result


def select(producer, consumer, index):
    if producer.callee.root_reference.data == "produce" and consumer.callee.root_reference.data == "consume":
        return PrivateTensorTransport("produce_compact", "consume_compact", bf16)
    return None


@pytest.mark.parametrize("index", [0, 1])
def test_private_views_change_only_transport_and_preserve_original_module(index):
    original = module(index=index)
    before = str(original)
    result, records = rewrite_private_transports(original, select)
    assert str(original) == before
    assert len(records) == 1 and records[0]["consumer_input"] == index
    calls = [o for o in result.walk() if o.name == "func.call"]
    assert [o.callee.root_reference.data for o in calls] == ["produce_compact", "consume_compact"]
    assert str(calls[0].results[0].type) == "tensor<2x16xbf16>"
    assert str(calls[1].arguments[index].type) == "tensor<1x2x16xbf16>"
    assert str(calls[1].results[0].type) == "tensor<1x2x16xf32>"
    assert records[0]["private_views"] == 2
    result.verify()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"barrier": "func.call @effect():()->()"},
        {"live": True},
        {"repeated": True},
    ],
)
def test_observed_fanout_and_unknown_effects_keep_original_transport(kwargs):
    original = module(**kwargs)
    result, records = rewrite_private_transports(original, select)
    assert not records and str(result) == str(original)


def test_mutable_external_input_is_not_a_private_transport_contract():
    original = module()
    declaration = next(o for o in original.body.block.ops if o.name == "func.func" and o.sym_name.data == "consume")
    declaration.properties.pop("arg_attrs")
    assert not private_call_edges(original)


def test_unselected_legal_edge_does_not_change():
    original = module()
    result, records = rewrite_private_transports(original, lambda *_: None)
    assert not records and str(result) == str(original)


def test_a_selector_cannot_bypass_privacy_or_effect_checks():
    def unexpected(*_):
        raise AssertionError("illegal edge reached provider selection")

    for kwargs in ({"live": True}, {"repeated": True}, {"barrier": "func.call @effect():()->()"}):
        _, records = rewrite_private_transports(module(**kwargs), unexpected)
        assert not records


def test_symbol_collision_refuses_without_mutating_input():
    original = module()
    before = str(original)

    def collision(*_):
        return PrivateTensorTransport("produce", "consume_compact", bf16)

    with pytest.raises(ValueError, match="collides"):
        rewrite_private_transports(original, collision)
    assert str(original) == before


def test_unused_static_view_branches_can_be_pruned_before_private_edge_selection():
    original = module(
        barrier="""
        %dead0=tensor.collapse_shape %p [[0,1]]:tensor<2x16xf32> into tensor<32xf32>
        %dead1=tensor.expand_shape %dead0 [[0,1]] output_shape [2,16]
          :tensor<32xf32> into tensor<2x16xf32>
        %dead_constant=arith.constant 2.0:f32
    """
    )
    before = str(original)
    assert not private_call_edges(original)
    cleaned, removed = prune_unobserved_transport_views(original)
    assert str(original) == before and removed == 2
    assert any(op.name == "arith.constant" for op in cleaned.walk())
    transformed, records = rewrite_private_transports(cleaned, select)
    assert len(records) == 1
    transformed.verify()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"live": True},
        {"repeated": True},
        {"barrier": "func.call @effect():()->()"},
        {"barrier": "%dead=tensor.cast %p:tensor<2x16xf32> to tensor<?x16xf32>"},
    ],
)
def test_view_cleanup_never_hides_observation_unknown_views_or_effect_barriers(kwargs):
    original = module(**kwargs)
    before = str(original)
    cleaned, removed = prune_unobserved_transport_views(original)
    assert not removed and str(cleaned) == before
    assert not rewrite_private_transports(cleaned, select)[1]

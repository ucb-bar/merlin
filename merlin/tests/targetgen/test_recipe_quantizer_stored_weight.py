"""Named owners retain stored-weight identity and source geometry through selection."""

import pytest

from merlin.targetgen import _recipe_quantizer as quant


@pytest.fixture
def selected_weight_model():
    import torch

    class Linear(torch.nn.Linear):
        def forward(self, value):
            selected = self.weight.reshape(1, 3, 4)[0]
            return value @ selected.t()

    model = torch.nn.Sequential()
    model.add_module("projection", Linear(4, 3, bias=False))
    return model.eval(), (torch.ones(2, 4),)


@pytest.fixture
def static_recipe():
    return {
        "schema": "quant_recipe_v1",
        "status": "derived",
        "families": ["contraction"],
        "weight": {"dtype": "int8", "granularity": "tensor", "symmetric": True, "quant_min": -127, "quant_max": 127},
        "activation": {
            "dtype": "int8",
            "granularity": "tensor",
            "symmetric": True,
            "quant_min": -128,
            "quant_max": 127,
            "mode": "static",
        },
    }


@pytest.mark.parametrize("matching_geometry", [True, False])
def test_named_owner_selection_shares_weight_identity_and_geometry(
    selected_weight_model, static_recipe, matching_geometry
):
    import torch

    model, inputs = selected_weight_model
    inventory = quant.layer_inventory(model)
    assert inventory[0]["fqn"] == "projection"
    assert inventory[0]["weight_shape"] == [3, 4]
    if not matching_geometry:
        inventory[0]["weight_shape"] = [4, 3]
    graph = torch.export.export(model, inputs).module()
    quantizer = quant.build_quantizer(
        static_recipe, layer_plan=quant._plan_layers(static_recipe, model), source_layer_inventory=inventory
    )
    graph = quantizer.transform_for_annotation(graph)
    quantizer.annotate(graph)
    assert quantizer.annotated == int(matching_geometry)
    if matching_geometry:
        contraction = next(node for node in graph.graph.nodes if node.target == torch.ops.aten.matmul.default)
        annotation = contraction.meta["quantization_annotation"]
        assert annotation.input_qspec_map[contraction.args[1]] is quantizer.weight
    else:
        assert quantizer.refused_by_placement

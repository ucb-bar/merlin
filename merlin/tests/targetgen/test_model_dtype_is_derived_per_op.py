"""A whole-model capture is routed under ONE declared dtype, and some captures are mixed.

This file replaces a ratchet. The ratchet pinned that ``model_op_demands`` stamped the caller's single
``in_fmt`` on every demand, and said why that was survivable *only* for as long as a captured
``linalg.generic`` contraction carried no extent: give it one and a tile WOULD be synthesized, at the
model's declared dtype, which for both captures below is the wrong one for exactly that op. The extent
reader now understands those generics (see ``test_generic_contraction_extents``), so the dtype had to be
derived first, and this states what was derived.

Two facts, not one:

* ``in_fmt`` is what the compile was ASKED for. It stays the caller's single declared format on every
  demand, because that is the question routing answers -- a capture is routed under the datapath the
  compiler will lower it to, and making legality follow the captured element type instead empties the
  mesh of every f32 capture on an int8 target (``M1_lstmnetvit_gemmini``: 47 mesh ops -> 0).
* ``elem_fmt`` is what the op's own operands ARE. It is per op, canonical, and UNKNOWN (``None``) when
  it cannot be read -- and ``tile_fmt`` falls back to ``in_fmt`` there, so unknown never widens.

The corpus is mixed because torchAO quantizes Linear weights and leaves Conv2d alone:

* ``M2_microvit_gemmini`` -- 12 ``linalg.matmul`` ops in **i8**, one reducing contraction-tagged
  ``linalg.generic`` in **f32**.
* ``SY_model_resnet50`` -- 53 ``linalg.matmul`` in **f32**, one reducing generic in **i8**.
"""

from __future__ import annotations

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen.capsule_source import linalg_summary, model_op_demands


def _capsule(name: str) -> str:
    path = next(merlin_dir().joinpath("contract/capsules").rglob(f"{name}/capsule.interface.mlir"), None)
    if path is None:
        pytest.skip(f"{name} is not in this checkout")
    return path.read_text(encoding="utf-8")


def _contraction_formats(text: str) -> dict[str, set[str | None]]:
    """``elem_fmt`` of each contraction demand, grouped by the operation that carries it."""
    summary = linalg_summary(text)
    carriers = [c for c, op in zip(summary["carrying_ops"], summary["prov_ops"]) if op != "fill"]
    out: dict[str, set[str | None]] = {}
    for carrier, demand in zip(carriers, model_op_demands(text, "int8"), strict=True):
        if demand.family == "contraction":
            out.setdefault(carrier, set()).add(demand.elem_fmt)
    return out


@pytest.mark.parametrize(
    "name,named,generic",
    [("M2_microvit_gemmini", "int8", "fp32"), ("SY_model_resnet50", "fp32", "int8")],
)
def test_a_mixed_capture_gives_each_contraction_its_own_format(name, named, generic):
    by_carrier = _contraction_formats(_capsule(name))
    assert by_carrier["linalg.matmul"] == {named}
    assert by_carrier["linalg.generic"] == {generic}


def test_the_declared_format_still_reaches_every_demand():
    """Routing legality is about the format the compile was asked for, so ``in_fmt`` must not follow the
    capture. It used to be the ONLY format on a demand; it is still the one the router reads."""
    text = _capsule("M2_microvit_gemmini")
    for declared in ("int8", "bf16"):
        assert {d.in_fmt for d in model_op_demands(text, declared)} == {declared}


def test_making_legality_follow_the_captured_format_would_empty_the_mesh():
    """Why ``elem_fmt`` is a second fact rather than a replacement for ``in_fmt``.

    ``M1_lstmnetvit_gemmini`` is captured entirely in f32 and declared ``compile_dtype: int8``; the
    gemmini mesh declares int8 alone. Routing on the captured format would leave that capsule -- whose
    point is mesh execution -- with nothing on the mesh at all.
    """
    routing = pytest.importorskip("merlin.targetgen.routing")
    try:
        text = _capsule("M1_lstmnetvit_gemmini")
        demands = model_op_demands(text, "int8")
        mesh = routing.route_plan(demands, "gemmini").get("mesh") or []
    except Exception as exc:  # noqa: BLE001 -- an unresolvable contract is not a verdict
        pytest.skip(f"gemmini contract not resolvable here: {exc}")
    assert mesh, "the declared format is what keeps this capsule's contractions on the mesh"
    assert {d.demand.elem_fmt for d in mesh} == {"fp32"}, "and every one of them is captured in f32"
    assert all(d.demand.in_fmt == "int8" for d in mesh)


def test_an_unreadable_format_keeps_the_declared_one():
    """UNKNOWN never widens. An op whose operands are not one registry format gets ``elem_fmt is None``
    and therefore the declared format for its tile, exactly as before this field existed."""
    text = (
        "module {\n"
        '  %0 = linalg.generic {indexing_maps = [], iterator_types = ["parallel"]} '
        "ins(%a, %b : tensor<4xf32>, tensor<4xi1>) outs(%c : tensor<4xf32>) "
        'attrs = {prov.op = "select", prov.family = "elementwise_map"} { ^bb0: }\n'
        '  %1 = linalg.generic {indexing_maps = [], iterator_types = ["parallel"]} '
        "ins(%d : tensor<4xi32>) outs(%e : tensor<4xi32>) "
        'attrs = {prov.op = "index_shuffle", prov.family = "elementwise_map"} { ^bb0: }\n'
        "}\n"
    )
    demands = {d.op: d for d in model_op_demands(text, "bf16")}
    # operands that disagree with each other have no single format...
    assert demands["select"].elem_fmt is None
    # ...and neither does an element type the format registry does not carry (i32 is not an operand format)
    assert demands["index_shuffle"].elem_fmt is None
    for d in demands.values():
        assert d.tile_fmt == "bf16"


def test_the_format_is_canonical_not_the_mlir_spelling():
    """``i8`` in the IR and ``int8`` in a contract are ONE format; resolving through the registry is what
    keeps the tile builder and the router from disagreeing about which."""
    text = (
        "module {\n"
        '  %0 = linalg.matmul {prov.op = "matmul", prov.family = "contraction"} '
        "ins(%a, %b : tensor<8x16xi8>, tensor<16x32xi8>) outs(%c : tensor<8x32xi32>) -> tensor<8x32xi32>\n"
        "}\n"
    )
    (demand,) = model_op_demands(text, "int8")
    assert demand.elem_fmt == "int8" and demand.tile_fmt == "int8"


def test_the_mesh_tile_is_synthesized_in_the_ops_own_format():
    """The consumer half. A tile built at the model's declared format certifies arithmetic the op does
    not perform -- which is only reachable now that such an op carries an extent at all."""
    import inspect

    from merlin.compile import mesh as MESH

    source = inspect.getsource(MESH._mesh_verify)
    assert "_mesh_tile_binding(target, d.tile_fmt, r.acc)" in source, (
        "the mesh tile must be bound to the op's own element format (falling back to the declared one "
        "when it is unknown), not to the model-level format every demand carries"
    )

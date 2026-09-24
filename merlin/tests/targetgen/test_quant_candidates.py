"""Multi-format candidate derivation does not borrow one format's readout for another."""

from __future__ import annotations

import json

from merlin.targetgen import quant_recipe as QR
from merlin.targetgen import readout_facet as RF


def _facts(element: str) -> dict:
    return {
        "facts": {
            "datapaths": [
                {"name": "input", "dtype": element, "evidence": "test datapath"},
                {"name": "accumulator", "dtype": "i32", "evidence": "test accumulator"},
            ],
            "memories": [{"name": "accumulator", "bytes": 4096}],
            "interfaces": [
                {
                    "name": "register_bundle_layouts",
                    "unresolved": {},
                    "bundles": {
                        "StoreConfig": {
                            "width": 64,
                            "fields": {"out_scale": {"offset": 32, "width": None, "slot_width": 32}},
                        }
                    },
                }
            ],
        }
    }


def _facet(unit: str, dtypes: list[str], element: str) -> RF.ReadoutFacet:
    return RF.derive(
        "t",
        facts=_facts(element),
        unit={"name": unit, "dtypes": dtypes},
        scalar_abi={
            "schema": RF.SCALAR_ABI_SCHEMA,
            "accumulator_dtype": "i32",
            "output_dtype": element,
            "scale_dtype": "f32",
            "clamp_min": -128,
            "clamp_max": 127,
            "provenance": {"scope": "test"},
        },
        readouts=[{"selector": element, "applies": ["acc_scale"]}],
    )


def _unit(name: str, formats: list[str], *, cap_formats: list[str] | None = None) -> dict:
    return {
        "name": name,
        "kind": "systolic",
        "dtypes": formats,
        "ops": ["matmul"],
        "semantic_capabilities": [
            {"family": "contraction", "dtypes": cap_formats if cap_formats is not None else formats}
        ],
    }


def test_declared_alternatives_are_inventoried_but_not_licensed_by_one_readout() -> None:
    contract = {"compute_units": [_unit("u", ["int8", "fp32"])]}
    facet = _facet("u", ["int8", "fp32"], "i8")
    candidates = QR.derive_candidates(contract, [facet])
    assert [(c.unit, c.format, c.status) for c in candidates] == [
        ("u", "int8", QR.DERIVED),
        ("u", "fp32", QR.UNDERIVABLE),
    ]
    assert candidates[0].recipe is not None and candidates[0].recipe.weight.dtype == "int8"
    assert "format-specific" in candidates[1].underivable["element_dtype"]
    assert candidates[1].to_dict()["framework_route"] == "not_evaluated"
    assert any(e["rung"] == "rtl_datapath" for e in candidates[0].to_dict()["evidence"])
    json.dumps([c.to_dict() for c in candidates])
    # The new inventory has no effect on the existing single-recipe capture selector.
    assert QR.select([QR.derive(facet)]).weight.dtype == "int8"


def test_distinct_units_can_independently_derive_distinct_formats() -> None:
    contract = {"compute_units": [_unit("integer", ["int8"]), _unit("float", ["fp32"])]}
    candidates = QR.derive_candidates(
        contract,
        [_facet("integer", ["int8"], "i8"), _facet("float", ["fp32"], "f32")],
    )
    assert [(c.format, c.status) for c in candidates] == [("int8", QR.DERIVED), ("fp32", QR.DERIVED)]
    assert candidates[1].recipe.weight.dtype == "fp32"


def test_format_rung_from_other_block_format_cannot_license_int8() -> None:
    contract = {"compute_units": [_unit("u", ["int8", "mxfp4"])]}
    facet = _facet("u", ["int8", "mxfp4"], "i8")
    candidates = QR.derive_candidates(contract, [facet])
    assert [c.status for c in candidates] == [QR.UNDERIVABLE, QR.UNDERIVABLE]
    assert "different declared format" in candidates[0].underivable["scale_granularities"]
    assert "format-specific" in candidates[1].underivable["element_dtype"]


def test_accumulate_and_semantic_capabilities_must_admit_candidate_format() -> None:
    contract = {
        "compute_units": [
            {
                **_unit("u", ["int8", "fp32"], cap_formats=["int8"]),
                "accumulate": [{"in": "int8", "weight": "fp32", "acc": "i32"}],
            }
        ]
    }
    candidates = QR.derive_candidates(contract, [_facet("u", ["int8", "fp32"], "i8")])
    assert "no int8 x int8 pairing" in candidates[0].underivable["accumulate"]
    assert "no standalone contraction" in candidates[1].underivable["contraction"]
    assert all(c.recipe is None for c in candidates)


def test_mismatched_scalar_abi_or_missing_facet_refuses_even_declared_format() -> None:
    facet = _facet("u", ["int8"], "i8")
    facet.scalar_abi["output_dtype"] = "f32"
    contract = {"compute_units": [_unit("u", ["int8"]), _unit("other", ["fp32"])]}
    first, second = QR.derive_candidates(contract, [facet])
    assert "does not prove" in first.underivable["scalar_abi"]
    assert "no readout facet" in second.underivable["readout"]
    assert (first.status, second.status) == (QR.UNDERIVABLE, QR.UNDERIVABLE)

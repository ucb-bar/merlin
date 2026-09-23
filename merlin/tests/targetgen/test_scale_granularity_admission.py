"""A scaled region is admitted only at a granularity the target's readout was derived to hold."""

from __future__ import annotations

from merlin.targetgen import readout_facet as RF

_SCALAR_SCALE = {
    "facts": {
        "interfaces": [
            {
                "name": "register_bundle_layouts",
                "unresolved": {},
                "bundles": {
                    "StoreConfig": {
                        "width": 64,
                        "fields": {
                            "out_scale": {"offset": 32, "width": None, "width_param": "scale_bits", "slot_width": 32}
                        },
                    }
                },
            }
        ]
    }
}


def _facts() -> dict:
    return _SCALAR_SCALE


def test_a_requantization_is_admitted_only_at_a_granularity_the_readout_holds() -> None:
    from merlin.targetgen import eligibility as E
    from merlin.targetgen.compute_units import SemanticCapability

    cap_map = {"elementwise_map": SemanticCapability(family="elementwise_map", dtypes=("int8",))}
    held = RF.TargetReadout((RF.derive("t", facts=_facts()),))
    unread = RF.TargetReadout((RF.derive("t", facts={}),))

    def verdict(granularity, readout):
        region = E.RegionDescriptor(
            op="requant", family="elementwise_map", in_dtype="int8", scale_granularity=granularity
        )
        return E.is_eligible(region, cap_map, readout=readout)

    assert verdict("tensor", held).eligible
    refused = verdict("column", held)
    assert (refused.eligible, refused.refusal, refused.undetermined) == (False, "scale_granularity", False)
    unknown = verdict("column", unread)
    assert (unknown.eligible, unknown.refusal, unknown.undetermined) == (False, "scale_granularity_unknown", True)
    # A region that states no granularity, or a caller with no facet, is decided as before.
    assert verdict(None, held).eligible and verdict("column", None).eligible
    # One unit that holds it is enough; an unread unit beside a refusal is unknown, not a refusal.
    mixed = RF.TargetReadout((RF.derive("t", facts=_facts()), RF.derive("t", facts={})))
    assert mixed.admits_granularity("tensor") is True and mixed.admits_granularity("column") is None

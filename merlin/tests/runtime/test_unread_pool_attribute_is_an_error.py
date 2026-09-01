"""A pooling attribute this runtime does not read configures nothing, and must say so.

`pool_params` already fails closed on a MISSING or wrong-arity attribute, for a stated reason: "a pooling
stage whose window silently defaulted would return a correctly-shaped tensor of numbers nobody computed."
An UNRECOGNISED attribute is the same failure and worse, because it is invisible -- the author believes
they configured a stage the engine never applies.

Measured 2026-09-01: an agent expressed a maxpool epilogue with an invented `pool_kind`. The capsule
schema leaves `operation.attributes` open -- it must, since attributes differ per op -- so the capsule
VALIDATED; this function dropped the key; and the agent concluded from the validation that its
formulation was legal, and from the wrong answer that no schema-valid formulation existed. Three capsules
(GP0/GP1/GP2) stayed frozen on that misdiagnosis across several rounds.
"""
from __future__ import annotations

import pytest

from merlin.runtime.commandbuffer import (
    POOL_ATTR_ARITY,
    POOL_PAD_VALUE_ATTR,
    pool_params,
)


def _valid() -> dict:
    return {"pool_in_dims": [8, 8], "pool_size": [2, 2], "pool_stride": [2, 2],
            "pool_padding": [0, 0, 0, 0]}


class TestTheMeasuredTrap:
    def test_an_invented_pool_attribute_raises_and_names_the_accepted_set(self):
        attrs = _valid() | {"pool_kind": "max"}
        with pytest.raises(ValueError) as ei:
            pool_params(attrs, op="conv2d")
        msg = str(ei.value)
        assert "pool_kind" in msg, msg
        assert "configure nothing" in msg or "configures nothing" in msg, msg
        # the message must be actionable: it names what IS read
        for k in POOL_ATTR_ARITY:
            assert k in msg, f"the accepted set must name {k}: {msg}"

    def test_several_unread_attributes_are_all_named(self):
        attrs = _valid() | {"pool_kind": "max", "pool_mode": "same"}
        with pytest.raises(ValueError) as ei:
            pool_params(attrs, op="matmul")
        assert "pool_kind" in str(ei.value) and "pool_mode" in str(ei.value)


class TestItDoesNotBreakWhatWorks:
    def test_the_declared_set_still_parses(self):
        got = pool_params(_valid(), op="conv2d")
        assert got["pool_size"] == (2, 2) and got["pool_stride"] == (2, 2)
        assert got["pool_in_dims"] == (8, 8) and got["pool_padding"] == (0, 0, 0, 0)
        assert got["pad_value"] is None

    def test_the_optional_pad_value_is_accepted(self):
        got = pool_params(_valid() | {POOL_PAD_VALUE_ATTR: -128}, op="conv2d")
        assert got["pad_value"] == -128

    def test_non_pool_attributes_are_left_alone(self):
        """This guard is scoped to the pooling family; it must not police unrelated attributes."""
        attrs = _valid() | {"epilogue": ["maxpool"], "output_dtype": "i32", "semantic": "conv"}
        pool_params(attrs, op="conv2d")            # must not raise

    def test_the_missing_attribute_error_is_unchanged(self):
        attrs = _valid()
        del attrs["pool_size"]
        with pytest.raises(ValueError, match="pool_size"):
            pool_params(attrs, op="conv2d")


def test_the_shipped_corpus_is_clean_under_this_rule():
    """No shipped capsule may already carry an unread pooling attribute."""
    import glob

    import yaml

    from merlin.common.paths import repo_root

    known = set(POOL_ATTR_ARITY) | {POOL_PAD_VALUE_ATTR}
    offenders = []
    root = repo_root() / "merlin" / "contract" / "capsules"
    for p in glob.glob(str(root / "**" / "capsule.yaml"), recursive=True):
        doc = yaml.safe_load(open(p, encoding="utf-8")) or {}
        attrs = ((doc.get("operation") or {}).get("attributes") or {})
        unread = [k for k in attrs if str(k).startswith("pool") and k not in known]
        if unread:
            offenders.append((p.split("/capsules/")[-1], unread))
    assert not offenders, f"capsules carrying pooling attributes nothing reads: {offenders}"

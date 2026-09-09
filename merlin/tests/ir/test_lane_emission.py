"""A lane plan that places work the emitted program does not contain must refuse.

MEASURED. SmolVLA's plan places 2,874 regions of dtype bf16 on the scalar lane, each with a recorded
lane and a recorded reason. The interface MLIR mentions bf16 21,971 times; the lowered MLIR mentions
it ZERO times and the emitted LLVM has no `bfloat` at all, while f32 arithmetic survives (2,250
`fmul float`). Nothing refused.

The dropped work was the cross-attention consuming the prefix KV-cache, so that input became dead
and was eliminated -- read 33 times in the export, referenced zero times in the kernel. The device
produced an exactly constant per-step delta of 0.092030 and a trajectory advancing 25x too little.
Diagnosing it took a 32-minute whole-model simulation and a five-step forensic chain; this check
answers it at compile time from two artifacts already on disk.

A performance campaign over such a program measures it as FASTER, correctly -- it does less work.
That is why this refuses rather than warns.
"""
from __future__ import annotations

import pytest

from merlin.verify import lane_emission as LE
from merlin.verify.lane_emission import LaneEmissionError


def _cb(*placements):
    return {"params": {"lane_placement": [dict(p) for p in placements]}}


def _p(region, dtype, lane="scalar_rvv_lane", **extra):
    row = {"region": region, "dtype": dtype, "lane": lane}
    row.update(extra)
    return row


class TestItCatchesWorkThatWasPlacedButNotEmitted:
    def test_a_placed_dtype_absent_from_the_text_is_DROPPED(self):
        got = LE.assess(_cb(_p("m0", "bf16"), _p("m1", "f32")),
                        "%1 = arith.mulf %a, %b : f32", dtype_tokens=LE.MLIR_DTYPE_TOKENS)
        assert got.status == "dropped" and got.refusing
        assert [p.dtype for p in got.dropped] == ["bf16"]

    def test_the_refusal_names_the_count_the_lane_and_examples(self):
        got = LE.assess(_cb(*[_p(f"m{i}", "bf16") for i in range(7)], _p("k", "f32")),
                        "f32", dtype_tokens=LE.MLIR_DTYPE_TOKENS)
        assert got.dropped[0].regions == 7
        assert got.dropped[0].lanes == ("scalar_rvv_lane",)
        assert got.dropped[0].examples == ("m0", "m1", "m2", "m3")
        assert "7 placed region(s)" in got.detail

    def test_the_refusal_says_WHY_it_matters_for_a_performance_campaign(self):
        got = LE.assess(_cb(_p("m0", "bf16")), "f32", dtype_tokens=LE.MLIR_DTYPE_TOKENS)
        assert "would score it as faster" in got.detail and "it does less" in got.detail

    def test_every_placed_dtype_present_is_EMITTED(self):
        got = LE.assess(_cb(_p("a", "f32"), _p("b", "i8"), _p("c", "bf16")),
                        "types: f32 i8 bf16 here", dtype_tokens=LE.MLIR_DTYPE_TOKENS)
        assert got.status == "emitted" and not got.refusing

    def test_a_plan_with_no_placement_is_NOT_APPLICABLE_rather_than_a_pass(self):
        got = LE.assess({"params": {}}, "anything", dtype_tokens=LE.MLIR_DTYPE_TOKENS)
        assert got.status == "not_applicable" and not got.refusing
        assert "no placement to hold" in got.detail

    def test_every_status_is_in_the_declared_vocabulary(self):
        for cb, text in ((_cb(_p("a", "f32")), "f32"), (_cb(_p("a", "bf16")), "f32"),
                         (_cb(_p("a", "mystery")), "f32"), ({"params": {}}, "x")):
            assert LE.assess(cb, text, dtype_tokens=LE.MLIR_DTYPE_TOKENS).status in LE.STATUSES

    def test_dropped_and_unknown_both_refuse(self):
        assert "dropped" in LE.REFUSING_STATUSES and "unknown" in LE.REFUSING_STATUSES
        assert "emitted" not in LE.REFUSING_STATUSES
        assert "not_applicable" not in LE.REFUSING_STATUSES


class TestATokenMustMatchAsATypeNotAsASubstring:
    """A substring test reports a dropped dtype as PRESENT, which is a false pass in the one
    direction that matters."""

    def test_i1_does_not_match_inside_i16(self):
        got = LE.assess(_cb(_p("a", "i1")), "%x = arith.addi %a, %b : i16",
                        dtype_tokens=LE.MLIR_DTYPE_TOKENS)
        assert got.status == "dropped", "i1 must not be found inside i16"

    def test_i1_does_match_when_really_present(self):
        got = LE.assess(_cb(_p("a", "i1")), "%x = arith.andi %a, %b : i1",
                        dtype_tokens=LE.MLIR_DTYPE_TOKENS)
        assert got.status == "emitted"

    def test_f32_does_not_match_inside_a_longer_identifier(self):
        got = LE.assess(_cb(_p("a", "f32")), "call @helper_f32_thing()",
                        dtype_tokens=LE.MLIR_DTYPE_TOKENS)
        assert got.status == "dropped"

    def test_llvm_float_does_not_match_inside_floating(self):
        got = LE.assess(_cb(_p("a", "f32")), "; floating point notes",
                        dtype_tokens=LE.LLVM_DTYPE_TOKENS)
        assert got.status == "dropped"

    def test_a_token_at_the_very_start_or_end_still_matches(self):
        assert LE.assess(_cb(_p("a", "f32")), "f32", dtype_tokens=LE.MLIR_DTYPE_TOKENS
                         ).status == "emitted"


class TestTheSpellingTableIsTheLANGUAGEsNotATargets:
    def test_mlir_and_llvm_spell_bf16_differently(self):
        assert LE.MLIR_DTYPE_TOKENS["bf16"] == ("bf16",)
        assert LE.LLVM_DTYPE_TOKENS["bf16"] == ("bfloat",)

    def test_bf16_present_in_llvm_spelling_is_emitted(self):
        got = LE.assess(_cb(_p("a", "bf16")), "%1 = fmul bfloat %a, %b",
                        dtype_tokens=LE.LLVM_DTYPE_TOKENS)
        assert got.status == "emitted"

    def test_the_mlir_spelling_would_MISS_it_in_llvm_text(self):
        """Which is why the table is a required argument, not a default."""
        got = LE.assess(_cb(_p("a", "bf16")), "%1 = fmul bfloat %a, %b",
                        dtype_tokens=LE.MLIR_DTYPE_TOKENS)
        assert got.status == "dropped"

    def test_the_table_covers_the_torch_spellings_the_plans_actually_emit(self):
        """Censused: the plans declare bool (72) and float64 (18) alongside the ABI spellings."""
        for name in ("bool", "float64", "f32", "bf16", "i8", "i64"):
            assert name in LE.MLIR_DTYPE_TOKENS, name
            assert name in LE.LLVM_DTYPE_TOKENS, name
        assert LE.MLIR_DTYPE_TOKENS["bool"] == ("i1",)
        assert LE.LLVM_DTYPE_TOKENS["float64"] == ("double",)

    def test_a_dtype_the_table_cannot_spell_is_UNKNOWN_not_assumed(self):
        got = LE.assess(_cb(_p("a", "some_future_type")), "anything",
                        dtype_tokens=LE.MLIR_DTYPE_TOKENS)
        assert got.status == "unknown" and got.refusing
        assert got.unplaceable == ["some_future_type"]
        assert "refused rather than assumed" in got.detail


class TestAPlacementWithNoDtypeIsCountedNotSwallowed:
    """lstmnetvit has 44 of them. Refusing the whole assessment would make the check unusable;
    skipping them silently would let a dtype vanish behind a blank field."""

    def test_they_are_counted_and_reported(self):
        got = LE.assess(_cb(_p("a", "f32"), {"region": "b", "lane": "scalar_rvv_lane"}),
                        "f32", dtype_tokens=LE.MLIR_DTYPE_TOKENS)
        assert got.status == "emitted"
        assert got.undeclared_placements == 1
        assert got.to_dict()["undeclared_placements"] == 1

    def test_they_do_not_mask_a_real_drop(self):
        got = LE.assess(_cb(_p("a", "bf16"), {"region": "b"}), "f32",
                        dtype_tokens=LE.MLIR_DTYPE_TOKENS)
        assert got.status == "dropped" and got.undeclared_placements == 1

    def test_a_plan_where_EVERY_placement_lacks_a_dtype_refuses_as_vacuous(self):
        with pytest.raises(LaneEmissionError, match="would be vacuous"):
            LE.assess(_cb({"region": "a"}, {"region": "b"}), "anything",
                      dtype_tokens=LE.MLIR_DTYPE_TOKENS)


class TestAnUnreadablePlanIsRefusedNotGuessed:
    def test_a_non_list_placement_is_refused(self):
        with pytest.raises(LaneEmissionError, match="not a list"):
            LE.assess({"params": {"lane_placement": "matmul_0"}}, "x",
                      dtype_tokens=LE.MLIR_DTYPE_TOKENS)

    def test_a_non_mapping_record_is_refused(self):
        with pytest.raises(LaneEmissionError, match="not a mapping"):
            LE.assess({"params": {"lane_placement": ["matmul_0"]}}, "x",
                      dtype_tokens=LE.MLIR_DTYPE_TOKENS)

    def test_a_buffer_with_no_params_is_not_applicable(self):
        assert LE.assess({}, "x", dtype_tokens=LE.MLIR_DTYPE_TOKENS).status == "not_applicable"

    def test_the_block_is_self_describing_about_what_it_does_NOT_check(self):
        block = LE.assess(_cb(_p("a", "f32")), "f32",
                          dtype_tokens=LE.MLIR_DTYPE_TOKENS).to_dict()
        assert block["schema"] == "merlin_lane_emission_v1"
        assert "region identity is not recoverable" in block["does_not_check"]


class TestItFiresOnTheRealEmissionsInThisTree:
    """The acceptance test: it must catch SmolVLA and clear the other three."""

    def _emissions(self):
        import glob
        import json

        from merlin.common.paths import artifacts_dir
        root = (artifacts_dir() / "perf-bench" / "gemmini"
                / "_global_phase2_baseline_emission_cache_v1")
        found = []
        for path in sorted(glob.glob(str(root / "*" / "command_buffer.json"))):
            lowered = pytest.importorskip("pathlib").Path(path).with_name("lowered.mlir")
            if not lowered.is_file():
                continue
            with open(path, encoding="utf-8") as handle:
                buffer = json.load(handle)
            n_args = len((buffer.get("kernel_abi") or {}).get("args") or [])
            found.append((n_args, buffer, lowered))
        if not found:
            pytest.skip("no emissions with a lowered sibling in this tree")
        return found

    def test_the_bf16_drop_is_caught(self):
        for n_args, buffer, lowered in self._emissions():
            if n_args != 1163:          # smolvla flow_denoise
                continue
            got = LE.assess(buffer, lowered.read_text(encoding="utf-8"),
                            dtype_tokens=LE.MLIR_DTYPE_TOKENS)
            assert got.status == "dropped" and got.refusing
            assert [p.dtype for p in got.dropped] == ["bf16"]
            assert got.dropped[0].regions > 1000, (
                "the whole bf16 subgraph is dropped, not only its contractions")
            return
        pytest.skip("no smolvla emission in this tree")

    def test_the_other_emissions_are_not_falsely_accused(self):
        checked = 0
        for n_args, buffer, lowered in self._emissions():
            if n_args == 1163:
                continue
            got = LE.assess(buffer, lowered.read_text(encoding="utf-8"),
                            dtype_tokens=LE.MLIR_DTYPE_TOKENS)
            assert got.status == "emitted", (
                f"{n_args}-argument emission wrongly accused: {got.detail[:200]}")
            checked += 1
        assert checked >= 1, "no emission exercised the passing path"

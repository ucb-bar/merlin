"""How much of a kernel's assembly we understand, and understand it AS WHAT.

The expert-vs-ours comparison is only meaningful if both sides are understood to the same depth, and
two failure modes make that silently untrue: a lifter handed a stream it cannot read fills nothing and
reports no divergence (which reads as agreement), and a disassembler's ignorance looks exactly like a
corpus with no structure. This audit exists so both are numbers rather than assumptions.
"""
from __future__ import annotations

import pytest

from merlin.kernels import asm_audit as A
from merlin.kernels import endpoints as EP


class _D:
    def __init__(self, i, roles=(), mnemonic="", claimed=False, identity=""):
        self.index, self.addr = i, i
        self.roles = tuple(roles)
        self.mnemonic = mnemonic
        self.from_endpoint = claimed
        self.identity = identity or mnemonic
        self.fields = {}


class TestTheSplitAlwaysAccountsForEveryInstruction:
    def test_the_four_way_split_sums_to_the_total(self):
        """A split that does not sum is not a measurement — it is a number with a hole in it."""
        decoded = [_D(0, roles=("accumulate",), claimed=True, identity="MAC"),
                   _D(1, mnemonic="addi"),
                   _D(2, claimed=True, identity="MYSTERY"),
                   _D(3, mnemonic="<unknown>")]
        a = A._classify(decoded, EP.load_endpoint("gemmini_rocc"), "t", "spatial")
        assert a.total == 4 and a.is_consistent()
        assert (a.role_tagged, a.named_by_tool, a.claimed_no_role, a.unaccounted) == (1, 1, 1, 1)

    def test_claimed_but_unroled_is_kept_apart_from_unaccounted(self):
        """They look alike in a total and want opposite responses: the first is a line missing from a
        role table we own, the second is an instruction nobody can explain."""
        a = A._classify([_D(0, claimed=True, identity="CUSTOM0"), _D(1, mnemonic="<unknown>")],
                        EP.load_endpoint("gemmini_rocc"), "t", "spatial")
        assert a.claimed_no_role == 1 and a.unaccounted == 1
        assert a.unroled_identities == ("CUSTOM0",), "the gap must be NAMED, not counted"

    def test_semantic_fraction_counts_meaning_not_disassembly(self):
        # 'the tool named it' is not 'we know what it does'.
        a = A._classify([_D(0, mnemonic="addi"), _D(1, roles=("accumulate",), claimed=True)],
                        EP.load_endpoint("gemmini_rocc"), "t", "spatial")
        assert a.semantic_fraction == 0.5 and a.named_by_tool == 1


class TestTheGapsAreNamedNotImplied:
    def test_a_stream_where_nothing_carries_a_role_says_so_loudly(self):
        """The load-bearing warning: a CCA lifted from such a stream compares equal to everything, so
        the comparison LOOKS clean precisely because one side was never read."""
        a = A._classify([_D(0, mnemonic="addi")], EP.load_endpoint("gemmini_rocc"), "t", "spatial")
        assert any("NOTHING in this stream carries a role" in g for g in a.gaps())

    def test_an_empty_stream_is_not_a_kernel_that_drives_nothing(self):
        a = A.AsmAudit(target="t")
        assert any("EMPTY" in g and "not the same as" in g for g in a.gaps())

    def test_unaccounted_words_are_never_counted_as_absent(self):
        a = A._classify([_D(0, mnemonic="<unknown>")], EP.load_endpoint("gemmini_rocc"), "t", "spatial")
        assert a.unaccounted == 1 and any("nothing could place" in g for g in a.gaps())


class TestComparabilityIsGuarded:
    def _aud(self, hist, total=100):
        a = A.AsmAudit(target="t", total=total, role_tagged=sum(hist.values()))
        a.role_histogram = dict(hist)
        return a

    def test_two_well_read_streams_are_comparable(self):
        rep = A.comparable(self._aud({"accumulate": 30}), self._aud({"accumulate": 20}))
        assert rep["comparable"] and rep["shared_roles"] == ["accumulate"]

    def test_a_stream_we_barely_read_is_refused(self):
        """Comparing a stream we understand 60% of against one we understand 0% of produces
        divergences that are artefacts of the second reading — and it looks CLEAN, because the unread
        side contributes nothing to disagree with."""
        rep = A.comparable(self._aud({"accumulate": 30}), self._aud({}, total=100))
        assert not rep["comparable"] and any("below the" in r for r in rep["reasons"])

    def test_an_empty_stream_is_refused_with_its_reason(self):
        rep = A.comparable(self._aud({"accumulate": 5}), A.AsmAudit(target="t"))
        assert not rep["comparable"] and any("empty stream" in r for r in rep["reasons"])

    def test_cross_target_comparison_is_refused(self):
        a, b = self._aud({"accumulate": 5}), self._aud({"accumulate": 5})
        b.target = "other"
        rep = A.comparable(a, b)
        assert not rep["comparable"] and any("different targets" in r for r in rep["reasons"])

    def test_the_role_diff_is_reported_both_ways(self):
        rep = A.comparable(self._aud({"accumulate": 3, "readout": 1}), self._aud({"accumulate": 3}))
        assert rep["expert_only_roles"] == ["readout"] and rep["ours_only_roles"] == []


class TestAgainstRealStreams:
    """The claim only means something if it holds on kernels nobody wrote for this test."""

    def _obj(self, pattern):
        import glob
        hits = sorted(glob.glob(pattern))
        if not hits:
            pytest.skip(f"no artifact matching {pattern} in this checkout")
        return hits[0]

    def test_a_real_vector_kernel_has_every_vector_instruction_roled(self):
        """The vector path used to carry NO declared meaning at all, so the one target the loop worked
        on was the one whose assembly could not be compared to anything."""
        from merlin.kernels.decode import grammar as G
        from merlin.kernels.decode import rvv as R
        obj = self._obj("out/runs/rvv_experiment/*/model.o")
        stream = R.decode(obj)
        decoded = G.decode_stream([i.raw for i in stream.insns], EP.load_endpoint("rvv_lanes"))
        vector = {i.raw.mnemonic for i in stream.insns if i.is_vector}
        unroled = set(G.unroled_mnemonics(decoded)) & vector
        assert not unroled, f"vector instructions with no declared role: {sorted(unroled)}"

    def test_a_real_vector_kernel_is_semantically_covered(self):
        a = A.audit_stream(self._obj("out/runs/rvv_experiment/*/model.o"), "rvv")
        assert a.is_consistent() and a.total > 0
        assert a.semantic_fraction > 0.1, a.to_dict()

    def test_two_kernels_written_in_different_languages_are_comparable(self):
        """The apples-to-apples claim, on real artifacts: hand-written C intrinsics against
        MLIR-compiled output, reduced to the same vocabulary."""
        hand = A.audit_stream(self._obj("out/runs/rvv_experiment/hand_v0_*/model.o"), "rvv")
        gen = A.audit_stream(self._obj("out/runs/rvv_experiment/impr_*/generated/model.o"), "rvv")
        rep = A.comparable(hand, gen)
        assert rep["comparable"], rep["reasons"]
        assert rep["shared_roles"], "two kernels for one target shared no role vocabulary at all"

    def test_pinning_the_disassembler_changes_what_we_see(self):
        """Measured: 76% unknown with the tool's default, 15% with the extensions given. A probe that
        does not pin its ISA settings reports the TOOL's ignorance as the corpus's nature."""
        import glob
        elfs = sorted(glob.glob("/scratch2/agustin/radiance-kernels/kernels/*/kernel.radiance.elf"))
        if not elfs:
            pytest.skip("no radiance ELF in this checkout")
        from merlin.kernels.decode import rvv as R
        bare = R.decode(elfs[0])
        pinned = R.decode(elfs[0], triple="riscv32", mattr="+m,+a,+f,+d,+c")
        n_bare = sum(1 for i in bare.insns if i.raw.mnemonic == "<unknown>")
        n_pin = sum(1 for i in pinned.insns if i.raw.mnemonic == "<unknown>")
        assert n_pin < n_bare, (
            f"pinning the ISA attributes did not reduce the unnamed count ({n_bare} -> {n_pin})")

    def test_the_declared_endpoint_settings_are_what_the_audit_uses(self):
        block = EP._spec()["endpoints"]["radiance_simt"]["encoding"]
        assert block.get("disasm_mattr"), "the endpoint must pin its disassembler attributes"
        assert int(block["stream_width"]) == 32, (
            "the width of a word in the OBJECT is not the ISA's internal instruction width; decoding "
            "at the internal width declines every architectural word")


class TestEveryTargetDecodesRealExpertCode:
    """The confidence check. A role table that only ever saw synthetic words proves nothing about
    whether we can read the kernels experts actually wrote."""

    def _first(self, pattern):
        import glob
        hits = sorted(glob.glob(pattern))
        if not hits:
            pytest.skip(f"no artifact matching {pattern} in this checkout")
        return hits[0]

    _GEMMINI = ("/scratch2/agustin/chipyard/generators/gemmini/software/gemmini-rocc-tests/"
                "build/bareMetalC/")

    def test_a_real_rocc_expert_binary_decodes_with_nothing_unaccounted(self):
        a = A.audit_stream(self._first(self._GEMMINI + "matmul*-baremetal"), "gemmini")
        assert a.is_consistent() and a.role_tagged > 0
        assert a.unaccounted == 0 and a.claimed_no_role == 0, a.to_dict()

    def test_the_two_instruction_levels_are_visible_in_real_expert_code(self):
        """The Phase 2 finding, confirmed on binaries: the same C call lowers either way — the library
        expands it — so the level is a property of the EMITTED stream and only disassembly can see it."""
        from merlin.kernels.decode import rocc as RC
        from merlin.kernels.decode import rvv as R
        ep = EP.load_endpoint("gemmini_rocc")
        fine = self._first(self._GEMMINI + "matmul-baremetal")
        fsm = self._first(self._GEMMINI + "conv-baremetal")
        a = RC.audit([i.raw for i in R.decode(fine).insns], "gemmini", ep)
        b = RC.audit([i.raw for i in R.decode(fsm).insns], "gemmini", ep)
        assert a.level == "fine_grained" and b.level == "fsm", (a.level, b.level)

    def test_an_offloaded_stream_is_not_judged_incomplete(self):
        """When the endpoint's own sequencer runs the loop, the steps never appear in the instruction
        stream. Reporting them missing flags a correct expert kernel as broken."""
        from merlin.kernels.decode import rocc as RC
        from merlin.kernels.decode import rvv as R
        obj = self._first(self._GEMMINI + "conv-baremetal")
        a = RC.audit([i.raw for i in R.decode(obj).insns], "gemmini",
                     EP.load_endpoint("gemmini_rocc"))
        assert a.level == "fsm" and a.missing_roles == ()

    def test_a_real_matrix_extension_binary_role_tags(self):
        a = A.audit_stream(
            self._first("/scratch2/agustin/chipyard/generators/saturn/benchmarks/opu-*.riscv"),
            "saturn")
        if not a.total:
            pytest.skip("saturn benchmark not decodable in this checkout")
        assert a.role_tagged > 0 and "accumulate" in a.role_histogram
        assert a.unaccounted == 0

    def test_a_hand_written_corpus_resolves_its_core_instructions(self):
        """The corpus and the model spell these differently; joined by encoding, not by name."""
        import glob
        files = sorted(glob.glob("/scratch2/agustin/mvp-lhwir/modeling/third_party/atlas-npu/"
                                 "baremetal/assembly/*.S"))
        if not files:
            pytest.skip("atlas corpus not present in this checkout")
        hist = {}
        for f in files[:60]:
            a = A.audit_text(open(f, errors="replace").read().splitlines(), "atlas")
            for k, v in a.role_histogram.items():
                hist[k] = hist.get(k, 0) + v
        # weight_load and readout are the MXU push/pop, which only appear once the assembler bridge
        # resolves the corpus spelling — without it the contraction facets are undecidable.
        assert {"accumulate", "weight_load", "readout"} <= set(hist), hist

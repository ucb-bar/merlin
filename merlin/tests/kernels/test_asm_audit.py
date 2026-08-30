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

import os

from merlin.common.paths import _dotenv


def _ext(pattern: str) -> str:
    """Expand ``${KEY}`` in an external-corpus path against the process env, then .env.

    These corpora are checkouts of other repos, so the location is per-machine. A missing key expands
    to "" and the caller's glob simply finds nothing — the same skip a machine without the checkout
    already took. Never a baked absolute path: this file is public.
    """
    out = pattern
    while "${" in out:
        i = out.index("${"); j = out.index("}", i)
        key = out[i + 2:j]
        out = out[:i] + (os.environ.get(key) or _dotenv().get(key) or "") + out[j + 1:]
    return out



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
        elfs = sorted(glob.glob(_ext("${MERLIN_RADIANCE_KERNELS}/kernels/*/kernel.radiance.elf")))
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

    _GEMMINI = _ext("${MERLIN_EXT_CHIPYARD}/generators/gemmini/software/gemmini-rocc-tests/"
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
            self._first(_ext("${MERLIN_EXT_CHIPYARD}/generators/saturn/benchmarks/opu-*.riscv")),
            "saturn")
        if not a.total:
            pytest.skip("saturn benchmark not decodable in this checkout")
        assert a.role_tagged > 0 and "accumulate" in a.role_histogram
        assert a.unaccounted == 0

    def test_a_hand_written_corpus_resolves_its_core_instructions(self):
        """The corpus and the model spell these differently; joined by encoding, not by name."""
        import glob
        files = sorted(glob.glob(_ext("${MERLIN_MLC_DIR}/../modeling/third_party/atlas-npu/"
                                 "baremetal/assembly/*.S")))
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


class TestAMultiEngineTargetIsNotAuditedThroughOneEngine:
    """The failure the engine model exists to prevent, reappearing inside the instrument built to
    detect it: audited through its array endpoint alone, a two-engine target's lane work does not
    appear at all — and it does not appear SILENTLY, because the histogram simply has no entry."""

    def _corpus(self):
        import glob
        files = sorted(glob.glob(_ext("${MERLIN_MLC_DIR}/../modeling/third_party/atlas-npu/"
                                 "baremetal/assembly/*.S")))
        if not files:
            pytest.skip("atlas corpus not present in this checkout")
        return files[:40]

    def test_auditing_every_endpoint_finds_both_engines(self):
        engines = {}
        for f in self._corpus():
            lines = open(f, errors="replace").read().splitlines()
            m = A.merge_audits(A.audit_every_endpoint(lines, "atlas", text=True))
            for eng, slot in m["per_engine"].items():
                engines.setdefault(eng, set()).update(slot["roles"])
        assert {"spatial", "vector"} <= set(engines), engines
        assert "accumulate" in engines["spatial"] and "elementwise" in engines["vector"]

    def test_one_endpoint_alone_misses_the_other_engines_work(self):
        # Stated as a test so nobody "simplifies" the audit back to a single endpoint.
        lines = []
        for f in self._corpus():
            lines += open(f, errors="replace").read().splitlines()
        arr = A.audit_text(lines, "atlas", EP.load_endpoint("atlas_isa"))
        vpu = A.audit_text(lines, "atlas", EP.load_endpoint("atlas_vpu"))
        assert "elementwise" not in arr.role_histogram
        assert vpu.role_histogram.get("elementwise", 0) > 0

    def test_the_instruction_total_is_not_multiplied_by_endpoint_count(self):
        """One stream read several ways is still one stream; counting the total per endpoint would
        inflate every coverage fraction by the number of engines."""
        lines = open(self._corpus()[0], errors="replace").read().splitlines()
        auds = A.audit_every_endpoint(lines, "atlas", text=True)
        m = A.merge_audits(auds)
        assert m["total"] == auds[0].total and len(auds) > 1

    def test_roles_are_kept_per_engine_not_pooled(self):
        lines = []
        for f in self._corpus():
            lines += open(f, errors="replace").read().splitlines()
        m = A.merge_audits(A.audit_every_endpoint(lines, "atlas", text=True))
        assert set(m["per_engine"]) >= {"spatial", "vector"}
        assert "elementwise" not in m["per_engine"]["spatial"]["roles"], (
            "a pooled histogram reads as one machine doing all of it")


class TestTheKernelRepoWasEnough:
    """The custom opcode spaces were called unguessable. They were not: the target's own kernel repo
    ships both tables — an intrinsics header whose inline assembly defines the SIMT control surface,
    and a RoCC ISA header defining the MX array's funct codes."""

    def _elfs(self):
        import glob
        f = sorted(glob.glob(_ext("${MERLIN_RADIANCE_KERNELS}/kernels/*/kernel.radiance.elf")))
        if not f:
            pytest.skip("radiance kernels not present in this checkout")
        return f[:6]

    def test_the_simt_control_surface_is_derived_from_the_targets_header(self):
        """`.insn r %0, 4, 0, ...` inside `vx_barrier` is the target saying, in its own words, that
        this encoding is a warp barrier. Reading it is the same act as reading a funct table out of
        RTL — the source differs, the derivation does not."""
        from merlin.kernels.decode import insn_header as H
        insns, problems = H.parse_insn_header(
            _ext("${MERLIN_RADIANCE_KERNELS}/lib/include/vx_intrinsics.h"))
        names = {i.name for i in insns}
        assert {"vx_barrier", "vx_split", "vx_join", "vx_wspawn"} <= names, sorted(names)
        assert problems == (), problems

    def test_barriers_and_divergence_appear_in_real_kernels(self):
        """`simt.barriers_in_loop` was an axis nothing could populate. It is decidable now."""
        hist = {}
        for f in self._elfs():
            for k, v in A.audit_stream(f, "radiance",
                                       EP.load_endpoint("radiance_simt")).role_histogram.items():
                hist[k] = hist.get(k, 0) + v
        assert hist.get("sync", 0) > 0 and hist.get("divergence", 0) > 0, hist

    def test_the_mx_array_is_a_second_endpoint_read_from_the_repos_own_isa_header(self):
        import glob
        mx = sorted(glob.glob(_ext("${MERLIN_RADIANCE_KERNELS}/kernels/*mxgemm*/*.elf")))
        if not mx:
            pytest.skip("no MX kernels in this checkout")
        ep = EP.load_endpoint("radiance_mx")
        assert "loop_descriptor" in ep.roles and not ep.missing
        hist = {}
        for f in mx:
            for k, v in A.audit_stream(f, "radiance", ep).role_histogram.items():
                hist[k] = hist.get(k, 0) + v
        assert hist.get("loop_descriptor", 0) > 0, (
            "the expert MX kernels drive the array through its hardware loop descriptor")

    def test_two_endpoints_sharing_one_opcode_space_do_not_claim_each_others_words(self):
        """Measured: the MX RoCC and the SIMT intrinsics share CUSTOM0, told apart by field — a RoCC
        command carries its operation in funct7, an intrinsic has funct7 == 0. Ignoring funct7
        mislabelled the array's instructions as SIMT control and inflated that role count."""
        from merlin.kernels.decode import insn_header as H
        table, _ = H.table_for("radiance", EP.load_endpoint("radiance_simt"))
        assert table, "the intrinsics table did not resolve"
        assert all(len(k) == 3 for k in table), "the table must key on funct7, not just funct3"
        assert all(k[2] == 0 for k in table), "every intrinsic declares funct7 == 0"

    def test_ceding_is_scoped_to_the_shared_space_only(self):
        """Applied to every opcode space it also cedes ordinary arithmetic — a standard R-type
        instruction has funct7 bits like anything else — and coverage falls while looking like a
        correctness fix."""
        import inspect

        from merlin.kernels.decode import derived_isa as D
        src = inspect.getsource(D.decode_stream)
        assert "space in cede_funct7_in" in src, "the cede must be scoped to named spaces"


class TestUnaccountedIsIntersectedNotMinimised:
    """A word is only 'unaccounted' for a TARGET when NO endpoint could place it.

    ``merge_audits`` used to take ``min()`` over each endpoint's unplaced COUNT. A minimum is an
    upper bound on an intersection, and the accumulator discarded a genuine zero, so the answer
    depended on the order the audits arrived in. Both are pinned here.
    """

    @staticmethod
    def _a(endpoint, engine, unplaced, *, total=1000, stream="k.elf", **kw):
        return A.AsmAudit(target="t", endpoint=endpoint, engine=engine, stream=stream,
                          total=total, unaccounted=len(unplaced),
                          unaccounted_indices=tuple(unplaced), **kw)

    def test_order_does_not_change_the_answer(self):
        a = self._a("epA", "vector", [])
        b = self._a("epB", "spatial", range(500))
        assert A.merge_audits([a, b])["unaccounted"] == A.merge_audits([b, a])["unaccounted"] == 0

    def test_one_endpoint_placing_everything_means_nothing_is_unaccounted(self):
        a = self._a("epA", "vector", [])
        b = self._a("epB", "spatial", range(500))
        assert A.merge_audits([a, b])["unaccounted"] == 0

    def test_disjoint_failures_are_not_a_shared_gap(self):
        # Each endpoint fails on 500 words, but on DIFFERENT words: every word was placed by someone.
        a = self._a("epA", "vector", range(0, 500))
        b = self._a("epB", "spatial", range(500, 1000))
        assert A.merge_audits([a, b])["unaccounted"] == 0

    def test_a_genuinely_shared_gap_survives(self):
        a = self._a("epA", "vector", range(0, 600))
        b = self._a("epB", "spatial", range(400, 1000))
        assert A.merge_audits([a, b])["unaccounted"] == 200      # the overlap 400..599

    def test_positions_from_different_streams_are_never_intersected(self):
        # Same positions, different kernels: two separate gaps, not one shared one.
        a = self._a("epA", "vector", range(0, 10), stream="k1.elf")
        b = self._a("epA", "vector", range(0, 10), stream="k2.elf")
        m = A.merge_audits([a, b])
        assert m["unaccounted"] == 20 and m["total"] == 2000 and m["n_streams"] == 2

    def test_a_count_without_positions_is_UNKNOWN_not_zero(self):
        """THE NEGATIVE CASE: the arithmetic path cannot say WHICH words were unplaced."""
        a = A.AsmAudit(target="t", endpoint="", engine="vector", stream="k.elf",
                       total=10, unaccounted=3, positions_known=False)
        assert A.merge_audits([a])["unaccounted"] is None

    def test_positions_known_with_nothing_unplaced_is_zero_not_unknown(self):
        a = self._a("epA", "vector", [])
        assert A.merge_audits([a])["unaccounted"] == 0

    def test_two_text_sources_are_two_streams_even_with_no_name(self):
        """A text source has no name to key on, and object identity will NOT do: a freed list's id
        is reused, so two files could alias to one stream and be counted once instead of twice."""
        lines = ["VMATMUL.MXU0 0, 0, 0"]
        got = {a.stream for a in A.audit_every_endpoint(list(lines), "atlas", text=True)}
        got |= {a.stream for a in A.audit_every_endpoint(list(lines), "atlas", text=True)}
        assert len(got) == 2, got

    def test_width_composition_covers_exactly_the_intersected_words(self):
        """A single 'unaccounted' number cannot distinguish a real instruction we failed to decode
        from a byte objdump could not form into one. Measured on the pinned radiance corpus: 878
        unplaced entries = 84 thirty-two-bit words (the genuine custom surface) + 484 sixteen-bit
        + 310 EIGHT-bit, and a RISC-V instruction is never 8 bits."""
        a = A.AsmAudit(target="t", endpoint="epA", engine="vector", stream="k.elf", total=100,
                       unaccounted=3, unaccounted_indices=(1, 2, 3),
                       unaccounted_width_at={1: 32, 2: 16, 3: 8})
        b = A.AsmAudit(target="t", endpoint="epB", engine="spatial", stream="k.elf", total=100,
                       unaccounted=2, unaccounted_indices=(1, 2),
                       unaccounted_width_at={1: 32, 2: 16})
        m = A.merge_audits([a, b])
        assert m["unaccounted"] == 2                       # 3 was placed by epB
        assert m["unaccounted_widths"] == {16: 1, 32: 1}   # and 8-bit entry 3 is NOT counted
        assert sum(m["unaccounted_widths"].values()) == m["unaccounted"]

    def test_a_position_no_decoder_sized_is_reported_as_unknown_width(self):
        """NEGATIVE CASE: width 0 means 'nobody said', and must not be guessed into a real width."""
        a = A.AsmAudit(target="t", endpoint="epA", engine="vector", stream="k.elf", total=10,
                       unaccounted=1, unaccounted_indices=(5,), unaccounted_width_at={})
        assert A.merge_audits([a])["unaccounted_widths"] == {0: 1}

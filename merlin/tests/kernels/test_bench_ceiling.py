"""Kernel-performance ceiling harness (S4.2): measure a curated kernel's spike cycle
count and join it 1:1 to a compiler-measured cycle count via the compare.py key.

Unit tests (no toolchain) cover the jsonl round-trip, attainment math, shape_regime
stability, the compare.py key contract, and graceful None paths. The real spike-run
tests are skipped when spike/riscv-gcc/saturn-corpus is unavailable.
"""
import json

import pytest

from merlin.kernels import bench_ceiling as bc
from merlin.kernels.compare import RvvFingerprint

_HAVE_SPIKE = bc.saturn_available()
_needs_spike = pytest.mark.skipif(
    not _HAVE_SPIKE,
    reason="spike/riscv-gcc/saturn-corpus/encoding.h unavailable (set MERLIN_CHIPYARD)")


# ---------------------------------------------------------------- toolchain-independent
def test_fingerprint_key_matches_compare():
    """The ceiling join key must be exactly compare.py's RvvFingerprint key shape."""
    key = bc.fingerprint_key("matmul", "i64", "square_small")
    assert key == {"op": "matmul", "dtype": "i64", "shape_regime": "square_small"}
    # and it is consumable as a compare.py key (same {op,dtype,shape_regime} dict)
    fp = RvvFingerprint.from_objdump("", key, "run")
    assert fp.key == key


def test_shape_regime_is_deterministic_string():
    assert bc.shape_regime("matmul", 8, 8, 8) == "square_small"
    assert bc.shape_regime("matmul", 32, 32, 32) == "square_medium"
    assert bc.shape_regime("matmul", 512, 512, 512) == "square_large"
    assert bc.shape_regime("dotprod", 512, 1, 1) == "vector"
    assert bc.shape_regime("matmul", 8, 256, 256) == "skinny"
    assert bc.shape_regime("matmul", 64, 128, 64) == "rectangular"
    # stable: same inputs -> same string
    assert bc.shape_regime("matmul", 32, 32, 32) == bc.shape_regime("matmul", 32, 32, 32)


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "ceiling.jsonl"
    row = {
        "op": "matmul", "dtype": "i64", "M": 32, "N": 32, "K": 32,
        "shape_regime": "square_medium", "source": "saturn", "target": "spike",
        "bench": "vec-igemm", "kernel_ref": "imatmul", "cycles": 26055,
        "isa": bc.DEFAULT_ISA,
        "fingerprint_key": bc.fingerprint_key("matmul", "i64", "square_medium"),
    }
    bc.append_ceiling(row, path)
    bc.append_ceiling({**row, "M": 64, "N": 64, "K": 64, "cycles": 200087}, path)
    rows = bc.load_ceiling(path)
    assert len(rows) == 2
    assert rows[0] == row
    assert rows[1]["cycles"] == 200087
    # each line is independently valid JSON
    for line in path.read_text().splitlines():
        json.loads(line)


def test_load_ceiling_missing_file(tmp_path):
    assert bc.load_ceiling(tmp_path / "absent.jsonl") == []


def test_attainment_math():
    row = {"cycles": 26055}
    # compiler matches the ceiling exactly -> 1.0
    assert bc.attainment(row, 26055) == pytest.approx(1.0)
    # compiler slower than expert -> < 1.0 (leaving performance on the table)
    assert bc.attainment(row, 30000) == pytest.approx(26055 / 30000)
    # compiler faster than expert -> > 1.0
    assert bc.attainment(row, 20000) > 1.0


def test_attainment_guards():
    assert bc.attainment({"cycles": 100}, 0) is None
    assert bc.attainment({"cycles": 100}, -5) is None
    assert bc.attainment({}, 100) is None
    assert bc.attainment({"cycles": 100}, None) is None


def test_find_ceiling_by_key():
    rows = [
        {"fingerprint_key": bc.fingerprint_key("matmul", "i64", "square_small"),
         "cycles": 573},
        {"fingerprint_key": bc.fingerprint_key("dotprod", "i64", "vector"),
         "cycles": 401},
    ]
    hit = bc.find_ceiling(rows, "dotprod", "i64", "vector")
    assert hit is not None and hit["cycles"] == 401
    assert bc.find_ceiling(rows, "matmul", "f32", "square_small") is None


def test_ceiling_joins_to_compiler_fingerprint():
    """End-to-end key contract: a ceiling row and a generated RvvFingerprint built for the
    SAME op-shape share the exact join key, so attainment pairs them 1:1."""
    regime = bc.shape_regime("matmul", 32, 32, 32)
    ceiling_row = {"cycles": 26055,
                   "fingerprint_key": bc.fingerprint_key("matmul", "i64", regime)}
    gen = RvvFingerprint.from_objdump("", bc.fingerprint_key("matmul", "i64", regime), "ours")
    assert ceiling_row["fingerprint_key"] == gen.key
    # pretend our compiler took 30k cycles for the same shape
    assert bc.attainment(ceiling_row, 30000) == pytest.approx(26055 / 30000)


def test_vopacc_excluded_and_bad_target():
    assert bc.run_kernel_ceiling("saturn", "vec-VOPACC-x", "matmul", "i64", (8, 8, 8)) is None
    assert bc.run_kernel_ceiling("saturn", "vec-igemm", "", "", (8, 8, 8),
                                 target="board") is None
    assert bc.run_kernel_ceiling("xnnpack", "vec-igemm", "", "", (8, 8, 8)) is None
    assert bc.run_kernel_ceiling("saturn", "no-such-bench", "", "", (8, 8, 8)) is None


# ------------------------------------------------------- console parsing (real captured output)
#
# These are the VERBATIM spike consoles of the three registered benches, captured by building and
# running them from this tree (riscv64-unknown-elf-gcc + spike, saturn-vectors corpus). They are the
# contract the structural readers are written against, so the parsers stay testable with no
# toolchain present.

_IGEMM_CONSOLE = """IMATMUL
Calculating a (4 x 4) x (4 x 4) matrix multiplication...
Calculating imatmul...
The execution took 152 cycles.
The performance is 842 OPs/1000 cycles.
Calculating a (32 x 32) x (32 x 32) matrix multiplication...
Calculating imatmul...
The execution took 26055 cycles.
The performance is 2515 OPs/1000 cycles.
Calculating a (64 x 64) x (64 x 64) matrix multiplication...
Calculating imatmul...
The execution took 200087 cycles.
The performance is 2620 OPs/1000 cycles.
Verifying result...
Passed.
"""

_DOTPROD_CONSOLE = """DOTP 512
Calulating 64b dotp with vectors with length = 8
Vector cycles: 29 instructions: 29
Calulating 64b dotp with vectors with length = 512
Vector cycles: 401 instructions: 401
Calulating 32b dotp with vectors with length = 512
Vector cycles: 210 instructions: 210
SUCCESS.
"""

# The WHOLE vec-sgemm console — there is no ": N cycles" line anywhere in it. saturn's
# common/syscalls.c prints "%s = %d" per counter and setStats(0) has already made each a delta over
# the measured region, so `mcycle` IS the sgemm's cycle count.
_SGEMM_CONSOLE = """sgemm M,N,K = 71,71,71
mcycle = 137509
minstret = 137514
"""


def test_igemm_console_parses_each_swept_size():
    for size, cycles in ((4, 152), (32, 26055), (64, 200087)):
        r = bc._parse_igemm(_IGEMM_CONSOLE, size, size, size)
        assert r.state == bc._PARSED and r.cycles == cycles and r.instructions is None


def test_igemm_unswept_size_is_absent_not_unparseable():
    """A size the bench never ran is ABSENT — a real "we did not measure this", not a defect."""
    for size in (7, 8, 128, 1):
        assert bc._parse_igemm(_IGEMM_CONSOLE, size, size, size).state == bc._ABSENT
    assert bc._parse_igemm(_IGEMM_CONSOLE, 4, 8, 4).state == bc._ABSENT     # not square


def test_igemm_header_does_not_match_a_different_size():
    """The old pattern keyed on the four-operand shape; the replacement must be no looser."""
    assert bc._square_shape_header("Calculating a (64 x 64) x (64 x 64) matrix...", 64)
    assert not bc._square_shape_header("Calculating a (64 x 64) x (64 x 64) matrix...", 4)
    assert not bc._square_shape_header("Calculating a (14 x 4) x (4 x 4) matrix...", 4)
    assert bc._square_shape_header("(4x4)x(4x4)", 4)          # the old `\s*` allowed zero spaces


def test_igemm_measured_but_unreadable_is_unparseable():
    """Header present, cycle line gone: the size WAS measured, so this is a tooling defect."""
    drifted = ("Calculating a (32 x 32) x (32 x 32) matrix multiplication...\n"
               "Calculating imatmul...\n"
               "The execution consumed 26055 ticks.\n")
    r = bc._parse_igemm(drifted, 32, 32, 32)
    assert r.state == bc._UNPARSEABLE
    assert r.cycles is None                       # UNKNOWN, never 0
    assert "(32 x 32)" in r.detail


def test_dotprod_console_parses_cycles_and_instructions():
    r = bc._parse_dotprod(_DOTPROD_CONSOLE, 512, 1, 1)
    assert r.state == bc._PARSED and (r.cycles, r.instructions) == (401, 401)
    r8 = bc._parse_dotprod(_DOTPROD_CONSOLE, 8, 1, 1)
    assert (r8.cycles, r8.instructions) == (29, 29)
    # first (64-bit) occurrence wins, so the key is deterministic across widths
    assert bc._parse_dotprod(_DOTPROD_CONSOLE, 512, 1, 1).cycles == 401


def test_dotprod_length_header_is_word_bounded():
    r"""`length = 512` must not answer a request for 51 (the old pattern's trailing `\b`)."""
    line = "Calulating 64b dotp with vectors with length = 512"
    assert bc._length_header(line, 512)
    assert not bc._length_header(line, 51)
    assert not bc._length_header(line, 5)
    assert bc._length_header("length=8", 8)                  # the old `\s*` allowed zero spaces
    assert bc._parse_dotprod(_DOTPROD_CONSOLE, 51, 1, 1).state == bc._ABSENT
    assert bc._parse_dotprod(_DOTPROD_CONSOLE, 512, 2, 1).state == bc._ABSENT   # not a vector shape


def test_sgemm_reads_the_console_the_bench_actually_prints():
    r"""REGRESSION: the retired `:\s*(\d+)\s+cycles` pattern matched nothing in this console, so
    vec-sgemm's ceiling was silently unmeasurable. It is measurable now."""
    assert "cycles" not in _SGEMM_CONSOLE          # the old pattern had nothing to match, ever
    r = bc._parse_sgemm(_SGEMM_CONSOLE, 71, 71, 71)
    assert r.state == bc._PARSED
    assert r.cycles == 137509 and r.instructions == 137514


def test_sgemm_still_accepts_the_pk_stats_spelling():
    """A pk-hosted console prints `<code>: N cycles, ...` — what the old pattern read. Still read."""
    r = bc._parse_sgemm("core   0: 137509 cycles, 137514 instructions, 1.00 CPI\n", 71, 71, 71)
    assert r.state == bc._PARSED and r.cycles == 137509


def test_sgemm_only_measures_its_one_fixed_shape():
    assert bc._parse_sgemm(_SGEMM_CONSOLE, 32, 32, 32).state == bc._ABSENT
    assert bc._parse_sgemm(_SGEMM_CONSOLE, 71, 71, 70).state == bc._ABSENT


def test_sgemm_drifted_console_is_unparseable_not_absent():
    r = bc._parse_sgemm("sgemm M,N,K = 71,71,71\nrun complete\n", 71, 71, 71)
    assert r.state == bc._UNPARSEABLE and r.cycles is None


def test_counts_are_validated_never_coerced():
    """Anchor, then validate the token: a non-decimal is refused rather than read as a number."""
    assert bc._took_cycles("The execution took 26055 cycles.") == 26055
    assert bc._took_cycles("The execution took -5 cycles.") is None
    assert bc._took_cycles("The execution took many cycles.") is None
    assert bc._took_cycles("The performance is 2515 OPs/1000 cycles.") is None   # no 'took'
    assert bc._vector_cycles("Vector cycles: 401 instructions: 401") == (401, 401)
    assert bc._vector_cycles("Vector cycles: n/a instructions: 401") is None
    assert bc._vector_cycles("Scalar cycles: 401 instructions: 401") is None
    assert bc._named_counter("mcycle = 137509", "mcycle") == 137509
    assert bc._named_counter("minstret = 137514", "mcycle") is None
    assert bc._named_counter("mcycle = ?", "mcycle") is None
    assert bc._named_counter("mcycle = 137509 extra", "mcycle") is None


def test_a_positional_read_cannot_wander_onto_another_line():
    """The trap on the other side of a too-narrow pattern: splitting on whitespace and taking
    parts[N] would happily read a number out of an unrelated line. Every reader here is anchored on
    a named literal, so a neighbouring line yields nothing."""
    for line in ("The performance is 2620 OPs/1000 cycles.", "DOTP 512", "Verifying result...",
                 "sgemm M,N,K = 71,71,71", "Calculating imatmul..."):
        assert bc._took_cycles(line) is None
        assert bc._vector_cycles(line) is None
        assert bc._named_counter(line, "mcycle") is None
        assert bc._colon_cycles(line) is None


# --------------------------------------------------------------------------- spike-gated
@_needs_spike
def test_igemm_ceiling_real_spike():
    row = bc.run_kernel_ceiling("saturn", "vec-igemm", "", "", (32, 32, 32))
    assert row is not None, "vec-igemm 32x32x32 should build + run on spike"
    assert row["op"] == "matmul" and row["target"] == "spike"
    assert row["cycles"] > 0
    assert row["fingerprint_key"] == bc.fingerprint_key("matmul", row["dtype"],
                                                        row["shape_regime"])
    # demonstrate attainment against a dummy compiler cycle count
    att = bc.attainment(row, row["cycles"] * 1.2)
    assert att == pytest.approx(1 / 1.2)


@_needs_spike
def test_igemm_unmeasured_size_returns_none():
    # vec-igemm only sweeps power-of-two squares up to M(=64); 7x7x7 is not measured.
    assert bc.run_kernel_ceiling("saturn", "vec-igemm", "", "", (7, 7, 7)) is None


@_needs_spike
def test_sgemm_ceiling_real_spike():
    """vec-sgemm's ONE fixed 71x71x71 shape yields a real ceiling row.

    Before the console reader was rewritten this bench always returned None: its pattern looked for
    a pk-style ": N cycles" line the saturn corpus never prints.
    """
    row = bc.run_kernel_ceiling("saturn", "vec-sgemm", "", "", (71, 71, 71))
    assert row is not None, "vec-sgemm 71x71x71 should build + run on spike"
    assert row["op"] == "matmul" and row["dtype"] == "f32"
    assert row["cycles"] > 0 and row.get("instructions", 0) > 0
    assert bc.run_kernel_ceiling("saturn", "vec-sgemm", "", "", (70, 70, 70)) is None


@_needs_spike
def test_dotprod_ceiling_real_spike():
    row = bc.run_kernel_ceiling("saturn", "vec-dotprod", "", "", (512, 1, 1))
    assert row is not None
    assert row["op"] == "dotprod" and row["shape_regime"] == "vector"
    assert row["cycles"] > 0 and row.get("instructions", 0) > 0

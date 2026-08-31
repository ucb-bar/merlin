"""The ingest scrapers' accept sets, pinned.

These four scrapers read a naming convention (OpenBLAS filenames, XNNPACK ukernel symbols, Autocomp
``test()`` signatures and hash filenames) and a Python source tree (Triton). They used to do it with
regexes; the failure mode a pattern has is that a too-narrow one silently drops valid-but-differently
spelled input, so what each parser accepts is pinned here — including the boundary cases that decide
it: an extra separator, a differently ordered token, an absent optional group, a case difference.

Each case is annotated with what the retired pattern did, so a future change to these parsers has to
be a deliberate change to the accept set rather than an accident.
"""
import pytest

from merlin.common.paths import merlin_dir
from merlin.kernels.types import normalize_dtype
from merlin.kernels.ingest import autocomp as ac
from merlin.kernels.ingest import openblas as ob
from merlin.kernels.ingest import triton as tri
from merlin.kernels.ingest import xnnpack as xn

OPENBLAS = str(merlin_dir() / "tests" / "data" / "kernels" / "openblas")


# --------------------------------------------------------------- OpenBLAS filenames

@pytest.mark.parametrize("name,expected", [
    # the shipped spellings: both trailing fields, both dtype-letter lengths
    ("dgemm_kernel_8x4_zvl128b.c", ("gemm", "f64", {"MR": 8, "NR": 4, "vlen_bits": 128})),
    ("sbgemm_kernel_16x2_zvl256b.c", ("gemm", "bf16", {"MR": 16, "NR": 2, "vlen_bits": 256})),
    ("ztrmm_kernel_8x4_zvl128b.c", ("trmm", "c64", {"MR": 8, "NR": 4, "vlen_bits": 128})),
    # c910v carries no vlen: the optional group is ABSENT, not zero
    ("dgemm_kernel_8x4_c910v.c", ("gemm", "f64", {"MR": 8, "NR": 4})),
    # the separator before "kernel_" is optional, and leading zeros are digits like any other
    ("dgemmkernel_8x4_zvl128b.c", ("gemm", "f64", {"MR": 8, "NR": 4, "vlen_bits": 128})),
    ("dgemm_kernel_08x04_zvl0128b.c", ("gemm", "f64", {"MR": 8, "NR": 4, "vlen_bits": 128})),
    # a one-letter prefix is all dtype and no root, so the op is the (empty) remainder
    ("x_kernel_1x1_zvl128b.c", ("x", "unknown", {"MR": 1, "NR": 1, "vlen_bits": 128})),
    # leftmost "kernel_" that parses wins -- the prefix capture was non-greedy
    ("kernel_kernel_2x2_zvl256b.c", ("kernel", "unknown", {"MR": 2, "NR": 2, "vlen_bits": 256})),
])
def test_openblas_gemm_family_names(name, expected):
    assert ob._parse_gemm_kernel(name) == expected


@pytest.mark.parametrize("name", [
    "kernel_8x4_zvl128b.c",          # the prefix is required (>= 1 char)
    "dgemm__kernel_8x4_zvl128b.c",   # exactly ONE optional separator, never two
    "d_gemm_kernel_8x4_zvl128b.c",   # the prefix itself may not contain a separator
    "Dgemm_kernel_8x4_zvl128b.c",    # the prefix alphabet is lowercase
    "dgemm_kernel_8x4_zvl128B.c",    # ... and so is the vlen suffix
    "dgemm_kernel_8x4_zvlb.c",       # the vlen digits are required
    "dgemm_kernel_8x_zvl128b.c",     # both tile extents are required
    "dgemm_kernel_8x4_zvl128b.cc",   # the extension is exactly ".c"
    "amax_rvv.c",                    # a BLAS1 kernel is not this family
])
def test_openblas_gemm_family_rejects(name):
    assert ob._parse_gemm_kernel(name) is None


@pytest.mark.parametrize("name,expected", [
    ("zaxpy_rvv.c", ("axpy", "c64")),
    ("amax_rvv.c", ("amax", "unknown")),
    ("iamax_rvv.c", ("amax", "unknown")),          # the index variant reduces to the base op
    ("izamax_rvv.c", ("amax", "c64")),             # ... including its dtype letter
    ("gemv_n_vector.c", ("gemv", "unknown")),
    ("gemv_t_vector_v2.c", ("gemv", "unknown")),   # the "_v2" goes with the "_vector" it follows
    ("a_vector_v2_rvv.c", ("a", "unknown")),       # back-to-back variant tokens all go
    ("dsdot_vector.c", ("dot", "f32")),
    ("sbdot_rvv.c", ("dot", "bf16")),
    ("omatcopy_cn_rvv.c", ("transpose", "unknown")),
    # a name outside the root table reports its own leading token rather than "unknown"
    ("a_rvvx.c", ("a", "unknown")),
    ("vector_rvv.c", ("vector", "unknown")),
])
def test_openblas_blas_family_names(name, expected):
    assert ob._parse_blas(name) == expected


@pytest.mark.parametrize("name,stripped", [
    ("zaxpy_rvv.c", "zaxpy.c"),
    ("a_c910v.c", "a.c"),
    ("gemv_t_vector_v2.c", "gemv_t.c"),
    ("a_vector_v2_rvv.c", "a.c"),          # back-to-back variants all go
    ("a_rvv_b.c", "a_b.c"),                # ... and one in the middle of the name
    ("vector_rvv.c", "vector.c"),          # never the LEADING token: that is the routine's name
    ("a_rvvx.c", "a_rvvx.c"),              # a whole token only, never a prefix of one
    ("a_vectorish.c", "a_vectorish.c"),
    ("a_vector_v2x.c", "a_v2x.c"),         # "_vector_v2" has no boundary here, but "_vector" does
    ("a_rvv.h", "a_rvv.h"),                # the boundary is "_" or the ".c" extension
])
def test_openblas_variant_token_stripping(name, stripped):
    """What the retired substitution removed, token for token: the vectorization variant, never a
    routine name that merely starts with one."""
    assert ob._drop_variant_tokens(name) == stripped


@pytest.mark.parametrize("name,is_vector", [
    ("amax.c", False), ("amax_rvv.c", True), ("gemv_n_vector.c", True),
    ("dgemm_kernel_8x4_c910v.c", True), ("dgemm_kernel_8x4_zvl128b.c", True),
    ("dgemm_kernel_8x4_zvlb.c", False),
])
def test_openblas_vector_kernel_selection(name, is_vector):
    assert ob._is_vector_kernel(name) is is_vector


def test_openblas_ingest_reports_ops_outside_the_root_table():
    """A vector kernel whose routine is not a known BLAS root is ingested under the token read off
    its own name AND listed, so an unknown spelling is a reported gap, never a silent mislabel."""
    diag = {}
    kernels = {k.path.split("/")[-1]: k for k in ob.ingest_openblas(OPENBLAS, diagnostics=diag)}
    assert set(kernels) == {"dgemm_kernel_8x4_zvl128b.c", "zaxpy_rvv.c"}
    assert diag["unrecognized_ops"] == {}
    assert ob._parse_blas("trsm_kernel_LN_rvv.c") == ("trsm", "unknown")
    assert "trsm" not in ob._KNOWN_OPS


# --------------------------------------------------------------- XNNPACK ukernel symbols

def _sym(text, isa="rvv"):
    return xn.find_ukernel_symbol(text, isa)


def test_xnnpack_symbol_reads_core_and_shape():
    assert _sym("void xnn_f32_gemm_minmax_ukernel_1x4v__rvv(size_t mr)") == (
        "f32_gemm_minmax", "1x4v")


def test_xnnpack_symbol_isa_is_a_prefix_not_the_whole_suffix():
    """``__rvvfp16arith`` is ingested under base ISA ``rvv`` -- this is how the fp16 microkernels
    reach the corpus at all, so the ISA test must stay a prefix test."""
    assert _sym("xnn_f16_gemm_ukernel_1x4v__rvvfp16arith(") == ("f16_gemm", "1x4v")


def test_xnnpack_symbol_takes_the_first_one_in_the_file():
    text = "xnn_f32_vadd_ukernel__rvv\nxnn_f32_gemm_ukernel_2x2__rvv\nxnn_qs8_gemm_ukernel_4x4__rvv"
    assert _sym(text) == ("f32_gemm", "2x2")   # the first line has an EMPTY shape: not a symbol


def test_xnnpack_symbol_takes_the_last_ukernel_infix_in_one_identifier():
    """The core capture was greedy, so a core that itself contains ``_ukernel_`` keeps it."""
    assert _sym("xnn_f32_ukernel_gemm_ukernel_2x2__rvv") == ("f32_ukernel_gemm", "2x2")


@pytest.mark.parametrize("text", [
    "xnn_ukernel_2x2__rvv",             # the core is required
    "xnn_f32_gemm_ukernel___rvv",       # the shape is required
    "xnn_f32_gemm_ukernel_2_2__rvv",    # the shape carries no separator
    "xnn_f32_gemm_ukernel_2x2__neon",   # a different base ISA
    "xnn_f32_gemm_ukernel_2X2__rvv",    # the symbol alphabet is lowercase: uppercase ends the run
    "XNN_F32_GEMM_UKERNEL_2x2__rvv",
    "xnn_f32_gemm_ukernel_2x2_rvv",     # exactly two separators before the ISA
])
def test_xnnpack_symbol_rejects(text):
    assert _sym(text) is None


def test_xnnpack_symbol_may_start_inside_a_longer_identifier():
    """The old pattern had no left boundary, so a wrapped symbol still matched. Kept: dropping it
    would lose every kernel whose symbol is built by a macro paste."""
    assert _sym("my_xnn_f32_gemm_ukernel_2x2__rvv") == ("f32_gemm", "2x2")


@pytest.mark.parametrize("token,expected", [
    ("4x4", {"MR": 4, "NR": "4"}),
    ("1x4v", {"MR": 1, "NR": "4v"}),
    ("1x4vc", {"MR": 1, "NR": "4vc"}),
    ("1x4c", {"MR": 1, "NR": "4c"}),
    ("9p", {"kernel_points": 9, "channel_tile": ""}),        # both optional groups absent
    ("9p8", {"kernel_points": 9, "channel_tile": "8"}),
    ("9p8v", {"kernel_points": 9, "channel_tile": "8v"}),
    ("25p2vc", {"kernel_points": 25, "channel_tile": "2vc"}),
    # `c` alone qualifies the MRxNR form but NOT the points form -- the two spellings differ
    ("9p8c", {"kernel_points": 9, "channel_tile": "8"}),
    # both forms are read as a PREFIX, so trailing text does not defeat the tile
    ("4x4vcx", {"MR": 4, "NR": "4vc"}),
    ("4x4p2", {"MR": 4, "NR": "4"}),
])
def test_xnnpack_shape_tokens(token, expected):
    assert xn.parse_shape_token(token) == expected


@pytest.mark.parametrize("token", ["4x", "x4", "4X4", "minmax", "u2v", "", "p", "v"])
def test_xnnpack_shape_token_rejects(token):
    assert xn.parse_shape_token(token) is None


def test_xnnpack_unparsed_shape_token_is_kept_verbatim():
    """An unreadable tile is preserved, not dropped: the shape is then visibly unparsed."""
    assert xn._parse_shape("minmax") == {"tile": "minmax"}


# --------------------------------------------------------------- Autocomp signatures

def test_autocomp_signature_reads_matmul_operands():
    sig = "void test(int8_t A[512][512], int8_t B[512][512], int8_t C[512][512]) {}"
    assert ac.parse_signature(sig) == ("matmul", "i8", {"M": 512, "K": 512, "N": 512}, ())


def test_autocomp_signature_skips_qualifiers_and_odd_whitespace():
    """``const`` is not a type here; the scan finds the parameter that follows it. Whitespace may
    sit anywhere, including inside the dimensions, BETWEEN them, and across newlines.

    Whitespace between two dimensions is a deliberate widening: the retired pattern allowed it
    inside a bracket but not between brackets, so ``A [ 8 ] [ 4 ]`` read as rank 1 and the operand
    was mis-measured with nothing to show for it."""
    sig = ("static void\ntest ( const int8_t A [ 8 ] [ 4 ] ,\n"
           "  const int8_t B[4][2], int8_t C[8][2] );")
    assert ac.parse_signature(sig) == ("matmul", "i8", {"M": 8, "K": 4, "N": 2}, ())


def test_autocomp_signature_has_no_word_boundary_before_void():
    """Pinned as-is: the retired pattern had no boundary, so a run-together ``avoid test(...)`` was
    an entry point. Nothing in the corpus depends on rejecting it, and tightening it silently would
    change which files parse."""
    assert ac.parse_signature("avoid test(float A[2][2], float B[2][2], float C[2][2]);")[0] == "matmul"


def test_autocomp_signature_conv_by_rank_or_by_name():
    rank = "void test(int8_t A[2][2][2][2], int8_t B[2][2], int8_t C[2][2]);"
    assert ac.parse_signature(rank)[0] == "conv"
    named = "void test(int8_t inp[2][2], int8_t weights[2][2], int8_t output[2][2]);"
    op, _dtype, shape, _skipped = ac.parse_signature(named)
    assert op == "conv" and shape == {"inp": [2, 2], "weights": [2, 2], "output": [2, 2]}


def test_autocomp_signature_under_three_operands_keeps_per_name_dims():
    assert ac.parse_signature("void test(float A[2][3], float B[3][4]);") == (
        "matmul", "f32", {"A": [2, 3], "B": [3, 4]}, ())


@pytest.mark.parametrize("text", [
    "voidtest(int a[2][2], int b[2][2], int c[2][2]);",   # the separator is required
    "void TEST(float A[2][2]);",                          # the entry point is spelled lowercase
    "int test(float A[2][2]);",                           # ... and returns void
    "nothing to see here",
])
def test_autocomp_signature_absent(text):
    assert ac.parse_signature(text) == ("unknown", "unknown", {}, ())


def test_autocomp_unclosed_parameter_list_keeps_looking():
    """A ``void test(`` with no ``)`` is not the entry point; a later well-formed one still is."""
    text = "void test(oops\n\nvoid test(float A[2][2], float B[2][2], float C[2][2]);"
    assert ac.parse_signature(text)[0] == "matmul"


def test_autocomp_unreadable_parameters_are_reported_not_dropped():
    """A pointer parameter carries no dimensions, so the shape is PARTIAL. It is named in the
    fourth field (and recorded in the record's meta) instead of vanishing."""
    op, dtype, shape, unreadable = ac.parse_signature(
        "void test(float *A, float B[3][3], float C[3][3], float D[3][3]);")
    assert (op, dtype, shape) == ("matmul", "f32", {"M": 3, "K": 3, "N": 3})
    assert unreadable == ("float *A",)


@pytest.mark.parametrize("name,expected", [
    ("kernel_0013bf80fb0a.c", "0013bf80fb0a"),
    ("kernels/kernel_00ff.c", "00ff"),
    ("kernel_kernel_abc.c", "abc"),        # leftmost "kernel_" whose remainder is all hex
    ("kernel_ABC.c", None),                # the hash is lowercase hex
    ("kernel_xyz.c", None),
    ("kernel_.c", None),                   # ... and non-empty
    ("kernel_ff.cpp", None),               # the extension is exactly ".c"
    ("kernel_ff.c.bak", None),
])
def test_autocomp_kernel_hash(name, expected):
    assert ac.kernel_hash(name) == expected


# --------------------------------------------------------------- Triton source scrape

def _names(src):
    return [n for n, _body in tri._functions(src)]


def test_triton_finds_jit_functions():
    assert _names("@triton.jit\ndef foo(a):\n    pass\n") == ["foo"]
    assert _names("@triton.jit(interpret=True)\ndef bar(a):\n    pass\n") == ["bar"]
    assert _names("@other\n@triton.jit\ndef qux(a):\n    pass\n") == ["qux"]
    assert _names("class K:\n    @triton.jit\n    def meth(a):\n        pass\n") == ["meth"]


def test_triton_finds_functions_the_line_pattern_could_not():
    """A decorator or a signature spanning lines: the retired pattern required the decorator and the
    ``def`` to sit on lines of their own, so these were invisible to it."""
    assert _names("@triton.jit(\n  interpret=True)\ndef multi(a):\n    pass\n") == ["multi"]
    assert _names("@triton.jit\n@foo(\n  bar=1)\ndef dec(a):\n    pass\n") == ["dec"]


@pytest.mark.parametrize("src", [
    "# @triton.jit\ndef commented(a):\n    pass\n",
    "raise RuntimeError('only in @triton.jit\\'d functions')\n\ndef __next__(self):\n    pass\n",
    "@triton.jitx\ndef weird(a):\n    pass\n",
])
def test_triton_prose_mentions_declare_nothing(src):
    """A mention of the decorator is not a declaration. The line pattern attached whatever ``def``
    came next to it and minted a kernel that does not exist."""
    assert _names(src) == []


def test_triton_kernels_embedded_in_a_template_string_are_ingested():
    """Codegen templates and docstring examples hold real kernels. They are parsed as the Python
    they are, and the body is the embedded source rather than a slice of the enclosing file."""
    src = 'TEMPLATE = r"""\n@triton.jit\ndef inner(a):\n    return tl.dot(a, a)\n"""\n'
    got = tri._functions(src)
    assert [n for n, _b in got] == ["inner"]
    assert got[0][1].startswith("@triton.jit\ndef inner")
    assert '"""' not in got[0][1]


def test_triton_unparseable_template_is_reported():
    """A template whose signature holds placeholders is not valid Python anywhere -- it is named in
    the scan's report so the drop is visible, instead of being lost between the quotes."""
    src = 'T = r"""\n@triton.jit\ndef gen({{argdefs()}}, a):\n    pass\n"""\n'
    functions, unparsed = tri.scan_functions(src)
    assert functions == []
    assert len(unparsed) == 1 and "line 1" in unparsed[0]


def test_triton_body_stops_at_the_next_kernel_or_top_level_def():
    src = ("@triton.jit\ndef a(x):\n    return tl.float16(x)\n\n"
           "@triton.jit\ndef b(x):\n    return tl.float8e4nv(x)\n\n"
           "def helper():\n    pass\n")
    got = dict(tri._functions(src))
    assert set(got) == {"a", "b"}
    assert "def b" not in got["a"] and "def helper" not in got["b"]
    assert tri._guess_dtype(got["a"]) == "f16"
    # the float8 spelling is open-ended, so the whole trailing word is read, not a fixed list
    assert tri._guess_dtype(got["b"]) == normalize_dtype("float8e4nv")


@pytest.mark.parametrize("body,expected", [
    ("x = tl.float32(0)", "f32"),
    ("tl.bfloat16", normalize_dtype("bfloat16")),
    ("tl.int8_t", "i8"),                     # a known name is read as a prefix
    ("y = atl.float32", "f32"),              # no left boundary, as before
    ("tl.dot(a, b)", "unknown"),
    ("", "unknown"),
])
def test_triton_dtype_scan(body, expected):
    assert tri._guess_dtype(body) == expected


def test_triton_ingest_reports_a_file_it_cannot_parse(tmp_path):
    """A file carrying the decorator that does not parse is listed, never silently skipped."""
    (tmp_path / "broken.py").write_text("@triton.jit\ndef f(:\n", encoding="utf-8")
    (tmp_path / "good.py").write_text("@triton.jit\ndef g(a):\n    pass\n", encoding="utf-8")
    diag = {}
    got = list(tri.ingest_triton(str(tmp_path), diagnostics=diag))
    assert [k.meta["function"] for k in got] == ["g"]
    assert list(diag["unparsed"]) == ["broken.py"]


def test_triton_ingest_warns_when_there_is_no_diagnostics_dict(tmp_path):
    """With nowhere to record them, the same facts go out as a warning. Silence is the one outcome
    that is not allowed: a corpus quietly short a kernel is how a mining loop under-counts."""
    (tmp_path / "template.py").write_text(
        'T = r"""\n@triton.jit\ndef gen({{argdefs()}}):\n    pass\n"""\n', encoding="utf-8")
    with pytest.warns(UserWarning, match="not valid Python"):
        assert list(tri.ingest_triton(str(tmp_path))) == []

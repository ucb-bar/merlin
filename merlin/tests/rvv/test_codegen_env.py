"""A configuration that does not determine the binary is not a configuration.

Two beam nodes carrying byte-identical knobs.yaml -- same compiler_features, same dtype_strategy,
same capture bundle -- emitted two different binaries (210dbfe9a01c44aa vs 2efd837676ff75cd) and ran
2,555,462 ns against 4,151,146 ns. Nothing in either run's artifacts named the environment the
compiler ran under, and the lowering path reads a couple of dozen MERLIN_* variables, several of
which steer codegen directly.
"""
from __future__ import annotations

from merlin.llvmlower import codegen_env as ce


def test_only_this_projects_variables_are_captured_and_unset_is_not_empty():
    env = {"MERLIN_PEROP_MR_CAP": "8", "PATH": "/usr/bin", "HOME": "/home/x", "MERLIN_VEC_RANK": ""}
    snap = ce.snapshot(env)
    assert snap == {"MERLIN_PEROP_MR_CAP": "8", "MERLIN_VEC_RANK": ""}
    # a variable that is NOT SET is omitted, never recorded as empty: "unset" and "set to empty" are
    # different inputs to the compiler and conflating them makes the record lie
    assert "MERLIN_OPU_ALIGN" not in ce.snapshot({})
    assert ce.snapshot({}) == {}


def test_a_secret_is_recorded_as_present_but_never_by_value():
    for name in ("MERLIN_K1_SSH_KEY", "MERLIN_API_TOKEN", "MERLIN_DB_PASSWORD", "MERLIN_SECRET_X"):
        snap = ce.snapshot({name: "s3kr3t-material"})
        assert snap[name] == "<redacted>", name
        assert "s3kr3t" not in str(snap)
    # that it was SET can be the thing that explains a difference, so presence survives redaction
    assert ce.differences(ce.snapshot({"MERLIN_K1_SSH_KEY": "a"}), ce.snapshot({}))


def test_a_payload_sized_value_is_summarised_not_inlined():
    big = "x" * 5000
    assert ce.snapshot({"MERLIN_ARGS": big})["MERLIN_ARGS"] == "<5000 chars>"


def test_capture_is_by_prefix_so_a_new_variable_cannot_be_silently_missed():
    """A curated name list goes stale exactly where it matters: the next variable someone adds is
    the one not on it, and its absence is silent."""
    src = open(ce.__file__).read()
    assert 'PREFIX = "MERLIN_"' in src
    assert "name.startswith(PREFIX)" in src
    # a variable this module has never heard of is still captured
    assert ce.snapshot({"MERLIN_SOMETHING_INVENTED_TOMORROW": "1"}) == {
        "MERLIN_SOMETHING_INVENTED_TOMORROW": "1"}


def test_differences_names_the_set_vs_unset_case():
    """The difference most likely to explain two binaries from one recorded configuration, and the
    easiest to miss, is a variable set on one side and absent on the other."""
    a = ce.snapshot({"MERLIN_PEROP_MR_CAP": "4"})
    b = ce.snapshot({})
    assert ce.differences(a, b) == {"MERLIN_PEROP_MR_CAP": ("4", None)}
    assert ce.differences(a, a) == {}
    assert ce.digest(a) != ce.digest(b)
    assert ce.digest(a) == ce.digest({"MERLIN_PEROP_MR_CAP": "4"})


def test_the_beam_and_the_runner_both_record_it():
    from merlin.common.paths import merlin_dir
    base = merlin_dir() / "python" / "merlin" / "mining"
    for name in ("beam.py", "runner.py"):
        src = (base / name).read_text()
        assert "codegen_env" in src, name
        assert "_codegen_env.digest()" in src, name

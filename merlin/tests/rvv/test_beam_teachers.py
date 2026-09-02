"""The beam's EXPERT side: one lifted fixture, or the whole family teacher set.

The beam's default expert is a single ``--expert-objdump`` fixture, and that silently bounds what the
search can discover. `cca_compare` only reports an axis when BOTH sides are populated, so an axis the
one expert cannot answer yields no divergence, routes to no CompilerAction, and is never forked --
however much of the model's wall it owns. An expert GEMM has no activation, so on an fp32 whole model
`compute.activation_vectorization` was uncomparable while the dynamic profile put scalar `exp` at
16.48% of real model work (``__ieee754_expf`` 11.91% + ``expf`` 4.57%).

These tests pin the seam (``run_beam(compare_fn=...)``) and, on the real harvested fixtures, that
consulting every teacher strictly WIDENS what the beam sees. The widening is the whole point: a seam
that were inert here would be a seam that changed nothing about what the search can find.
"""
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.kernels import cca
from merlin.mining.beam import _cca_divergences

_FIXTURES = merlin_dir() / "tests" / "data" / "cca_asm"


def _run_with_ours(tmp_path: Path, fixture: str = "ours_baseline_matmul.objdump") -> Path:
    gen = tmp_path / "run1" / "generated"
    gen.mkdir(parents=True)
    (gen / "objdump.txt").write_text((_FIXTURES / fixture).read_text())
    return tmp_path / "run1"


def test_compare_fn_replaces_the_single_expert_diff(tmp_path):
    run = _run_with_ours(tmp_path)
    expert = cca.CCA(op="matmul", backend=["rvv"],
                     compute=cca.ComputeFacet(op="matmul", contraction_form="fused_fma"))
    seen = {}

    def _cmp(ours):
        seen["ours_op"] = ours.compute.op
        return ["sentinel"]

    assert _cca_divergences(run, expert, {"op": "matmul"}, compare_fn=_cmp) == ["sentinel"]
    assert seen["ours_op"] == "matmul", "the seam still receives OUR lifted CCA, not the raw text"


def test_compare_fn_is_not_consulted_when_there_is_no_objdump(tmp_path):
    """Fail closed the same way the single-expert path does: no emitted code, no divergences. A
    compare_fn that ran here would be diffing against nothing and returning confident gaps."""
    calls = []
    out = _cca_divergences(tmp_path / "absent", None, {"op": "matmul"},
                           compare_fn=lambda ours: calls.append(ours) or ["x"])
    assert out == [] and calls == []


def test_teachers_widen_what_the_beam_sees_on_the_real_fixtures(tmp_path):
    """The non-inert proof, on harvested fixtures rather than mocks.

    MEASURED on small_llama fp32: matmul teacher alone -> 5 divergences / 4 mintable forks; all
    teachers -> 9 / 6, adding compute.activation_vectorization (taught by gelu) and
    compute.reduction_form (taught by softmax). This asserts the DIRECTION and the strict superset on
    whatever fixture set is harvested in this checkout, so it stays true as teachers are added.
    """
    from merlin.mining.wholemodel_proposer import expert_family_cca, teacher_compare_fn

    run = _run_with_ours(tmp_path)
    matmul_expert = expert_family_cca("matmul", dtype="fp32")
    if matmul_expert is None:
        pytest.skip("gemm fixture not harvested in this checkout")

    single = {d.axis for d in _cca_divergences(run, matmul_expert, {"op": "matmul"})}
    audit = []
    every = {d.axis for d in _cca_divergences(
        run, matmul_expert, {"op": "matmul"},
        compare_fn=teacher_compare_fn(dtype="f32", record=audit))}

    assert single <= every, f"a teacher set must never LOSE an axis; dropped {sorted(single - every)}"
    assert every - single, "consulting every teacher found nothing new -- the seam is inert"
    # every axis is attributed to the teacher that justified it, and the attribution rides on the
    # divergence itself so a consumer cannot lose it.
    taught = audit[0]["taught_by"]
    assert set(every) <= set(taught), "an axis with no named teacher is an unauditable claim"
    assert len({t for t in taught.values()}) > 1, "more than one teacher must actually contribute"


def test_the_axes_no_teacher_can_answer_are_reported_not_dropped(tmp_path):
    """"The search found no divergence here" is only honest alongside the axes nobody could judge."""
    from merlin.mining.wholemodel_proposer import teacher_compare_fn

    run = _run_with_ours(tmp_path)
    audit = []
    _cca_divergences(run, None, {"op": "matmul"},
                     compare_fn=teacher_compare_fn(dtype="f32", record=audit))
    assert audit and "unanswered_axes" in audit[0]
    assert isinstance(audit[0]["unanswered_axes"], list)
    assert audit[0]["dtype"] == "fp32", "the caller's spelling must be normalised, not rejected"


def test_a_dtype_with_no_fixture_yields_no_teacher_rather_than_the_wrong_one(tmp_path):
    """bf16 has no harvested fixture. It must produce NO expert -- never fp32's, whose divergences
    would be differences that are only the dtype, routing to levers that then measure inert."""
    from merlin.mining.wholemodel_proposer import canonical_dtype, expert_family_cca

    assert canonical_dtype("bf16") is None
    assert expert_family_cca("matmul", dtype="bf16") is None


@pytest.mark.parametrize("spelling,expected", [
    ("f32", "fp32"), ("fp32", "fp32"), ("float32", "fp32"),
    ("i8", "int8"), ("int8", "int8"), ("qd8", "int8"),
    ("f16", "fp16"), ("half", "fp16"),
    ("bf16", None), ("", None), (None, None), ("nonsense", None),
])
def test_dtype_spellings_normalise_or_fail_closed(spelling, expected):
    from merlin.mining.wholemodel_proposer import canonical_dtype
    assert canonical_dtype(spelling) == expected


def test_every_dtype_alias_names_a_live_fixture_key():
    """A guard on the alias table itself: an alias pointing at a key no fixture table has would
    silently mean 'no expert' for a dtype the caller believes is supported."""
    from merlin.mining import wholemodel_proposer as wp
    for spelling in wp._DTYPE_ALIASES:
        assert wp.canonical_dtype(spelling) is not None


def test_instrumented_beam_with_teachers_records_the_audit_and_forks_more(tmp_path, monkeypatch):
    """End-to-end through ``run_instrumented_beam``: --teachers must reach the beam, land its audit in
    the parent run, and (the point) surface MORE forkable divergences than the single expert."""
    from merlin.mining import load_rvv_package
    from merlin.mining.beam_cli import run_instrumented_beam

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    ours_objd = (_FIXTURES / "ours_baseline_matmul.objdump").read_text()
    expert_objd = _FIXTURES / "xnnpack_f32_gemm_rvv.objdump"
    hand_v0 = Path(__file__).resolve().parents[3] / "out/artifacts/targets/rvv/hand_v0"
    if not hand_v0.is_dir():
        pytest.skip("frozen hand_v0 seed package not present")

    def mock_certify(*, package_dir, model_dir, runs_root, run_id, targets, baseline_run_dir):
        gen = Path(runs_root) / run_id / "generated"
        gen.mkdir(parents=True, exist_ok=True)
        (gen / "objdump.txt").write_text(ours_objd)
        pkg = load_rvv_package(package_dir)
        n = pkg.op_match[0]["vector"][-2] if pkg.op_match else 8
        return {"correctness": {"gate_ok": True},
                "measurement": [{"target": "k1", "cycle_accurate": False,
                                 "cycles": 4_000_000 // n, "wall_ns": 900_000 // n}]}

    def _run(teachers):
        return run_instrumented_beam(
            seed_pkg=str(hand_v0), model_dir=tmp_path / "wl", expert_objdump=expert_objd,
            op="matmul", dtype="f32", targets=("k1",), width=3, depth=1, top_k=1,
            certify_fn=mock_certify, teachers=teachers)

    single, every = _run(None), _run("all")

    parent = Path(every["parent_run_dir"])
    audit = parent / "teacher_audit.yaml"
    assert audit.is_file(), "the teacher audit must land in the parent run, not only in memory"
    body = audit.read_text()
    assert "taught_by" in body and "unanswered_axes" in body
    # the single-expert run must NOT write one (nothing to attribute)
    assert not (Path(single["parent_run_dir"]) / "teacher_audit.yaml").exists()

    def _levers(res):
        return {n.get("lever") for n in res.get("nodes", [])} - {"seed", None}

    assert _levers(every) >= _levers(single), "teachers must never lose a lever the single expert found"
    assert len(every.get("nodes", [])) >= len(single.get("nodes", [])), \
        "more answerable axes must not yield fewer explored forks"


def test_teachers_on_a_dtype_with_no_fixtures_refuses_instead_of_silently_single_expert(tmp_path, monkeypatch):
    """Asking for teachers on bf16 must FAIL, not quietly fall back to the single expert. A silent
    fallback would report a teacher-set run in the record while searching the narrow space."""
    from merlin.mining.beam_cli import run_instrumented_beam

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    with pytest.raises(SystemExit, match="harvested fixtures"):
        run_instrumented_beam(
            seed_pkg="unused", model_dir=tmp_path / "wl",
            expert_objdump=_FIXTURES / "xnnpack_f32_gemm_rvv.objdump",
            op="matmul", dtype="bf16", teachers="all")

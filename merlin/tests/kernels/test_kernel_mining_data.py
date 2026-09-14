"""Kernel-mining facts that used to be Python literals are DATA, so a new ISA family or expert corpus is a
data edit, not a core one: the motif-marker table and the target -> ISA-family map live in
``framework_contracts/feature_extraction/<family>.yaml``; expert-corpus locations and source aliases live in
the corpus registry ``merlin/contract/corpora.yaml`` (read only by ``merlin.targetgen.corpora``)."""
import pytest

from merlin.common.paths import merlin_dir
from merlin.kernels import build_asm as B
from merlin.kernels import framework_contracts as FC
from merlin.kernels import markers as M
from merlin.targetgen import corpora as C


def _clear_marker_caches():
    FC.load_feature_contract.cache_clear()
    M._target_families.cache_clear()
    M._compiled_for_family.cache_clear()


@pytest.fixture
def feature_dir(tmp_path, monkeypatch):
    """A private feature_extraction dir, swapped in for the shipped one; caches rebuilt both ways."""
    monkeypatch.setattr(FC, "_FEATURE_DIR", tmp_path)
    _clear_marker_caches()
    yield tmp_path
    monkeypatch.undo()
    _clear_marker_caches()


def test_every_declared_motif_is_in_the_vocabulary():
    for fam in FC.feature_families():
        assert set(FC.load_feature_contract(fam).get("markers") or {}) <= set(M.MOTIFS), fam


def test_a_new_isa_family_is_one_data_file(feature_dir):
    for f in (FC.feature_families.__globals__["_DIR"] / "feature_extraction").glob("*.yaml"):
        (feature_dir / f.name).write_text(f.read_text())
    (feature_dir / "toyisa.yaml").write_text(
        "family: toyisa\ntargets: [toyisa, Toy_Accel]\n"
        "markers:\n  accumulator_lifetime: ['toy_mac\\w*']\n")
    assert M.target_family("toy_accel") == "toyisa"
    assert M.target_family("rvv") == "rvv"                       # the shipped families still resolve
    fired = M.fired_markers("for (i = 0; i < n; i++) toy_mac_acc(x);", "TOY_ACCEL")
    assert fired == {"accumulator_lifetime": ["toy_mac_acc"], "tiling_blocking": ["for ("]}


def test_a_target_claimed_by_two_families_is_refused(feature_dir):
    (feature_dir / "a.yaml").write_text("targets: [dup]\n")
    (feature_dir / "b.yaml").write_text("targets: [DUP]\n")
    with pytest.raises(ValueError, match="claimed by two"):
        M.target_family("dup")


def test_a_misspelled_motif_is_refused_not_silently_dropped(feature_dir):
    (feature_dir / "x.yaml").write_text("targets: [x]\nmarkers:\n  pakced_rhs: ['a']\n")
    with pytest.raises(ValueError, match="unknown motif"):
        M.markers_for("x")


def test_source_aliases_name_exactly_one_corpus():
    seen = {}
    for name, spec in C.kernel_corpora().items():
        for alias in spec.get("sources") or []:
            assert seen.setdefault(alias.lower(), name) == name, alias
            assert C.kernel_corpus_for_source(alias.upper()) == name
    assert C.kernel_corpus_for_source("") is None
    assert C.kernel_corpus_for_source("no_such_source") is None


def test_canonical_capsule_corpus_is_first():
    assert C.capsule_corpus_roots()[0] == merlin_dir() / "contract" / "capsules"


def test_a_new_corpus_is_a_registry_entry(tmp_path, monkeypatch):
    reg = {"capsule_corpora": [], "kernel_corpora": {"toyblas": {
        "sources": ["toyblas", "toy-blas"], "layout": "single_tu", "checkout": "ToyBLAS",
        "include_subdirs": ["", "inc"]}}}
    monkeypatch.setattr(C, "_registry", lambda: reg)
    monkeypatch.setenv(C.kernel_corpus_env("toyblas"), str(tmp_path))
    assert C.kernel_corpus_env("toyblas") == "MERLIN_TOYBLAS_REPO"
    assert B.framework_include_roots("Toy-BLAS") == [tmp_path, tmp_path / "inc"]
    assert B.benchmark_source() is None                  # no standalone-benchmark corpus declared
    assert B.benchmarks_dir() is None
    assert B.corpus_root("nope") is None


def test_env_var_wins_over_every_other_location(tmp_path, monkeypatch):
    name = B.benchmark_source()
    assert name is not None
    monkeypatch.setenv(C.kernel_corpus_env(name), str(tmp_path))
    assert B.corpus_root(name) == tmp_path
    spec = C.kernel_corpora()[name]
    assert B.benchmarks_dir() == tmp_path / spec["benchmarks"]

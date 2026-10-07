"""A cached golden must be invalidated by everything that could change it.

Goldens are derived and deterministic -- operands come from a name-salted fill with no RNG -- so the
engines are memoizable. They are also deliberately slow: ``fp_reduce`` accumulates in the device's own
order, one step at a time, in pure Python, because a numpy dot product rounds differently from the
hardware. A repeated generation must reuse an answer only when every numerical source is identical.

⚠️ The risk is not a slow cache, it is a STALE one. A stale golden does not fail loudly -- it grades a
backend against the wrong answer, the exact failure class this corpus exists to catch. So these tests
check INVALIDATION by mutation rather than checking that a hit is fast: entry, binding, engine,
operand synthesis, external oracle, Phase 0 source, and device facts must move the key when changed.
"""

from __future__ import annotations

import os
import types
from functools import cache

import external_sources
import pytest
from merlin_experiments.phase0 import numerics as NUMERICS
from merlin_experiments.phase0.declarations import for_target

pytestmark = pytest.mark.target("atlas", "gemmini", "muon", "mx_gemmini", "radiance", "saturn")


@pytest.fixture(scope="module")
def gen():
    from merlin_experiments.phase0 import golden_cache

    return golden_cache


class _Binding:
    operand_dtype = "f32"
    tile_dim = 16


def _entry(**over):
    base = {
        "name": "GOLDEN_CACHE_PROBE",
        "op": "conv2d",
        "ci": 4,
        "N": 8,
        "Himg": 6,
        "Wimg": 6,
        "kh": 3,
        "kw": 3,
        "stride": [1, 1],
        "padding": [1, 1, 1, 1],
        "dilation": [1, 1],
        "layout": "nhwc",
        "ifm": "IFM",
        "weight": "W",
        "out": "Y0",
    }
    base.update(over)
    return base


class TestTheCacheAnswersWithTheSameThingItComputed:
    def test_a_hit_returns_the_computed_result_unchanged(self, gen):
        first = gen._golden_cached(NUMERICS._simt_golden, _entry(), _Binding())
        second = gen._golden_cached(NUMERICS._simt_golden, _entry(), _Binding())
        assert first == second

    def test_a_hit_matches_the_uncached_engine(self, gen):
        """The cache must be indistinguishable from calling the engine directly."""
        cached = gen._golden_cached(NUMERICS._simt_golden, _entry(), _Binding())
        direct = NUMERICS._simt_golden(_entry(), _Binding())
        assert cached == direct


class TestEveryInputThatMovesAGoldenMovesTheKey:
    def test_a_different_entry_misses(self, gen):
        assert gen._golden_cache_key(NUMERICS._simt_golden, _entry(), _Binding()) != gen._golden_cache_key(
            NUMERICS._simt_golden, _entry(kh=5), _Binding()
        )

    def test_a_different_binding_misses(self, gen):
        class Other(_Binding):
            operand_dtype = "bf16"

        assert gen._golden_cache_key(NUMERICS._simt_golden, _entry(), _Binding()) != gen._golden_cache_key(
            NUMERICS._simt_golden, _entry(), Other()
        )

    def test_a_different_engine_misses(self, gen):
        assert gen._golden_cache_key(NUMERICS._simt_golden, _entry(), _Binding()) != gen._golden_cache_key(
            NUMERICS._float_golden, _entry(), _Binding()
        )

    def test_an_edit_to_the_engine_SOURCE_misses(self, gen, tmp_path, monkeypatch):
        """THE LOAD-BEARING ONE, checked by MUTATION.

        A conv2d branch was added to the SIMT engine on 2026-09-05 and an attention statement to the
        composed micro model. A key that did not read the engine's bytes would have served the
        pre-change goldens straight through both edits.
        """
        before = gen._golden_cache_key(NUMERICS._simt_golden, _entry(), _Binding())
        real = gen._source_digest_of

        monkeypatch.setattr(
            gen, "_source_digest_of", lambda obj: "MUTATED" if obj is NUMERICS._simt_golden else real(obj)
        )
        after = gen._golden_cache_key(NUMERICS._simt_golden, _entry(), _Binding())
        assert after != before, "an engine source change did not invalidate the cache"

    def test_an_edit_to_the_operand_SYNTHESIS_misses(self, gen, monkeypatch):
        """Operands are half the answer; a changed fill changes every golden built from it."""
        from merlin.targetgen import corpus_operands as CO

        before = gen._golden_cache_key(NUMERICS._simt_golden, _entry(), _Binding())
        real = gen._source_digest_of
        monkeypatch.setattr(gen, "_source_digest_of", lambda obj: "MUTATED" if obj is CO else real(obj))
        assert gen._golden_cache_key(NUMERICS._simt_golden, _entry(), _Binding()) != before


class TestTheCacheFailsOpenNeverWrong:
    def test_a_damaged_entry_is_a_miss_not_an_error(self, gen, monkeypatch, tmp_path):
        """A corrupt cache file must recompute, never propagate or raise."""
        monkeypatch.setattr("merlin.common.artifacts.cache_dir", lambda _ns: tmp_path)
        key = gen._golden_cache_key(NUMERICS._simt_golden, _entry(), _Binding())
        victim = tmp_path / key[:2] / f"{key}.json"
        victim.parent.mkdir(parents=True, exist_ok=True)
        victim.write_text("{ this is not json", encoding="utf-8")
        assert gen._golden_cached(NUMERICS._simt_golden, _entry(), _Binding()) == NUMERICS._simt_golden(
            _entry(), _Binding()
        )

    def test_the_escape_hatch_bypasses_the_cache(self, gen, monkeypatch):
        monkeypatch.setattr(gen, "_GOLDEN_CACHE_DISABLED", True)
        assert gen._golden_cached(NUMERICS._simt_golden, _entry(), _Binding()) == NUMERICS._simt_golden(
            _entry(), _Binding()
        )


class TestItIsTargetAgnostic:
    def test_the_key_names_no_target(self, gen):
        """The cardinal rule: a cache keyed on a target name would be an overfit by construction."""
        import inspect

        src = inspect.getsource(gen._golden_cache_key) + inspect.getsource(gen._golden_cached)
        for name in ("gemmini", "atlas", "radiance", "saturn", "muon", "mx_gemmini"):
            assert name not in src.lower(), f"the cache mentions {name!r}"


class TestTheDeviceIsPartOfTheKey:
    """A changed RTL must not be answered from a cache built against the previous one.

    The goldens are deliberately independent of the RTL -- an oracle derived from the device would be
    the device grading itself -- but the RTL reaches them INDIRECTLY: the binding's tile edge, dtypes
    and subnormal handling come from the capability manifest, and an entry's extents come from facts
    like memory capacity and array geometry. So the key carries the target's RTL-facts digest and
    invalidates conservatively.
    """

    def test_a_different_facts_digest_misses(self, gen):
        a = gen._golden_cache_key(NUMERICS._simt_golden, _entry(), _Binding(), "facts-rev-A")
        b = gen._golden_cache_key(NUMERICS._simt_golden, _entry(), _Binding(), "facts-rev-B")
        assert a != b, "a changed RTL revision did not invalidate the cache"

    def test_an_unknown_digest_is_its_own_key_not_a_wildcard(self, gen):
        """An empty digest means the caller could not establish which device this is. It must not
        collide with a known revision, or an unprovenanced run would be served a provenanced answer."""
        unknown = gen._golden_cache_key(NUMERICS._simt_golden, _entry(), _Binding(), "")
        known = gen._golden_cache_key(NUMERICS._simt_golden, _entry(), _Binding(), "facts-rev-A")
        assert unknown != known


class TestTheMemoizedProductIsBitIdentical:
    @external_sources.requires_rtl("atlas")
    def test_the_product_cache_changes_no_value(self, gen):
        """``rnd(dec(a)*dec(b))`` is a pure function of the operand code pair, so memoizing it on that
        pair is identical by construction. Pinned because it sits inside a golden engine, where a
        'small' numeric difference is a wrong answer that grades a backend."""
        import yaml

        from merlin.common.paths import repo_root as _rr
        from merlin.targetgen.corpus_spec import derive_binding
        from merlin.targetgen.target_experiment import load_target_experiment

        desc = _rr() / "merlin/experiments/capsule_bench/targets/atlas/target_experiment.yaml"
        prof = for_target("atlas").recipe
        if not desc.is_file() or not prof.is_file():
            pytest.skip("this checkout has no float-regime target to exercise")
        eb = derive_binding(load_target_experiment(desc), (yaml.safe_load(prof.read_text()) or {}).get("datapath", {}))
        entry = {
            "name": "PRODCACHE_PROBE",
            "op": "matmul",
            "M": eb.tile_dim,
            "K": 64,
            "N": eb.tile_dim,
            "lhs": "A0",
            "weight": "W",
            "out": "Y0",
        }
        outputs, _prov = NUMERICS._float_golden(entry, eb)
        assert outputs, "the engine produced no output tensor"
        # `outputs` is keyed by tensor name; flatten the ROWS, not the keys
        flat = [v for block in outputs.values() for row in block for v in row]
        assert flat, "the output tensor is empty"
        assert len(set(flat)) > 1, "a constant golden grades nothing"


def test_phase0_source_identity_participates_in_cache_key(gen, monkeypatch):
    monkeypatch.setattr(gen, "source_digest", lambda: "first-source-closure")
    first = gen._golden_cache_key(NUMERICS._simt_golden, _entry(), _Binding())
    monkeypatch.setattr(gen, "source_digest", lambda: "second-source-closure")
    assert gen._golden_cache_key(NUMERICS._simt_golden, _entry(), _Binding()) != first


@pytest.mark.parametrize(
    "source",
    [
        "__init__.py",
        "__main__.py",
        "profiles.py",
        "numerics.py",
        "golden_cache.py",
        "writer.py",
        "sweeps.py",
        "provenance.py",
        "generation.py",
    ],
)
def test_each_extracted_helper_source_changes_key_and_missing_owner_disables_cache(gen, monkeypatch, tmp_path, source):
    for original in gen.source_files():
        relative = original.relative_to(original.parent)
        (tmp_path / relative).write_bytes(original.read_bytes())
    monkeypatch.setattr(gen, "__file__", str(tmp_path / "golden_cache.py"))
    before = gen._golden_cache_key(NUMERICS._simt_golden, _entry(), _Binding())
    helper = tmp_path / source
    helper.write_bytes(helper.read_bytes() + b"\n# synthetic helper mutation\n")
    assert gen._golden_cache_key(NUMERICS._simt_golden, _entry(), _Binding()) != before
    helper.unlink()
    assert gen.source_digest() == "unresolvable"


def test_unknown_source_closure_recomputes_without_reading_or_writing_cache(gen, monkeypatch):
    calls = []

    def engine(entry, binding):
        calls.append(entry)
        return {"result": len(calls)}, {"source": "synthetic"}

    monkeypatch.setattr(gen, "_GOLDEN_CACHE_DISABLED", False)
    monkeypatch.setattr(gen, "_source_digest_of", lambda obj: "known-engine")
    monkeypatch.setattr(gen, "source_digest", lambda: "unresolvable")
    monkeypatch.setattr("merlin.common.artifacts.cache_dir", lambda *args: pytest.fail("unresolved cache accessed"))
    assert gen._golden_cached(engine, _entry(), _Binding())[0] == {"result": 1}
    assert gen._golden_cached(engine, _entry(), _Binding())[0] == {"result": 2}
    assert len(calls) == 2


def test_selected_external_specir_source_mutation_invalidates_float_golden_key(gen, monkeypatch, tmp_path):
    """The independent oracle is external: either edited module must move the key."""
    dtypes_path = tmp_path / "dtypes.py"
    refmodel_path = tmp_path / "refmodel.py"
    dtypes_path.write_text("# codec v1\n", encoding="utf-8")
    refmodel_path.write_text("def fp_reduce(): return 1\n", encoding="utf-8")
    dtypes = types.ModuleType("specir.oracle.dtypes")
    dtypes.__file__ = str(dtypes_path)
    namespace = {}
    exec(compile(refmodel_path.read_bytes(), str(refmodel_path), "exec"), namespace)
    monkeypatch.setattr(NUMERICS, "_specir", lambda **_kw: (dtypes, namespace["fp_reduce"]))
    selected = {
        "model": {"engine": "specir_fp_reduce", "source_root_path": str(tmp_path)},
        "selection_status": "explicit",
    }

    first_identity = NUMERICS.specir_oracle_source_identity(selected)
    first = gen._golden_cache_key(NUMERICS._float_golden, _entry(), _Binding(), oracle_source=first_identity)
    dtypes_path.write_text("# codec v2\n", encoding="utf-8")
    second_identity = NUMERICS.specir_oracle_source_identity(selected)
    second = gen._golden_cache_key(NUMERICS._float_golden, _entry(), _Binding(), oracle_source=second_identity)
    assert first != second

    refmodel_path.write_text("def fp_reduce(): return 2\n", encoding="utf-8")
    third_identity = NUMERICS.specir_oracle_source_identity(selected)
    third = gen._golden_cache_key(NUMERICS._float_golden, _entry(), _Binding(), oracle_source=third_identity)
    assert second != third
    assert set(third_identity["modules"]) == {"specir.oracle.dtypes", "specir.oracle.refmodel"}


def test_selected_external_specir_source_unavailable_fails_closed(gen, monkeypatch, tmp_path):
    dtypes_path = tmp_path / "dtypes.py"
    refmodel_path = tmp_path / "refmodel.py"
    dtypes_path.write_text("# codec\n", encoding="utf-8")
    refmodel_path.write_text("def fp_reduce(): return 1\n", encoding="utf-8")
    dtypes = types.ModuleType("specir.oracle.dtypes")
    dtypes.__file__ = str(dtypes_path)
    namespace = {}
    exec(compile(refmodel_path.read_bytes(), str(refmodel_path), "exec"), namespace)
    monkeypatch.setattr(NUMERICS, "_specir", lambda **_kw: (dtypes, namespace["fp_reduce"]))
    selected = {
        "model": {"engine": "specir_fp_reduce", "source_root_path": str(tmp_path)},
        "selection_status": "explicit",
    }
    refmodel_path.unlink()
    with pytest.raises(OSError, match="source.*unavailable"):
        NUMERICS.specir_oracle_source_identity(selected)
    with pytest.raises(ValueError, match="source identity is required"):
        gen._golden_cached(NUMERICS._float_golden, _entry(), _Binding())


@pytest.mark.parametrize("rm", ["rne", "rmm", "rtz", "rdn", "rup"])
@pytest.mark.parametrize(
    "order,cadence",
    [
        ("index_sequential", "per_step"),
        ("tree", "per_step"),
        ("index_sequential", "single_final"),
    ],
)
def test_exact_pair_fold_preserves_specir_special_values_and_rounding(rm, order, cadence):
    try:
        D, fp_reduce = NUMERICS._specir()
    except ImportError:
        pytest.skip("independent SpecIR oracle is not installed")
    reduce, clear = NUMERICS._float_reducer(fp_reduce, D.BF16, order=order, cadence=cadence, rm=rm)
    sequences = [
        [],
        [0x8000],
        [0x8000, 0x0000],
        [0x0000, 0x8000],
        [0x7FC1, 0x0000],
        [0x7F80, 0xFF80, 0x0000],
        [0x3F80, 0xBF80, 0x8000, 0x0000],
        [0x3F80, 0x0001, 0xBF80],
    ]
    for values in sequences:
        assert reduce(values) == fp_reduce(values, D.BF16, order=order, cadence=cadence, rm=rm)
    clear()


def test_exact_pair_fold_matches_full_selected_32x32_tile():
    """Opt-in long differential check over a real full-tile deterministic stimulus."""
    if os.environ.get("MERLIN_TEST_FULL_FLOAT_TILE") != "1":
        pytest.skip("set MERLIN_TEST_FULL_FLOAT_TILE=1 for the 32x32x4096 differential check")
    try:
        D, fp_reduce = NUMERICS._specir()
    except ImportError:
        pytest.skip("independent SpecIR oracle is not installed")
    m = n = 32
    k = 4096
    name = "PR01_fits_double_k4096"
    a, _ = NUMERICS._det_fp8(D, "A0", (m, k), name, "fp8_e4m3", D.FP8_E4M3)
    w, _ = NUMERICS._det_fp8(D, "W", (k, n), name, "fp8_e4m3", D.FP8_E4M3)
    decode = NUMERICS._operand_decoder(D, D.FP8_E4M3, flush_subnormals=True)

    @cache
    def product(x, y):
        return D.round_to_format(decode(x) * decode(y), D.BF16, "rne")

    reduce, clear = NUMERICS._float_reducer(fp_reduce, D.BF16, order="index_sequential", cadence="per_step", rm="rne")
    for i in range(m):
        for j in range(n):
            values = [product(a[i * k + p], w[p * n + j]) for p in range(k)]
            expected = fp_reduce(values, D.BF16, order="index_sequential", cadence="per_step", rm="rne")
            assert reduce(values) == expected, f"mismatched full-tile output ({i}, {j})"
    clear()


def test_vectorized_fold_equals_the_scalar_ordered_fold():
    """`ordered_fold` runs every lane at once but must fold in exactly the scalar index order."""
    import numpy as np

    rng = np.random.default_rng(7)
    m, k, n = 5, 37, 4
    lhs_codes, rhs_codes = rng.integers(0, 9, size=(m, k)), rng.integers(0, 7, size=(k, n))
    table = rng.integers(0, 1 << 16, size=(9, 7))

    def step(acc, addend):  # deliberately order-sensitive and non-associative
        return (acc * 31 + addend * 7 + (acc ^ addend)) % 65521

    folded = NUMERICS.ordered_fold(table, lhs_codes, rhs_codes, step)
    for i in range(m):
        for j in range(n):
            acc = int(table[lhs_codes[i, 0], rhs_codes[0, j]])
            for p in range(1, k):
                acc = step(acc, int(table[lhs_codes[i, p], rhs_codes[p, j]]))
            assert folded[i, j] == acc
    with pytest.raises(ValueError, match="raw codes"):
        NUMERICS.ordered_fold([[-1]], np.zeros((1, 1), dtype=int), np.zeros((1, 1), dtype=int), step)


def test_vectorized_specir_fold_matches_the_scalar_reducer():
    try:
        D, fp_reduce = NUMERICS._specir()
    except ImportError:
        pytest.skip("independent SpecIR oracle is not installed")
    import numpy as np

    m, k, n = 4, 64, 3
    a, _ = NUMERICS._det_fp8(D, "A0", (m, k), "fold_check", "fp8_e4m3", D.FP8_E4M3)
    w, _ = NUMERICS._det_fp8(D, "W", (k, n), "fold_check", "fp8_e4m3", D.FP8_E4M3)
    decode = NUMERICS._operand_decoder(D, D.FP8_E4M3, flush_subnormals=True)
    reduce, clear = NUMERICS._float_reducer(fp_reduce, D.BF16, order="index_sequential", cadence="per_step", rm="rne")
    lhs, rhs = np.asarray(a).reshape(m, k), np.asarray(w).reshape(k, n)
    lc, li = np.unique(lhs, return_inverse=True)
    rc, ri = np.unique(rhs, return_inverse=True)
    table = [[D.round_to_format(decode(int(x)) * decode(int(y)), D.BF16, "rne") for y in rc] for x in lc]
    folded = NUMERICS.ordered_fold(table, li.reshape(m, k), ri.reshape(k, n), reduce.step)
    for i in range(m):
        for j in range(n):
            values = [D.round_to_format(decode(a[i * k + p]) * decode(w[p * n + j]), D.BF16, "rne") for p in range(k)]
            assert folded[i, j] == reduce(values)
    clear()

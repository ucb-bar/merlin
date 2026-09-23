"""A cached golden must be invalidated by everything that could change it.

Goldens are derived and deterministic -- operands come from a name-salted fill with no RNG -- so the
engines are memoizable. They are also deliberately slow: ``fp_reduce`` accumulates in the device's own
order, one step at a time, in pure Python, because a numpy dot product rounds differently from the
hardware. Measured 2026-09-05: an atlas regeneration spends ~90 minutes in those engines and a radiance
one ~40, nearly all of it recomputing capsules nothing changed.

⚠️ The risk is not a slow cache, it is a STALE one. A stale golden does not fail loudly -- it grades a
backend against the wrong answer, the exact failure class this corpus exists to catch. So these tests
check INVALIDATION by mutation rather than checking that a hit is fast: each of the four inputs that
determines a golden (entry, binding, engine, operand synthesis) must move the key, and an engine edit
must be caught from the bytes ON DISK so that work in progress invalidates its own cached results.
"""

from __future__ import annotations

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

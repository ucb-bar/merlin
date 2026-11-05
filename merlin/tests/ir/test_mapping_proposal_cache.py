"""Only option paths survive mapping reuse; current IR and effects stay fresh."""
from dataclasses import replace
import hashlib

import pytest
from test_pass_exploration import emission, facts, families, source

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.pass_exploration import (
    ExplorationRefused, MappingProposalCache, MappingProposalKey,
    explore_candidates, generate_candidates,
)
from merlin.transforms.pipeline_policy import RegisteredPass, ShapePredicate


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def key():
    return MappingProposalKey(sha('operation semantics and attrs'), ((32, 64), (64, 48), (32, 48)),
        ('row_major',)*3, ('i8', 'i8', 'i32'), ('bias',),
        sha('selected hardware'), sha('selected library'), sha('exact numerical policy'))


def test_cached_search_rebinds_current_factories_and_emits_both_independent_sources():
    cache = MappingProposalCache(max_entries=2)
    counts = []
    def current_families():
        def wrap(family):
            def bind(options):
                counts.append(family.name)
                return family.bind(options)
            return replace(family, bind=bind)
        return tuple(wrap(family) for family in families())
    first = generate_candidates(current_families(), output='program', proposal_cache=cache, proposal_key=key())
    original_calls = len(counts)
    second = generate_candidates(current_families(), output='program', proposal_cache=cache, proposal_key=key())
    assert not first.proposal_cache_hit and second.proposal_cache_hit
    assert second.search_steps < first.search_steps and len(counts) == 2*original_calls
    assert first.variants == second.variants and (cache.hits, cache.misses, len(cache)) == (1, 1, 1)
    # Independent SSA/source content, identical operation mapping geometry.
    independent = parse_ir('''builtin.module {func.func @g() -> i32 {
      %a = arith.constant 7 : i32
      %b = arith.constant 4 : i32
      %v = arith.addi %a, %b : i32
      func.return %v : i32
    }}''')
    a = list(explore_candidates(source(), first, facts, output='program', emission=emission))
    b = list(explore_candidates(independent, second, facts, output='program', emission=emission))
    assert all(row.evaluation.status == 'compiled' for row in (*a, *b))
    assert {row.emission.data for row in a}.isdisjoint(row.emission.data for row in b)
    # Cache data is bounded immutable option keys, without IR/actions/evidence.
    assert all(type(name) is str for paths, truncated in cache._records.values()
               for path in paths for name in path)
    assert all(type(truncated) is bool for paths, truncated in cache._records.values())


def test_cached_paths_use_current_predicates_and_current_prerequisites():
    cache = MappingProposalCache()
    generate_candidates(families(), output='program', proposal_cache=cache, proposal_key=key())
    guarded = generate_candidates(families((ShapePredicate(0, 0, at_least=128),)),
        output='program', proposal_cache=cache, proposal_key=key())
    assert guarded.proposal_cache_hit
    rows = list(explore_candidates(source(), guarded, facts, output='program', emission=emission))
    assert all(row.evaluation.status == 'unsupported' and row.emission is None for row in rows)
    old = families()
    def missing(options):
        bound = old[-1].bind(options)
        return RegisteredPass(replace(bound.spec, requires=('missing_current_analysis',)), bound.action)
    with pytest.raises(ExplorationRefused):
        generate_candidates((*old[:-1], replace(old[-1], bind=missing)), output='program',
            proposal_cache=cache, proposal_key=key())
    assert cache.invalidations == 1


@pytest.mark.parametrize('field,value', [
    ('semantic_sha256', sha('changed semantics')), ('shapes', ((33, 64), (64, 48), (33, 48))),
    ('layouts', ('blocked',)*3), ('dtypes', ('i16', 'i8', 'i32')), ('epilogue', ('relu',)),
    ('hardware_sha256', sha('other hardware')), ('library_sha256', sha('other library')),
    ('numerical_sha256', sha('other numerics')),
])
def test_changed_mapping_domain_misses(field, value):
    cache = MappingProposalCache()
    generate_candidates(families(), output='program', proposal_cache=cache, proposal_key=key())
    value = generate_candidates(families(), output='program', proposal_cache=cache,
        proposal_key=replace(key(), **{field: value}))
    assert not value.proposal_cache_hit and cache.misses == 2


def test_search_limits_domains_and_eviction_do_not_reuse_incomplete_requests():
    cache = MappingProposalCache(max_entries=1)
    a = generate_candidates(families(), output='program', max_candidates=1,
        proposal_cache=cache, proposal_key=key())
    b = generate_candidates(families(), output='program', max_candidates=2,
        proposal_cache=cache, proposal_key=key())
    c = generate_candidates(families(), output='program', max_candidates=1,
        proposal_cache=cache, proposal_key=key())
    assert not any(row.proposal_cache_hit for row in (a, b, c)) and len(cache) == 1
    assert len(a.variants) == len(c.variants) == 1 and len(b.variants) == 2
    with pytest.raises(ValueError, match='cache and complete domain'):
        generate_candidates(families(), output='program', proposal_cache=cache)
    with pytest.raises(ValueError, match='exact semantic'):
        replace(key(), hardware_sha256='unknown')

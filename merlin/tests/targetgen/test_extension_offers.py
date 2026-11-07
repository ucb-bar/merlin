from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.llvmlower.matrix_route_cost import CostModelSelector, UnitGeometry
from merlin.targetgen.extension_offers import (
    MatrixExtension,
    OfferError,
    OperationOffer,
    load_extensions,
    select_extension,
    select_offer,
)


def test_two_offers_select_cheaper_and_reject_tie():
    a = OperationOffer("kodiak", "opu", 10)
    b = OperationOffer("iris", "vme", 9)
    assert select_offer((a, b), fallback_cycles=20) == b
    with pytest.raises(OfferError, match="equal-cost offers"):
        select_offer((a, OperationOffer("iris", "vme", 10)), fallback_cycles=20)


def test_fallback_and_duplicate_offers():
    a = OperationOffer("kodiak", "opu", 10)
    assert select_offer((a,), fallback_cycles=10) is None
    assert select_offer((), fallback_cycles=10) is None
    with pytest.raises(OfferError, match="duplicate offer"):
        select_offer((a, a), fallback_cycles=20)


class _Provider:
    API_VERSION = 1

    def geometry(self, *, unit, config):
        assert (unit, config) == ("opu", "cfg")
        return 64, 16

    def rewrite_prepared_file(self, prepared, work, *, select, tile_edge):
        return prepared, work, select, tile_edge


class _Routing:
    support_target = "kodiak"
    unit = "opu"
    config = "cfg"
    select = None
    policy = None

    def __init__(self, provider=None):
        self._provider = provider or _Provider()

    def provider(self):
        return self._provider

    def tile_edge(self):
        return 64


@pytest.mark.parametrize("m,n,k", [(8, 2048, 64), (64, 64, 128), (1, 1, 1)])
def test_matrix_adapter_preserves_existing_route_and_cost_receipt(m, n, k):
    shape = SimpleNamespace(parallel=(m, n), reduction=(k,))
    old = CostModelSelector(UnitGeometry(64, 16)).decide(shape)
    ext = MatrixExtension(_Routing())
    new = ext.selector().decide(shape)
    assert new.to_dict() == old.to_dict()
    assert ext.selector()(shape) == old.route
    assert ext.lower("ir", "work", select=lambda _: True)[0:2] == ("ir", "work")


def test_matrix_extension_rejects_incompatible_api():
    provider = _Provider()
    provider.API_VERSION = 2
    with pytest.raises(OfferError, match="API version"):
        MatrixExtension(_Routing(provider))

    class Missing(_Provider):
        API_VERSION = None

    with pytest.raises(OfferError, match="API version"):
        MatrixExtension(_Routing(Missing()))


def test_matrix_extension_keeps_provider_identity():
    ext = MatrixExtension(_Routing())
    offer = ext.offers(SimpleNamespace(parallel=(64, 64), reduction=(64,)))[0]
    assert (offer.owner, offer.unit) == ("kodiak", "opu")


def test_explicit_boolean_selector_keeps_legacy_behavior():
    routing = _Routing()
    routing.select = lambda shape: shape.parallel[0] >= 8
    selector = MatrixExtension(routing).selector()
    assert not hasattr(selector, "decide")
    assert selector(SimpleNamespace(parallel=(8, 3)))
    assert not selector(SimpleNamespace(parallel=(7, 3)))


def test_two_extensions_compose_and_dispatch_lowering():
    class Extension:
        API_VERSION = 1

        def __init__(self, owner, cycles):
            self.owner, self.cycles = owner, cycles

        def offers(self, operation, *, layout_elements=None):
            assert operation == "contraction"
            return (OperationOffer(self.owner, "unit", self.cycles),)

        def lower(self, prepared, work, *, select):
            assert select("contraction")
            return self.owner, prepared, work

    slower, faster = Extension("slower", 12), Extension("faster", 8)
    selected = select_extension((slower, faster), "contraction", fallback_cycles=15)
    assert selected is not None
    extension, offer = selected
    assert offer.owner == "faster"
    assert extension.lower("ir", "work", select=lambda op: op == "contraction") == ("faster", "ir", "work")
    assert select_extension((slower, faster), "contraction", fallback_cycles=7) is None


def test_extension_api_version_is_checked_before_offers():
    class Incompatible:
        API_VERSION = 2

        def offers(self, operation, *, layout_elements=None):
            raise AssertionError("must fail before loading offers")

    with pytest.raises(OfferError, match="API version"):
        select_extension((Incompatible(),), "op", fallback_cycles=10)


def test_declared_extension_chain_binds_from_each_owner(monkeypatch):
    from merlin.targetgen import plugins

    context = object()
    bound = []

    def factory(owner):
        def bind(got):
            assert got is context
            bound.append(owner)
            return SimpleNamespace(API_VERSION=1, offers=lambda *_args, **_kw: (), lower=lambda *_args, **_kw: ())

        return SimpleNamespace(API_VERSION=1, bind=bind)

    def chain(target, key):
        return (("rvv", factory("rvv")), ("chip", factory("chip"))) if (target, key) == ("chip", "extension") else ()

    monkeypatch.setattr(plugins, "load_declared_chain", chain)
    assert len(load_extensions("chip", context)) == 2
    assert bound == ["rvv", "chip"]


def test_declared_extension_requires_version_and_factory(monkeypatch):
    from merlin.targetgen import plugins

    monkeypatch.setattr(plugins, "load_declared_chain", lambda *_: (("chip", SimpleNamespace(API_VERSION=2)),))
    with pytest.raises(OfferError, match="API version"):
        load_extensions("chip", object())
    monkeypatch.setattr(plugins, "load_declared_chain", lambda *_: (("chip", SimpleNamespace(API_VERSION=1)),))
    with pytest.raises(OfferError, match="bind"):
        load_extensions("chip", object())

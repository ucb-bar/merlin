from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.llvmlower.matrix_route_cost import CostModelSelector, UnitGeometry
from merlin.targetgen.extension_offers import MatrixExtension, OfferError, OperationOffer, select_offer


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


def test_matrix_extension_keeps_provider_identity():
    ext = MatrixExtension(_Routing())
    offer = ext.offers(SimpleNamespace(parallel=(64, 64), reduction=(64,)))[0]
    assert (offer.owner, offer.unit) == ("kodiak", "opu")

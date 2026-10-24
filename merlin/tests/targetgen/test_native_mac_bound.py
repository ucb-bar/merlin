import pytest

from merlin.targetgen.native_mac_bound import require_mesh_mac_bound


def proof(k=16, bound=262144, bits=20):
    return dict(bound_proof=dict(primitive_k=k, max_mesh_partial_sum=bound, required_signed_mac_bits=bits))


def test_full_signed_byte_prefix_bound_covers_ws_and_os():
    result = require_mesh_mac_bound(proof(), mesh_signed_bits=20)
    assert result["conservative_absolute_bound"] == 262144
    assert result["dataflows"] == ["WS", "OS"]


@pytest.mark.parametrize("numeric", [{}, proof(0), proof(32, 0), proof(16, 1 << 19), proof(16, 1, 32)])
def test_missing_or_overwide_mesh_bound_refuses(numeric):
    with pytest.raises(ValueError):
        require_mesh_mac_bound(numeric, mesh_signed_bits=20)


def test_missing_mesh_declaration_refuses():
    with pytest.raises(ValueError):
        require_mesh_mac_bound(proof())

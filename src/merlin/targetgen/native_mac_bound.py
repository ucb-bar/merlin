"""Every byte MAC partial, in either dataflow, must fit the 20-bit mesh."""


def require_mesh_mac_bound(numeric, *, mesh_signed_bits=None):
    """Validate a whole-prefix bound, never just a cancellation-prone final sum.

    Both signed-byte inputs include -128, so a product can be +16384. No D
    payload is licensed here. The provider separately checks garbage mesh D,
    unit scales, and full-width accumulator reads on the actual schedule.
    """
    if type(mesh_signed_bits) is not int or mesh_signed_bits <= 1:
        raise ValueError("native arithmetic requires declared signed mesh MAC width")
    limit = 1 << (mesh_signed_bits - 1)
    proof = numeric.get("bound_proof")
    if not isinstance(proof, dict):
        raise ValueError("native arithmetic requires a per-primitive MAC bound")
    k = proof.get("primitive_k")
    if type(k) is not int or k <= 0:
        raise ValueError("native arithmetic requires a positive primitive K")
    bound = k * 128 * 128
    if bound >= limit:
        raise ValueError("a mesh MAC partial can wrap at the selected mesh width in WS or OS")
    declared = proof.get("max_mesh_partial_sum")
    bits = proof.get("required_signed_mac_bits")
    if (
        type(declared) is not int
        or not 0 <= declared < limit
        or type(bits) is not int
        or not 1 <= bits <= mesh_signed_bits
    ):
        raise ValueError("provider MAC bound is absent or exceeds the declared mesh width")
    return dict(
        method="all_prefixes_full_signed_byte_products",
        primitive_k=k,
        conservative_absolute_bound=bound,
        mesh_signed_bits=mesh_signed_bits,
        dataflows=["WS", "OS"],
    )

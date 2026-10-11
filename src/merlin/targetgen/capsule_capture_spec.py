"""Capture specifications projected from typed abstract capsule entries."""

def _entry_seed(name: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(name)) or 1



def _capture_spec(entry: dict, binding) -> dict:
    """Build the PytorchRefSource capture spec (op + shapes + dtype + seed) from an abstract entry."""
    dim = binding.tile_dim
    op = entry.get("op", "matmul")
    M = entry.get("M", entry.get("M_tiles", 1) * dim)
    K = entry.get("K", entry.get("K_tiles", 1) * dim)
    N = entry.get("N", entry.get("N_tiles", 1) * dim)
    if op == "attention_qk" and "N" not in entry and "N_tiles" not in entry:
        N = M  # Preserve the square default, while honoring explicit rectangular source geometry.
    spec = {
        "op": entry.get("capture_op", op),
        "dtype": entry.get("capture_dtype", binding.operand_dtype),
        "seed": _entry_seed(entry["name"]),
        "M": M,
        "K": K,
        "N": N,
    }
    if "shape" in entry:
        spec["shape"] = list(entry["shape"])
    # WHICH QUANTIZATION the captured PROGRAM should carry, when the entry names one. The dtype default
    # for int8 is weight-only, which emits a float matmul over dequantized weights -- correct for a
    # model ladder, wrong for a capsule meant to exercise an integer datapath, and not fixable by
    # substituting a golden: the program itself contains no integer contraction.
    if entry.get("quant_scheme"):
        spec["quant_scheme"] = str(entry["quant_scheme"])
    if op in ("rmsnorm", "gemma_4norm", "layernorm"):
        spec["eps"] = entry.get("eps", 1.0 / 65536.0)
    # optional per-op fields (batch / soft-cap / conv geometry) pass through verbatim when present
    for f in (
        "B",
        "cap",
        "Cin",
        "Himg",
        "Wimg",
        "P",
        "Pool",
        "Dv",
        "causal",
        "eps",
        "input_palette",
        "producer_scale",
        "quant_scale",
        "native_input_dtype",
        "native_output_dtype",
        "native_source_abi",
        "frontend_op",
    ):
        if f in entry:
            spec[f] = entry[f]
    if any(key in entry for key in ("native_input_dtype", "native_output_dtype", "native_source_abi")):
        spec["generalization"] = entry.get("generalization")
    return spec

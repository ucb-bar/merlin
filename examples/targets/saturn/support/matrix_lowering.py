"""Explicit-selection adapter to the provider-owned OPU shim.

This adapter does not certify compilation, hardware, or the transitive source closure.
"""

from . import opu_shim

signature_sidecar_name = opu_shim.SIDECAR_NAME
SIDECAR_NAME = opu_shim.SIDECAR_NAME
shim = opu_shim


def load_contract(unit, *, path=None):
    return opu_shim.load_contract(unit, path=path)


def derive_encodings(contract):
    return opu_shim.derive_encodings(contract)


def geometry(*, unit, config):
    return opu_shim.load_contract(unit).geometry(config)


def selector(tile_edge):
    from merlin.llvmlower import int8_contractions

    return int8_contractions.tile_filling_selector(tile_edge)


def rewrite_prepared_file(prepared, work, *, select, tile_edge):
    from merlin.llvmlower import int8_contractions

    return int8_contractions.rewrite_prepared_file(
        prepared, work, select=select, tile_edge=tile_edge,
        symbol_prefix=opu_shim.SYMBOL_PREFIX, sidecar_name=opu_shim.SIDECAR_NAME,
    )


def load_signatures(work):
    from merlin.llvmlower import int8_contractions

    return int8_contractions.load_sidecar(work, opu_shim.SIDECAR_NAME)


def build_object(signatures, work, *, unit, config, cc, cflags, scalar_tile=False, parallel_tiles=False):
    return opu_shim.build_object(
        signatures, work, unit=unit, config=config, cc=cc, cflags=cflags,
        scalar_tile=scalar_tile, parallel_tiles=parallel_tiles,
    )

"""Iris's matrix_lowering plugin: route int8 contractions to the VME unit (see vme_shim.py).

The callables are the ones merlin.runtime.backends.zephyr_model.MatrixRouting requires, plus the optional
``device_simulator`` that lets a build run its SHIPPED image on a simulator implementing the unit. This
adapter certifies nothing by itself; the gates are the build's audit, the twin and the simulator run.
"""

from . import vme_shim

signature_sidecar_name = vme_shim.SIDECAR_NAME
SIDECAR_NAME = vme_shim.SIDECAR_NAME
shim = vme_shim


def load_contract(unit, *, path=None):
    return vme_shim.load_contract(unit, path=path)


def derive_encodings(contract):
    return vme_shim.derive_encodings(contract)


def geometry(*, unit, config):
    return vme_shim.load_contract(unit).geometry(config)


def selector(tile_edge):
    from merlin.llvmlower import int8_contractions

    return int8_contractions.tile_filling_selector(tile_edge)


def rewrite_prepared_file(prepared, work, *, select, tile_edge):
    from merlin.llvmlower import int8_contractions

    return int8_contractions.rewrite_prepared_file(
        prepared,
        work,
        select=select,
        tile_edge=tile_edge,
        symbol_prefix=vme_shim.SYMBOL_PREFIX,
        sidecar_name=vme_shim.SIDECAR_NAME,
    )


def load_signatures(work):
    from merlin.llvmlower import int8_contractions

    return int8_contractions.load_sidecar(work, vme_shim.SIDECAR_NAME)


def build_object(signatures, work, *, unit, config, cc, cflags, scalar_tile=False, parallel_tiles=False):
    return vme_shim.build_object(
        signatures,
        work,
        unit=unit,
        config=config,
        cc=cc,
        cflags=cflags,
        scalar_tile=scalar_tile,
        parallel_tiles=parallel_tiles,
    )


def device_simulator(*, unit, config):
    return vme_shim.device_simulator(unit=unit, config=config)

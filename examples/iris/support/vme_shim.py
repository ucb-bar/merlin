"""Iris's VME matrix unit, end to end: read its facts, emit its kernel, build the routed object.

Everything here is a fact about Iris's unit or the code that drives it; everything generic lives in Merlin:

* the encoding derivation and its cross-checks: :mod:`merlin.targetgen.rtl.matrix_isa`, fed the
  declaration names in ``contracts/matrix_units.yaml``;
* the descriptor-ABI bridge, operand pack and scalar fallback around the kernel:
  :mod:`merlin.kernels.matrix_shim`;
* the shipped/twin pair, its audit and the routing report: ``merlin.runtime.backends.matrix_image``.

What this module adds is the reading of THIS unit's sources (which file declares what), its geometry from
the elaborated configuration (the vector unit whose params enable the unit, not the first one), the
kernel (:mod:`vme_kernel`), and the declaration of the simulator that implements the unit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import vme_kernel

SIDECAR_NAME = "vme_signatures.json"
SYMBOL_PREFIX = "merlin_vme_gemm_i8"
#: Namespace of the shim's own symbols (scratch, counters), distinct from the entry prefix.
SHIM_PREFIX = "merlin_vme"

CONTRACT_PATH = Path(__file__).resolve().parent / "contracts" / "matrix_units.yaml"

#: The ABI of the routed contraction (int8 operands, int32 accumulation): not a hardware fact.
OPERAND_BITS = 8
ACC_BITS = 32


def _env(name: str) -> str | None:
    from merlin.common.paths import env

    return env(name)


@dataclass(frozen=True)
class UnitContract:
    """One unit's derivation entry points (addresses, never values), read from :data:`CONTRACT_PATH`."""

    unit: str
    pin: str
    root_env: str
    path: str
    sources: dict[str, str]
    core: dict[str, Any]
    declarations: dict[str, Any]
    configs: dict[str, Any]
    kernel_roles: dict[str, str]
    operand_layout: dict[str, int]
    simulator: dict[str, Any] = field(default_factory=dict)

    def root(self) -> Path:
        got = _env(self.root_env)
        if not got:
            raise FileNotFoundError(
                f"${self.root_env} is unset, so the sources of {self.unit!r} cannot be located and its encodings "
                "cannot be derived; refusing to emit a kernel from guessed words"
            )
        return Path(got)

    def checkout(self) -> Path:
        return self.root() / self.path if self.path else self.root()

    def source(self, name: str) -> Path:
        if name not in self.sources:
            raise KeyError(f"{self.unit!r} declares no {name!r} source in {CONTRACT_PATH}")
        return self.checkout() / self.sources[name]

    def core_checkout(self) -> Path:
        return self.root() / str(self.core["path"])

    def core_source(self) -> Path:
        return self.core_checkout() / str(self.core["source"])

    def config_scala(self) -> list[Path]:
        out = [self.checkout() / str(p) for p in self.configs.get("config_scala", ())]
        out += [self.root() / str(p) for p in self.configs.get("host_config_scala", ())]
        return [p for p in out if p.is_file()]

    def mixin_scala(self) -> list[Path]:
        return [p for p in (self.checkout() / str(m) for m in self.configs.get("mixin_scala", ())) if p.is_file()]

    def vector_params(self, config: str) -> dict[str, int]:
        """``{vLen, dLen}`` of the vector unit in ``config`` whose params ENABLE this unit."""
        from merlin.targetgen.rtl import matrix_isa

        d = self.declarations
        cfg_text = "\n".join(p.read_text(encoding="utf-8") for p in self.config_scala())
        got, notes = matrix_isa.matrix_unit_params(
            cfg_text,
            config,
            mixin_text="\n".join(p.read_text(encoding="utf-8") for p in self.mixin_scala()),
            params_text=self.source("params").read_text(encoding="utf-8"),
            params_object=str(d["params_object"]),
            enable_flag=str(d["enable_flag"]),
        )
        if "vLen" not in got or "dLen" not in got:
            raise ValueError(f"cannot derive {self.unit!r} geometry for {config!r}: {notes}")
        return got

    def geometry(self, config: str) -> tuple[int, int]:
        """``(tile_edge, operand_alignment_bytes)``: VLMAX at the operand width, and the datapath width."""
        from merlin.kernels.opu_cert import logical_tile_edge

        p = self.vector_params(config)
        return logical_tile_edge(int(p["vLen"]), sew_bits=OPERAND_BITS), max(1, int(p["dLen"]) // 8)

    def matrix_types(self) -> list[tuple[int, int]]:
        """``[(sew, widen)]`` the RTL accepts as matrix configurations (e.g. ``def opuMatrixTypes``)."""
        from merlin.targetgen.rtl.matrix_isa import seq_int_tuples

        name = str(self.declarations["matrix_types"])
        out = [t for t in seq_int_tuples(self.source("params").read_text(encoding="utf-8"), name) if len(t) == 2]
        if not out:
            raise ValueError(f"`def {name}` in {self.source('params')} declares no (sew, widen) pairs")
        return out

    def kmax(self, sew: int, widen: int) -> int:
        """The reduction depth one matmul takes at ``(sew, widen)``, from the host core's ``kmax`` table."""
        from merlin.targetgen.rtl.matrix_isa import match_case_table

        table = match_case_table(
            self.core_source().read_text(encoding="utf-8"), str(self.core["config_object"]), str(self.core["kmax_def"])
        )
        if (sew, widen) not in table:
            raise ValueError(f"kmax({sew}, {widen}) is not in the RTL's table {self.core_source()}")
        return table[(sew, widen)]

    def kernel_spec(self) -> vme_kernel.VmeKernelSpec:
        types = self.matrix_types()
        operand = [t for t in types if t[0] == OPERAND_BITS and t[0] * t[1] == ACC_BITS]
        readout = [t for t in types if t[1] == 1 and t[0] == ACC_BITS]
        if len(operand) != 1 or len(readout) != 1:
            raise ValueError(f"the RTL's matrix configurations {types} have no unique int8->int32 operand/readout pair")
        return vme_kernel.VmeKernelSpec(
            roles=dict(self.kernel_roles),
            operand_type=operand[0],
            readout_type=readout[0],
            kstep=self.kmax(*operand[0]),
            layout=vme_kernel.OperandLayout(**{k: int(v) for k, v in self.operand_layout.items()}),
            acc_bits=ACC_BITS,
            operand_bits=OPERAND_BITS,
        )


def load_contract(unit: str, *, path: str | Path | None = None) -> UnitContract:
    import yaml

    p = Path(path) if path is not None else CONTRACT_PATH
    units = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("units") or {}
    if unit not in units:
        raise KeyError(f"{p} declares no matrix unit {unit!r}; declared: {sorted(units)}")
    b = units[unit]
    need = ("pin", "root_env", "sources", "core", "declarations", "configs", "kernel_roles", "operand_layout")
    missing = [k for k in need if k not in b]
    if missing:
        raise ValueError(f"{p}: unit {unit!r} is missing {missing}")
    return UnitContract(
        unit=unit,
        pin=str(b["pin"]),
        root_env=str(b["root_env"]),
        path=str(b.get("path", "")),
        sources=dict(b["sources"]),
        core=dict(b["core"]),
        declarations=dict(b["declarations"]),
        configs=dict(b["configs"]),
        kernel_roles=dict(b["kernel_roles"]),
        operand_layout=dict(b["operand_layout"]),
        simulator=dict(b.get("simulator") or {}),
    )


def derive_encodings(contract: UnitContract):
    """The unit's encodings as MATCH/MASK pairs, derived from its RTL and cross-checked.

    Always against the unit author's ``vme.h``; also against the declared simulator's opcode table when its
    source checkout is present (a third source, and the one that says the simulator decodes the same words).
    """
    from merlin.targetgen.rtl import matrix_isa

    d = contract.declarations
    isa = matrix_isa.derive(
        consts=contract.source("consts"),
        instructions=contract.source("instructions"),
        params=contract.source("params"),
        insn_seq=str(d["insn_seq"]),
        consts_container=str(d["consts_container"]),
        funct6_enum=str(d["funct6_enum"]),
        opcode_name=str(d["opcode_name"]),
        opcode_flags={str(k): str(v) for k, v in (d.get("opcode_flags") or {}).items()},
        decode_fields={str(k): str(v) for k, v in d["decode_fields"].items()},
        form_funct3={str(k): str(v) for k, v in (d.get("form_funct3") or {}).items()},
        vtype_fields=tuple(str(v) for v in d.get("vtype_fields") or ()),
        core_source=contract.core_source(),
        core_container=str(contract.core["container"]),
        core_insns=[str(n) for n in contract.core["instructions"]],
    )
    isa = matrix_isa.crosscheck_header(
        isa, contract.source("crosscheck_header"), pairs={str(k): str(v) for k, v in d["crosscheck_pairs"].items()}
    )
    sim = contract.simulator
    src = _env(str(sim.get("source_env") or "")) if sim.get("source_env") else None
    if src and sim.get("opcode_table"):
        isa = matrix_isa.crosscheck_opcode_table(
            isa, Path(src) / str(sim["opcode_table"]), pairs={str(k): str(v) for k, v in sim["opcode_pairs"].items()}
        )
    return isa


def device_simulator(*, unit: str, config: str, contract_path: str | Path | None = None) -> dict[str, Any]:
    """The functional simulator that implements this unit, as the contract declares it.

    ``available`` is False (with the reason) when its binary is not configured or not present; a caller
    must not substitute another simulator, which would trap on the unit's words or decode them as
    something else. ``isa_extensions`` are spelled with this configuration's derived tile edge.
    """
    from merlin.common import provenance as PROV

    contract = load_contract(unit, path=contract_path)
    sim = contract.simulator
    if not sim:
        return {"available": False, "reason": f"{unit!r} declares no simulator"}
    env_name = str(sim["binary_env"])
    binary = _env(env_name)
    if not binary or not Path(binary).is_file():
        return {
            "available": False,
            "reason": (
                f"${env_name} is {'unset' if not binary else f'{binary!r}, which is not a file'}: build the "
                f"simulator pinned as {sim.get('pin')!r} (see the iris package README) and point ${env_name} at it"
            ),
        }
    tile_edge, _align = contract.geometry(config)
    exts = [str(e).format(tile_edge=tile_edge) for e in sim.get("isa_extensions") or ()]
    out: dict[str, Any] = {
        "available": True,
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        "isa_extensions": exts,
    }
    src = _env(str(sim.get("source_env") or "")) if sim.get("source_env") else None
    if sim.get("pin") and src:
        ver = PROV.verify(str(sim["pin"]), checkout=src)
        out["provenance"] = {"pin": str(sim["pin"]), "ok": bool(ver.ok), "drift": list(ver.drift)}
    else:
        out["provenance"] = {"pin": sim.get("pin"), "ok": None, "note": "simulator source checkout not configured"}
    return out


@dataclass(frozen=True)
class VmeBuild:
    """What a build produced and what it is a result about (the matrix_lowering build contract)."""

    object_path: Path
    source_path: Path
    signatures: dict[str, tuple[int, ...]]
    alignment_bytes: int
    scratch_bytes: int
    tile_edge: int | None = None
    scalar_tile: bool = False
    parallel_tiles: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    gaps: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "object": str(self.object_path),
            "source": str(self.source_path),
            "signatures": {k: list(v) for k, v in self.signatures.items()},
            "alignment_bytes": self.alignment_bytes,
            "scratch_bytes": self.scratch_bytes,
            "tile_edge": self.tile_edge,
            "scalar_tile": self.scalar_tile,
            "parallel_tiles": self.parallel_tiles,
            "provenance": self.provenance,
            "gaps": list(self.gaps),
        }


def emit_translation_unit(signatures, contract: UnitContract, *, alignment_bytes: int, scratch_bytes=None):
    """Derive, cross-check, emit: the kernel inside Merlin's generic descriptor-ABI shim."""
    from merlin.kernels import matrix_shim

    derived = derive_encodings(contract)
    if not derived.ok:
        raise ValueError(
            f"the encoding derivation for {contract.unit!r} is not resolved (gaps {list(derived.gaps)}, failed checks "
            f"{[c for c in derived.crosschecks if not c.get('agrees')]}); refusing to build from it"
        )
    kernel = vme_kernel.emit_kernel(derived.encodings, contract.kernel_spec(), derivation_ok=derived.ok)
    unit = matrix_shim.emit_translation_unit(
        kernel,
        vme_kernel.KERNEL_FUNC,
        signatures,
        alignment_bytes=alignment_bytes,
        prefix=SHIM_PREFIX,
        scratch_bytes=scratch_bytes,
    )
    return unit, derived


def build_object(
    signatures: Mapping[str, Sequence[int]],
    work: str | Path,
    *,
    unit: str,
    config: str,
    cc: str | Path,
    cflags: Sequence[str],
    scalar_tile: bool = False,
    parallel_tiles: bool = False,
    scratch_bytes: int | None = None,
    contract_path: str | Path | None = None,
) -> VmeBuild:
    """Derive the unit's facts, emit the translation unit, compile it, record what it came from."""
    from merlin.common import provenance as PROV

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    contract = load_contract(unit, path=contract_path)
    gaps: list[str] = []
    reads = [*contract.sources.values(), *(str(m) for m in contract.configs.get("mixin_scala", ()))]
    pins = {contract.pin: PROV.verify(contract.pin, checkout=contract.checkout(), reads=reads)}
    if contract.core.get("pin"):
        pins[str(contract.core["pin"])] = PROV.verify(
            str(contract.core["pin"]), checkout=contract.core_checkout(), reads=[str(contract.core["source"])]
        )
    for name, ver in pins.items():
        if not ver.ok:
            gaps.append(f"pin {name} drifted: {'; '.join([*ver.drift, *map(str, ver.missing_paths)])}")

    tile_edge, alignment = contract.geometry(config)
    unit_src, derived = emit_translation_unit(
        signatures, contract, alignment_bytes=alignment, scratch_bytes=scratch_bytes
    )
    src = work / "merlin_vme_shim.c"
    src.write_text(str(unit_src), encoding="utf-8")
    obj = work / "merlin_vme_shim.o"
    cmd = [str(cc), *cflags, "-c", str(src), "-o", str(obj)]
    if parallel_tiles:
        cmd[1:1] = ["-fopenmp"]
    if scalar_tile:
        cmd[1:1] = ["-DMERLIN_MATRIX_SCALAR_TILE", f"-DMERLIN_MATRIX_TILE_EDGE={tile_edge}"]
    got = subprocess.run(cmd, capture_output=True, text=True)
    if got.returncode != 0 or not obj.is_file():
        raise RuntimeError(f"the emitted VME shim did not compile:\ncmd: {' '.join(cmd)}\n{got.stderr[-3000:]}")
    (work / "encodings.json").write_text(json.dumps(derived.to_dict(), indent=2), encoding="utf-8")
    prov = PROV.record(
        pins=pins,
        sources=[contract.source(k) for k in sorted(contract.sources)] + [contract.core_source()],
        artifacts={"shim_object": obj, "shim_source": src},
    )
    return VmeBuild(
        object_path=obj,
        source_path=src,
        signatures={k: tuple(int(x) for x in v) for k, v in signatures.items()},
        alignment_bytes=alignment,
        scratch_bytes=unit_src.scratch_bytes,
        tile_edge=tile_edge,
        scalar_tile=scalar_tile,
        parallel_tiles=parallel_tiles,
        provenance=prov,
        gaps=tuple(gaps),
    )

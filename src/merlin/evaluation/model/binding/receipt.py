"""Bind evaluation to the selected machine, tools and observed artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from merlin.evaluation.model.binding import device_sources as S

RECEIPT_SCHEMA = "merlin.evaluation.model.binding.receipt.v1"
SPEC_SCHEMA = "merlin.evaluation.model.binding_spec.v1"
SPEC_FILE = "binding.yaml"
PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"


class BindingError(RuntimeError):
    """The binding inputs cannot be selected (no contract, no spec, a malformed spec)."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def serialize(receipt: Mapping[str, Any]) -> bytes:
    """The receipt's canonical bytes: what ``binding.json`` holds and what :func:`digest` hashes."""
    return (json.dumps(receipt, indent=1, sort_keys=True, default=str) + "\n").encode()


def digest(receipt: Mapping[str, Any]) -> str:
    return hashlib.sha256(serialize(receipt)).hexdigest()


def record(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """``{receipt_sha256, status}``: what a report or a bundle manifest carries of a receipt."""
    return {"receipt_sha256": digest(receipt), "status": receipt["status"]}


def write(receipt: Mapping[str, Any], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialize(receipt))
    return {"receipt": path.name, **record(receipt)}


# ------------------------------------------------------------------------------------------- the engine


@dataclass(frozen=True)
class Engine:
    """How the receipt reaches the toolchain and the functional simulator (replaced in tests).

    ``canonical_isa`` expands an ISA string to the extension set the toolchain gives it (implied
    extensions included); ``image_arch`` reads the ISA an ELF records; ``spike_argv`` is the bound
    backend's Spike command for an image; ``run`` executes a command and returns (code, console);
    ``disassemble`` lists an image's code with every instruction in its canonical (non-alias) form.
    """

    canonical_isa: Callable[[str], list[str]]
    image_arch: Callable[[Path], list[str]]
    spike_argv: Callable[[Path], list[str]]
    run: Callable[[Sequence[str]], tuple[int, str]]
    probe_image: Callable[[], Path]
    march: str | None = None
    disassemble: Callable[[Path], str] | None = None


def toolchain_engine(
    target: str, host_package: Path, *, dram_base: int, dram_bytes: int, work: Path, timeout: float
) -> Engine:
    """The real engine: the host package's replay toolchain, the device compiler and the bound Spike."""
    from merlin.compiler.backend.host import load_host_package
    from merlin.compiler.lower.model import toolchain as LT
    from merlin.evaluation.model.kernels import replay as RP
    from merlin.runtime.backends.baremetal import image_link as SM
    from merlin.runtime.backends.baremetal import model_build as _model_elf_metadata

    toolchain = RP.toolchain_for(host_package)
    target_flag = [f for f in toolchain.device_cflags if f.startswith("--target=")] or [SM.CLANG_TARGET]
    marches = [f.removeprefix("-march=") for f in load_host_package(host_package).cflags if f.startswith("-march=")]
    work.mkdir(parents=True, exist_ok=True)

    def canonical(isa: str) -> list[str]:
        name = hashlib.sha256(isa.encode()).hexdigest()[:16]
        source, obj = work / "isa_probe.c", work / f"isa_{name}.o"
        source.write_text("int merlin_isa_probe;\n")
        done = subprocess.run(
            [str(LT.clang()), *target_flag, f"-march={isa}", "-c", str(source), "-o", str(obj)],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            raise S.SourceReadError(f"the device compiler refuses -march={isa}: {done.stderr.strip()[-300:]}")
        return _model_elf_metadata.arch_extensions(obj)

    def probe() -> Path:
        image = work / "probe"
        image.mkdir(parents=True, exist_ok=True)
        main_c, data_s, _ = RP._write_image_sources(image, [], {}, {})  # noqa: SLF001
        device_c = image / "device_probe.c"
        device_c.write_text("void merlin_binding_device_probe(void) {}\n")
        device_o = image / "device_probe.o"
        done = subprocess.run(
            [str(LT.clang()), *toolchain.device_cflags, "-c", str(device_c), "-o", str(device_o)],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            raise RP.ReplayError(f"device compiler failed on the probe object: {done.stderr[-500:]}")
        return RP.link_image(image, main_c, data_s, [device_o], toolchain)

    def argv(elf: Path) -> list[str]:
        return RP.spike_argv(target, elf, dram_base=dram_base, dram_bytes=dram_bytes)[0]

    def run(command: Sequence[str]) -> tuple[int, str]:
        _, engine = RP.spike_argv(target, Path(command[-1]), dram_base=dram_base, dram_bytes=dram_bytes)
        done = subprocess.run(
            list(command), cwd=engine["cwd"], env=engine["environment"], capture_output=True, timeout=timeout
        )
        return done.returncode, done.stdout.decode("latin-1") + done.stderr.decode("latin-1")

    return Engine(
        canonical_isa=canonical,
        image_arch=_model_elf_metadata.arch_extensions,
        spike_argv=argv,
        run=run,
        probe_image=probe,
        march=marches[0] if len(marches) == 1 else None,
        disassemble=lambda elf: disassemble(toolchain.gcc, elf),
    )


def disassemble(compiler: Path, elf: Path) -> str:
    """``objdump -d -M no-aliases`` of ``elf`` by the disassembler the image's compiler driver names."""
    named = subprocess.run([str(compiler), "-print-prog-name=objdump"], capture_output=True, text=True, check=False)
    objdump = Path(named.stdout.strip())
    if named.returncode != 0 or not objdump.is_file():
        raise S.SourceReadError(f"{compiler} names no installed objdump ({named.stdout.strip()!r})")
    done = subprocess.run(
        [str(objdump), "-d", "-M", "no-aliases", str(elf)], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise S.SourceReadError(f"{objdump.name} failed on {elf}: {done.stderr.strip()[-300:]}")
    return done.stdout


# ------------------------------------------------------------------------------------------- the receipt


@dataclass
class _Builder:
    checks: list[dict[str, Any]] = field(default_factory=list)
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    machine: dict[str, Any] = field(default_factory=dict)

    def check(self, name: str, status: str, detail: str, **values: Any) -> str:
        self.checks.append({"name": name, "status": status, "detail": detail, "values": _plain(values)})
        return status

    @contextmanager
    def guard(self, name: str):
        """Run one check; any failure to derive its facts records it as unknown."""
        try:
            yield
        except Exception as error:  # noqa: BLE001 -- an underivable fact is an unknown, never a crash
            self.check(name, UNKNOWN, f"cannot be derived: {type(error).__name__}: {error}")

    def file(self, key: str, path: Path | None, *, pinned: str | None = None) -> str | None:
        if path is None:
            self.inputs[key] = {"path": None, "sha256": None}
            return None
        path = Path(path)
        sha = sha256_file(path) if path.is_file() else None
        self.inputs[key] = {"path": str(path), "sha256": sha, **({"pinned_sha256": pinned} if pinned else {})}
        return sha


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _dig(doc: Any, path: Iterable[str]) -> Any:
    for key in path:
        if not isinstance(doc, Mapping) or key not in doc:
            return None
        doc = doc[key]
    return doc


def _span(base: int, size: int) -> dict[str, Any]:
    return {"base": base, "bytes": size, "hex": f"{hex(base)}+{hex(size)}"}


def load_spec(path: Path, target: str) -> dict[str, Any]:
    if not Path(path).is_file():
        raise BindingError(f"no binding spec at {path}")
    spec = yaml.safe_load(Path(path).read_text()) or {}
    if spec.get("schema") != SPEC_SCHEMA:
        raise BindingError(f"{path}: schema must be {SPEC_SCHEMA!r}")
    if spec.get("target") != target:
        raise BindingError(f"{path}: binding spec is for {spec.get('target')!r}, not {target!r}")
    return spec


def _resolver(checkout: Path | None) -> Callable[[Any], Path | None]:
    from merlin.common.paths import repo_root

    def resolve(value: Any) -> Path | None:
        if value is None:
            return None
        if isinstance(value, Mapping) and "repo" in value:
            return repo_root() / str(value["repo"])
        path = Path(str(value))
        if path.is_absolute():
            return path
        if checkout is None:
            raise BindingError(f"{value}: no RTL source checkout to resolve it against")
        return checkout / path

    return resolve


def receipt(
    *,
    target: str,
    contract_path: Path,
    facts_path: Path,
    spec_path: Path,
    board: str | None,
    host_package: Path | None,
    dram_base: int,
    dram_bytes: int,
    images: Sequence[Path] = (),
    work: Path,
    engine: Engine | None = None,
) -> dict[str, Any]:
    """Build the binding receipt; never raises for a fact it cannot derive (that check is ``unknown``)."""
    from merlin.common.paths import ext_path
    from merlin.evaluation.model.binding.checks import (
        _citations,
        _csr_checks,
        _dram_checks,
        _fpga_memory,
        _header_equality,
        _isa_checks,
        _param_check,
        _partial_extensions,
    )
    from merlin.runtime.boards import catalog as BR
    from merlin.target.rtl.facts import facts_body, load_facts

    out = _Builder()
    contract = yaml.safe_load(Path(contract_path).read_text()) or {}
    if contract.get("name") != target:
        raise BindingError(f"{contract_path}: contract is for {contract.get('name')!r}, not {target!r}")
    spec = load_spec(spec_path, target)
    out.file("contract", Path(contract_path))
    out.file("rtl_facts", Path(facts_path))
    out.file("binding_spec", Path(spec_path))
    facts_doc = load_facts(target, explicit=facts_path, regenerate=False)
    facts = facts_body(facts_doc, target, needs="the binding receipt")
    source = contract.get("rtl_source") or {}
    checkout = None
    with out.guard("rtl_source_checkout"):
        checkout = ext_path(str(source["ext_root"]))
    resolve = _resolver(checkout)
    with out.guard("spec_citations"):
        _citations(out, spec.get("citations") or (), resolve)
    partial = _partial_extensions(out, spec.get("partial_extensions") or {}, resolve)

    # -- the functional simulator and its accelerator extension, against the contract's pins
    runner = contract.get("runner") or {}
    block = runner.get(str(runner.get("backend"))) or {}
    spike = _dig(block, ("engines", "spike", "binary")) or {}
    extension = runner.get("spike_extension") or {}
    for key, entry, pin in (
        ("spike_binary", spike.get("path"), spike.get("sha256")),
        ("spike_extension", extension.get("extlib"), extension.get("sha256")),
    ):
        with out.guard(f"{key}_pin"):
            sha = out.file(key, Path(entry), pinned=pin)
            status = PASS if sha is not None and sha == pin else (UNKNOWN if sha is None else FAIL)
            out.check(f"{key}_pin", status, f"{entry} sha256 {sha} vs contract pin {pin}", sha256=sha, pinned=pin)
    build = (spec.get("spike") or {}).get("extension_build")
    if build:
        with out.guard("spike_extension_build"):
            built = out.file("spike_extension_build", resolve(build))
            installed = out.inputs.get("spike_extension", {}).get("sha256")
            status = PASS if built and built == installed else (UNKNOWN if not built or not installed else FAIL)
            out.check(
                "spike_extension_build",
                status,
                "the installed extension library is the build beside Spike's parameter header",
                built=built,
                installed=installed,
            )

    # -- the selected FIRRTL and the elaboration behind it
    elaboration = spec.get("elaboration") or {}
    directory = resolve(elaboration.get("directory")) if elaboration.get("directory") else None
    fir_path = Path(_dig(facts_doc, ("inputs", "fir_path")) or "")
    with out.guard("firrtl_identity"):
        fir_sha = out.file("firrtl", fir_path) if str(fir_path) else None
        elab_fir = out.file("elaboration_firrtl", directory / elaboration["firrtl"]) if directory else None
        declared = {
            "rtl_facts.inputs.fir_sha256": _dig(facts_doc, ("inputs", "fir_sha256")),
            "contract.rtl_source.artifact_sha256.fir": _dig(source, ("artifact_sha256", "fir")),
            "elaboration directory FIRRTL": elab_fir,
        }
        configs = {"rtl_facts.source.config": _dig(facts, ("source", "config")), "contract": source.get("config")}
        bad = [k for k, v in declared.items() if v != fir_sha] + (["config"] if len(set(configs.values())) != 1 else [])
        status = UNKNOWN if fir_sha is None else (FAIL if bad else PASS)
        out.check(
            "firrtl_identity",
            status,
            "selected FIRRTL bytes == facts and contract pins == the elaboration directory's FIRRTL"
            + (f"; differs: {bad}" if bad else ""),
            firrtl_sha256=fir_sha,
            declared=declared,
            config=configs,
        )
    # The target elaboration above remains the source of target/FPGA facts.
    # A selected simulation may execute different SoC FIRRTL only through the
    # exact unit correspondence and its independently verified engine receipt.
    execution_config = block.get("config")
    if execution_config and execution_config != source.get("config"):
        with out.guard("simulation_firrtl_identity"):
            from merlin.runtime.execution import model_inputs as MI
            from merlin.target.rtl.extraction.unit_correspondence import execution_binding

            correspondence = execution_binding(facts_doc, config=execution_config)
            execution = MI.selected_firrtl(facts_path, target=target, config=execution_config)
            from merlin.target.rtl.extraction.source_selection import load_selection

            selection = correspondence["selection"]
            if (
                out.file("simulation_source_selection", Path(selection["path"]), pinned=selection["sha256"])
                != selection["sha256"]
            ):
                raise BindingError("simulation source selection bytes changed")
            selected = load_selection(selection["path"], target=target)
            expected = {
                "firrtl": correspondence["firrtl"],
                "hierarchy": correspondence["hierarchy"],
                "core_hw": correspondence["core_hw"],
            }
            if selected["config"] != execution_config or any(
                selected["sources"][role] != pin for role, pin in expected.items()
            ):
                raise BindingError("simulation bundle differs from the facts correspondence")
            # Bind with the explicit contract/facts, never a different ambient selection.
            from merlin.runtime.backends.rocc.tool_selection import bind

            backend = bind(target=target, contract=contract, facts=facts_doc)
            citation = backend.execution_engine_identity("gsim")
            if citation["receipt"]["firrtl_sha256"] != execution["firrtl_sha256"]:
                raise BindingError("simulation engine receipt differs from correspondence FIRRTL")
            out.check(
                "simulation_firrtl_identity",
                PASS,
                "verified target-unit correspondence binds the selected simulation bundle and engine",
                facts_config=correspondence["facts_config"],
                execution_config=execution_config,
                firrtl_sha256=execution["firrtl_sha256"],
            )

    # -- the board: its DTS pin and its memory span
    board_row = None
    catalog = resolve(spec.get("board_catalog")) if spec.get("board_catalog") else None
    boards: dict[str, Any] = {}
    with out.guard("board"):
        out.file("board_catalog", catalog)
        boards = BR.load_boards(catalog)
        board_row = boards[board or spec["board"]]
        out.machine["board"] = {"name": board_row.name, **_span(board_row.dram_base, board_row.dram_bytes)}
    dts_root = None
    dts_isa: list[str] = []
    with out.guard("dts_pin"):
        dts_path = directory / elaboration["dts"]
        sha = out.file("dts", dts_path, pinned=board_row.host_dts_sha256 if board_row else None)
        pin = board_row.host_dts_sha256 if board_row else None
        status = UNKNOWN if sha is None or pin is None else (PASS if sha == pin else FAIL)
        out.check("dts_pin", status, f"DTS sha256 {sha} vs board pin {pin}", sha256=sha, pinned=pin)
        dts_root = S.read_dts(dts_path.read_text())
        dts_isa = S.dts_isas(dts_root)
        out.machine["dts"] = {
            "isa": dts_isa,
            "memory": [{**m, **_span(m["base"], m["bytes"])} for m in S.dts_memory(dts_root)],
        }

    # -- parameter headers: Spike's, the elaboration's, the FPGA build's, and the RTL facts
    headers: dict[str, S.Header | None] = {"spike": None, "elaboration": None}
    for key, where in (
        ("spike", (spec.get("spike") or {}).get("params_header")),
        ("elaboration", elaboration.get("params_header")),
    ):
        with out.guard(f"{key}_params_header"):
            path = resolve(where)
            out.file(f"{key}_params_header", path)
            headers[key] = S.read_header(path.read_text())
    _header_equality(out, headers)
    fpga = spec.get("fpga") or {}
    if fpga.get("params_header_sha256"):
        with out.guard("fpga_params_header"):
            entry = fpga["params_header_sha256"]
            declared_doc = yaml.safe_load(resolve(entry["file"]).read_text())
            declared = _dig(declared_doc, entry["key"])
            spike_sha = out.inputs.get("spike_params_header", {}).get("sha256")
            status = UNKNOWN if not declared or not spike_sha else (PASS if declared == spike_sha else FAIL)
            out.check(
                "fpga_params_header",
                status,
                "Spike's parameter header is the one the FPGA build's elaboration generated",
                spike=spike_sha,
                fpga_build=declared,
                declared_by=f"{resolve(entry['file'])}: {'.'.join(entry['key'])}",
            )
    for name, param in (spec.get("params") or {}).items():
        with out.guard(f"param.{name}"):
            _param_check(out, name, param, headers, facts, contract)

    # -- ISA: DTS vs host package vs replay images vs Spike
    images = [Path(p) for p in images]
    if engine is None and host_package is not None:
        with out.guard("engine"):
            target_engine = toolchain_engine(
                target,
                Path(host_package),
                dram_base=dram_base,
                dram_bytes=dram_bytes,
                work=work,
                timeout=float(_dig(block, ("engines", "spike", "max_timeout_s")) or 0) or None,
            )
            engine = target_engine
    if host_package is not None:
        out.file("host_package_knobs", Path(host_package) / "knobs.yaml")
        out.file("host_package_manifest", Path(host_package) / "manifest.yaml")
    probe = None
    if engine is not None:  # the replay probe image is always checked, beside any image passed in
        with out.guard("probe_image"):
            probe = engine.probe_image()
            out.file("probe_image", probe)
            images = [probe, *images]
    for image in images:
        if image != probe:
            out.file(f"image:{image}", image)
    _isa_checks(
        out, engine, dts_isa, images, contract_argv=_dig(block, ("engines", "spike", "argv")) or [], partial=partial
    )
    _csr_checks(out, engine, images, partial)

    # -- DRAM: board span vs Spike -m vs the FPGA target's memory
    fpga_span = _fpga_memory(out, fpga, source.get("config"), resolve, boards)
    _dram_checks(out, board_row, fpga_span, images)
    if probe is not None and engine is not None:
        with out.guard("spike_runs_probe"):
            argv = engine.spike_argv(probe)
            code, console = engine.run(argv)
            status = PASS if code == 0 and "MCDONE" in console else FAIL
            out.check("spike_runs_probe", status, f"Spike exited {code} on the probe image", argv=argv)

    statuses = [c["status"] for c in out.checks]
    return {
        "schema": RECEIPT_SCHEMA,
        "target": target,
        "status": PASS if statuses and all(s == PASS for s in statuses) else FAIL,
        "inputs": out.inputs,
        "machine": _plain(out.machine),
        "checks": out.checks,
        "mismatches": [f"{c['name']}: {c['detail']}" for c in out.checks if c["status"] == FAIL],
        "unknown": [f"{c['name']}: {c['detail']}" for c in out.checks if c["status"] == UNKNOWN],
    }


# ------------------------------------------------------------------------------------------ the command


def replay_memory_defaults() -> tuple[int, int]:
    """The ``--dram-base``/``--dram-bytes`` a replay passes Spike when not told otherwise."""
    from merlin.evaluation.model import cli

    parser = argparse.ArgumentParser(add_help=False)
    cli._engine(parser)  # noqa: SLF001 -- the replay commands' own option defaults
    defaults = parser.parse_args([])
    return defaults.dram_base, defaults.dram_bytes


def receipt_for(args: argparse.Namespace) -> dict[str, Any]:
    """The receipt for a command's arguments: explicit options, else the bundle's engine, else the environment."""
    from merlin.common.paths import build_dir
    from merlin.evaluation.model.bundles import seal as B
    from merlin.evaluation.model.bundles.models import load as load_models

    engine_inputs: dict[str, Any] = {}
    if getattr(args, "bundle", None) and (Path(args.bundle) / "MANIFEST.json").is_file():
        # A bundle being built has no manifest yet; its engine inputs are the command's own options.
        engine_inputs = B.Bundle(args.bundle).manifest().get("engine") or {}
    chosen = {
        key: getattr(args, key, None) or (Path(engine_inputs[key]) if engine_inputs.get(key) else None)
        for key in ("contract", "rtl_facts", "host_package")
    }
    contract = chosen["contract"] or (
        Path(os.environ["MERLIN_TARGET_CONTRACT"]) if os.environ.get("MERLIN_TARGET_CONTRACT") else None
    )
    facts = chosen["rtl_facts"] or (
        Path(os.environ["MERLIN_RTL_FACTS"]) if os.environ.get("MERLIN_RTL_FACTS") else None
    )
    if contract is None or facts is None:
        raise BindingError("binding needs a target contract and its RTL facts (--contract/--rtl-facts or a bundle)")
    os.environ["MERLIN_TARGET_CONTRACT"] = str(contract)
    os.environ["MERLIN_RTL_FACTS"] = str(facts)
    models = load_models(getattr(args, "models", None))
    base, size = replay_memory_defaults()
    dram_base = getattr(args, "dram_base", None)
    dram_bytes = getattr(args, "dram_bytes", None)
    work = Path(getattr(args, "work", None) or (build_dir() / "model_check")) / "binding"
    return receipt(
        target=models.target,
        contract_path=Path(contract),
        facts_path=Path(facts),
        spec_path=Path(getattr(args, "spec", None) or models.path.with_name(SPEC_FILE)),
        board=getattr(args, "board", None),
        host_package=chosen["host_package"],
        dram_base=base if dram_base is None else dram_base,
        dram_bytes=size if dram_bytes is None else dram_bytes,
        images=list(getattr(args, "image", None) or ()),
        work=work,
    )


def gate(doc: Mapping[str, Any]) -> dict[str, Any]:
    passed = sum(c["status"] == PASS for c in doc["checks"])
    return {
        "status": doc["status"],
        "detail": f"{passed}/{len(doc['checks'])} binding checks pass; "
        f"{len(doc['mismatches'])} mismatch(es), {len(doc['unknown'])} unknown",
    }


def failures(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"kind": "gate", "gate": "binding", "check": c["name"], "status": c["status"], "detail": c["detail"]}
        for c in doc["checks"]
        if c["status"] != PASS
    ]

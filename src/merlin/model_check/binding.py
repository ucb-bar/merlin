"""The Spike / RTL / FPGA binding receipt (contract §7): is the functional model the machine the FPGA runs?

Every compared value is read from a file: the target contract (the Spike binary, its accelerator extension
library and their pins), the RTL facts and the FIRRTL they were extracted from, the elaboration's DTS and
parameter header, Spike's parameter header, the board catalog, the host package, the replay image the
toolchain links, and the FPGA build's configuration sources. Which files those are is the target's binding
spec (``examples/<target>/numerics/binding.yaml``). A check whose fact cannot be derived is ``unknown``,
and an unknown fails the receipt just as a mismatch does.
"""

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

from merlin.model_check import binding_sources as S

RECEIPT_SCHEMA = "merlin.model_check.binding.v1"
SPEC_SCHEMA = "merlin.model_check.binding_spec.v1"
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
    from merlin.llvmlower import toolchain as LT
    from merlin.mining import registry
    from merlin.model_check import replay as RP
    from merlin.runtime.backends import spike_model as SM

    toolchain = RP.toolchain_for(host_package)
    target_flag = [f for f in toolchain.device_cflags if f.startswith("--target=")] or [SM.CLANG_TARGET]
    marches = [
        f.removeprefix("-march=") for f in registry.load_rvv_package(host_package).cflags if f.startswith("-march=")
    ]
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
        return SM.arch_extensions(obj)

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
        image_arch=SM.arch_extensions,
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
    from merlin.runtime import boards as BR
    from merlin.targetgen.rtl.facts import facts_body, load_facts

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


# --------------------------------------------------------------------------------------- the checks


def _cited_value(path: Path, key: Sequence[str]) -> Any:
    return _dig(yaml.safe_load(path.read_text()), [str(k) for k in key])


def _reread(out: _Builder, rows: Sequence[Mapping[str, Any]], resolve) -> tuple[list[str], list[str]]:
    """``(missing, changed)`` citations among ``rows``.

    A citation is ``line`` + ``contains`` (the text is on that line), ``key`` + ``equals``/``contains``
    (a YAML/JSON value), or ``absent`` (the text occurs nowhere in the file: what a source does not do).
    """
    missing, changed = [], []
    for row in rows:
        path = resolve(row["file"])
        if "line" in row:
            where = f"{path}:{row['line']}"
        elif "key" in row:
            where = f"{path}: {'.'.join(map(str, row['key']))}"
        else:
            where = f"{path}: nowhere {row.get('absent')!r}"
        if path is None or not path.is_file():
            missing.append(f"{where} ({row['claim']})")
            continue
        out.file(f"cited:{path}", path)
        text = path.read_text(encoding="utf-8", errors="replace")
        if "line" in row:
            lines = text.splitlines()
            held = 0 < int(row["line"]) <= len(lines) and str(row["contains"]) in lines[int(row["line"]) - 1]
        elif "key" in row:
            value = _cited_value(path, row["key"])
            held = value == row["equals"] if "equals" in row else str(row["contains"]) in str(value or "")
        elif "absent" in row:
            held = bool(str(row["absent"])) and str(row["absent"]) not in text
        else:
            held = False
        if not held:
            changed.append(f"{where} ({row['claim']})")
    return missing, changed


def _citations(out: _Builder, rows: Sequence[Mapping[str, Any]], resolve) -> None:
    """Re-read every citation of the spec: a missing source is unknown, a changed one fails."""
    if not rows:
        out.check("spec_citations", UNKNOWN, "the binding spec cites no source for its assertions")
        return
    missing, changed = _reread(out, rows, resolve)
    status = UNKNOWN if missing else (FAIL if changed else PASS)
    out.check(
        "spec_citations",
        status,
        f"{len(rows) - len(missing) - len(changed)}/{len(rows)} citations hold"
        + (f"; missing: {missing}" if missing else "")
        + (f"; no longer say it: {changed}" if changed else ""),
        missing=missing,
        changed=changed,
    )


def _partial_extensions(out: _Builder, declared: Mapping[str, Any], resolve) -> dict[str, list[str]]:
    """``{extension: absent CSRs}`` for each extension the hardware implements in part, as its cited sources say.

    An extension whose citations do not all hold is left out, so an ISA check that needs it fails or
    stays unknown exactly as if nothing had been declared.
    """
    accepted: dict[str, list[str]] = {}
    for extension, entry in declared.items():
        with out.guard(f"partial.{extension}"):
            absent = [str(c) for c in entry.get("absent_csrs") or ()]
            rows = entry.get("citations") or ()
            missing, changed = _reread(out, rows, resolve) if rows else ([], [])
            if not rows or not absent:
                status, detail = UNKNOWN, f"{extension} is declared partial without citations and absent CSRs"
            elif missing or changed:
                status = UNKNOWN if missing else FAIL
                detail = f"{extension}: citations missing {missing}, no longer say it {changed}"
            else:
                status = PASS
                detail = f"{extension} is implemented in part ({len(rows)} citations hold); absent CSRs {absent}"
                accepted[str(extension)] = absent
            out.check(f"partial.{extension}", status, detail, absent_csrs=absent, missing=missing, changed=changed)
    return accepted


def _csr_checks(out: _Builder, engine: Engine | None, images: Sequence[Path], partial: Mapping[str, list[str]]) -> None:
    """No image touches a CSR that a partially implemented extension leaves out (it would trap on hardware)."""
    absent = sorted({csr for csrs in partial.values() for csr in csrs})
    if not absent:
        return
    for index, image in enumerate(images):
        with out.guard(f"csr_absent[{index}]"):
            if engine is None or engine.disassemble is None:
                raise BindingError("no disassembler: the image's CSR accesses cannot be read")
            hits = S.csr_accesses(engine.disassemble(image), absent)
            functions = sorted({h["function"] or "?" for h in hits})
            out.check(
                f"csr_absent[{index}]",
                FAIL if hits else PASS,
                f"{image.name}: "
                + (f"accesses {sorted({h['csr'] for h in hits})} in {functions}" if hits else f"no access to {absent}"),
                image=image,
                absent_csrs=absent,
                hits=hits[:20],
            )


def _fpga_identity(out: _Builder, fpga: Mapping[str, Any], resolve) -> None:
    """The spec's FPGA configuration names are the pinned bitstream's (its config and deploy quintuplet)."""
    if not fpga.get("pin"):
        out.check("fpga.identity", UNKNOWN, "the binding spec names no bitstream pin for the FPGA build")
        return
    path = resolve(fpga["pin"]["file"])
    pin = _cited_value(path, fpga["pin"]["key"]) or {}
    quintuplet = Path(str(pin.get("path") or "")).parent.name.split("-")
    stated = {"config": pin.get("config"), "quintuplet": quintuplet[-2:] if len(quintuplet) >= 2 else None}
    good = pin.get("config") == fpga.get("config") and stated["quintuplet"] == [
        fpga.get("config"),
        fpga.get("platform_config"),
    ]
    out.check(
        "fpga.identity",
        PASS if good else FAIL,
        f"bitstream pin {'.'.join(fpga['pin']['key'])} states config {stated['config']} and deploy quintuplet "
        f"{'-'.join(quintuplet)}; the spec says {fpga.get('config')} on {fpga.get('platform_config')}",
        pin=stated,
    )


def _header_equality(out: _Builder, headers: Mapping[str, S.Header | None]) -> None:
    spike, elaboration = headers.get("spike"), headers.get("elaboration")
    if spike is None or elaboration is None:
        out.check("params_header_equal", UNKNOWN, "Spike's or the elaboration's parameter header is unavailable")
        return
    differ = sorted(
        name
        for table in ("defines", "typedefs")
        for name in set(getattr(spike, table)) | set(getattr(elaboration, table))
        if getattr(spike, table).get(name) != getattr(elaboration, table).get(name)
    )
    same_bytes = out.inputs["spike_params_header"]["sha256"] == out.inputs["elaboration_params_header"]["sha256"]
    out.check(
        "params_header_equal",
        PASS if not differ else FAIL,
        "every #define and typedef of Spike's header equals the elaboration's"
        + (f"; differs: {differ}" if differ else "")
        + ("" if same_bytes else " (bytes differ)"),
        differ=differ,
        same_bytes=same_bytes,
        defines=len(spike.defines),
        typedefs=len(spike.typedefs),
    )


def _dtype(text: str) -> dict[str, Any] | None:
    """``i8`` / ``si8`` / ``ui32`` / ``f32`` / ``bf16`` -> ``{kind, bits[, signed]}``."""
    letters = text.rstrip("0123456789")
    digits = text[len(letters) :]
    if not digits:
        return None
    kinds = {
        "i": ("int", None),
        "si": ("int", True),
        "ui": ("int", False),
        "f": ("float", None),
        "bf": ("bfloat", None),
    }
    if letters not in kinds:
        return None
    kind, signed = kinds[letters]
    out: dict[str, Any] = {"kind": kind, "bits": int(digits)}
    if signed is not None:
        out["signed"] = signed
    return out


def _header_value(header: S.Header | None, spec: Mapping[str, Any]) -> Any:
    if header is None:
        return None
    if "define" in spec:
        return header.integer(spec["define"])
    value = header.c_type(spec["typedef"])
    if value is None:
        return None
    if spec.get("carrier"):
        carrier = header.c_type(spec["carrier"]) or {}
        if carrier.get("bits") is not None:
            value["bits"] = carrier["bits"]
    for key in ("exp_bits", "sig_bits"):
        if spec.get(key) is not None:
            value[key] = header.integer(spec[key])
    return value


def _fact_value(facts: Mapping[str, Any], selector: Mapping[str, Any]) -> Any:
    rows = [r for r in facts.get(selector["collection"]) or () if isinstance(r, Mapping)]
    rows = [r for r in rows if r.get("name") == selector["name"]]
    if len(rows) != 1:
        return None
    row = rows[0]
    if "product" in selector:
        parts = [_dig(row, str(f).split(".")) for f in selector["product"]]
        if any(not isinstance(p, int) for p in parts):
            return None
        value = 1
        for part in parts:
            value *= part
        return value
    value = _dig(row, str(selector["field"]).split("."))
    return _as(value, selector.get("as"))


def _as(value: Any, how: str | None) -> Any:
    if value is None:
        return None
    if how == "bits":
        return {"bits": value} if isinstance(value, int) else None
    if how == "signedness":
        return {"signed": value == "signed"} if value in ("signed", "unsigned") else None
    if isinstance(value, str):
        return _dtype(value)
    return value


def _agrees(expected: Any, value: Any) -> bool | None:
    """``True`` when ``value`` states nothing ``expected`` contradicts; ``None`` when it cannot be confirmed."""
    if isinstance(expected, int) and isinstance(value, int):
        return expected == value
    if isinstance(expected, Mapping) and isinstance(value, Mapping):
        verdict: bool | None = True
        for key, stated in value.items():
            if key == "c_type":
                continue
            if expected.get(key) is None:
                verdict = None  # the source states something Spike's header does not: unconfirmed
            elif expected[key] != stated:
                return False
        return verdict
    return None if expected is None or value is None else False


def _param_check(
    out: _Builder,
    name: str,
    param: Mapping[str, Any],
    headers: Mapping[str, S.Header | None],
    facts: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    expected = _header_value(headers.get("spike"), param["header"])
    sources: list[tuple[str, str, Any]] = []
    elaboration = _header_value(headers.get("elaboration"), param["header"])
    if elaboration is not None:
        sources.append(("rtl", "elaboration params header", elaboration))
    not_recorded = []
    for selector in param.get("rtl_facts") or ():
        label = f"rtl_facts.{selector['collection']}[{selector['name']}]." + (
            "*".join(selector["product"]) if "product" in selector else str(selector["field"])
        )
        value = _fact_value(facts, selector)
        if value is None:
            not_recorded.append(label)
        else:
            sources.append(("rtl", label, value))
    for selector in param.get("contract") or ():
        value = _as(_dig(contract, selector["path"]), selector.get("as"))
        label = "contract." + ".".join(selector["path"])
        if value is None:
            not_recorded.append(label)
        else:
            sources.append(("contract", label, value))
    verdicts = [(label, value, _agrees(expected, value)) for _, label, value in sources]
    differ = [label for label, _, verdict in verdicts if verdict is False]
    unconfirmed = [label for label, _, verdict in verdicts if verdict is None]
    rtl_side = [label for kind, label, _ in sources if kind == "rtl"]
    if expected is None:
        status, detail = UNKNOWN, f"Spike's parameter header does not state {name}"
    elif differ:
        status, detail = FAIL, f"{name}: Spike {expected} differs from {differ}"
    elif not rtl_side:
        status, detail = UNKNOWN, f"no RTL-side source states {name}"
    elif unconfirmed:
        status, detail = UNKNOWN, f"{name}: {unconfirmed} cannot be compared with Spike's {expected}"
    else:
        status, detail = PASS, f"{name} = {expected} in Spike's header and {len(sources)} RTL/contract source(s)"
    out.check(
        f"param.{name}",
        status,
        detail,
        spike=expected,
        sources={label: value for label, value, _ in verdicts},
        not_recorded=not_recorded,
    )


def _vendor_free(isa: str) -> tuple[str, list[str]]:
    """``isa`` without its non-standard (``x``-prefixed) extensions, which a toolchain cannot spell."""
    head, *tail = isa.split("_")
    kept = [t for t in tail if not t.startswith("x")]
    return "_".join([head, *kept]), [t for t in tail if t.startswith("x")]


def _isa_set(tokens: Sequence[str]) -> tuple[str, frozenset[str]]:
    head = tokens[0]
    digits = head[2:]
    xlen_digits = digits[: len(digits) - len(digits.lstrip("0123456789"))]
    base = digits[len(xlen_digits) :]
    return "rv" + xlen_digits, frozenset({*base, *tokens[1:]})


def _isa_checks(
    out: _Builder,
    engine: Engine | None,
    dts_isa: Sequence[str],
    images: Sequence[Path],
    *,
    contract_argv: Sequence[str],
    partial: Mapping[str, list[str]] | None = None,
) -> None:
    partial = dict(partial or {})
    if engine is None:
        out.check("isa", UNKNOWN, "no host package: the replay toolchain and Spike command cannot be derived")
        return
    available: tuple[str, frozenset[str]] | None = None
    with out.guard("isa.dts"):
        if len(set(dts_isa)) != 1:
            raise BindingError(f"the DTS declares {len(set(dts_isa))} distinct CPU ISAs")
        spelled, vendor = _vendor_free(dts_isa[0])
        available = _isa_set(engine.canonical_isa(spelled))
        out.machine["dts_isa"] = {"declared": dts_isa[0], "vendor_extensions": vendor, "canonical": available[1]}

    def within(name: str, label: str, required: tuple[str, frozenset[str]], **values: Any) -> None:
        if available is None:
            out.check(name, UNKNOWN, "the DTS ISA is unavailable", **values)
            return
        beyond = sorted(required[1] - available[1])
        cited = [e for e in beyond if e in partial]  # implemented in part; its absent CSRs are checked per image
        missing = [e for e in beyond if e not in partial]
        same_xlen = required[0] == available[0]
        status = PASS if same_xlen and not missing else FAIL
        if status == PASS:
            detail = f"{label} within the DTS ISA" + (
                f" (cited partial: {', '.join(f'{e} without {partial[e]}' for e in cited)})" if cited else ""
            )
        else:
            detail = f"{label} needs {missing or ['XLEN ' + required[0]]} that the DTS riscv,isa does not declare"
        out.check(name, status, detail, required=required[1], missing=missing, cited_partial=cited, **values)

    with out.guard("isa.host_package"):
        if engine.march is None:
            raise BindingError("the host package declares no single -march")
        within(
            "isa.host_package",
            f"host package -march={engine.march}",
            _isa_set(engine.canonical_isa(engine.march)),
            march=engine.march,
        )
    for index, image in enumerate(images):
        with out.guard(f"isa.image[{index}]"):
            tokens = engine.image_arch(image)
            if not tokens:
                raise BindingError(f"{image} records no ISA")
            within(f"isa.image[{index}]", f"{image.name} Tag_RISCV_arch", _isa_set(tokens), image=image)
        with out.guard(f"isa.spike[{index}]"):
            argv = engine.spike_argv(image)
            spelled = [a.removeprefix("--isa=") for a in argv if a.startswith("--isa=")]
            if len(spelled) != 1:
                raise BindingError(f"the Spike command states {len(spelled)} --isa flags")
            out.machine.setdefault("spike", {}).update({"argv": argv, "isa": spelled[0]})
            within(
                f"isa.spike[{index}]",
                f"Spike --isa={spelled[0]}",
                _isa_set(engine.canonical_isa(spelled[0])),
                argv=argv,
            )
    spelled = [a.removeprefix("--isa=") for a in contract_argv if str(a).startswith("--isa=")]
    if spelled:
        with out.guard("isa.contract_engine"):
            within(
                "isa.contract_engine",
                f"contract Spike engine --isa={spelled[0]}",
                _isa_set(engine.canonical_isa(spelled[0])),
                isa=spelled[0],
            )


def spike_memory(argv: Sequence[str]) -> list[tuple[int, int]]:
    """The ``-m<base>:<bytes>[,...]`` spans of a Spike command (a bare ``-m<MiB>`` has no stated base)."""
    spans = []
    for arg in argv:
        if not arg.startswith("-m") or arg.startswith("--"):
            continue
        for part in arg[2:].split(","):
            base, sep, size = part.partition(":")
            if not sep:
                raise BindingError(f"Spike memory {arg!r} states no base; its span is Spike's default")
            spans.append((int(base, 0), int(size, 0)))
    return spans


def _fpga_memory(
    out: _Builder, fpga: Mapping[str, Any], elaborated: str | None, resolve, boards: Mapping[str, Any]
) -> tuple[int, int] | None:
    """The FPGA target's DRAM span from its configuration sources, calibrated on the elaborated config."""
    if not fpga:
        out.check("fpga.target_memory", UNKNOWN, "the binding spec names no FPGA build")
        return None
    with out.guard("fpga.identity"):
        _fpga_identity(out, fpga, resolve)
    sources = fpga.get("sources") or {}
    memory = fpga.get("target_memory") or {}
    reader = S.ConfigSources(
        [resolve(r) for r in sources.get("roots") or ()],
        config_class=str(sources.get("config_class")),
        interpolators={str(k): int(v) for k, v in (sources.get("interpolators") or {}).items()},
    )
    calibrated = False
    with out.guard("fpga.reader_calibration"):
        if "dts" not in out.machine:
            raise BindingError("the elaborated DTS is unavailable, so the reader cannot be calibrated")
        base = reader.resolve(str(elaborated), memory["key"], memory["base"])
        size = reader.resolve(str(elaborated), memory["key"], memory["bytes"])
        dts = [(m["base"], m["bytes"]) for m in out.machine["dts"]["memory"] if m["enabled"]]
        calibrated = dts == [(base.value, size.value)]
        out.check(
            "fpga.reader_calibration",
            PASS if calibrated else FAIL,
            f"the configuration reader gives {elaborated} {hex(base.value)}+{hex(size.value)}; its elaborated DTS "
            f"declares {[f'{hex(b)}+{hex(s)}' for b, s in dts]}",
            derived=_span(base.value, size.value),
            dts=[_span(b, s) for b, s in dts],
            size_from=[*size.via, size.origin],
        )
    span = None
    with out.guard("fpga.target_memory"):
        base = reader.resolve(str(fpga["config"]), memory["key"], memory["base"])
        size = reader.resolve(str(fpga["config"]), memory["key"], memory["bytes"])
        span = (base.value, size.value)
        out.machine["fpga"] = {
            "config": fpga["config"],
            "target_memory": _span(*span),
            "base_from": [*base.via, base.origin],
            "size_from": [*size.via, size.origin],
        }
        out.check(
            "fpga.target_memory",
            PASS if calibrated else UNKNOWN,
            f"{fpga['config']} external memory {hex(span[0])}+{hex(span[1])}"
            + ("" if calibrated else " (the reader is not calibrated on the elaborated configuration)"),
            **out.machine["fpga"],
        )
    host = fpga.get("host_memory") or {}
    if host and span is not None:
        with out.guard("fpga.host_capacity"):
            platform = str(fpga["platform_config"])
            channel = reader.resolve(platform, host["channel_key"], host["channel_bytes"])
            channels = reader.resolve(platform, host["channels_key"], None)
            capacity = channel.value * channels.value
            out.machine.setdefault("fpga", {})["host_memory"] = {
                "platform_config": platform,
                "channel_bytes": channel.value,
                "channels": channels.value,
                "bytes": capacity,
                "channel_from": [*channel.via, channel.origin],
                "channels_from": [*channels.via, channels.origin],
            }
            out.check(
                "fpga.host_capacity",
                PASS if span[1] <= capacity else FAIL,
                f"target DRAM {hex(span[1])} within the host's {channels.value} x {hex(channel.value)} B",
                **out.machine["fpga"]["host_memory"],
            )
    for path in sorted(reader.read_files):
        out.file(f"fpga_source:{path}", Path(path))
    if fpga.get("board"):
        with out.guard("dram.fpga_board_entry"):
            row = boards[fpga["board"]]
            entry = (row.dram_base, row.dram_bytes)
            out.check(
                "dram.fpga_board_entry",
                UNKNOWN if span is None else (PASS if entry == span else FAIL),
                f"board {row.name} declares {hex(entry[0])}+{hex(entry[1])}",
                board=_span(*entry),
                derived=_span(*span) if span else None,
            )
    if fpga.get("dts"):
        with out.guard("dram.fpga_dts"):
            path = resolve(fpga["dts"])
            row = boards.get(fpga.get("board")) if fpga.get("board") else None
            out.file("fpga_dts", path, pinned=getattr(row, "host_dts_sha256", None))
            enabled = [(m["base"], m["bytes"]) for m in S.dts_memory(S.read_dts(path.read_text())) if m["enabled"]]
            out.check(
                "dram.fpga_dts",
                PASS if span is not None and enabled == [span] else FAIL,
                f"the FPGA build's DTS declares {[f'{hex(b)}+{hex(s)}' for b, s in enabled]}",
                dts=[_span(b, s) for b, s in enabled],
            )
    return span if calibrated else None


def _dram_checks(out: _Builder, board, fpga_span: tuple[int, int] | None, images: Sequence[Path]) -> None:
    board_span = (board.dram_base, board.dram_bytes) if board is not None else None
    argv = (out.machine.get("spike") or {}).get("argv")
    spans: list[tuple[int, int]] | None = None
    with out.guard("dram.spike_vs_board"):
        if argv is None:
            raise BindingError("no Spike command was derived")
        spans = spike_memory(argv)
        out.machine["spike"]["memory"] = [_span(b, s) for b, s in spans]
        status = UNKNOWN if board_span is None else (PASS if spans == [board_span] else FAIL)
        out.check(
            "dram.spike_vs_board",
            status,
            f"Spike -m {[f'{hex(b)}+{hex(s)}' for b, s in spans]} vs the board's "
            + (f"{hex(board_span[0])}+{hex(board_span[1])}" if board_span else "unknown span"),
            spike=[_span(b, s) for b, s in spans],
            board=_span(*board_span) if board_span else None,
        )
    status = UNKNOWN if board_span is None or fpga_span is None else (PASS if fpga_span == board_span else FAIL)
    out.check(
        "dram.fpga_vs_board",
        status,
        "the FPGA target's DRAM "
        + (f"{hex(fpga_span[0])}+{hex(fpga_span[1])}" if fpga_span else "(underived)")
        + " vs the board's "
        + (f"{hex(board_span[0])}+{hex(board_span[1])}" if board_span else "(unknown)"),
        fpga=_span(*fpga_span) if fpga_span else None,
        board=_span(*board_span) if board_span else None,
    )
    regions = {
        "board": [board_span] if board_span else None,
        "spike": spans,
        "fpga": [fpga_span] if fpga_span else None,
    }
    for index, image in enumerate(images):
        with out.guard(f"footprint[{index}]"):
            segments = S.elf_load_segments(image)
            outside = {
                name: [
                    f"{hex(a)}+{hex(n)}" for a, n in segments if not any(b <= a and a + n <= b + s for b, s in spans_)
                ]
                for name, spans_ in regions.items()
                if spans_
            }
            missing = [name for name, spans_ in regions.items() if not spans_]
            bad = {k: v for k, v in outside.items() if v}
            status = FAIL if bad else (UNKNOWN if missing else PASS)
            out.check(
                f"footprint[{index}]",
                status,
                f"{image.name}: every loadable segment within the board, Spike and FPGA memory"
                + (f"; outside: {bad}" if bad else "")
                + (f"; unknown span: {missing}" if missing else ""),
                image=image,
                segments=[f"{hex(a)}+{hex(n)}" for a, n in segments],
            )


# ------------------------------------------------------------------------------------------ the command


def replay_memory_defaults() -> tuple[int, int]:
    """The ``--dram-base``/``--dram-bytes`` a replay passes Spike when not told otherwise."""
    from merlin.model_check import cli

    parser = argparse.ArgumentParser(add_help=False)
    cli._engine(parser)  # noqa: SLF001 -- the replay commands' own option defaults
    defaults = parser.parse_args([])
    return defaults.dram_base, defaults.dram_bytes


def receipt_for(args: argparse.Namespace) -> dict[str, Any]:
    """The receipt for a command's arguments: explicit options, else the bundle's engine, else the environment."""
    from merlin.common.paths import build_dir
    from merlin.model_check import bundle as B
    from merlin.model_check.models import load as load_models

    engine_inputs: dict[str, Any] = {}
    if getattr(args, "bundle", None):
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

"""Check numerical, ISA, memory and source facets of a machine binding.

Receipt assembly calls these checks after its record types are initialized.
The check predicates do not choose compiler implementations or acceptance policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from merlin.evaluation.model.binding import device_sources as S
from merlin.evaluation.model.binding.receipt import FAIL, PASS, UNKNOWN, BindingError, _dig, _span
from merlin.evaluation.model.binding.scala_config import ConfigSources

if TYPE_CHECKING:
    from merlin.evaluation.model.binding.receipt import Engine, _Builder


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
    reader = ConfigSources(
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

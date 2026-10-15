"""What the package is asked to lower, per model: interface modules, call ABI and stated entries, no data.

``kernels-export`` writes, for every unique device symbol of every program of a model, exactly the
interface module the device build hands the package (the catalog's), the host call's buffers (role,
shape, dtype in call order) and the kernel's logical argument order, the stated group entry (op,
extents or convolution geometry, epilogue, ``acc_scale``) with the rewrite's notes, and how many routed
groups call it. It reads only a program's catalog, its rewrite sidecar and its census: no operand,
result or expected value, and nothing of any input, sealed or visible. The output is deterministic
(sorted, no timestamps, no paths) so the harness can stage it into an agent sandbox and diff it::

    <out>/index.md                                   every exported model's kernel forms
    <out>/<model>/index.md, kernels.json             one model: forms, per-kernel table, host census
    <out>/<model>/kernels/<program>/<symbol>/interface.mlir, abi.json
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

EXPORT_SCHEMA = "merlin.model_check.kernels_export.v1"

#: Entry fields that name where a group came from rather than what it computes.
_PROVENANCE_FIELDS = frozenset({"name", "cat", "label", "source_role", "source_reference"})


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _dump(value: Any) -> str:
    return json.dumps(value, indent=1, sort_keys=True) + "\n"


def stated_entry(entry: Mapping[str, Any] | None) -> dict[str, Any]:
    """The stated group program without its provenance fields."""
    return {k: v for k, v in sorted((entry or {}).items()) if k not in _PROVENANCE_FIELDS}


def _pair(value, default) -> list[int]:
    values = list(value) if isinstance(value, (list, tuple)) else [value if value is not None else default] * 2
    return [int(v) for v in values]


def _stride_text(stride: Sequence[int]) -> str:
    return f"s{stride[0]}" if len(set(stride)) == 1 else "s" + "x".join(str(v) for v in stride)


def _output_dtypes(call_buffers: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(b.get("dtype")) for b in call_buffers if str(b.get("role", "")).startswith("out_")]


def kernel_form(entry: Mapping[str, Any] | None, call_buffers: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """``(category, form)``: a short class of kernel and its full one-line statement, from the entry alone."""
    entry = dict(entry or {})
    op = str(entry.get("op") or entry.get("kind") or "unknown")
    epilogue = [str(s) for s in entry.get("epilogue") or ()]
    outs = _output_dtypes(call_buffers)
    if op == "conv2d":
        kh, kw = int(entry.get("kh", 0)), int(entry.get("kw", 0))
        stride = _pair(entry.get("stride"), 1)
        padding = [int(v) for v in entry.get("padding") or ()]
        category = f"conv2d {kh}x{kw}/{_stride_text(stride)}"
        form = f"{category} Cin={entry.get('ci')} Cout={entry.get('N')} on {entry.get('Himg')}x{entry.get('Wimg')}" + (
            f" pad {padding}" if any(padding) else ""
        )
    elif op in ("matmul", "batched_matmul", "batch_matmul"):
        category = "matmul" if op == "matmul" else "batched matmul"
        if entry.get("stationary_operand") == "activation":
            category = f"activation-stationary {category}"
        batch = f"B={entry['B']} " if entry.get("B") is not None else ""
        form = f"{category} {batch}M={entry.get('M')} K={entry.get('K')} N={entry.get('N')}"
    else:
        geometry = {k: v for k, v in sorted(entry.items()) if k not in {"op", "kind", "epilogue", "scale_granularity"}}
        category = op
        form = f"{op} {json.dumps(geometry, sort_keys=True)}"
    if epilogue:
        category += " + " + "+".join(epilogue)
        scale = entry.get("acc_scale")
        form += " + " + "+".join(epilogue) + (f" (acc_scale {float(scale):.9g})" if scale is not None else "")
    if outs:
        form += " -> " + ",".join(outs)
        category += " -> " + ",".join(outs)
    return category, form


def interface_opcodes(interface: str) -> list[str]:
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    try:
        cb = parse_interface_mlir(interface)
    except Exception:  # noqa: BLE001 -- an unparsable interface is reported, not raised
        return []
    return [str(c.get("opcode")) for c in cb.get("commands") or ()]


def kernel_arguments(interface: str) -> tuple[list[dict[str, Any]] | None, str]:
    from merlin.llvmlower.device_shim import logical_buffers

    buffers, why = logical_buffers(interface)
    if buffers is None:
        return None, why
    return [
        {"position": i, "name": b.name, "kind": b.kind, "role": b.role, "shape": list(b.shape), "dtype": b.dtype}
        for i, b in enumerate(buffers)
    ], ""


def natural_key(text: str) -> tuple[str, int, str]:
    """``matmul_10`` after ``matmul_9``: a trailing ``_<number>`` sorts by its value."""
    head, sep, tail = text.rpartition("_")
    return (head, int(tail), "") if sep and tail.isdigit() else (text, -1, text)


def _regions(ids: Sequence[str], shown: int = 3) -> str:
    return ", ".join(ids) if len(ids) <= shown + 1 else ", ".join(ids[:shown]) + f", ... ({len(ids)} regions)"


def routed_groups(sidecar: Mapping[str, Any], symbol: str) -> list[dict[str, Any]]:
    rows = [r for r in sidecar.get("routed") or () if str(r.get("symbol")) == symbol]
    return [{"source_region_id": r.get("source_region_id"), "group": r.get("group")} for r in rows]


def describe(catalog, symbol: str) -> dict[str, Any]:
    """Everything about one kernel that does not depend on data: the form the package must lower."""
    sidecar = catalog.sidecar
    interface = catalog.interfaces[symbol]
    entry = (sidecar.get("entries") or {}).get(symbol)
    call = list((sidecar.get("call_buffers") or {}).get(symbol) or ())
    program = (sidecar.get("programs") or {}).get(symbol) or {}
    category, form = kernel_form(entry, call)
    groups = routed_groups(sidecar, symbol)
    arguments, why = kernel_arguments(interface)
    record = {
        "symbol": symbol,
        "category": category,
        "form": form,
        "interface_sha256": _sha(interface),
        "opcodes": interface_opcodes(interface),
        "entry": stated_entry(entry),
        "call_buffers": [{"role": b.get("role"), "shape": b.get("shape"), "dtype": b.get("dtype")} for b in call],
        "arg_access": list((sidecar.get("arg_access") or {}).get(symbol) or ()),
        "kernel_arguments": arguments,
        "routed_groups": len(groups),
        "source_region_ids": sorted({str(g["source_region_id"]) for g in groups}, key=natural_key),
        "notes": [str(n) for n in program.get("notes") or ()],
    }
    for key in ("stored_operand", "transposed", "column_order"):
        if program.get(key) is not None:
            record[key] = program[key]
    if arguments is None:
        record["kernel_arguments_refusal"] = why
    return record


def host_contractions(census: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The census's contractions kept on the host, grouped by op, operand dtypes and reason."""
    total = sum(int(r.get("macs") or 0) for r in census.get("contractions") or ()) or 1
    groups: dict[tuple, dict[str, Any]] = {}
    for row in census.get("contractions") or ():
        if row.get("placement") == "device":
            continue
        key = (str(row.get("op")), tuple(str(d) for d in row.get("dtypes") or ()), str(row.get("placement")))
        slot = groups.setdefault(
            key, {"op": key[0], "dtypes": list(key[1]), "placement": key[2], "count": 0, "macs": 0}
        )
        slot["count"] += 1
        slot["macs"] += int(row.get("macs") or 0)
    rows = [dict(g, mac_fraction=round(g["macs"] / total, 6)) for g in groups.values()]
    return sorted(rows, key=lambda r: (-r["macs"], r["op"], r["placement"]))


def export_model(bundle, model, out_dir: Path, *, programs: Iterable[str] | None = None) -> dict[str, Any]:
    """Write ``<out_dir>/<model>/`` and refresh ``<out_dir>/index.md``; returns the model's record."""
    out_dir = Path(out_dir)
    root = out_dir / model.name
    previous = root / "kernels.json"
    if root.exists():
        if not previous.is_file() or json.loads(previous.read_text()).get("schema") != EXPORT_SCHEMA:
            raise FileExistsError(f"{root} exists and is not a kernels export; refusing to replace it")
        shutil.rmtree(root)
    record: dict[str, Any] = {"schema": EXPORT_SCHEMA, "model": model.name, "programs": []}
    for program in programs or [p.name for p in model.programs]:
        directory = bundle.program_dir(model.name, program)
        census_path = directory / "census.json"
        census = json.loads(census_path.read_text()) if census_path.is_file() else {}
        catalog = bundle.catalog(model.name, program)
        kernels = []
        for symbol in sorted(catalog.interfaces):
            row = describe(catalog, symbol)
            target_dir = root / "kernels" / program / symbol
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "interface.mlir").write_text(catalog.interfaces[symbol])
            (target_dir / "abi.json").write_text(_dump({"program": program, **row}))
            kernels.append({**row, "path": f"kernels/{program}/{symbol}/interface.mlir"})
        record["programs"].append(
            {
                "program": program,
                "routed": len(catalog.sidecar.get("routed") or ()),
                "host_only": bool(catalog.sidecar.get("host_only")),
                "census": {k: census.get(k) for k in ("denominator", "device", "host", "device_mac_fraction")},
                "host_contractions": host_contractions(census),
                "kernels": kernels,
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "kernels.json").write_text(_dump(record))
    (root / "index.md").write_text(model_index(record))
    write_index(out_dir)
    return record


def _forms(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Unique interfaces of a model, grouped by category (one row per category)."""
    seen: dict[str, dict[str, Any]] = {}
    for program in record["programs"]:
        for kernel in program["kernels"]:
            slot = seen.setdefault(kernel["interface_sha256"], {**kernel, "programs": [], "groups": 0})
            slot["programs"].append(program["program"])
            slot["groups"] += kernel["routed_groups"]
    categories: dict[str, dict[str, Any]] = {}
    for kernel in seen.values():
        row = categories.setdefault(kernel["category"], {"category": kernel["category"], "kernels": [], "groups": 0})
        row["kernels"].append(kernel)
        row["groups"] += kernel["groups"]
    return sorted(categories.values(), key=lambda r: (-r["groups"], r["category"]))


def model_index(record: Mapping[str, Any]) -> str:
    unique = len({k["interface_sha256"] for p in record["programs"] for k in p["kernels"]})
    routed = sum(p["routed"] for p in record["programs"])
    lines = [
        f"# Device kernels of `{record['model']}`",
        "",
        f"{unique} unique interface module(s) over {len(record['programs'])} program(s); {routed} routed group(s).",
        "Each kernel's `interface.mlir` is exactly what the package is asked to lower; `abi.json` gives the",
        "host call (`call_buffers`, call order), the kernel's logical argument order and the stated entry.",
        "",
        "## Kernel forms",
        "",
        "| form | unique kernels | routed groups | instances |",
        "|---|---|---|---|",
    ]
    for row in _forms(record):
        instances = "; ".join(sorted({k["form"] for k in row["kernels"]}))
        lines.append(f"| {row['category']} | {len(row['kernels'])} | {row['groups']} | {instances} |")
    notes = sorted({(n, k["category"]) for p in record["programs"] for k in p["kernels"] for n in k["notes"]})
    if notes:
        lines += ["", "## Notes from the group rewrite", ""]
        lines += [f"- {category}: {note}" for note, category in notes]
    lines += ["", "## Per program", ""]
    for program in record["programs"]:
        census = program["census"]
        lines.append(
            f"### `{program['program']}`: {len(program['kernels'])} symbol(s), {program['routed']} routed group(s); "
            f"census {census.get('device')}/{census.get('denominator')} contractions on the device, "
            f"device MAC fraction {census.get('device_mac_fraction')}"
        )
        lines.append("")
        if program["kernels"]:
            lines += ["| symbol | form | groups | regions | opcodes |", "|---|---|---|---|---|"]
            for k in program["kernels"]:
                regions = _regions(k["source_region_ids"])
                lines.append(
                    f"| `{k['symbol']}` | {k['form']} | {k['routed_groups']} | {regions} | {' '.join(k['opcodes'])} |"
                )
            lines.append("")
        for host in program["host_contractions"]:
            lines.append(
                f"- host: {host['count']} x `{host['op']}` {'x'.join(host['dtypes'])} ({host['placement']}), "
                f"{host['mac_fraction']:.2%} of the program's MACs"
            )
        if program["host_contractions"]:
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_index(out_dir: Path) -> Path:
    """``<out_dir>/index.md`` over every model export found under ``out_dir`` (deterministic)."""
    out_dir = Path(out_dir)
    records = []
    for path in sorted(out_dir.glob("*/kernels.json")):
        doc = json.loads(path.read_text())
        if doc.get("schema") == EXPORT_SCHEMA:
            records.append(doc)
    lines = [
        "# Device kernels the package must lower",
        "",
        "Generated by `python -m merlin.model_check kernels-export`. Per model: `<model>/index.md`;",
        "per kernel: `<model>/kernels/<program>/<symbol>/interface.mlir` and `abi.json`.",
        "",
        "| model | form | unique kernels | routed groups |",
        "|---|---|---|---|",
    ]
    for record in records:
        for row in _forms(record):
            lines.append(f"| {record['model']} | {row['category']} | {len(row['kernels'])} | {row['groups']} |")
    host = [
        (record["model"], program["program"], h)
        for record in records
        for program in record["programs"]
        for h in program["host_contractions"]
    ]
    if host:
        lines += ["", "## Contractions kept on the host", ""]
        for model, program, h in host:
            lines.append(
                f"- {model}/{program}: {h['count']} x `{h['op']}` {'x'.join(h['dtypes'])} ({h['placement']}), "
                f"{h['mac_fraction']:.2%} of the program's MACs"
            )
    path = out_dir / "index.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path

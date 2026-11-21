"""Localize a wrong kernel result: the failing call re-runs in a second image that compares on the device.

A replay image prints two digests per result: enough to judge, not to say where a result is wrong. For
each kernel whose result differs, its first failing call (a visible input's when there is one) runs
again in a diagnostic image that also carries the expected result. After the call the device compares
the two element by element and prints the number of differing elements, the first and last, how many
rows (all but the last dimension) and columns (the last dimension) they touch, a coarse map of where
they lie, the first few values the kernel produced and a window from the first difference. Only those
lines cross the console, whatever the result's size.

The host turns them into the kernel report entry's ``diagnosis``: the kernel's form (symbol, source
regions, interface opcodes, buffers, stated entry and epilogue, from :func:`merlin.evaluation.model.kernels.diagnostics.describe`)
and, per result, the first differing element by flat and multi-index with the expected and the produced
value. A refused kernel's diagnosis is its form and the package's or build's refusal text.

A diagnosis is shown at the top of the entry only for a visible input's call; any other (a sealed input, a sealed
shape variant) goes under the entry's ``sealed`` key, which the agent view removes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from merlin.evaluation.model.kernels import replay as RP

# ---- diagnostics -------------------------------------------------------------------------------

#: Failing kernels diagnosed per program (each costs one more call on the simulator).
MAX_KERNELS = 8
#: Differing elements listed per result, and the window shown from the first difference.
LISTED = 16
WINDOW = 32
BEFORE = 4
#: Cells of the mismatch map over a result's flat index.
MAP_CELLS = 32
#: Characters of a refusal kept in a diagnosis.
REFUSAL_CHARS = 4000


def split_of(label: str) -> str:
    """The split of a replay case, the first component of its label (``<split>/<input>/...``)."""
    return label.partition("/")[0]


def choose(cases: Sequence[RP.ReplayCase], verdicts: Sequence[Mapping[str, Any]]) -> dict[str, RP.ReplayCase]:
    """The call to diagnose for each failing symbol: its first failing call, a visible one if any."""
    chosen: dict[str, RP.ReplayCase] = {}
    for case, verdict in zip(cases, verdicts, strict=True):
        if verdict["exact"]:
            continue
        current = chosen.get(case.symbol)
        if current is None or (split_of(current.label) != "visible" and split_of(case.label) == "visible"):
            chosen[case.symbol] = case
    return dict(sorted(chosen.items())[:MAX_KERNELS])


def _blob(work: Path, blobs: dict[str, int], files: list[Path], data: bytes) -> int:
    key = hashlib.sha256(data).hexdigest()
    if key not in blobs:
        blobs[key] = len(files)
        path = work / f"dblob_{len(files)}.bin"
        path.write_bytes(data)
        files.append(path)
    return blobs[key]


_C_HELPERS = """
static void mc_poison(unsigned char *p, unsigned long n) {
  for (unsigned long i = 0; i < n; ++i) p[i] = MC_POISON;
}
static void mc_tag(const char *tag, long c, long o) {
  htif_puts(tag); htif_putc(' '); htif_putd(c); htif_putc(' '); htif_putd(o);
}
static void mc_num(long v) { htif_putc(' '); htif_putd(v); }
static int mc_same(const unsigned char *a, const unsigned char *b, unsigned long es) {
  switch (es) {
    case 1: return a[0] == b[0];
    case 2: return *(const uint16_t *)a == *(const uint16_t *)b;
    case 4: return *(const uint32_t *)a == *(const uint32_t *)b;
    case 8: return *(const uint64_t *)a == *(const uint64_t *)b;
    default: break;
  }
  for (unsigned long i = 0; i < es; ++i) if (a[i] != b[i]) return 0;
  return 1;
}
static unsigned long long mc_value(const unsigned char *p, unsigned long es) {
  unsigned long long v = 0;
  for (unsigned long i = es; i-- > 0;) v = (v << 8) | p[i];
  return v;
}
static void mc_compare(long c, long o, const unsigned char *got, const unsigned char *want,
                       unsigned long n, unsigned long es, unsigned long cols) {
  unsigned long count = 0, first = 0, last = 0, unwritten = 0, rows = 0, ncols = 0, listed = 0;
  unsigned long lastrow = ~0ul, cmin = ~0ul, cmax = 0;
  unsigned long cells[MC_CELLS];
  for (unsigned long i = 0; i < MC_CELLS; ++i) cells[i] = 0;
  for (unsigned long i = 0; i < (cols + 7) / 8; ++i) mc_cols[i] = 0;
  for (unsigned long i = 0; i < n; ++i) {
    const unsigned char *g = got + i * es;
    if (mc_same(g, want + i * es, es)) continue;
    unsigned long poisoned = 1;
    for (unsigned long b = 0; b < es; ++b) if (g[b] != MC_POISON) poisoned = 0;
    if (!count) first = i;
    last = i;
    ++count;
    unwritten += poisoned;
    unsigned long row = i / cols, col = i - row * cols;
    if (row != lastrow) { ++rows; lastrow = row; }
    if (!(mc_cols[col >> 3] & (1u << (col & 7)))) { mc_cols[col >> 3] |= (unsigned char)(1u << (col & 7)); ++ncols; }
    if (col < cmin) cmin = col;
    if (col > cmax) cmax = col;
    ++cells[(i * MC_CELLS) / n];
    if (listed < MC_LISTED) {
      ++listed;
      mc_tag("MCDIFF", c, o); mc_num((long)i); htif_putc(' '); htif_puthex(mc_value(g, es)); htif_putc('\\n');
    }
  }
  mc_tag("MCDSUM", c, o);
  mc_num((long)count); mc_num((long)first); mc_num((long)last); mc_num((long)unwritten);
  mc_num((long)rows); mc_num((long)ncols); mc_num(count ? (long)cmin : 0); mc_num((long)cmax);
  htif_putc(' '); htif_puthex(count ? mc_value(got + last * es, es) : 0);
  htif_putc('\\n');
  mc_tag("MCDMAP", c, o);
  for (unsigned long i = 0; i < MC_CELLS; ++i) mc_num((long)cells[i]);
  htif_putc('\\n');
  if (count) {
    unsigned long start = first > MC_BEFORE ? first - MC_BEFORE : 0;
    unsigned long end = start + MC_WINDOW < n ? start + MC_WINDOW : n;
    mc_tag("MCDWIN", c, o); mc_num((long)start);
    for (unsigned long i = start; i < end; ++i) { htif_putc(' '); htif_puthex(mc_value(got + i * es, es)); }
    htif_putc('\\n');
  }
}
"""


def write_sources(
    work: Path,
    cases: Sequence[RP.ReplayCase],
    kernels: Mapping[str, str],
    interfaces: Mapping[str, str],
) -> tuple[Path, Path, list[dict[str, Any]]]:
    """``main.c`` + ``data.S`` of a diagnostic image for ``cases``, and its per-case plan."""
    from merlin.evaluation.model.linalg_semantics import storage_dtype

    blobs: dict[str, int] = {}
    files: list[Path] = []
    plan: list[dict[str, Any]] = []
    calls: list[str] = []
    used: dict[str, int] = {}
    max_out = 0
    max_cols = 1
    for ci, case in enumerate(cases):
        buffers, _by_role = RP._kernel_order(interfaces[case.symbol], case.roles)  # noqa: SLF001
        value_of = dict(zip(case.roles, case.operands, strict=True))
        expected_of = dict(zip([r for r in case.roles if r.startswith("out_")], case.expected, strict=True))
        used[kernels[case.symbol]] = len(buffers)
        args: list[str] = []
        outs: list[dict[str, Any]] = []
        offset = 0
        for buffer in buffers:
            dtype = np.dtype(storage_dtype(buffer.dtype))
            if buffer.kind == "output":
                want = np.ascontiguousarray(np.asarray(expected_of[buffer.role]).astype(dtype)).reshape(buffer.shape)
                cols = int(buffer.shape[-1]) if buffer.shape else 1
                outs.append(
                    {
                        "role": buffer.role,
                        "offset": offset,
                        "shape": list(buffer.shape),
                        "dtype": buffer.dtype,
                        "elements": int(want.size),
                        "itemsize": dtype.itemsize,
                        "cols": max(cols, 1),
                        "want": _blob(work, blobs, files, want.tobytes()),
                        "expected": want,
                    }
                )
                max_cols = max(max_cols, cols)
                args.append(f"(void *)(mc_out + {offset})")
                offset += (int(want.nbytes) + 63) // 64 * 64
                continue
            data = np.ascontiguousarray(np.asarray(value_of[buffer.role]).astype(dtype)).tobytes()
            args.append(f"(void *)mc_ptr[{_blob(work, blobs, files, data)}]")
        max_out = max(max_out, offset)
        lines = [f"  mc_poison(mc_out, {offset});", f"  {kernels[case.symbol]}({', '.join(args)});"]
        for oi, out in enumerate(outs):
            lines.append(
                f"  mc_compare({ci}, {oi}, mc_out + {out['offset']}, (const unsigned char *)mc_ptr[{out['want']}], "
                f"{out['elements']}ul, {out['itemsize']}ul, {out['cols']}ul);"
            )
        calls.append("\n".join(lines))
        plan.append({"case": ci, "symbol": case.symbol, "label": case.label, "outputs": outs})
    data = ['.section .weights, "a"', ".balign 64"]
    for index, path in enumerate(files):
        data += [f".global mc_dblob_{index}", f"mc_dblob_{index}:", f'.incbin "{path}"', ".balign 64"]
    data_s = work / "data.S"
    data_s.write_text("\n".join(data) + "\n")
    externs = "\n".join(f"extern const unsigned char mc_dblob_{i}[];" for i in range(len(files)))
    externs += "\nconst void *volatile mc_ptr[] = {" + ", ".join(f"mc_dblob_{i}" for i in range(len(files))) + "};"
    decls = "\n".join(f"extern void {k}({', '.join('void *' for _ in range(n))});" for k, n in sorted(used.items()))
    defines = "\n".join(
        f"#define {name} {value}"
        for name, value in (
            ("MC_POISON", RP.POISON),
            ("MC_CELLS", MAP_CELLS),
            ("MC_LISTED", LISTED),
            ("MC_WINDOW", WINDOW),
            ("MC_BEFORE", BEFORE),
        )
    )
    main_c = work / "main.c"
    main_c.write_text(
        f'#include <stdint.h>\n#include "htif.h"\n{defines}\n{externs}\n{decls}\n'
        f"static unsigned char mc_out[{max(max_out, 64)}] __attribute__((aligned(64)));\n"
        f"static unsigned char mc_cols[{(max_cols + 7) // 8}];\n"
        f"{_C_HELPERS}\n"
        "int main(unsigned long hartid) {\n"
        '  if (hartid) for (;;) { __asm__ volatile("wfi"); }\n'
        "  console_init();\n"
        "  htif_line_flush(0);\n"
        + "\n".join(calls)
        + '\n  htif_line_flush(1);\n  htif_puts("MCDONE\\n");\n  htif_exit(0);\n}\n'
    )
    return main_c, data_s, plan


def _decode(raw: int, itemsize: int, dtype: np.dtype):
    return np.frombuffer(int(raw).to_bytes(8, "little")[:itemsize], dtype=dtype)[0].item()


def _element(out: Mapping[str, Any], flat: int, got=None, *, position=None) -> dict[str, Any]:
    want = out["expected"]
    row = {"flat": int(flat), "index": [int(v) for v in np.unravel_index(int(flat), want.shape)]}
    row["expected"] = want.reshape(-1)[int(flat)].item()
    if got is not None:
        row["got"] = got
    if position is not None:
        row["position"] = position(int(flat))
    return row


def _map(cells: Sequence[int], elements: int) -> str:
    """One character per cell of the flat index: ``.`` none differ, ``#`` all differ, ``1``-``9`` tenths."""
    text = []
    for cell, count in enumerate(cells):
        lo = -(-cell * elements // MAP_CELLS)
        hi = -(-(cell + 1) * elements // MAP_CELLS)
        size = max(hi - lo, 1)
        if count <= 0:
            text.append(".")
        elif count >= size:
            text.append("#")
        else:
            text.append(str(min(9, max(1, -(-9 * count // size)))))
    return "".join(text)


def conv_position(entry: Mapping[str, Any] | None, shape: Sequence[int]) -> Callable[[int], dict[str, int]] | None:
    """For a convolution result stored as ``[N*Ho*Wo, Co]``: flat index -> output pixel and channel."""
    from merlin.runtime.engine.command_buffer import conv_out_dims

    entry = dict(entry or {})
    if entry.get("op") != "conv2d" or len(shape) != 2:
        return None
    stride = list(entry.get("stride") or [1, 1])
    padding = list(entry.get("padding") or [0, 0, 0, 0])
    ho, wo = conv_out_dims(
        int(entry["Himg"]), int(entry["Wimg"]), int(entry["kh"]), int(entry["kw"]), stride, padding, [1, 1]
    )
    cols = int(shape[1])

    def where(flat: int) -> dict[str, int]:
        row, co = divmod(flat, cols)
        n, pixel = divmod(row, ho * wo)
        oh, ow = divmod(pixel, wo)
        return {"n": n, "oh": oh, "ow": ow, "co": co}

    return where if ho * wo > 0 else None


def parse_console(console: str, plan: Sequence[Mapping[str, Any]], entries: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The diagnostic image's lines, per planned case and result."""
    from merlin.evaluation.model.linalg_semantics import storage_dtype

    lines: dict[tuple[str, int, int], list[list[str]]] = {}
    for line in console.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] in ("MCDIFF", "MCDSUM", "MCDMAP", "MCDWIN"):
            lines.setdefault((parts[0], int(parts[1]), int(parts[2])), []).append(parts[3:])
    found = []
    for entry in plan:
        results = []
        for oi, out in enumerate(entry["outputs"]):
            dtype = np.dtype(storage_dtype(out["dtype"]))
            summary = lines.get(("MCDSUM", entry["case"], oi))
            row: dict[str, Any] = {
                "role": out["role"],
                "shape": out["shape"],
                "dtype": out["dtype"],
                "elements": out["elements"],
            }
            if not summary:
                results.append({**row, "error": "the diagnostic image printed no comparison for this result"})
                continue
            count, first, last, unwritten, rows, ncols, cmin, cmax = (int(v) for v in summary[0][:8])
            last_got = int(summary[0][8], 16) if len(summary[0]) > 8 else None
            cells = [int(v) for v in (lines.get(("MCDMAP", entry["case"], oi)) or [[]])[0]]
            position = conv_position(entries.get(entry["symbol"]), out["shape"])
            row.update(
                differing=count,
                fraction=round(count / max(out["elements"], 1), 6),
                unwritten=unwritten,
                rows_differing=rows,
                rows=out["elements"] // out["cols"],
                cols_differing=ncols,
                cols=out["cols"],
            )
            if count:
                listed = [
                    _element(out, int(i), _decode(int(v, 16), out["itemsize"], dtype), position=position)
                    for i, v in lines.get(("MCDIFF", entry["case"], oi), [])
                ]
                row["first"] = listed[0] if listed else _element(out, first, position=position)
                row["last"] = _element(
                    out,
                    last,
                    None if last_got is None else _decode(last_got, out["itemsize"], dtype),
                    position=position,
                )
                row["col_range"] = [cmin, cmax]
                row["map"] = _map(cells, out["elements"])
                row["listed"] = listed
                window = (lines.get(("MCDWIN", entry["case"], oi)) or [[]])[0]
                if window:
                    start = int(window[0])
                    got = [_decode(int(v, 16), out["itemsize"], dtype) for v in window[1:]]
                    flat = out["expected"].reshape(-1)
                    row["window"] = {
                        "start": _element(out, start)["index"],
                        "expected": [flat[start + i].item() for i in range(len(got))],
                        "got": got,
                    }
                if unwritten == count:
                    row["hint"] = "every differing element still holds the poison byte: the kernel never wrote them"
            else:
                row["hint"] = "the result matched on the diagnostic run: the kernel's result is not reproducible"
            results.append(row)
        found.append({"symbol": entry["symbol"], "label": entry["label"], "results": results})
    return found


def refusal(catalog, symbol: str, why: str) -> dict[str, Any]:
    """The diagnosis of a refused kernel: its form and the refusal text."""

    return {"kind": "refused", **_context(describe(catalog, symbol)), "refusal": str(why)[:REFUSAL_CHARS]}


def _context(described: Mapping[str, Any]) -> dict[str, Any]:
    keep = ("form", "source_region_ids", "routed_groups", "opcodes", "entry", "kernel_arguments", "interface_sha256")
    context = {k: described.get(k) for k in keep}
    entry = described.get("entry") or {}
    context["epilogue"] = list(entry.get("epilogue") or [])
    if entry.get("acc_scale") is not None:
        context["acc_scale"] = entry["acc_scale"]
    return context


def localize(
    cases: Sequence[RP.ReplayCase],
    verdicts: Sequence[Mapping[str, Any]],
    *,
    catalog,
    kernels: Mapping[str, str],
    objects: Sequence[Path],
    work: Path,
    run: Callable[[Path, Path, Path, Sequence[Path]], str],
    image_bytes: int = 768 << 20,
) -> dict[str, dict[str, Any]]:
    """``symbol -> diagnosis`` for every failing kernel (at most :data:`MAX_KERNELS`).

    ``cases`` and ``verdicts`` are the replay's runnable cases and their verdicts, in the same order;
    ``run(work, main_c, data_s, objects)`` links and runs an image and returns its console. One image
    carries at most ``image_bytes`` of operands and expected results.
    """

    chosen = choose(cases, verdicts)
    diagnoses: dict[str, dict[str, Any]] = {}
    for symbol, case in chosen.items():
        diagnoses[symbol] = {
            "kind": "mismatch",
            **_context(describe(catalog, symbol)),
            "case": case.label,
            "split": split_of(case.label),
        }
    images: list[list[RP.ReplayCase]] = []
    size = 0
    for case in chosen.values():
        nbytes = sum(int(np.asarray(a).nbytes) for a in (*case.operands, *case.expected))
        if not images or (images[-1] and size + nbytes > image_bytes):
            images.append([])
            size = 0
        images[-1].append(case)
        size += nbytes
    for index, batch in enumerate(images):
        image_dir = Path(work) / f"diagnose_{index}"
        image_dir.mkdir(parents=True, exist_ok=True)
        try:
            main_c, data_s, plan = write_sources(image_dir, batch, kernels, catalog.interfaces)
            console = run(image_dir, main_c, data_s, objects)
        except (RP.ReplayError, OSError, ValueError, KeyError, TimeoutError) as error:
            for case in batch:
                diagnoses[case.symbol]["error"] = (
                    f"diagnostic replay failed: {type(error).__name__}: {str(error)[:600]}"
                )
            continue
        for found in parse_console(console, plan, catalog.sidecar.get("entries") or {}):
            diagnoses[found["symbol"]]["results"] = found["results"]
    return diagnoses


def summary(diagnosis: Mapping[str, Any]) -> str:
    """One line per diagnosis: where the first result differs, and how much of it."""
    if diagnosis.get("kind") == "refused":
        return f"{diagnosis.get('form')}: refused: {str(diagnosis.get('refusal'))[:200]}"
    if diagnosis.get("error"):
        return f"{diagnosis.get('form')}: {diagnosis['error'][:200]}"
    parts = []
    for result in diagnosis.get("results") or ():
        first = result.get("first")
        if not first:
            parts.append(f"{result['role']}: {result.get('hint') or result.get('error')}")
            continue
        got = first.get("got", "?")
        parts.append(
            f"{result['role']}{first['index']}: expected {first['expected']}, got {got}; "
            f"{result['differing']}/{result['elements']} elements differ in "
            f"{result['rows_differing']}/{result['rows']} rows x {result['cols_differing']}/{result['cols']} cols, "
            f"map {result['map']}"
        )
    return f"{diagnosis.get('form')} on {diagnosis.get('case')}: " + "; ".join(parts)


def attach(rows: Sequence[dict[str, Any]], diagnoses: Mapping[str, Mapping[str, Any]]) -> None:
    """Put each diagnosis on its kernel row: ``diagnosis`` for a visible input's call; any other call's
    (a sealed input, a sealed shape variant) under ``sealed.diagnosis``, which the agent view removes."""
    for row in rows:
        diagnosis = diagnoses.get(row.get("symbol"))
        if diagnosis is None:
            continue
        if diagnosis.get("split") == "visible":
            row["diagnosis"] = dict(diagnosis)
        else:
            row.setdefault("sealed", {})["diagnosis"] = dict(diagnosis)


# ---- export ------------------------------------------------------------------------------------
# What the package is asked to lower, per model: interface modules, call ABI and stated entries, no data.
#
# ``kernels-export`` writes, for every unique device symbol of every program of a model, exactly the
# interface module the device build hands the package (the catalog's), the host call's buffers (role,
# shape, dtype in call order) and the kernel's logical argument order, the stated group entry (op,
# extents or convolution geometry, epilogue, ``acc_scale``) with the rewrite's notes, and how many routed
# groups call it. It reads only a program's catalog, its rewrite sidecar and its census: no operand,
# result or expected value, and nothing of any input, sealed or visible. The output is deterministic
# (sorted, no timestamps, no paths) so the harness can stage it into an agent sandbox and diff it::
#
#     <out>/index.md                                   every exported model's kernel forms
#     <out>/<model>/index.md, kernels.json             one model: forms, per-kernel table, host census
#     <out>/<model>/kernels/<program>/<symbol>/interface.mlir, abi.json

EXPORT_SCHEMA = "merlin.evaluation.model.kernels_export.v1"

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
    from merlin.target.contract.interfaces.grammar import parse_interface_mlir

    try:
        cb = parse_interface_mlir(interface)
    except Exception:  # noqa: BLE001 -- an unparsable interface is reported, not raised
        return []
    return [str(c.get("opcode")) for c in cb.get("commands") or ()]


def kernel_arguments(interface: str) -> tuple[list[dict[str, Any]] | None, str]:
    from merlin.compiler.lower.device.device_shim import logical_buffers

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
    if program.get("batch_shape"):
        # The host call carries this leading batch and the kernel states ONE slice: the host calls the
        # kernel once per slice, so ``call_buffers`` lead with the batch and ``kernel_arguments`` do not.
        record["batch_shape"] = [int(v) for v in program["batch_shape"]]
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
        "Generated by `python -m merlin.evaluation.model kernels-export`. Per model: `<model>/index.md`;",
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

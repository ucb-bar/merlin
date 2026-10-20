"""Localize a wrong kernel result: the failing call re-runs in a second image that compares on the device.

A replay image prints two digests per result: enough to judge, not to say where a result is wrong. For
each kernel whose result differs, its first failing call (a visible input's when there is one) runs
again in a diagnostic image that also carries the expected result. After the call the device compares
the two element by element and prints the number of differing elements, the first and last, how many
rows (all but the last dimension) and columns (the last dimension) they touch, a coarse map of where
they lie, the first few values the kernel produced and a window from the first difference. Only those
lines cross the console, whatever the result's size.

The host turns them into the kernel report entry's ``diagnosis``: the kernel's form (symbol, source
regions, interface opcodes, buffers, stated entry and epilogue, from :func:`merlin.model_check.export.describe`)
and, per result, the first differing element by flat and multi-index with the expected and the produced
value. A refused kernel's diagnosis is its form and the package's or build's refusal text.

A diagnosis is shown at the top of the entry only for a visible input's call; any other (a sealed input, a sealed
shape variant) goes under the entry's ``sealed`` key, which the agent view removes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from merlin.model_check import replay as RP

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
    from merlin.runtime.linalg_numpy import storage_dtype

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
    from merlin.runtime.commandbuffer import conv_out_dims

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
    from merlin.runtime.linalg_numpy import storage_dtype

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
    from merlin.model_check import export as EX

    return {"kind": "refused", **_context(EX.describe(catalog, symbol)), "refusal": str(why)[:REFUSAL_CHARS]}


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
    from merlin.model_check import export as EX

    chosen = choose(cases, verdicts)
    diagnoses: dict[str, dict[str, Any]] = {}
    for symbol, case in chosen.items():
        diagnoses[symbol] = {
            "kind": "mismatch",
            **_context(EX.describe(catalog, symbol)),
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

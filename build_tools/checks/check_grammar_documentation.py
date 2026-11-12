#!/usr/bin/env python3
"""Gate: the interface grammar spec states exactly what the parser and the dialect accept.

``src/merlin/resources/contracts/interface_grammar.md`` is the spec of the ``merlin_iface`` input format, and it is in
the agent's bundle. Which ops a module may legally contain is decided elsewhere, by the reference
parser's own tables (``interface_emit``), which the parser treats as authoritative and fails CLOSED on.
Two more statements of the same grammar exist: the dialect contract (``interface_dialect_contract.yaml``,
the ops a package MUST accept and the opcode each maps to) and the registered dialect
(``merlin_iface.irdl.mlir``, generated from the reference ODS and loaded by a C++ tool). Nothing kept the
four in step.

Measured 2026-09-24: the parser defined 15 mnemonics and the spec documented 7 -- ``attention_pv``,
``attention_qk``, ``conv2d``, ``movement``, ``residual_add``, ``rmsnorm``, ``rope`` and ``softmax`` were
accepted, emitted into graded capsules, and described nowhere the agent could read, leaving it to
reverse-engineer its obligation from the capsule corpus.

What fails, each compared by CONTENT so a renamed opcode or a reordered operand list is a failure rather
than a stale table nobody notices:

* the spec's ``## Op index`` against the parser: every op it defines has a row whose opcode and operand
  order are the parser's, and no row names an op the parser does not define;
* every op the parser defines has its own ``### merlin_iface.<op>`` prose section;
* the dialect contract: every op it requires is one the parser defines, documented, and maps to the
  parser's own opcode;
* the registered dialect: every op it declares is one the parser defines and is documented, a named
  whole-op's IRDL operands are the parser's operand keys in order, and the index's "registered dialect"
  column says, op by op, exactly which ops the IRDL declares.

Parsed structurally with ``str`` operations (the no-regex rule applies to the checker too). FAILS CLOSED:
an unreadable source is reported as a failure, because a check that could not run established nothing.

    python build_tools/checks/check_grammar_documentation.py
    python build_tools/checks/check_grammar_documentation.py --json
    python build_tools/checks/check_grammar_documentation.py --print-index   # the table, to paste in
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _source_layout  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
for _package in reversed(_source_layout.source_packages(_REPO)):
    if str(_package.parent) not in sys.path:
        sys.path.insert(0, str(_package.parent))

_CONTRACT = _REPO / "src/merlin/resources/contracts"
_SPEC = _CONTRACT / "interface_grammar.md"
_DIALECT_CONTRACT = _CONTRACT / "interface_dialect_contract.yaml"
_IRDL = _CONTRACT / "merlin_iface.irdl.mlir"
_INDEX_HEADING = "## Op index"
_OP_PREFIX = "merlin_iface."
_ROW_MARK = "|"
_NO_OPCODE = "— (declares a leaf, issues no command)"
_EMPTY = ("—", "-", "")


def parser_table() -> dict[str, dict]:
    """``{mnemonic: {opcode, operands}}`` from the reference parser's own tables."""
    from merlin.development.targetgen.contract import interface_emit as IE

    opcodes = {**IE._OP_TO_OPCODE, **IE._NAMED_OP_TO_OPCODE}
    return {
        mnemonic: {
            "opcode": opcodes.get(mnemonic),
            "operands": list(IE._NAMED_OP_OPERAND_KEYS.get(mnemonic) or ()),
        }
        for mnemonic in sorted(IE.defined_mnemonics())
    }


def irdl_operations(text: str) -> dict[str, list[str]]:
    """``{op: [operand names in order]}`` declared by an IRDL dialect file."""
    ops: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("irdl.operation @"):
            current = stripped[len("irdl.operation @") :].split()[0].strip('"{')
            ops[current] = []
        elif current is not None and stripped.startswith("irdl.operands("):
            inner = stripped[len("irdl.operands(") :].rpartition(")")[0]
            ops[current] = [piece.partition(":")[0].strip() for piece in inner.split(",") if piece.strip()]
    return ops


def _documented_headings(text: str) -> set[str]:
    """Mnemonics given a prose section: every ``merlin_iface.<op>`` named on a heading line."""
    found: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        for chunk in stripped.split(_OP_PREFIX)[1:]:
            name = ""
            for char in chunk:
                if not (char.isalnum() or char == "_"):
                    break
                name += char
            if name:
                found.add(name)
    return found


def _index_rows(text: str) -> dict[str, dict]:
    """The generated op-index table as ``{mnemonic: {opcode, operands, registered}}``.

    Rows are ``| `op` | `OPCODE` | `a`, `b` | yes |``; a missing heading yields an empty mapping."""
    _, marker, tail = text.partition(_INDEX_HEADING)
    if not marker:
        return {}
    rows: dict[str, dict] = {}
    for line in tail.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):  # the index ends at the next heading
            break
        if not stripped.startswith(_ROW_MARK):
            continue
        cells = [cell.strip() for cell in stripped.strip(_ROW_MARK).split(_ROW_MARK)]
        if len(cells) < 4:
            continue
        mnemonic = cells[0].strip("`").strip()
        if not mnemonic or mnemonic == "op" or set(mnemonic) <= {"-", ":"}:
            continue
        # A cell may carry a gloss after the em dash; the dash is the value and the rest is prose.
        opcode = cells[1].split()[0].strip("`") if cells[1].strip() else ""
        operands = (
            [] if cells[2] in _EMPTY else [piece.strip().strip("`") for piece in cells[2].split(",") if piece.strip()]
        )
        rows[mnemonic] = {
            "opcode": None if opcode in _EMPTY else opcode,
            "operands": operands,
            "registered": cells[3].strip().lower(),
        }
    return rows


def render_index(table: dict[str, dict], registered: set[str]) -> str:
    """The op-index rows, generated from the parser and the IRDL. Regenerate when either gains an op."""
    lines = []
    for mnemonic, row in table.items():
        opcode = f"`{row['opcode']}`" if row["opcode"] else _NO_OPCODE
        operands = ", ".join(f"`{key}`" for key in row["operands"]) or "—"
        lines.append(f"| `{mnemonic}` | {opcode} | {operands} | {'yes' if mnemonic in registered else 'no'} |")
    return "\n".join(lines)


def audit() -> dict:
    try:
        table = parser_table()
    except Exception as exc:  # noqa: BLE001 -- reported: a check that could not run established nothing
        return {"status": "unverifiable", "detail": f"the reference parser tables are unreadable: {exc}"}
    if not table:
        return {"status": "unverifiable", "detail": "the reference parser defines no mnemonics"}
    try:
        text = _SPEC.read_text(encoding="utf-8")
        irdl = irdl_operations(_IRDL.read_text(encoding="utf-8"))
        contract = yaml.safe_load(_DIALECT_CONTRACT.read_text(encoding="utf-8"))
        required = {
            str(op["name"]).partition(_OP_PREFIX)[2]: op.get("maps_to") for op in contract["dialect"]["required_ops"]
        }
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        return {"status": "unverifiable", "detail": f"a grammar source is unreadable: {type(exc).__name__}: {exc}"}
    if not irdl:
        return {"status": "unverifiable", "detail": f"{_IRDL.name} declares no operation"}

    headings = _documented_headings(text)
    index = _index_rows(text)
    documented = set(index) | headings
    parsed = {name: {"opcode": row["opcode"], "operands": row["operands"]} for name, row in index.items()}
    registered_claim = {name for name, row in index.items() if row["registered"] == "yes"}
    named = {name for name, row in table.items() if row["operands"]}
    return {
        "status": "ok",
        "n_defined": len(table),
        "undocumented": sorted(set(table) - documented),
        "documented_but_undefined": sorted(documented - set(table)),
        "index_missing": sorted(set(table) - set(index)),
        "index_drifted": sorted(name for name, row in table.items() if name in parsed and parsed[name] != row),
        "no_prose_section": sorted(set(table) - headings),
        "contract_undefined": sorted(set(required) - set(table)),
        "contract_undocumented": sorted(set(required) & set(table) - documented),
        "contract_opcode_drift": sorted(
            name
            for name, maps_to in required.items()
            if name in table and table[name]["opcode"] is not None and maps_to != table[name]["opcode"]
        ),
        "dialect_undefined": sorted(set(irdl) - set(table)),
        "dialect_undocumented": sorted(set(irdl) & set(table) - documented),
        "dialect_operand_drift": sorted(name for name in set(irdl) & named if irdl[name] != table[name]["operands"]),
        "dialect_column_drift": sorted((registered_claim ^ set(irdl)) & set(index)),
        "with_prose_section": sorted(headings & set(table)),
        "registered": sorted(set(irdl) & set(table)),
    }


_FINDINGS = (
    ("undocumented", "the parser accepts op(s) the grammar spec describes nowhere"),
    ("documented_but_undefined", "the grammar spec describes op(s) the parser does not define"),
    ("index_missing", "the op index omits op(s) the parser defines"),
    ("index_drifted", "the op index disagrees with the parser (opcode or operand order)"),
    ("no_prose_section", "op(s) the parser defines have no `### merlin_iface.<op>` section"),
    ("contract_undefined", "interface_dialect_contract.yaml requires op(s) the parser does not define"),
    ("contract_undocumented", "interface_dialect_contract.yaml requires op(s) the spec does not document"),
    ("contract_opcode_drift", "interface_dialect_contract.yaml maps op(s) to an opcode the parser does not emit"),
    ("dialect_undefined", "the registered dialect declares op(s) the parser does not define"),
    ("dialect_undocumented", "the registered dialect declares op(s) the spec does not document"),
    ("dialect_operand_drift", "the registered dialect's operands disagree with the parser's operand keys"),
    ("dialect_column_drift", "the op index's 'registered dialect' column disagrees with merlin_iface.irdl.mlir"),
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--print-index", action="store_true", help="print the index rows and exit")
    args = parser.parse_args(argv)

    if args.print_index:
        print(render_index(parser_table(), set(irdl_operations(_IRDL.read_text(encoding="utf-8")))))
        return 0

    report = audit()
    if report["status"] != "ok":
        print(f"[FAIL] grammar-documentation: {report['detail']}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))

    failed = False
    for key, message in _FINDINGS:
        if report[key]:
            failed = True
            print(f"[FAIL] {message}: {', '.join(report[key])}", file=sys.stderr)
    if failed:
        print(
            f"[hint] regenerate the rows with --print-index and paste them under '{_INDEX_HEADING}' in "
            f"{_SPEC.relative_to(_REPO) if _SPEC.is_relative_to(_REPO) else _SPEC}",
            file=sys.stderr,
        )
        return 1
    if not args.json:
        print(
            f"[  ok] grammar-documentation: {report['n_defined']} ops defined, every one indexed and given a "
            f"section; {len(report['registered'])} declared by the registered dialect, as the index says."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

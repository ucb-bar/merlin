#!/usr/bin/env python3
"""Validate emitted NPU text ISA against model_npu ISA contract.

Checks:
  1. Each emitted mnemonic exists in model_npu ISA definitions.
  2. Required argument keys exist for each mnemonic.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

LINE_RE = re.compile(r"^([a-zA-Z0-9_.]+)(?:\s+(.*))?$")
KV_RE = re.compile(r"([a-zA-Z0-9_]+)\s*=\s*([+-]?[0-9]+)")


def _extract_instr_mnemonic(node: ast.FunctionDef) -> str | None:
    for dec in node.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        if not isinstance(dec.func, ast.Name) or dec.func.id != "instr":
            continue
        if not dec.args:
            continue
        first = dec.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    return None


def _extract_arg_keys(node: ast.FunctionDef) -> set[str]:
    keys: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Subscript):
            continue
        if not isinstance(sub.value, ast.Name) or sub.value.id != "args":
            continue

        key = None
        sl = sub.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            key = sl.value
        elif hasattr(ast, "Index") and isinstance(sl, ast.Index):
            inner = sl.value
            if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                key = inner.value

        if key is not None:
            keys.add(key)
    return keys


def parse_isa_definition(isa_definition_py: Path) -> dict[str, set[str]]:
    tree = ast.parse(isa_definition_py.read_text(), filename=str(isa_definition_py))
    contract: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        mnemonic = _extract_instr_mnemonic(node)
        if mnemonic is None:
            continue
        contract[mnemonic] = _extract_arg_keys(node)
    return contract


def parse_text_isa(isa_text: Path) -> list[tuple[int, str, set[str]]]:
    parsed: list[tuple[int, str, set[str]]] = []
    for idx, raw in enumerate(isa_text.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE_RE.match(line)
        if not m:
            raise ValueError(f"Line {idx}: invalid ISA line: {line}")
        mnemonic = m.group(1)
        rest = m.group(2) or ""
        keys = {k for k, _ in KV_RE.findall(rest)}
        parsed.append((idx, mnemonic, keys))
    return parsed


def main() -> int:
    repo_root = Path(__file__).resolve().parents[6]
    legacy_isa_def = repo_root / "third_party" / "npu_model" / "model_npu" / "configs" / "isa_definition.py"
    modern_isa_def = repo_root / "third_party" / "npu_model" / "npu_model" / "configs" / "isa_definition.py"
    default_isa_def = modern_isa_def if modern_isa_def.exists() else legacy_isa_def

    parser = argparse.ArgumentParser(description="Check emitted ISA against model_npu contract")
    parser.add_argument("isa_text", type=Path, help="Path to emitted text ISA file")
    parser.add_argument(
        "--isa-definition",
        type=Path,
        default=default_isa_def,
        help="Path to model_npu isa_definition.py",
    )
    parser.add_argument(
        "--strict-keys",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require exact key sets (default: false)",
    )
    args = parser.parse_args()

    contract = parse_isa_definition(args.isa_definition)
    emitted = parse_text_isa(args.isa_text)

    errors: list[str] = []
    for line_no, mnemonic, emitted_keys in emitted:
        if mnemonic not in contract:
            errors.append(f"Line {line_no}: mnemonic '{mnemonic}' is not defined in {args.isa_definition}")
            continue

        expected = contract[mnemonic]
        if args.strict_keys:
            if emitted_keys != expected:
                errors.append(
                    f"Line {line_no}: mnemonic '{mnemonic}' keys mismatch. "
                    f"expected={sorted(expected)} emitted={sorted(emitted_keys)}"
                )
        else:
            if not expected.issubset(emitted_keys):
                errors.append(
                    f"Line {line_no}: mnemonic '{mnemonic}' missing required keys. "
                    f"required={sorted(expected)} emitted={sorted(emitted_keys)}"
                )

    if errors:
        print("ISA contract check FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("ISA contract check PASSED: " f"{len(emitted)} instruction(s), {len({m for _, m, _ in emitted})} mnemonic(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Gate: the command-buffer epilogue vocabulary has ONE definition, and nothing has drifted from it.

The definition is :data:`merlin.runtime.commandbuffer.EPILOGUE_STAGES`. Six representations derive from
it and are checked here:

  1. the strict JSON validator's ``epilogue`` enum (``merlin/contract/schemas/command_buffer.schema.json``)
     — must equal the tuple, IN ORDER (``--write`` regenerates it);
  2/3. the two prose ABI documents' COMMIT ``epilogue`` line — must name the same set;
  4/5. the two in-tree dialect verifiers' ``KNOWN_EPILOGUE`` — must be the same object;
  6. the code template the target generator emits — must render from the same tuple.

Every other ``ordered subset of [...]`` line in the ABI documents is a per-opcode SUBSET (an engine that
implements fewer stages for one opcode) and is checked to be a subset, never widened silently.

WHY. There were six hand-maintained copies and no two agreed: the validator rejected ``maxpool``, which
both ABI documents instruct an author to emit and all three engines implement; the dialect copies
rejected ``acc_scale``, which the validator and both documents admit. A capsule whose commit fused a
pooling readout was failed with a schema violation for following the documented ABI.

Parsing is STRUCTURAL (yaml/json loads, ``str.partition``/``split``) — no regex, per the repo rule.

Usage::

    python build_tools/scripts/check_epilogue_vocabulary.py            # check (exit 1 on drift)
    python build_tools/scripts/check_epilogue_vocabulary.py --write    # regenerate the JSON enum
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT / "merlin" / "python"))

from merlin.runtime.commandbuffer import EPILOGUE_STAGES  # noqa: E402

_SCHEMA = _ROOT / "merlin" / "contract" / "schemas" / "command_buffer.schema.json"
#: The prose ABI documents, and which opcode's epilogue line in each carries the FULL vocabulary.
_ABI_DOCS = ((_ROOT / "merlin" / "contract" / "command_buffer_abi.yaml", "COMMIT"),
             (_ROOT / "merlin" / "contract" / "interface_dialect_contract.yaml", "merlin_iface.commit"))
_MARKER = "ordered subset of ["


def schema_enum(doc: dict) -> list[str] | None:
    """The validator's epilogue enum, or None when the schema does not define one (fail closed: the
    caller reports 'could not be located', never 'matches')."""
    node = doc
    for key in ("properties", "commands", "items", "properties", "attributes",
                "properties", "epilogue", "items"):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node.get("enum") if isinstance(node, dict) else None


def _bracketed(text: str) -> list[str] | None:
    """The comma-separated names inside the first ``[...]`` of an ``ordered subset of [...]`` string."""
    _, _, rest = text.partition(_MARKER)
    if not rest:
        return None
    inner, sep, _ = rest.partition("]")
    if not sep:
        return None
    return [t.strip() for t in inner.split(",") if t.strip()]


def abi_epilogue_lines(node, path: tuple = ()) -> list[tuple[tuple, list[str]]]:
    """Every ``epilogue: "ordered subset of [...]"`` in a loaded ABI document, with its key path."""
    out: list[tuple[tuple, list[str]]] = []
    if isinstance(node, dict):
        # A list entry identifies itself by its own `name` field (the interface-dialect document lists
        # ops); a mapping identifies its children by key (the command-buffer document keys by opcode).
        # Folding the self-name into the path makes one lookup work for both shapes.
        own = node.get("name")
        if isinstance(own, str):
            path = path + (own,)
        for k, v in node.items():
            if k == "epilogue" and isinstance(v, str):
                names = _bracketed(v)
                if names is not None:
                    out.append((path + (k,), names))
            else:
                out.extend(abi_epilogue_lines(v, path + (str(k),)))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.extend(abi_epilogue_lines(v, path + (str(i),)))
    return out


def check(write: bool = False) -> list[str]:
    canon = list(EPILOGUE_STAGES)
    problems: list[str] = []

    # 1. the JSON validator's enum
    doc = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    enum = schema_enum(doc)
    if enum is None:
        problems.append(f"{_SCHEMA.name}: the epilogue enum could not be located — the schema shape "
                        f"changed; this gate cannot certify it and does NOT report success")
    elif list(enum) != canon:
        if write:
            node = doc["properties"]["commands"]["items"]["properties"]["attributes"]["properties"]
            node["epilogue"]["items"]["enum"] = canon
            _SCHEMA.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            print(f"rewrote {_SCHEMA} epilogue enum -> {canon}")
        else:
            problems.append(f"{_SCHEMA.name}: epilogue enum {list(enum)} != EPILOGUE_STAGES {canon} "
                            f"(regenerate with --write)")

    # 2/3. the prose ABI documents
    import yaml
    for path, full_key in _ABI_DOCS:
        lines = abi_epilogue_lines(yaml.safe_load(path.read_text(encoding="utf-8")))
        if not lines:
            problems.append(f"{path.name}: no '{_MARKER}…]' epilogue line found — the document shape "
                            f"changed; this gate cannot certify it and does NOT report success")
            continue
        full = [names for keys, names in lines if full_key in keys]
        if not full:
            problems.append(f"{path.name}: no epilogue line under {full_key!r} — cannot certify")
        for names in full:
            if set(names) != set(canon):
                problems.append(f"{path.name}: {full_key} epilogue documents {names}, "
                                f"EPILOGUE_STAGES is {canon}")
        for keys, names in lines:
            extra = sorted(set(names) - set(canon))
            if extra:
                problems.append(f"{path.name}: {'/'.join(keys)} documents stage(s) {extra} that are not "
                                f"in EPILOGUE_STAGES — a per-opcode list may only be a SUBSET")

    # 4/5. the in-tree dialect verifiers
    from merlin.xdsl_dialects import interface as IFACE
    from merlin.xdsl_dialects.targets import factory as FACTORY
    for mod in (IFACE, FACTORY):
        got = set(getattr(mod, "KNOWN_EPILOGUE"))
        if got != set(canon):
            problems.append(f"{mod.__name__}.KNOWN_EPILOGUE {sorted(got)} != EPILOGUE_STAGES {canon}")

    # 6. the emitted target-dialect template
    from merlin.targetgen.generate import xdsl as GEN
    arts = GEN.generate({"dialect_name": "t", "ops": [{"name": o} for o in sorted(GEN.KNOWN_OPS)],
                         "types": [{"name": t} for t in sorted(GEN.KNOWN_TYPES)]})
    src = next((a.content for a in arts if a.relpath.endswith("_dialect.py")), "")
    marker = "_KNOWN_EPILOGUE = "
    if marker not in src:
        problems.append("generate.xdsl: the emitted dialect no longer defines _KNOWN_EPILOGUE — "
                        "this gate cannot certify it and does NOT report success")
    else:
        rendered = src.split(marker, 1)[1].split("\n", 1)[0]
        if set(eval(rendered, {"__builtins__": {}}, {})) != set(canon):  # noqa: S307 - our own literal
            problems.append(f"generate.xdsl: emitted _KNOWN_EPILOGUE {rendered} != {canon}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="regenerate the JSON validator's enum from EPILOGUE_STAGES")
    ap.add_argument("--staged", action="store_true", help="accepted for pre-commit symmetry (no-op)")
    a = ap.parse_args(argv)
    problems = check(write=a.write)
    if problems:
        print("epilogue vocabulary drift:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"epilogue vocabulary OK: {list(EPILOGUE_STAGES)} (6 representations agree)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

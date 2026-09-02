#!/usr/bin/env python3
"""Gate: does every interface op the parser accepts have ABI semantics the agent can implement?

`merlin/contract/command_buffer_abi.yaml` is handed to the graded agent as the command-buffer contract
(see `targetgen.generate_prompt`). The set of ops a capsule may legally contain is decided elsewhere --
by `interface_emit._NAMED_OP_OPERAND_KEYS`, which `boundary.grammar_mnemonics` calls the parser's own
tables and treats as authoritative. Nothing kept the two in step.

So an op could be added to the parser, emitted into capsules, and left with no operands, no attributes
and no semantics in the contract the agent reads. The agent's only remaining option is to guess the
ABI, and this repo has measured what that produces: an arm invented an instruction encoding because
the prompt never named the shipped ISA files.

Found on introduction: five ops in that state (`attention_qk`, `matmul_batched`, `rmsnorm`, `rope`,
`softmax`), plus a sixth caught in the act -- `bias_add` was added to the parser and documented in
`interface_grammar.md` in one change and to the ABI only after this gate was written.

Reporting-only by default, with a ratchet, for the reason the sibling gates are: the five predate the
check, and turning inherited debt into a hard failure on day one only teaches everyone to skip hooks.
`--fail-on-undocumented` is the CI form.

  --json                 machine-readable
  --ratchet PATH         pre-existing debt that MAY ONLY SHRINK
  --fail-on-undocumented exit non-zero on any undocumented op outside the ratchet
  --fail-on-unverifiable exit non-zero when the comparison COULD NOT RUN (unreadable ABI or parser
                         tables). A check that could not run has established nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
for _p in (_REPO / "merlin" / "python",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_DEFAULT_RATCHET = _HERE.parent / "undocumented_opcodes_ratchet.txt"


def _ratchet(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line.split()[0])
    return out


def audit() -> dict:
    """``{documented, undocumented, structural, status}`` -- mnemonic -> opcode, both directions."""
    import yaml

    from merlin.common.paths import merlin_dir
    from merlin.targetgen.contract import interface_emit as IE

    abi_path = merlin_dir() / "contract" / "command_buffer_abi.yaml"
    if not abi_path.is_file():
        return {"status": "unverifiable", "detail": f"no ABI contract at {abi_path}"}
    try:
        abi = yaml.safe_load(abi_path.read_text(encoding="utf-8")) or {}
        opcodes = set(abi.get("opcodes") or ())
    except yaml.YAMLError as exc:
        return {"status": "unverifiable", "detail": f"ABI contract unparseable: {exc}"}
    if not opcodes:
        return {"status": "unverifiable", "detail": "the ABI contract declares no opcodes"}

    # Both tables, because the two kinds of op reach an opcode differently: the residency-decomposed
    # core ops via `_OP_TO_OPCODE`, the whole-op classes via `_NAMED_OP_TO_OPCODE` (which applies the
    # spelling overrides). Reading only one would report the other kind as structural.
    mapping = {**IE._OP_TO_OPCODE, **IE._NAMED_OP_TO_OPCODE}
    documented, undocumented, structural = {}, {}, []
    for mnem in sorted(IE.defined_mnemonics()):
        opcode = mapping.get(mnem)
        if opcode is None:
            structural.append(mnem)          # `tensor` declares a leaf; it issues no command
            continue
        (documented if opcode in opcodes else undocumented)[mnem] = opcode
    # The mirror direction: an ABI opcode no mnemonic reaches is documentation for an op no capsule can
    # contain. Reported, not failed -- a target's own codegen may still emit it.
    reachable = set(mapping.values())
    return {
        "status": "ok",
        "documented": documented,
        "undocumented": undocumented,
        "structural": structural,
        "abi_opcodes_no_mnemonic_reaches": sorted(opcodes - reachable),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ratchet", type=Path, default=_DEFAULT_RATCHET)
    ap.add_argument("--fail-on-undocumented", action="store_true")
    ap.add_argument("--fail-on-unverifiable", action="store_true")
    a = ap.parse_args(argv)

    rep = audit()
    if rep["status"] != "ok":
        print(f"[FAIL] opcode-documentation: {rep['detail']}", file=sys.stderr)
        return 2 if a.fail_on_unverifiable else 0

    allowed = _ratchet(a.ratchet)
    new = {m: oc for m, oc in rep["undocumented"].items() if m not in allowed}
    stale = sorted(allowed - set(rep["undocumented"]))
    rep["ratcheted"] = sorted(set(rep["undocumented"]) & allowed)
    rep["new_undocumented"] = new
    rep["stale_ratchet_entries"] = stale

    if a.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"[opcode-doc] {len(rep['documented'])} documented, "
              f"{len(rep['undocumented'])} undocumented "
              f"({len(rep['ratcheted'])} ratcheted), {len(rep['structural'])} structural")
        for m, oc in sorted(new.items()):
            print(f"  [NEW] {m} -> {oc}: the parser accepts it and the ABI states no semantics; the "
                  f"agent would have to guess the operands")
        for m in rep["ratcheted"]:
            print(f"  [debt] {m} -> {rep['undocumented'][m]}")
        if stale:
            print(f"  [ratchet] {len(stale)} entry/entries now documented — delete them: "
                  f"{', '.join(stale)}")
        for oc in rep["abi_opcodes_no_mnemonic_reaches"]:
            print(f"  [note] ABI documents {oc}, which no interface mnemonic reaches")

    if a.fail_on_undocumented and new:
        print(f"[FAIL] {len(new)} interface op(s) the parser accepts have no ABI semantics",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

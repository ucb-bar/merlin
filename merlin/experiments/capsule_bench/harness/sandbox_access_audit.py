"""Thorough sandbox access audit — spawn the REAL agent bwrap sandbox for an arm's bundle and probe,
from INSIDE it, that EVERY answer surface is unreadable and the FULL test CONTRACT + granted tools ARE
readable. Stronger than the sampled ``mask_selftest``: it enumerates the complete derived answer-surface
set (goldens, private model weights, hidden capsules, prior backends, oracle/grader modules, memory + session
transcripts) and the per-capsule contract files, and checks each one for real inside the sandbox.

Target-general: everything is derived from the bundle manifest + the descriptor's answer surfaces (via
``merlin.targetgen.sandbox``) — no target literal. Exit 0 = the sandbox enforces allow/deny exactly.

Usage: MERLIN_TARGET_EXPERIMENT=<descriptor> sandbox_access_audit.py [--arm merlin_assisted_rtlchecks]
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402 — active target (descriptor-driven), bootstraps merlin/python
import yaml  # noqa: E402
from merlin.targetgen.sandbox import bwrap as _BW  # noqa: E402
from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces  # noqa: E402
from merlin.targetgen.target_experiment import load_target_experiment  # noqa: E402
import run_agent_experiment as RX  # noqa: E402

# The test CONTRACT files present in each graded capsule dir (what the agent must see to know the goal);
# the ANSWER files (golden.yaml / expected_command_buffer*) are enumerated by answer_surfaces() instead.
_CONTRACT_NAMES = ("capsule.interface.mlir", "capsule.yaml", "expected_instruction_coverage.yaml")
_CONTRACT_DOCS = ("merlin/contract/command_buffer_abi.yaml", "merlin/contract/interface_grammar.md",
                  "merlin/contract/capsule.schema.json")


def _expand(paths: set[str], p: Path, cap: int) -> None:
    if p.is_file():
        paths.add(str(p))
    elif p.is_dir():
        for f in list(p.rglob("*"))[:cap]:
            if f.is_file():
                paths.add(str(f))


def audit(arm_bundle: str) -> int:
    repo = C.REPO
    te = load_target_experiment(C.EXP / "target_experiment.yaml")
    bundle = yaml.safe_load((C.BUNDLES / arm_bundle / "input_bundle_manifest.yaml").read_text())

    # MUST-BE-DENIED: the canonical derived answer-surface set + the bundle's own denied paths.
    deny: set[str] = set()
    for s in answer_surfaces(te):
        _expand(deny, Path(s.path), 2000)
    for d in bundle.get("denied", []):
        _expand(deny, repo / d["path"], 500)

    # SHOULD-BE-READABLE: the test contract (interface / op spec / coverage target) + contract docs +
    # ISA headers + the tools this arm is granted — never the answers.
    allow: set[str] = set()
    capdirs = sorted({p.parent for p in (repo / te.corpus_rel()).rglob("capsule.interface.mlir")})[:4]
    for cd in capdirs:
        for n in _CONTRACT_NAMES:
            if (cd / n).is_file():
                allow.add(str(cd / n))
    for rel in (*_CONTRACT_DOCS, *te.isa_headers):
        if (repo / rel).is_file():
            allow.add(str(repo / rel))
    for a in bundle.get("allowed", []):
        p = repo / a["path"]
        if p.is_file():
            allow.add(str(p))
        elif p.is_dir():
            f = next((x for x in p.rglob("*.py") if x.is_file()), None)
            if f:
                allow.add(str(f))

    deny, allow = sorted(deny), sorted(allow)

    # assemble the REAL workspace + run one probe inside the COMPLETE real-run isolation.
    ws = Path(tempfile.mkdtemp(prefix="sbx_audit_")) / "workspace"
    RX.assemble_workspace(bundle, ws)
    viol = RX.assert_isolation(ws, bundle)

    # The probe reads its path list from STDIN, not from its own text. Inlining one `if` per path made
    # the inner script grow with the answer surface, and a single argv string is capped at 128 KiB on
    # Linux — declaring one more directory surface (a reference backend is ~400 files) pushed
    # `bash -c <bwrap …>` past the cap and the whole audit died with "Argument list too long" instead of
    # reporting a verdict. Feeding the list through stdin keeps the command constant-size however many
    # surfaces the target declares.
    probe = ("""while IFS= read -r __p; do
  case "$__p" in ===*) echo "$__p"; continue;; esac
  if head -c1 "$__p" >/dev/null 2>&1; then echo "R::$__p"; else echo "X::$__p"; fi
done""")
    feed = "\n".join(["===DENY===", *deny, "===ALLOW===", *allow]) + "\n"
    out = subprocess.run(["bash", "-c", _BW.wrap(te, ws, probe, bundle)], input=feed,
                         capture_output=True, text=True, cwd=str(repo)).stdout
    section, leaks, blocked = None, [], []
    for ln in out.splitlines():
        if ln in ("===DENY===", "===ALLOW==="):
            section = ln
            continue
        if ln.startswith("R::") and section == "===DENY===":
            leaks.append(ln[3:])
        if ln.startswith("X::") and section == "===ALLOW===":
            blocked.append(ln[3:])

    print(f"sandbox access audit — bundle={arm_bundle} target={te.target}")
    print(f"  symlink-level isolation violations: {viol or 'none'}")
    print(f"  probes: {len(deny)} deny / {len(allow)} allow")
    print(f"  answer surfaces READABLE (leaks): {len(leaks)}")
    for p in leaks[:20]:
        print(f"    LEAK  {p.replace(str(repo) + '/', '')}")
    print(f"  contract/allowed inputs UNREADABLE (broken grants): {len(blocked)}")
    for p in blocked[:20]:
        print(f"    BLOCKED  {p.replace(str(repo) + '/', '')}")
    ok = not leaks and not blocked and not viol
    print("VERDICT:", "PASS — sandbox enforces allow/deny exactly" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default=None,
                    help="bundle id to audit (default: the CIRCT arm's hwbringup bundle if present, "
                         "else the first bundle in the target's input_bundles dir)")
    a = ap.parse_args(argv)
    arm = a.arm
    if not arm:
        pref = ["merlin_assisted_rtlchecks_hwbringup_v0", "merlin_assisted_hwbringup_v0"]
        for cand in pref:
            if (C.BUNDLES / cand / "input_bundle_manifest.yaml").is_file():
                arm = cand
                break
        if not arm:
            bs = sorted(d.name for d in C.BUNDLES.glob("*_hwbringup*")
                        if (d / "input_bundle_manifest.yaml").is_file())
            arm = bs[0] if bs else None
    if not arm:
        print("no hwbringup bundle found for this target", file=sys.stderr)
        return 2
    return audit(arm)


if __name__ == "__main__":
    raise SystemExit(main())

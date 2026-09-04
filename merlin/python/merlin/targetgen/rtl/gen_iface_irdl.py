"""Single-source IRDL bridge for the `merlin_iface` contract dialect.

The C++ OOT backend and the Merlin/xDSL infra must agree on ONE dialect spec. Rather than hand-write
(and hand-sync) a C++ ODS dialect, this emits the canonical **IRDL** description of `merlin_iface` —
which `mlir-opt --irdl-file=...` registers DYNAMICALLY. The same IRDL is the portable spec the
xDSL side already mirrors (the contract grammar in `merlin/contract/interface_grammar.md`).

WHAT THIS DOES AND DOES NOT BUY (measured 2026-09-04, LLVM-23, all 269 tracked capsules):
  * A dynamically-registered dialect has NO custom parser, so mlir-opt can read only the GENERIC
    form (`"merlin_iface.matmul"(%a, %b) : (...) -> ...`). Capsules are written in the pretty form,
    so `--irdl-file` parses 0/269 of them -- it FAILS with rc=1 and an EMPTY stderr, which is why
    this went unnoticed. `verify()` reports that number; do not read a green run as coverage.
  * Of the constraints tblgen-to-irdl emits, only `irdl.base` (type identity) is enforced by the
    IRDL interpreter. `irdl.c_pred` carries a C++ predicate STRING the interpreter cannot evaluate,
    so shape/attribute/element-type constraints are inert: a `commit` returning `i32` instead of a
    tensor, and `epilogue = 42 : i64` against a string-array constraint, both verify clean.
So this file is a registration + type-identity spec. Structural checking remains the grader's job.

Pipeline (stock LLVM-23 tools, no custom C++):
  ODS (.td, the contract's reference dialect)  --tblgen-to-irdl-->  raw IRDL
                                               --normalize-------->  merlin_iface.irdl.mlir
  C++ OOT tool:  mlir-opt --irdl-file=merlin_iface.irdl.mlir  <capsule.interface.mlir>   (dynamic register)

Normalization — three upstream `tblgen-to-irdl` quirks, each of which silently breaks registration:
  1. `irdl.base "!builtin.string"` (from `StrAttr`) is not registrable; relaxed to `irdl.any`.
  2. Type symbols are spelled WITH their `!` sigil (`irdl.type @"!acc"`), which names the type
     `!acc` -- the parser then looks for `!merlin_iface.!acc` and rejects every valid module.
     The sigil belongs to the printer, not the symbol, so it is stripped.
  3. `let parameters` is dropped entirely, so `!merlin_iface.acc<bf16>` (what the corpus writes)
     fails with "expected 0 type arguments, but had 1". The parameters are read back from the ODS
     via llvm-tblgen's JSON record dump and re-attached -- whatever the ODS declares, nothing
     about this dialect assumed.

Usage: python -m merlin.targetgen.rtl.gen_iface_irdl [--out <irdl.mlir>] [--verify]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from merlin.common.paths import artifacts_dir, repo_root

_REPO = repo_root()
_LLVM = _REPO / "third_party" / "llvm-install"
_T2I = _LLVM / "bin" / "tblgen-to-irdl"
_MLIROPT = _LLVM / "bin" / "mlir-opt"
# merlin_iface is the SHARED contract dialect (not target-specific) — its pinned spec lives next to
# interface_grammar.md in merlin/contract/, NOT under a per-target dir.
_DEFAULT_OUT = _REPO / "merlin/contract/merlin_iface.irdl.mlir"
_CAPSULES = _REPO / "merlin/contract" / "capsules"


# tblgen-to-irdl renders every `StrAttr` operand constraint as the literal line
#     %2 = irdl.base "!builtin.string"
# and mlir-opt's IRDL runtime refuses to register that base, so the whole dialect fails to
# load. Relaxing it to `irdl.any` keeps REGISTRATION working (attribute-VALUE checking is
# the grader's job, unchanged). Only spaces/tabs after the literal are absorbed, never the
# newline — swallowing it would join two SSA defs onto one line and corrupt the IRDL.
_STR_BASE = 'irdl.base "!builtin.string"'


def _relax_string_base(raw: str) -> str:
    head, *rest = raw.split(_STR_BASE)
    return "irdl.any".join([head] + [chunk.lstrip(" \t") for chunk in rest])


def _strip_type_sigil(raw: str) -> str:
    """Drop the `!` tblgen-to-irdl bakes into type SYMBOLS (`irdl.type @"!acc"`).

    The symbol is the bare mnemonic; the leading `!` is added by the printer when the type is
    spelled. Left in, the dialect declares a type literally named `!acc`, so a module writing the
    correct `!merlin_iface.acc` is rejected with "expected dynamic type" -- i.e. the IRDL rejects
    every valid module, including the ones it was generated from."""
    return raw.replace('@"!', '@"')


def _ods_type_params(td: Path, includes: list[Path]) -> dict[str, list[str]]:
    """Map each ODS TypeDef mnemonic to its declared parameter names.

    Read from llvm-tblgen's JSON record dump (a real parser over the same .td tblgen-to-irdl
    consumed) rather than assumed, so this reflects whatever the contract dialect declares.
    IRDL rejects uppercase parameter names, so ODS camelCase is folded down; the name is
    positional-only in IRDL, so the fold is cosmetic."""
    args = [str(_LLVM / "bin" / "llvm-tblgen"), "--dump-json", str(td)]
    for inc in includes:
        args += ["-I", str(inc)]
    rec = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    out: dict[str, list[str]] = {}
    for name in rec.get("!instanceof", {}).get("TypeDef", []):
        entry = rec[name]
        params = (entry.get("parameters") or {}).get("args", [])
        names = [pair[1].lower() for pair in params if isinstance(pair, list) and len(pair) == 2]
        out[entry["mnemonic"]] = names
    return out


def _restore_type_params(raw: str, params: dict[str, list[str]]) -> str:
    """Re-attach the `irdl.parameters` tblgen-to-irdl drops.

    Without them a parameterised type is declared arity-0, so the corpus's `!merlin_iface.acc<bf16>`
    fails with "expected 0 type arguments, but had 1". Each parameter is `irdl.any`: the ODS type
    is `mlir::Type`, and the IRDL interpreter cannot evaluate a narrower `c_pred` anyway."""
    lines = []
    for line in raw.splitlines(keepends=True):
        body = line.strip()
        if body.startswith("irdl.type @"):
            mnemonic = body.split('@"', 1)[1].split('"', 1)[0] if '@"' in body else body.split("@", 1)[1].strip()
            names = params.get(mnemonic) or []
            if names:
                indent = line[: len(line) - len(line.lstrip())]
                inner = indent + "  "
                decls = "".join(f"{inner}%param{i} = irdl.any\n" for i, _ in enumerate(names))
                sig = ", ".join(f"{n}: %param{i}" for i, n in enumerate(names))
                lines.append(f'{indent}irdl.type @"{mnemonic}" {{\n{decls}{inner}irdl.parameters({sig})\n{indent}}}\n')
                continue
        lines.append(line)
    return "".join(lines)


def _iface_spec_digest(iface_dir: Path) -> str:
    """Digest of the whole ``MerlinIface`` ODS include dir -- the bytes that define the dialect."""
    h = hashlib.sha256()
    for f in sorted(iface_dir.rglob("*.td")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _discover_ref_ods_inc() -> Path:
    """Locate the reference ``merlin_iface`` ODS include dir, FAILING CLOSED on disagreement.

    ``merlin_iface`` is the SHARED contract dialect, so every OOT package under the generated targets
    tree must ship the SAME spec -- that premise is what makes "any package serves as the reference"
    sound. It is therefore checked rather than assumed: packages are grouped by the digest of their
    ODS, and a divergence raises instead of resolving by sort order.

    This is not hypothetical. Taking the first sorted match silently selected a 5-op ``agent_spec_v0``
    over the 7-op ``agent_spec_v1`` the tracked IRDL was generated from, which would have dropped
    ``conv2d`` and ``movement`` from the contract with no diagnostic. Pass ``--ref-ods`` to choose
    deliberately."""
    base = artifacts_dir() / "targets"
    cands = sorted(base.rglob("MerlinIface/MerlinIfaceOps.td"))
    if not cands:
        raise FileNotFoundError(
            f"no MerlinIface/MerlinIfaceOps.td found under {base}; build an OOT package first "
            f"(merlin-targetgen) or pass --ref-ods <mlir_oot/include>.")
    by_digest: dict[str, list[Path]] = {}
    for td in cands:
        by_digest.setdefault(_iface_spec_digest(td.parent), []).append(td)
    if len(by_digest) > 1:
        detail = "\n".join(
            f"  {d[:12]}: {len(tds)} package(s), e.g. {tds[0].relative_to(base)}"
            for d, tds in sorted(by_digest.items()))
        raise RuntimeError(
            "the merlin_iface contract dialect DIVERGES across OOT packages, so no single one is "
            "the reference; regenerating from an arbitrary pick would silently change the contract."
            f"\n{detail}\nRe-run naming the intended spec: --ref-ods <mlir_oot/include>")
    return cands[0].parent.parent


def generate(out: Path, ref_ods_inc: Path | None = None) -> Path:
    """tblgen-to-irdl on the reference ODS + normalize -> canonical merlin_iface.irdl.mlir."""
    ref_ods_inc = ref_ods_inc or _discover_ref_ods_inc()
    _REF_ODS_INC = ref_ods_inc
    td = _REF_ODS_INC / "MerlinIface" / "MerlinIfaceOps.td"
    raw = subprocess.run(
        [str(_T2I), str(td), "-I", str(_REF_ODS_INC), "-I", str(_LLVM / "include"),
         "--gen-dialect-irdl-defs", "--dialect=merlin_iface"],
        capture_output=True, text=True, check=True).stdout
    norm = _restore_type_params(
        _strip_type_sigil(_relax_string_base(raw)),
        _ods_type_params(td, [_REF_ODS_INC, _LLVM / "include"]),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(norm)
    return out


def verify(irdl: Path) -> tuple[int, int, list[str]]:
    """Round-trip every tracked capsule through the IRDL-registered dialect.

    NOT a coverage gate, and must not be read as one. A dynamically registered dialect has no custom
    parser, so mlir-opt can only read the GENERIC op form; the corpus is written in the pretty form
    and therefore does not parse here AT ALL (measured 0/269). That failure is silent by nature --
    rc=1 with an empty stderr -- so it is classified explicitly below rather than reported as a
    capsule defect, which is how it would otherwise read.

    The tool's real, exercised guarantee is pinned by ``merlin/tests/ir/test_iface_irdl_contract.py``
    instead: the dialect registers, and its type constraints reject non-conformant modules.
    """
    # Scope the denominator: `capsule.interface.mlir` is also the filename for MODEL capsules, which
    # are linalg-on-tensors and contain no merlin_iface op at all. Counting those as failures blamed
    # the corpus for a dialect this contract does not describe (measured: 139 of 509).
    every = sorted(_CAPSULES.rglob("capsule.interface.mlir"))
    caps = [c for c in every if "merlin_iface." in c.read_text()]
    not_applicable = len(every) - len(caps)
    ok = 0
    fails = []
    for c in caps:
        # No --verify-diagnostics: it would mask a parse failure. rc==0 AND merlin_iface ops printed.
        r = subprocess.run([str(_MLIROPT), f"--irdl-file={irdl}", str(c)],
                           capture_output=True, text=True)
        if r.returncode == 0 and "merlin_iface." in r.stdout:
            ok += 1
            continue
        msg = (r.stderr or "").strip().splitlines()
        if not msg:
            why = "custom/pretty assembly: an IRDL-registered dialect parses generic form only"
        else:
            why = msg[0][:90]
        fails.append(f"{c.parent.name}: {why}")
    return ok, len(caps), fails, not_applicable


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_DEFAULT_OUT))
    ap.add_argument("--ref-ods", default=None,
                    help="reference merlin_iface ODS include dir (mlir_oot/include); "
                         "auto-discovered from any OOT package under out/artifacts/targets if omitted")
    ap.add_argument("--verify", action="store_true", help="parse every capsule via the generated IRDL")
    a = ap.parse_args(argv)
    out = generate(Path(a.out), Path(a.ref_ods) if a.ref_ods else None)
    nops = out.read_text().count("irdl.operation")
    print(f"wrote {out} ({out.read_text().count(chr(10))} lines, {nops} ops)")
    if a.verify:
        ok, n, fails, n_a = verify(out)
        print(f"IRDL parse check: {ok}/{n} merlin_iface capsules parsed+verified via "
              f"mlir-opt --irdl-file ({n_a} non-merlin_iface capsules out of scope)")
        syntax_gap = sum(1 for f in fails if "generic form only" in f)
        if syntax_gap:
            print(f"  {syntax_gap}/{n} unreadable by this route (custom assembly, not a capsule defect);"
                  f" the enforced guarantee is merlin/tests/ir/test_iface_irdl_contract.py")
        for f in fails[:10]:
            print(f"  FAIL {f}")
        return 0 if ok == n and n > 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

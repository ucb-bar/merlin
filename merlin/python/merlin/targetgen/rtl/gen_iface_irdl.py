"""Single-source IRDL bridge for the `merlin_iface` contract dialect.

The C++ OOT backend and the Merlin/xDSL infra must agree on ONE dialect spec. Rather than hand-write
(and hand-sync) a C++ ODS dialect, this emits the canonical **IRDL** description of `merlin_iface` —
which `mlir-opt --irdl-file=...` registers DYNAMICALLY, so a C++ OOT tool parses the frozen
`*.interface.mlir` grammar with zero hand-written dialect code. The same IRDL is the portable spec the
xDSL side already mirrors (the contract grammar in `merlin/contract/interface_grammar.md`).

Pipeline (stock LLVM-23 tools, no custom C++):
  ODS (.td, the contract's reference dialect)  --tblgen-to-irdl-->  raw IRDL
                                               --normalize-------->  merlin_iface.irdl.mlir
  C++ OOT tool:  mlir-opt --irdl-file=merlin_iface.irdl.mlir  <capsule.interface.mlir>   (dynamic register)

Normalization: tblgen-to-irdl emits `irdl.base "!builtin.string"` for `StrAttr`, which mlir-opt's IRDL
runtime won't register; we relax those to `irdl.any` (IRDL here does dialect REGISTRATION so the parser
accepts the grammar — attribute-VALUE checking remains the grader's job, unchanged).

Usage: python -m merlin.targetgen.rtl.gen_iface_irdl [--out <irdl.mlir>] [--verify]
"""
from __future__ import annotations
import argparse
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


def _discover_ref_ods_inc() -> Path:
    """Locate the reference ``merlin_iface`` ODS include dir. Because ``merlin_iface`` is the SHARED
    contract dialect, ANY target's OOT package that ships ``MerlinIface/MerlinIfaceOps.td`` serves as
    the reference to DERIVE the IRDL from — we take the first match under the generated targets tree
    (deterministic, sorted). Override with ``--ref-ods``."""
    base = artifacts_dir() / "targets"
    for td in sorted(base.rglob("MerlinIface/MerlinIfaceOps.td")):
        return td.parent.parent  # .../mlir_oot/include
    raise FileNotFoundError(
        f"no MerlinIface/MerlinIfaceOps.td found under {base}; build an OOT package first "
        f"(merlin-targetgen) or pass --ref-ods <mlir_oot/include>.")


def generate(out: Path, ref_ods_inc: Path | None = None) -> Path:
    """tblgen-to-irdl on the reference ODS + normalize -> canonical merlin_iface.irdl.mlir."""
    ref_ods_inc = ref_ods_inc or _discover_ref_ods_inc()
    _REF_ODS_INC = ref_ods_inc
    td = _REF_ODS_INC / "MerlinIface" / "MerlinIfaceOps.td"
    raw = subprocess.run(
        [str(_T2I), str(td), "-I", str(_REF_ODS_INC), "-I", str(_LLVM / "include"),
         "--gen-dialect-irdl-defs", "--dialect=merlin_iface"],
        capture_output=True, text=True, check=True).stdout
    norm = _relax_string_base(raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(norm)
    return out


def verify(irdl: Path) -> tuple[int, int, list[str]]:
    """mlir-opt --irdl-file=<irdl> must dynamically register + parse+verify EVERY public capsule."""
    caps = sorted(_CAPSULES.rglob("capsule.interface.mlir"))
    ok = 0
    fails = []
    for c in caps:
        # HONEST test: round-trip the capsule through the IRDL-registered dialect (NO --verify-diagnostics,
        # which would mask a parse failure). rc==0 AND the printed module must contain merlin_iface ops.
        r = subprocess.run([str(_MLIROPT), f"--irdl-file={irdl}", str(c)],
                           capture_output=True, text=True)
        if r.returncode == 0 and "merlin_iface." in r.stdout:
            ok += 1
        else:
            msg = (r.stderr or "").strip().splitlines()
            fails.append(f"{c.parent.name}: {msg[0][:90] if msg else 'rc=%d / no merlin_iface in output' % r.returncode}")
    return ok, len(caps), fails


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
        ok, n, fails = verify(out)
        print(f"IRDL parse check: {ok}/{n} capsules parsed+verified via mlir-opt --irdl-file")
        for f in fails[:10]:
            print(f"  FAIL {f}")
        return 0 if ok == n and n > 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

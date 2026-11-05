#!/usr/bin/env python3
"""Run Merlin's OPU certification corpus on spike, executed by this package's OPU model.

    python spike/certify.py [--config KodiakOPUConfig] [--out <report dir>]

The corpus, the image, the in-image reference and the host-side digests are Merlin's
(``merlin.kernels.opu_cert`` / ``opu_corpus``), exactly as on the RTL; only the substrate differs. The model
is resolved the way every other Merlin runner resolves it -- ``merlin.targetgen.spike_extension`` reading
this package's ``runner.spike_extension`` (``$MERLIN_KODIAK_OPU_SPIKE_EXTLIB``, sha256-verified) -- so a
missing or rebuilt-but-undeclared .so fails here instead of certifying something else.

A pass here says the KERNEL is right on the MODEL; it is not an RTL certification, and the report says
which substrate ran.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load_shim():
    spec = importlib.util.spec_from_file_location("kodiak_opu_shim", ROOT / "opu_shim.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    from merlin.common import provenance as PROV
    from merlin.common.paths import artifacts_dir
    from merlin.kernels import opu_cert, opu_corpus
    from merlin.runtime.backends import spike as spike_backend
    from merlin.targetgen import spike_extension as SE
    from merlin.targetgen.providers import read_provider

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="elaborated config (default: the contract's rtl_source.config)")
    ap.add_argument("--out", type=Path, help="report directory (default: out/artifacts/verification/<target>/...)")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args(argv)

    target = read_provider(ROOT).target
    model = yaml.safe_load((HERE / "model.yaml").read_text(encoding="utf-8"))
    contract_doc = yaml.safe_load((ROOT / "contracts" / "target_contract.yaml").read_text(encoding="utf-8"))
    config = args.config or (contract_doc.get("rtl_source") or {}).get("config")
    shim = _load_shim()
    contract = shim.load_contract(str(model["unit"]), path=ROOT / "contracts" / "matrix_units.yaml")
    derived = shim.derive_encodings(contract)
    tile, align = contract.geometry(config)
    cfg_text = "\n".join(p.read_text(encoding="utf-8") for p in contract.config_scala())
    mixin_text = "\n".join(p.read_text(encoding="utf-8") for p in contract.mixin_scala())
    from merlin.targetgen.rtl import opu_isa

    vlen = opu_isa.vector_unit_params(cfg_text, config, mixin_text=mixin_text)["vLen"]

    # The declared model, digest-verified, or nothing. The selected contract must be THIS package's.
    if SE.declared_extension(target) is None:
        print(f"error: the selected {target} contract declares no runner.spike_extension", file=sys.stderr)
        return 1
    ext = SE.resolve(target, default_library_dir=spike_backend.spike_path().parent, default_extension_name="")

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = args.out or artifacts_dir() / "verification" / target / "opu_corpus_spike_model" / stamp
    out.mkdir(parents=True, exist_ok=True)
    console_path = out / "console.txt"

    def run(elf: Path) -> str:
        cmd = [str(spike_backend.spike_path()), f"--isa=rv64gcv_zvl{vlen}b", *ext.spike_flags(), str(elf)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        console_path.write_text(p.stdout + p.stderr, encoding="utf-8")
        return p.stdout + p.stderr

    verification = PROV.verify(contract.pin, checkout=contract.checkout(), reads=list(contract.sources.values()))
    provenance = PROV.record(
        pins={contract.pin: verification},
        sources=[contract.source(k) for k in sorted(contract.sources)],
        artifacts={"spike": spike_backend.spike_path(), "spike_extension": ext.extlib},
        extra={"substrate": "spike + functional model (not RTL)", "extension": ext.describe()},
    )
    runnable, deferred = opu_corpus.select(tile)
    t0 = time.time()
    report = opu_cert.certify(
        runnable, derived.encodings, contract.spec(), out / "build", config=config, tile_edge=tile, run=run,
        deferred=deferred, derivation_ok=derived.ok, operand_align=align, provenance=provenance,
    )
    wall = time.time() - t0
    doc = report.to_dict()
    doc["substrate"] = {"simulator": "spike", "extension": ext.describe(), "sha256": ext.sha256, "wall_s": round(wall, 1)}
    (out / "report.json").write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    passed = sum(1 for r in report.results if r.in_image_agrees and r.digest_agrees)
    print(f"{passed}/{len(report.results)} cases pass on spike+{ext.extension_name} "
          f"(deferred {len(report.deferred)}, tile {tile}, {wall:.0f}s); gaps: {list(report.gaps)}")
    for r in report.results:
        if not (r.in_image_agrees and r.digest_agrees):
            print(f"  FAIL {r.name} m={r.m} n={r.n} k={r.k} mismatches={r.mismatches} first_bad={r.first_bad} {r.note}")
    print(f"report: {out / 'report.json'}")
    return 0 if passed == len(report.results) and not report.gaps else 1


if __name__ == "__main__":
    raise SystemExit(main())

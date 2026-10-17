"""``python -m merlin_experiments.timing_probes``: the post-Phase-1, pre-Phase-2 calibration stage.

  identity  print the hardware identity a label is cached under (FIRRTL, emulator, toolchain)
  select    derive the probe set from the timing space, existing labels and the Phase 2 domain
  run       select (or take --selection), build and measure on gSIM, write the label archive, recheck
  check     re-run selection over labels including probe archives; exit 1 while any gap remains

Inputs are explicit or the target's selected ones: ``--facts`` (else the selected RTL facts),
``--contract`` (else the selected capability contract), ``--profile`` (the timing provider's
profile; required) and ``--space`` (else ``examples/<target>/verification/timing_probes/``).
Outputs are operator-private (directories 0700). No author, grade or corpus is involved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import backends, coverage, domain, runner, selection, space


def _contract(target: str, explicit: str | None) -> dict:
    if explicit:
        text = Path(explicit).read_text()
        import yaml

        return json.loads(text) if explicit.endswith(".json") else yaml.safe_load(text)
    from merlin.targetgen import target_registry

    return target_registry.resolve(target).load_contract()


def _facts(target: str, explicit: str | None) -> dict:
    from merlin.targetgen.rtl.facts import rtl_facts_path

    return json.loads(Path(rtl_facts_path(target, explicit=explicit)).read_text())


def _space_path(target: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    from merlin.common.paths import repo_root

    return repo_root() / "examples" / target / "verification" / "timing_probes" / "timing_space.yaml"


def _budget(values: list[str]) -> dict[str, int]:
    out = {}
    for value in values or ():
        name, _, number = value.partition("=")
        if name not in domain.QUANTITIES or not number.isdigit():
            raise SystemExit(f"--budget expects one of {domain.QUANTITIES}=<integer>, got {value!r}")
        out[name] = int(number)
    return out


def _load(args):
    facts = _facts(args.target_hardware, args.facts)
    profile = json.loads(Path(args.profile).read_text())
    contract = _contract(args.target_hardware, args.contract)
    timing_space = space.load_space(
        _space_path(args.target_hardware, args.space), facts=facts, profile=profile, target=args.target_hardware
    )
    backend = backends.load_backend(timing_space, facts=facts, contract=contract)
    return timing_space, backend, contract


def _select(args, timing_space, backend, extra_labels=()):
    labels, census = [], []
    for archive in [*(args.labels or ()), *extra_labels]:
        got, counted = coverage.read_archive(archive, timing_space)
        labels += got
        census.append(counted)
    document = selection.select(timing_space, labels, domain.load(args.domain), backend, budget=_budget(args.budget))
    document["label_archives"] = census
    document["coverage"] = coverage.coverage_summary(timing_space, labels)
    return document


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True, default=str) + "\n")
    path.chmod(0o600)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="python -m merlin_experiments.timing_probes",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("identity", "select", "run", "check"):
        command = commands.add_parser(name)
        command.add_argument("--target-hardware", required=True, help="the target whose selected design is probed")
        if name == "identity":
            continue
        command.add_argument("--profile", required=True, help="the timing provider's profile (JSON)")
        command.add_argument("--facts")
        command.add_argument("--contract")
        command.add_argument("--space")
        command.add_argument("--labels", action="append", help="timing archive (repeatable)")
        command.add_argument("--domain", required=name != "run", help="Phase 2 corpus ledger or ranges")
        command.add_argument("--budget", action="append", help="clip a domain maximum, e.g. macs=16777216")
        command.add_argument("--out", required=True)
        if name == "run":
            command.add_argument("--selection", help="a selection document from `select` (else select now)")
            command.add_argument("--repeats", type=int, default=2)
            command.add_argument("--jobs", type=int, default=1)
            command.add_argument("--max-cycles", type=int, default=200_000_000)
            command.add_argument("--timeout", type=float, default=3600.0)
            command.add_argument("--spike", action="store_true", help="also run each ELF on the functional model")
            command.add_argument("--expect-identity", help="refuse unless the derived hardware identity matches")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "identity":
        identity = runner.hardware_identity(args.target_hardware)
        print(json.dumps({k: v for k, v in identity.items() if k != "engine_citation"}, indent=1, sort_keys=True))
        return 0
    timing_space, backend, contract = _load(args)
    if args.command in ("select", "check"):
        document = _select(args, timing_space, backend)
        _write(Path(args.out), document)
        print(
            json.dumps(
                {
                    "probes": [p["id"] for p in document["probes"]],
                    "gaps": document["gaps"],
                    "design_check": document["design_check"],
                },
                indent=1,
            )
        )
        return 0 if args.command == "select" or document["gaps"] == 0 else 1
    out = runner.private_dir(Path(args.out))
    if args.selection:
        chosen = json.loads(Path(args.selection).read_text())
    else:
        if not args.domain:
            raise SystemExit("run without --selection needs --domain")
        chosen = _select(args, timing_space, backend)
        _write(out / "selection.json", chosen)
    report = runner.run(
        chosen,
        space=timing_space,
        backend=backend,
        contract=contract,
        target=args.target_hardware,
        out_dir=out,
        repeats=args.repeats,
        jobs=args.jobs,
        max_cycles=args.max_cycles,
        timeout_s=args.timeout,
        spike=args.spike,
        expect_identity=args.expect_identity,
    )
    if args.domain:
        recheck = _select(args, timing_space, backend, extra_labels=[report["archive"]["path"]])
        _write(out / "recheck.json", recheck)
        report["recheck"] = {
            "gaps": recheck["gaps"],
            "probes_still_requested": [p["id"] for p in recheck["probes"]],
            "design_check": recheck["design_check"],
        }
    _write(out / "run_report.json", report)
    print(json.dumps(report, indent=1, sort_keys=True, default=str))
    failed = [p for p in report["probes"] if p["status"] != "pass" or not p["deterministic"]]
    return 1 if failed or report.get("recheck", {}).get("gaps") else 0


if __name__ == "__main__":
    sys.exit(main())

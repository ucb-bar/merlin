"""The installed Merlin command; optional workflows are imported only when selected."""

from __future__ import annotations

import argparse
import importlib

_COMMANDS = {
    "experiment": (
        "merlin.experiments.cli",
        "merlin[experiments]",
        "derive tests, author compilers, optimize performance",
    ),
    "compile": ("merlin.service_cli", None, "compile explicit source or a model"),
    "run": ("merlin.service_cli", None, "run an existing artifact or prepare external execution"),
    "inspect": ("merlin.service_cli", None, "inspect products, software specifications or installed backends"),
    "collect": ("merlin.service_cli", None, "collect existing external observations"),
    "lower": ("merlin.llvmlower.cli", None, "lower supplied MLIR and inspect intermediate stages"),
    "target": ("merlin.targetgen.cli", None, "inspect and derive target support"),
    "storage": ("merlin.common.storage_cli", None, "inspect and manage generated output"),
    "verify": ("merlin.verify.cli", None, "verify compiler passes"),
}


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="merlin", description="Compiler generation and three-phase experiments")
    parser.add_argument("command", choices=_COMMANDS, help="; ".join(f"{k}: {v[2]}" for k, v in _COMMANDS.items()))
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    selected = parser.parse_args(argv[:1]).command
    module, requirement, _ = _COMMANDS[selected]
    try:
        entry = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if requirement and exc.name in {"aet", "jsonschema"}:
            parser.error(f"'{selected}' requires the optional '{exc.name}' dependency; install '{requirement}'")
        raise
    handler = getattr(entry, f"{selected}_main") if module == "merlin.service_cli" else entry.main
    return handler(argv[1:]) or 0


if __name__ == "__main__":
    raise SystemExit(main())

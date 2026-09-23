"""The installed Merlin command; optional workflows are imported only when selected."""

from __future__ import annotations

import argparse
import importlib

_COMMANDS = {
    "experiment": (
        "merlin_experiments.cli",
        "merlin-experiments",
        "derive tests, author compilers, optimize performance",
    ),
    "compile": ("merlin.compile_cli", None, "compile a model"),
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
    module, distribution, _ = _COMMANDS[selected]
    try:
        entry = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if distribution and exc.name == module.split(".")[0]:
            parser.error(
                f"'{selected}' requires the optional {distribution} distribution; "
                f"from a Merlin checkout run: uv pip install -e packages/{distribution}"
            )
        raise
    return entry.main(argv[1:]) or 0


if __name__ == "__main__":
    raise SystemExit(main())

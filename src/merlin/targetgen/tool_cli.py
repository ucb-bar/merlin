"""Installed operator entrypoints for target-neutral support primitives.

These tools are also imported by selected OOT providers. Explicit input and output arguments keep
their use independent of a Merlin checkout or any in-tree accelerator implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _boot(args: argparse.Namespace) -> int:
    from .fixed_format import boot

    preamble = Path(args.asm_preamble).read_text() if args.asm_preamble else ""
    result = boot.build_boot_object(
        args.source,
        args.out,
        target=args.target,
        clang=args.clang,
        march=args.march,
        mabi=args.mabi,
        asm_preamble=preamble,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _link(args: argparse.Namespace) -> int:
    from .fixed_format import link

    result = link.link_fork_free(
        args.object,
        args.linker_script,
        args.out,
        target=args.target,
        linker=args.linker,
    )
    print(result)
    return 0


def _register_slices(args: argparse.Namespace) -> int:
    from .rtl import register_slices

    raw = json.loads(Path(args.inputs).read_text())
    if not isinstance(raw, dict) or any(
        not isinstance(value, list) or len(value) != 2 or not isinstance(value[0], str) or type(value[1]) is not int
        for value in raw.values()
    ):
        raise ValueError("--inputs must map hardware SSA names to [input_label, bit_width]")
    result = register_slices.derive_register_slices(
        Path(args.hw).read_text(),
        module=args.module,
        registers=args.register,
        selector=args.selector,
        selector_value=int(args.selector_value, 0),
        inputs={key: (value[0], value[1]) for key, value in raw.items()},
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(rendered)
    else:
        print(rendered, end="")
    return 0


def _spike_extension(args: argparse.Namespace) -> int:
    from . import spike_extension

    result = spike_extension.resolve(
        args.target,
        default_library_dir=args.default_library_dir,
        default_extension_name=args.default_extension_name,
    )
    print(
        json.dumps(
            {
                "target": result.target,
                "declared": result.declared,
                "extension_name": result.extension_name,
                "extlib": str(result.extlib) if result.extlib else None,
                "library_dir": str(result.library_dir),
                "sha256": result.sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="merlin-target-tools", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    boot = sub.add_parser("fixed-boot", help="stock-assemble and transcode a fixed-format boot object")
    boot.add_argument("--target", required=True)
    boot.add_argument("--source", required=True)
    boot.add_argument("--out", required=True)
    boot.add_argument("--clang", required=True)
    boot.add_argument("--asm-preamble", help="target-owned assembler preamble file")
    boot.add_argument("--march", default="rv32im")
    boot.add_argument("--mabi", default="ilp32")
    boot.set_defaults(func=_boot)

    link = sub.add_parser("fixed-link", help="stock-link and patch derived fixed-format relocations")
    link.add_argument("--target", required=True)
    link.add_argument("--object", action="append", required=True)
    link.add_argument("--linker-script", required=True)
    link.add_argument("--out", required=True)
    link.add_argument("--linker")
    link.set_defaults(func=_link)

    slices = sub.add_parser("register-slices", help="derive conditional RTL register input slices")
    slices.add_argument("--hw", required=True, help="elaborated HW MLIR file")
    slices.add_argument("--module", required=True)
    slices.add_argument("--register", action="append", required=True, help="hardware SSA register reference")
    slices.add_argument("--selector", required=True, help="hardware SSA selector reference")
    slices.add_argument("--selector-value", required=True, help="integer selector value (base 0)")
    slices.add_argument("--inputs", required=True, help="JSON mapping SSA refs to [input_label, bit_width]")
    slices.add_argument("--out", help="JSON output path; stdout when omitted")
    slices.set_defaults(func=_register_slices)

    spike = sub.add_parser("spike-extension", help="verify the selected target's L2 model identity")
    spike.add_argument("--target", required=True)
    spike.add_argument("--default-library-dir", required=True)
    spike.add_argument("--default-extension-name", required=True)
    spike.set_defaults(func=_spike_extension)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

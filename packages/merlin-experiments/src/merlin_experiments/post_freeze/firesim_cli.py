"""Command-line preparation and collection of frozen FireSim observations."""

import argparse
import json
from pathlib import Path

import yaml

from .firesim import __doc__, collect, load, prepare


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--selection", type=Path, required=True)
    c = sub.add_parser("collect")
    c.add_argument("--request", type=Path, required=True)
    c.add_argument("--result-dir", type=Path, required=True)
    c.add_argument("--ssm-invocation", type=Path)
    c.add_argument("--ssm-command", type=Path)
    c.add_argument("--elf-upload", type=Path)
    for command in (p, c):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = (
            prepare(args.selection.absolute())
            if args.command == "prepare"
            else collect(
                args.request.absolute(),
                args.result_dir.absolute(),
                args.ssm_invocation.absolute() if args.ssm_invocation else None,
                ssm_command_path=args.ssm_command.absolute() if args.ssm_command else None,
                elf_upload_path=args.elf_upload.absolute() if args.elf_upload else None,
            )
        )
        source = value if args.command == "prepare" else load(args.request)
        if args.out.resolve().is_relative_to(Path(source["compiler"]["path"]).resolve()):
            raise ValueError("observation output cannot modify the frozen compiler")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"status": value["status"], "output": str(args.out), "paid_jobs_launched": 0}))
        return 0 if value["status"] in {"prepared", "observation_imported"} else 2
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        print(json.dumps({"status": "refused", "error": str(error), "paid_jobs_launched": 0}))
        return 2

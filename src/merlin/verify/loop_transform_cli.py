"""Scoped ordered unroll/1D segmentation checker and guarded output consumer."""

import argparse
import json
from pathlib import Path

from .loop_transform_gate import NotEstablished, check, deliver, replay
from .loop_transform_intake import Settings, Unavailable, Unsupported, prepare, produce


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path)
    inputs.add_argument("--buffers", type=Path, nargs=3)
    inputs.add_argument("--replay", type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--lean", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--tile", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        if args.replay:
            if args.output:
                raise Unsupported("replay is evidence; output delivery requires a fresh actual request")
            result = replay(args.replay, reference=args.reference, lean=args.lean, destination=args.destination)
        else:
            settings = Settings(args.factor, args.tile)
            buffers = (
                tuple(p.read_text() for p in args.buffers)
                if args.buffers
                else produce(args.source.read_text(), reference=args.reference, settings=settings)
            )
            prepared = prepare(buffers, reference=args.reference, settings=settings)
            result = check(prepared, lean=args.lean, destination=args.destination)
            if args.output:

                def consume(text):
                    with args.output.open("x") as handle:
                        handle.write(text)

                deliver(result, prepared, consume)
        print(json.dumps({"status": "checked", "report": result.report, "general_tiling": False}))
        return 0
    except (Unsupported, NotEstablished, Unavailable, OSError, ValueError) as error:
        status = (
            "unsupported"
            if isinstance(error, Unsupported)
            else "not_established"
            if isinstance(error, NotEstablished)
            else "unavailable"
        )
        print(json.dumps({"status": status, "reason": str(error), "delivered": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

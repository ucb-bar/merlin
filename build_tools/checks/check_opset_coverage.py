#!/usr/bin/env python3
"""Check explicit public op/form metadata; no implicit corpus or debt creation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from merlin.targetgen.opset_contract import problems
from merlin.targetgen.opset_coverage_input import read_request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--ratchet", type=Path)
    args = parser.parse_args(argv)
    _, report, _ = read_request(args.input)
    debt = (
        ()
        if args.ratchet is None
        else tuple(
            line.split("#", 1)[0].strip()
            for line in args.ratchet.read_text().splitlines()
            if line.split("#", 1)[0].strip()
        )
    )
    errors = problems(report, ratchet=debt)
    print(json.dumps({"report": report, "problems": errors}, sort_keys=True, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())

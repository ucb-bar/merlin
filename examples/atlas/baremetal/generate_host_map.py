"""Generate adapter constants from Atlas runtime/selected-host-map.json."""
import argparse
import json
from pathlib import Path


def generate(source, destination):
    record = json.loads(Path(source).read_text())
    keys = {"IMEM_BASE": "imem_base", "CSR_BASE": "csr_base",
            "IMEM_CAPACITY_BYTES": "imem_capacity_bytes"}
    lines = ["#ifndef EE290_HOST_MAP_H", "#define EE290_HOST_MAP_H"]
    for name, key in keys.items():
        value = record[key]
        if type(value) is not int or value <= 0:
            raise ValueError("positive host map address/extent required")
        lines.append(f"#define EE290_{name} {value:#x}ULL")
    Path(destination).write_text("\n".join([*lines, "#endif", ""]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    generate(args.source, args.destination)

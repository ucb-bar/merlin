"""Fixed original transpose observer with distinct finite storage observations."""

import importlib.util
import json
import sys
from pathlib import Path


def observe(loader, metadata, stimulus):
    spec = importlib.util.spec_from_file_location(
        "fixed_original_storage_observer", Path(__file__).with_name("original_reference_observer.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.observe(loader, metadata, stimulus, version=3)


if __name__ == "__main__":
    loader, metadata, stimulus, destination = map(Path, sys.argv[1:])
    output = observe(loader, json.loads(metadata.read_bytes()), json.loads(stimulus.read_bytes()))
    destination.write_text(json.dumps(output, sort_keys=True, allow_nan=False) + "\n")
    destination.chmod(0o600)

"""Fixed complete original pointwise probe transport; emits values, no verdict."""

import importlib.util
import json
import sys
from pathlib import Path


def main(request, destination):
    selected = json.loads(request.read_bytes())
    if set(selected) != {"schema", "members"} or selected["schema"] != "merlin.original_pointwise_probe_request.v1":
        raise ValueError("pointwise probe transport requires its exact fixed request")
    spec = importlib.util.spec_from_file_location(
        "fixed_original_pointwise_storage", Path(__file__).with_name("original_pointwise_reference_observer.py")
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    slots = []
    for member in selected["members"]:
        if set(member) != {"slot", "source", "metadata", "inputs", "actual"} or member["slot"] in slots:
            raise ValueError("pointwise probe transport changed complete unique original slots")
        output = worker.observe(
            Path(member["source"]),
            json.loads(Path(member["metadata"]).read_bytes()),
            json.loads(Path(member["inputs"]).read_bytes()),
        )
        actual = Path(member["actual"])
        actual.write_text(json.dumps(output, sort_keys=True, allow_nan=False) + "\n")
        actual.chmod(0o600)
        slots.append(member["slot"])
    destination.write_text(
        json.dumps({"schema": "merlin.original_pointwise_probe_observation.v1", "slots": slots}) + "\n"
    )
    destination.chmod(0o600)


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))

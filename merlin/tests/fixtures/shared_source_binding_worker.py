"""Isolated Python algorithm execution from an explicitly reviewed source SDK."""

import hashlib
import json
import runpy
import sys
from pathlib import Path

assert sys.dont_write_bytecode
config = json.loads(Path(sys.argv[1]).read_bytes())
bundle = Path(config["bundle"])
assert not any(name == "merlin" or name.startswith("merlin.") for name in sys.modules)
sys.path[:0] = [str(bundle), *config["dependencies"]]

for name in (
    "merlin.llvmlower.direct_source_contraction",
    "merlin.llvmlower.late_quant_rne",
    "merlin_experiments.phase0.connected_integer_bounds",
    "merlin.runtime.reference",
):
    try:
        __import__(name)
    except ImportError:
        pass
    else:
        raise AssertionError("private/unselected import became available: " + name)

if config["selected"]:
    value = runpy.run_path(config["candidate"])["execute"](
        Path(config["source"]), config["command_buffer"], config["analysis"]
    )
    assert value["removed_rhs_loads"] == 2 and value["outputs"] == 25 and value["products"] == 100
    assert value["token_text"] == ["%a", "=", "add", "i32", "%b", ",", "1"]
    assert value["binding"] == dict(ordinal=2, lhs="a", rhs="b", output="out", m=5, k=4, n=5)
    for name, module in tuple(sys.modules.items()):
        if name == "merlin" or name.startswith("merlin."):
            assert Path(module.__file__).resolve().is_relative_to(bundle)
    value["loaded_merlin_modules"] = sorted(
        name for name in sys.modules if name == "merlin" or name.startswith("merlin.")
    )
    value["candidate_sha256"] = hashlib.sha256(Path(config["candidate"]).read_bytes()).hexdigest()
else:
    try:
        runpy.run_path(config["candidate"])
    except ImportError:
        value = {"unselected_import": "refused"}
    else:
        raise AssertionError("V1 release unexpectedly supplied source-binding API")
Path(config["result"]).write_text(json.dumps(value, sort_keys=True) + "\n")

"""Private neutral test executor for a queued policy and frozen shared release.

This isolates import origins, not candidate filesystem/network access. Ordinary
sandbox admission remains a separate requirement; no hardware/timing role issues.
"""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

assert sys.dont_write_bytecode
config = json.loads(Path(sys.argv[1]).read_bytes())
root, work = Path(config["bundle"]), Path(config["work"])
assert not any(name == "merlin" or name.startswith("merlin.") for name in sys.modules)
sys.path[:0] = [str(root), config["fixtures"], *config["dependencies"]]
# The selected compiler may own new pass modules; the frozen SDK keeps precedence.
sys.path.insert(1, str(Path(config["candidate"]).parent))

import numpy as np  # noqa: E402
import test_configured_pointwise as configured  # noqa: E402
import test_pointwise_source as source  # noqa: E402

from merlin.compile.scheduling.library_sources import shared_library_sources  # noqa: E402

choose = runpy.run_path(config["candidate"])["schedule"]
observations = []
native_root = work / "native"
native_root.mkdir()
native = source.execute.__wrapped__(native_root, SimpleNamespace(param=dict(source=configured.EXECUTOR, counters=5)))
for index, (m, n, separate) in enumerate(((3, 5, False), (2, 11, True))):
    case_root = work / ("shared-pass-" + str(index))
    case_root.mkdir()
    options = configured.case(case_root, m=m, n=n, separate=separate)
    facts = json.loads(options["selected_contracts"].facts_path.read_bytes())
    selected = choose((m, n), {"persistent_fields": facts["facts"]["state_fields"]})
    budgets, rewrites = selected if type(selected) is tuple else (selected, ())
    assert budgets.max_fusions == 0 and budgets.optimize_storage
    model, operation = source.source_program(m=m, n=n)
    compiled = configured.compile_source_pointwise(model, operation, budgets=budgets, rewrites=rewrites, **options)
    image = b"".join(compiled.images())
    (case_root / "program.bin").write_bytes(image)
    x = np.arange(m * n, dtype=np.int32) * 37 - 350
    bias = np.full(m * n, 17, dtype=np.int32)
    expected = np.clip(x.astype(np.int64) + bias, -128, 127).astype("i1").tobytes()
    output, counts = native(compiled.images(), {"x": x.tobytes(), "b0": bias.tobytes()}, separate=separate)
    assert output == expected + bytes([193] * (1032 - m * n))
    assert counts[3:] == (configured.LOW, configured.HIGH)
    calls = options["encoding"].decode_calls((case_root / "program.bin").read_bytes())
    assert len(calls) == counts[0] == (17 if rewrites else 4 if budgets.eliminate_configuration else 18)
    assert counts[1:3] == ((8 * m * n, m * n) if rewrites else (12 * m * n, 5 * m * n))
    observations.append(
        dict(
            shape=[m, n],
            separate=separate,
            commands=counts[0],
            transfer_bytes=list(counts[1:3]),
            final_settings=list(counts[3:]),
            image_path=str(case_root / "program.bin"),
            output_hex=output.hex(),
            rewrites=[row["name"] for row in compiled.rewrite_checks],
        )
    )
origins = {}
for name, module in tuple(sys.modules.items()):
    if name == "merlin" or name.startswith("merlin."):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(root), (name, path)
        assert all(Path(p).resolve().is_relative_to(root) for p in getattr(module, "__path__", ()))
        origins[name] = str(path)
manifest = json.loads((root / "shared-library.json").read_bytes())
sources = shared_library_sources()
assert sources == {row["module"]: row["sha256"] for row in manifest["members"] if row["module"] is not None}
Path(config["result"]).write_text(
    json.dumps(
        dict(
            observations=observations,
            source_owners=sources,
            module_origins=origins,
            library_sha256=manifest["library_sha256"],
            bytecode_disabled=sys.dont_write_bytecode,
        )
    )
)

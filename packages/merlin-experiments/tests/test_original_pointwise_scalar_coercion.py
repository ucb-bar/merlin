"""Direct original integer Scalar conversion from selected native public c10."""

import hashlib
import json
import os
import struct
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.targetgen.original_pointwise_reference import OriginalPointwiseReferencePolicy, scalar_bound


def test_actual_original_native_scalar_conversion_full_signed64_tie_roster(tmp_path):
    required = ("MERLIN_TEST_TORCH_PYTHON", "MERLIN_TEST_TORCH_SCALAR_HEADER", "MERLIN_TEST_TORCH_TYPECAST_HEADER")
    if not all(os.environ.get(name) for name in required):
        pytest.skip("direct Scalar controls need explicitly selected native Torch and public SDK headers")
    python, scalar, typecast = (Path(os.environ[name]).absolute() for name in required)
    headers = []
    for path in (scalar, typecast):
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("direct Scalar controls require exact ordinary selected public SDK headers")
        headers.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pairs = []
    for power in (53, 60):
        step = 2.0 ** (power - 23)
        for sign in (-1, 1):
            for odd in (False, True):
                lower = 2.0**power + (step if odd else 0.0)
                midpoint = (1 << power) + (3 if odd else 1) * (1 << (power - 24))
                pairs += [
                    (sign * (midpoint - 1), sign * lower),
                    (sign * midpoint, sign * (lower + step if odd else lower)),
                    (sign * (midpoint + 1), sign * (lower + step)),
                ]
    pairs += [(-(1 << 63), -(2.0**63)), ((1 << 63) - 1, 2.0**63), (-1, -1.0), (0, 0.0), (1, 1.0)]
    request, script, output = (tmp_path / name for name in ("request.json", "observe.py", "outputs.json"))
    request.write_text(json.dumps({"headers": headers, "bounds": [bound for bound, _ in pairs]}, allow_nan=False))
    script.write_text("""import json,sys,hashlib
from pathlib import Path
import torch
request=json.loads(Path(sys.argv[1]).read_bytes())
include=Path(torch.__file__).absolute().parent/'include'
expected=[include/'c10/core/Scalar.h',include/'c10/util/TypeCast.h']
for declared,path in zip(request['headers'],expected,strict=True):
 assert declared=={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
rows=[]
for bound in request['bounds']:
 value=torch.ops.aten.clamp.default(torch.tensor([-2.0**100],dtype=torch.float32),min=bound)
 rows.append({'bound':bound,'data_hex':value.numpy().tobytes().hex()})
Path(sys.argv[2]).write_text(json.dumps({'headers':request['headers'],'rows':rows,
 'runtime':{'torch_version':torch.__version__,'git_version':torch.version.git_version,'package':torch.__file__}},sort_keys=True))
""")
    native = I.run(
        [str(python), "-I", str(script), str(request), str(output)],
        directory=tmp_path,
        cwd=tmp_path,
        stage="original_signed64_scalar_direct_f32_native",
        inputs=(script, request, scalar, typecast),
        outputs=(output,),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        capture_output=True,
        timeout=60,
    )
    native.check_returncode()
    actual = json.loads(output.read_bytes())
    assert actual["headers"] == headers and len(actual["rows"]) == len(pairs) == 29
    chosen = OriginalPointwiseReferencePolicy(
        "aten.clamp.default",
        ("float32",),
        ("float32",),
        "float32",
        "finite_f32",
        "not_applicable",
        "elementwise",
        "not_applicable",
        "rne",
        True,
        False,
        "not_applicable",
        0.0,
        0.0,
        "preserve",
    )
    chosen.verify()
    for row, (bound, expected) in zip(actual["rows"], pairs, strict=True):
        assert row == {"bound": bound, "data_hex": struct.pack("<f", expected).hex()}
        assert struct.pack("<f", scalar_bound(bound, chosen)).hex() == row["data_hex"]

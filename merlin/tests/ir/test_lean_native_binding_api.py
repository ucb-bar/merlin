"""The gate's domain visitor uses the actual pinned MLIR Python object API.

This exercises parsing and domain admission only, without invoking Lean, compiler
passes, native lowering or execution. Mock transport tests cannot qualify this API.
"""

import hashlib
import json

from merlin.llvmlower import proof_stage_transport as transport

PROBE = """import json, sys
from pathlib import Path
from torch_mlir import ir
request = json.loads(Path(sys.argv[1]).read_bytes())
namespace = {'ir': ir, '_run_stages': lambda *args: None, '_PG_MAX_BYTES': 1000000}
exec(request['visitor'], namespace)
results = []
for case in request['cases']:
    namespace['_PG_BINDING'] = {'index_bits': case['bits']}
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(case['source'])
        try:
            namespace['_pg_domain'](module)
        except ValueError as exc:
            results.append({'name': case['name'], 'verdict': 'refused', 'reason': str(exc)})
        else:
            results.append({'name': case['name'], 'verdict': 'admitted'})
print(json.dumps(results))
"""


def test_actual_binding_visits_scalar_tensor_and_index_abis(tmp_path, lean_tools):
    scalar = "func.func @forward(%x: i8) -> i8 { return %x : i8 }"
    tensor = "func.func @forward(%x: tensor<2xi32>) -> tensor<2xi32> { return %x : tensor<2xi32> }"
    index = "func.func @forward(%x: index) -> index { return %x : index }"
    tensor_index = "func.func @forward(%x: tensor<2xindex>) -> tensor<2xindex> { return %x : tensor<2xindex> }"
    local_index = "func.func @forward() { %c = arith.constant 1 : index return }"
    nested_index = """func.func @forward(%x: tensor<i32>) -> tensor<i32> {
      %r = linalg.generic {indexing_maps = [affine_map<() -> ()>], iterator_types = []}
        outs(%x : tensor<i32>) {
        ^bb0(%v: i32):
          %c = arith.constant 1 : index
          linalg.yield %v : i32
        } -> tensor<i32>
      return %r : tensor<i32>
    }"""
    cases = [
        {"name": name, "bits": bits, "source": source}
        for name, bits, source in (
            ("scalar32", 32, scalar),
            ("tensor32", 32, tensor),
            ("index-abi32", 32, index),
            ("tensor-index-abi32", 32, tensor_index),
            ("index-local32", 32, local_index),
            ("index-nested32", 32, nested_index),
            ("index-abi64", 64, index),
            ("tensor-index-abi64", 64, tensor_index),
        )
    ]
    script, request = tmp_path / "probe.py", tmp_path / "request.json"
    script.write_text(PROBE)
    request.write_text(json.dumps({"visitor": transport.NATIVE, "cases": cases}))
    command = [str(lean_tools.m2m_python), str(script), str(request)]
    result = transport.run_owned(command, capture_output=True, text=True, check=False, timeout=10)
    (tmp_path / "stdout.log").write_text(result.stdout)
    (tmp_path / "stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    expected = ["admitted", "admitted", "refused", "refused", "refused", "refused", "admitted", "admitted"]
    assert [item["verdict"] for item in actual] == expected
    for item in actual:
        if item["verdict"] == "refused":
            assert "index-typed" in item["reason"]
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "compiler": transport.fingerprint(lean_tools.m2m_python),
                "command": command,
                "visitor_sha256": hashlib.sha256(transport.NATIVE.encode()).hexdigest(),
                "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                "results": actual,
                "scope": "real pinned Python binding domain visitor; no Lean/passes/native execution",
            },
            indent=2,
        )
    )

"""Parser-level key injectivity probes (pinned producer parser, local xDSL 0.68.0).

Graham independent review (agent), 2026-10-26. Records (1) whether `str()` of
float attributes distinguishes NaN payloads and signed zero, as used by the loops
captures' metadata keys, and (2) the process-global renaming of a repeated
dense_resource handle across separate parse_ir calls. Run with
PYTHONPATH=<pinned producer>/src.
"""
import json
from pathlib import Path

from merlin.transforms.certificates import ir_text, parse_ir

OUT = Path(__file__).resolve().parent.parent / "probes/parser-keys"
scalar = "builtin.module {\n  func.func @k() -> f32 {\n    %c = arith.constant VALUE : f32\n    func.return %c : f32\n  }\n}"
dense = ("builtin.module {\n  func.func @k() -> tensor<1xf32> {\n    %c = arith.constant dense<[VALUE]> : tensor<1xf32>\n"
         "    func.return %c : tensor<1xf32>\n  }\n}")
resource = ("builtin.module {\n  func.func @k() -> tensor<2xi8> {\n    %c = arith.constant dense_resource<blob1> : tensor<2xi8>\n"
            "    func.return %c : tensor<2xi8>\n  }\n}\n{-#\n  dialect_resources: {\n    builtin: {\n      blob1: \"0x010000000102\"\n"
            "    }\n  }\n#-}")
rows = {"float_keys": [], "resource_handles": []}
for template in (scalar, dense):
    for value in ("0x7FC00001", "0x7FC00002", "-0.0", "0.0") if template is scalar else ("0x7FC00001", "0x7FC00002"):
        module = parse_ir(template.replace("VALUE", value))
        op = list(module.walk())[2]
        rows["float_keys"].append({"literal": value, "str_key": [(k, str(v)) for k, v in op.properties.items()]})
for k in range(3):
    module = parse_ir(resource)
    op = list(module.walk())[2]
    rows["resource_handles"].append({"parse": k, "handle": op.value.resource_handle.data,
                                     "printed_has_payload": "dialect_resources" in ir_text(module)})
OUT.joinpath("RESULT.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=1))

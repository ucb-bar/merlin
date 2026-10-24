"""Compiled C-interface descriptors, not expected answers, determine observations."""

import copy
import json
import shutil
import subprocess

import numpy as np
import pytest
from targetgen_fixtures import measured_routing

from merlin.llvmlower import c_runtime
from merlin.runtime.backends.spike_model import parse_console
from merlin.targetgen.core_aten_semantics import validate_full


def run_boundary(tmp_path, *, stride=(2, 1), offset=0, aliases=True, contract_aliases=True):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native C compiler unavailable")
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.mlir").write_text("""builtin.module {
      func.func @forward(%x: tensor<4xf32>) -> (tensor<2x2xf32>, tensor<4xf32>) {
        %v = "fixture.view"(%x) : (tensor<4xf32>) -> tensor<2x2xf32>
        func.return %v, %x : tensor<2x2xf32>, tensor<4xf32>
      }
    }""")
    (model / "weights.safetensors.manifest.json").write_text(json.dumps({"0": {"kind": "input", "name": "x"}}))
    (model / "input_order.json").write_text(json.dumps({"x": 0}))
    values = [1.0, 2.0, 3.0, 4.0]
    tensor = dict(
        kind="tensor", dtype="float32", shape=[4], stride=[1], storage_offset=0, requires_grad=False, values=values
    )
    expected = {**tensor, "shape": [2, 2], "stride": [2, 1], "values": [[1.0, 2.0], [3.0, 4.0]]}
    arguments = dict(args=dict(kind="tuple", items=[tensor]), kwargs={})
    record = dict(
        input_indices=[0],
        output_indices=[0],
        output_abi=[dict(dtype="f32", shape=[2, 2])],
        semantic_boundary=dict(post_indices=[1], post_abi=[dict(dtype="f32", shape=[4])]),
        case=dict(
            arguments=arguments,
            post_arguments=copy.deepcopy(arguments),
            expected=expected,
            comparison="torch_close",
            mutated_arguments=[],
            output_input_aliases=["result->args[0]"] if contract_aliases else [],
        ),
    )
    # Deliberately unrelated answer fields: none may construct the observation.
    (model / "semantic_io.json").write_text(
        json.dumps(
            dict(
                input_indices=[0],
                post_indices=[1],
                outputs=[
                    dict(
                        shape=[999],
                        stride=[999],
                        storage_offset=999,
                        requires_grad=True,
                        alias_outputs=[1] if contract_aliases else [],
                    ),
                    dict(shape=[999], stride=[999], storage_offset=999, requires_grad=True, alias_outputs=[]),
                ],
            )
        )
    )
    np.savez(model / "inputs.npz", in0=np.array(values, np.float32))
    cgen = tmp_path / "cgen"
    c_runtime.generate(model, cgen, model / "inputs.npz", dump_all_outputs=True)
    # This is a compiled program implementing the ranked memref C ABI. Its output
    # descriptor can disagree with the private contract independently of its values.
    storage = "input" if aliases else "fresh"
    source = tmp_path / "main.c"
    source.write_text(f"""
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "model_gen.h"
#include "model_io.h"
static void htif_puts(const char *s) {{ fputs(s,stdout); }}
static void htif_putd(long v) {{ printf("%ld",v); }}
static void htif_putc(char v) {{ putchar(v); }}
#include "semantic_io.h"
typedef struct {{ void *allocated, *aligned; int64_t offset, sizes[2], strides[2]; }} rank2;
typedef struct {{ void *allocated, *aligned; int64_t offset, sizes[1], strides[1]; }} rank1;
typedef struct {{ rank2 view; rank1 post; }} result;
void _mlir_ciface_forward(result *out, rank1 *in) {{
 static float fresh[8]={{{"0,1,2,3,4,0,0,0" if offset else "1,2,3,4,0,0,0,0"}}};
 float *input=in->aligned;
 out->view.allocated={storage}; out->view.aligned={storage}; out->view.offset={offset};
 out->view.sizes[0]=2; out->view.sizes[1]=2;
 out->view.strides[0]={stride[0]}; out->view.strides[1]={stride[1]};
 out->post=*in;
}}
int main(void) {{
 rank1 in={{MERLIN_INPUT_PTR[0],MERLIN_INPUT_PTR[0],0,{{4}},{{1}}}};
 void *d[]={{&in}};
 merlin_semantic_before(); merlin_invoke(d); merlin_semantic_after();
 for(int i=0;i<2;i++) {{ printf("OUT_BYTES %d %zu",i,MERLIN_OUTPUT_NBYTES[i]);
  for(size_t j=0;j<MERLIN_OUTPUT_NBYTES[i];j++) printf(" %u",((unsigned char*)MERLIN_OUTPUT_PTR[i])[j]); puts(""); }}
 puts("DONE"); return 0;
}}
""")
    from merlin.common.paths import repo_root

    subprocess.run(
        [
            compiler,
            "-I",
            str(cgen),
            "-I",
            str(repo_root() / "merlin/runtime/c"),
            str(source),
            str(cgen / "model_call.c"),
            "-lm",
            "-o",
            str(tmp_path / "run"),
        ],
        check=True,
        capture_output=True,
    )
    result = subprocess.run([str(tmp_path / "run")], check=True, capture_output=True, text=True)
    observed = parse_console(result.stdout)
    return record, observed


def test_actual_alias_and_layout_pass_with_honest_scope(tmp_path):
    record, result = run_boundary(tmp_path)
    verdict = validate_full(record, result["output_bytes"], result["semantic_readback"])
    assert verdict["status"] == "pass"
    assert verdict["semantic_scope"] == "values+mutation+observed-metadata"
    assert verdict["imposed_by_harness"] == ["requires_grad"]
    assert result["semantic_readback"]["aliases"][0] == [1]
    assert result["semantic_readback"]["metadata"][0]["storage_offset"] == 0


@pytest.mark.parametrize("defect", ["stride", "offset", "fresh_instead_of_alias", "alias_instead_of_fresh"])
def test_compiled_boundary_defects_fail(tmp_path, defect):
    options = {}
    if defect == "stride":
        options["stride"] = (1, 2)
    elif defect == "offset":
        # Keep all values correct, so only the descriptor offset can catch this.
        options.update(offset=1, aliases=False, contract_aliases=False)
    elif defect == "fresh_instead_of_alias":
        options["aliases"] = False
    else:
        options["contract_aliases"] = False
    record, result = run_boundary(tmp_path, **options)
    if defect == "stride":
        # Match the actual gathered values to isolate the metadata verdict.
        record["case"]["expected"]["values"] = [[1.0, 3.0], [2.0, 4.0]]
    verdict = validate_full(record, result["output_bytes"], result["semantic_readback"])
    assert verdict["status"] == "mismatch"
    assert ("metadata mismatch" if defect in ("stride", "offset") else "output_input_aliases mismatch") in verdict[
        "reason"
    ]
    assert verdict["values_mutation_status"] == "pass"


def test_strided_values_are_gathered_from_actual_memref(tmp_path):
    record, result = run_boundary(tmp_path, stride=(1, 2))
    record["case"]["expected"].update(stride=[1, 2], values=[[1.0, 3.0], [2.0, 4.0]])
    assert validate_full(record, result["output_bytes"], result["semantic_readback"])["status"] == "pass"


def test_aggregate_keeps_numeric_and_mutation_passes_when_aliases_fail(tmp_path):
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    record, observed = run_boundary(tmp_path, aliases=False)
    record.update(overload="fixture", status="bundled")
    verdict = grade_core_aten_batch(
        {"cases": [record]}, observed["output_bytes"], semantic_readback=observed["semantic_readback"]
    )
    assert verdict["passed_count"] == 0
    assert verdict["full_semantic_passed_count"] == 0
    assert verdict["values_mutation_passed_count"] == 1
    assert verdict["numeric_passed_count"] == 1
    assert verdict["numeric_only_passed_count"] == 0


@pytest.mark.parametrize("defect", [None, "output", "post", "alias", "metadata", "missing_reference"])
def test_rational_reference_survives_full_semantic_evaluation(tmp_path, monkeypatch, defect):
    pytest.importorskip("torch")
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch
    from merlin.targetgen.core_aten_eval import _validate_observation

    record, observed = run_boundary(tmp_path)
    record.update(overload="fixture", case_id="fixture", status="bundled", numeric_policy="spec_int8")
    record["case"].update(case_id="fixture", comparison_parameters={"rtol": 0, "atol": 0})
    # Capture-time values deliberately disagree with the independently prepared
    # rational contract; metadata, mutation and aliases remain source-owned.
    record["case"]["expected"]["values"] = [[9.0, 9.0], [9.0, 9.0]]
    reference = {str(i): raw.hex() for i, raw in enumerate(observed["output_bytes"])}
    record["numeric_admission"] = dict(
        status="admitted",
        quantized_contract={"scale_denominator": 64},
        quantized_non_contraction_sites=1,
        integer_reference=dict(status="prepared", engine="independent_source", expected_output_bytes=reference),
    )
    record["routing"] = measured_routing("fixture")
    monkeypatch.setattr("merlin.targetgen.core_aten_device.verify_execution_evidence", lambda _: True)
    if defect in {"output", "post"}:
        index = 0 if defect == "output" else 1
        observed["output_bytes"][index] = bytes(len(observed["output_bytes"][index]))
    elif defect == "alias":
        observed["semantic_readback"]["aliases"][0] = []
    elif defect == "metadata":
        observed["semantic_readback"]["metadata"][0]["storage_offset"] = 1
    elif defect == "missing_reference":
        reference.pop("1")
    verdict = grade_core_aten_batch(
        {"cases": [record]}, observed["output_bytes"], semantic_readback=observed["semantic_readback"]
    )["cases"]["fixture"]
    if defect:
        assert verdict["status"] != "pass"
    else:
        assert verdict["status"] == "pass"
        _validate_observation(
            record["case"],
            {
                **verdict["observation"],
                "semantic_scope": verdict["semantic_scope"],
                "imposed_by_harness": verdict["imposed_by_harness"],
            },
            allow_unobserved_autograd=True,
        )
        assert verdict["observation"]["output"]["values"] == [[1.0, 2.0], [3.0, 4.0]]

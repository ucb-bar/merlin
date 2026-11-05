"""Original typed member transport on actual ordinary upstream host products.

The native source/LLVM/object/link/output records are real. Upstream owner
substitutions isolate this transport algorithm and issue no semantic, source
preparation, target, runtime or compiler qualification.
"""

import copy
import importlib.util
import json
import math
import os
from pathlib import Path

import pytest
from merlin_experiments.phase0 import original_reference_standard_ir as S
from merlin_experiments.phase1 import component_original_members as M

from merlin.common import invocation_record as I
from merlin.common.jsonio import canonical_json
from merlin.common.paths import runtime_dir
from merlin.runtime.backends.base import decode_float_readback
from merlin.runtime.out_b64 import OutB64Decoder
from merlin.targetgen.contract.compile_only import require_pointer_entry
from merlin.targetgen.original_operator_reference import OriginalReferenceBudget, prepare_original_reference
from merlin.targetgen.original_pointwise_reference import OriginalPointwiseReferencePolicy
from merlin.targetgen.original_pointwise_sources import pointwise_source
from merlin.targetgen.original_reference_values import TypedReferenceTensor as T


def _fixture(name):
    spec = importlib.util.spec_from_file_location("member_host_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_unit = _fixture("test_component_original_members")
_native = _fixture("test_component_original_pointwise_execution")
_native.WORKER = _native.WORKER.replace(
    "compile_host(module,build)", "compile_host(module,build,retain_llvm_dialect=True)"
)
isolated, ordinary_pointwise = _unit.isolated, _native.ordinary_pointwise


@pytest.fixture(scope="module")
def native_publication(tmp_path_factory):
    compiler = os.environ.get("MERLIN_TEST_CLANG") or os.environ.get("MERLIN_CLANG")
    if not compiler:
        pytest.skip("native complete-value publication needs the explicitly selected host C compiler")
    owner = tmp_path_factory.mktemp("original-member-publication")
    source, executable = owner / "publish.c", owner / "publish"
    source.write_text(r"""
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "out_b64.h"
static void sink(const char *line) { fputs(line, stdout); }
int main(int argc, char **argv) {
  if (argc!=6) return 11;
  unsigned width=strtoul(argv[2],0,10), rows=strtoul(argv[3],0,10), cols=strtoul(argv[4],0,10);
  int sign=atoi(argv[5]); if (!width || width>8 || !rows || !cols) return 12;
  FILE *input=fopen(argv[1],"rb"); if(!input) return 13;
  printf("OUT_B64_BEGIN v1 Y %u %u %u %s\n",rows,cols,width,sign?"s":"u");
  merlin_out_b64 out; merlin_out_b64_init(&out,rows*cols,width,sign,sink);
  for (unsigned i=0;i<rows*cols;++i) {
    uint64_t word=0; for(unsigned byte=0;byte<width;++byte) {
      int value=fgetc(input); if(value==EOF) return 14; word|=(uint64_t)value<<(8*byte);
    }
    if(!merlin_out_b64_word(&out,word)) return 15;
  }
  if(fgetc(input)!=EOF || !merlin_out_b64_finish(&out)) return 16;
  fclose(input); puts("OUT_B64_END"); puts("DONE"); return 0;
}
""")
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    result = I.run(
        [compiler, "-std=c99", "-I", str(runtime_dir() / "baremetal"), str(source), "-o", str(executable)],
        directory=owner,
        stage="original_member_publication_build",
        cwd=owner,
        env=environment,
        inputs=(source, runtime_dir() / "baremetal/out_b64.h"),
        outputs=(executable,),
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode()
    I.require_environment(next(owner.rglob("invocation.json")), environment=environment)
    return executable, environment


@pytest.mark.parametrize("case", _native.CASES, ids=[case["name"] for case in _native.CASES])
def test_original_member_full_native_storage_and_final_element_comparison(
    isolated, ordinary_pointwise, native_publication, tmp_path, monkeypatch, case
):
    produced, native_rows = ordinary_pointwise
    original = produced / case["name"]
    form, extent = case["form"], case["extent"]
    fields = {key: value for key, value in form["source_numerical_semantics"].items() if key != "schema"}
    for name in ("operand_dtypes", "readout_dtypes"):
        fields[name] = tuple(fields[name])
    source = pointwise_source(form, extent=extent, max_tensor_elements=10000)
    contract = prepare_original_reference(
        form,
        source,
        extent=extent,
        policy=OriginalPointwiseReferencePolicy(**fields),
        budget=OriginalReferenceBudget(10000, 100000, 100000, 20000),
        output_byteorder="little",
    )
    metadata = contract.verify()
    # Read the actual standard source, not the loader or a handcrafted IR.
    abi = S.ordered_abi(
        original / "original.mlir",
        metadata,
        {
            "max_source_bytes": 20000,
            "max_nesting": 64,
            "max_integer_bits": 64,
            "max_dense_elements": 10000,
            "max_dense_payload_bytes": 100000,
        },
    )
    standard, document, references, _, _ = isolated
    monkeypatch.setattr(S, "_contract", lambda *_args: contract)
    shape, dtype = metadata["inputs"][0]["shape"], metadata["inputs"][0]["dtype"]
    input_tensor = T("X", dtype, tuple(shape), (original / "input.bin").read_bytes(), "little")
    input_tensor.verify()
    for reference, emitted in zip(references, document["members"], strict=True):
        reference["target"] = form["target"]
        reference["call"] = copy.deepcopy(form)
        input_path = Path(reference["products"]["inputs"]["path"])
        input_path.write_bytes(canonical_json([M.R._tensor_record(input_tensor)]))
        reference["products"]["inputs"] = M.R._pin(input_path)
        source_path = Path(emitted["products"]["source"]["path"])
        source_path.write_bytes((original / "original.mlir").read_bytes())
        emitted.update(
            original={key: reference[key] for key in M._IDENTITY},
            reference_member_sha256=M._sha(M.R._json(reference)),
            products={"source": M.R._pin(source_path)},
            ordered_abi=abi,
        )
    owner = M.prepare(standard_ir=standard, destination=tmp_path / "members")
    member = M.OriginalCandidateMember(owner, 0)
    actual = T("Y", dtype, tuple(shape), (original / "actual.bin").read_bytes(), "little")
    values = list(actual.values())
    assert member.compare_values({"Y": values})["status"] == "pass"
    # Actual ordinary native values pass through the shared C container writer
    # and reader, retaining int64 precision and f32 zero sign before comparison.
    from merlin.targetgen.original_reference_values import format_record

    emitted_format = format_record(dtype)
    executable, environment = native_publication
    rows, cols = (math.prod(shape[:-1]), shape[-1]) if shape else (1, 1)
    publication = I.run(
        [
            str(executable),
            str(original / "actual.bin"),
            str(emitted_format["element_bits"] // 8),
            str(rows),
            str(cols),
            str(int(emitted_format["kind"] == "int_affine" and emitted_format["signed"])),
        ],
        directory=tmp_path,
        stage="original_member_native_complete_publication",
        cwd=tmp_path,
        env=environment,
        inputs=(original / "actual.bin",),
        capture_output=True,
        timeout=30,
    )
    assert publication.returncode == 0, publication.stderr.decode()
    decoder, transported, done = OutB64Decoder(), {}, 0
    for line in publication.stdout.decode().splitlines():
        if line == "DONE":
            done += 1
        else:
            assert decoder.consume(line.split(), transported)
    decoder.require_closed()
    assert done == 1
    I.require_environment(next(tmp_path.rglob("invocation.json")), environment=environment)
    transported = decode_float_readback(transported, {"Y": dtype})
    assert member.compare_values(transported)["status"] == "pass"
    changed = list(values)
    changed[-1] += 1
    wrong = member.compare_values({"Y": changed})
    assert wrong["status"] == "fail" and wrong["original_reference"]["mismatches"][0]["index"] == len(values) - 1
    with pytest.raises(ValueError):
        member.compare_values({})
    work = original / "ordinary"
    assert (work / "model_host.o").read_bytes().startswith(b"\x7fELF")
    assert (work / "model_host.so").read_bytes().startswith(b"\x7fELF")
    assert list(work.rglob("*.mlir")) and list(work.glob("llvm-dialect-*"))
    assert native_rows[case["name"]]["comparison"]["passed"]
    (tmp_path / "original-comparison.json").write_bytes(canonical_json(wrong) + b"\n")


def test_opaque_ciface_pointer_arity_is_not_a_flat_tensor_storage_proof(ordinary_pointwise, tmp_path):
    """Keep the actual representation gap visible; no new adapter or authority."""
    owner, _ = ordinary_pointwise
    llvm = next((owner / "round_tail" / "ordinary").glob("llvm-dialect-*/*.mlir"))
    # Actual C interface has one input/output descriptor pointer. The existing
    # entry gate establishes opaque pointer arity, not the pointed-to storage.
    signature = "UNKNOWN"
    try:
        require_pointer_entry(llvm.read_text(), entry_symbol="_mlir_ciface_forward", pointer_arity=2)
        signature = "opaque_pointer_arity_only"
    except ValueError as error:
        assert str(error) == "emitted entry has malformed LLVM pointer ABI IR"
    text = (owner / "round_tail" / "ordinary/model.ll").read_text()
    assert "define void @_mlir_ciface_forward(ptr" in text and "getelementptr" in text
    assert "load {" in text and "extractvalue {" in text
    (tmp_path / "storage-gap.json").write_text(
        json.dumps(
            {
                "entry_symbol": "_mlir_ciface_forward",
                "pointer_arity": 2,
                "existing_pointer_signature_reader": signature,
                "descriptor_to_flat_tensor_storage_correspondence": "UNKNOWN",
                "target_runtime_and_stage_authority": "UNKNOWN",
            },
            sort_keys=True,
        )
        + "\n"
    )

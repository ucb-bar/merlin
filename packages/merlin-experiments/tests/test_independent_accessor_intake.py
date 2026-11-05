"""Native source accessor replay, drift and authority exclusion controls."""

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from merlin_experiments.phase0 import accessor_intake as A
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal

from merlin.common import invocation_record


@pytest.fixture
def sources(tmp_path):
    selected = shutil.which("g++")
    if not selected:
        pytest.skip("actual native C++ compiler not installed")
    compiler = Path(selected).resolve()
    checkout = tmp_path / "public"
    checkout.mkdir()
    header = checkout / "accessor.h"
    header.write_text(
        """#include <cstdint>
#define SELECTED_CONSTANT 57
struct PublicWord { uint64_t word; PublicWord(uint64_t w):word(w) {}
 int length() {return (word & 1) ? 4 : 2;} };
struct PublicFields { uint64_t first:5; uint64_t second:7; uint64_t unused:52; };
union PublicUnion { PublicWord bits; PublicFields fields;
 PublicUnion():bits(0) {} };
"""
    )
    for args in (
        ("init", "-q"),
        ("config", "user.name", "native-test"),
        ("config", "user.email", "native-test@example.invalid"),
        ("remote", "add", "origin", "https://example.invalid/public-source"),
        ("add", "accessor.h"),
        ("commit", "-qm", "independent native accessor fixture"),
    ):
        subprocess.run(["git", "-C", str(checkout), *args], check=True)
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    includes = tmp_path / "includes"
    includes.mkdir()
    shutil.copyfile(header, includes / header.name)
    spec = tmp_path / "minimal-accessor-spec.json"
    spec.write_text(
        json.dumps(
            {
                "header": "accessor.h",
                "union_type": "PublicUnion",
                "word_member": "bits",
                "fields_member": "fields",
                "conversion_type": "PublicWord",
                "length_member": "length",
                "fields": ["first", "second"],
                "constants": ["SELECTED_CONSTANT"],
            }
        )
    )
    return dict(
        public_checkout=checkout,
        commit=commit,
        include_root=includes,
        native_compiler=compiler,
        reviewed_spec=spec,
        forbidden_roots=(),
        output_root=tmp_path / "observed",
    )


def test_actual_native_basis_and_fresh_decode_invocation(sources, tmp_path):
    authority = A.issue_independent_accessor_intake(**sources)
    facts = authority.public_facts()
    assert facts["fields"] == {"first": {"offset": 0, "width": 5}, "second": {"offset": 5, "width": 7}}
    assert facts["source_constants"] == {"SELECTED_CONSTANT": 57}
    assert facts["observed_words"] == 130
    assert "physical_cpu_encoding_and_byte_order" in facts["unknowns"]
    words = [0, 1, 64, 130, (1 << 64) - 1]
    observed = authority.decode_words(words, tmp_path / "actual-calls")
    assert [row["word"] for row in observed["rows"]] == words
    assert [row["length"] for row in observed["rows"]] == [2, 4, 2, 2, 4]
    assert observed["rows"][3]["second"] == 4
    assert Path(observed["receipt_path"]).is_file()
    assert (tmp_path / "actual-calls" / "input.txt").read_text() == "".join(str(w) + "\n" for w in words)
    invocation_files = list((tmp_path / "actual-calls" / "invocations").rglob("invocation.json"))
    assert len(invocation_files) == 1
    actual_call = invocation_record.verify(invocation_files[0])
    assert actual_call["stage"] == "native_accessor_decode"
    assert actual_call["argv"] == [str(authority.executable)]
    assert actual_call["inputs"][0]["path"] == str(tmp_path / "actual-calls" / "input.txt")
    stdout = Path(actual_call["stdout"]["path"])
    assert stdout.read_bytes().startswith(b"META ")
    assert list((tmp_path / "actual-calls").rglob("stdout*")) == [stdout]
    facts["fields"]["first"]["offset"] = 63
    assert authority.public_facts()["fields"]["first"]["offset"] == 0
    authority.verify_public_facts(sources["output_root"] / "facts.json")


def test_saved_records_or_copied_object_cannot_mint_authority(sources):
    authority = A.issue_independent_accessor_intake(**sources)
    copied = replace(authority)
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        copied.verify()
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        copied.decode_words([0], sources["output_root"].parent / "laundered")


@pytest.mark.parametrize("role", ["binary", "header", "fact", "issuer_spec"])
def test_changes_to_actual_selected_inputs_or_outputs_refuse(sources, role):
    authority = A.issue_independent_accessor_intake(**sources)
    path = {
        "binary": authority.executable,
        "header": sources["include_root"] / "accessor.h",
        "fact": sources["output_root"] / "facts.json",
        "issuer_spec": sources["reviewed_spec"],
    }[role]
    path.write_bytes(path.read_bytes() + b"\nchanged")
    with pytest.raises(RtlIntakeRefusal, match="changed"):
        authority.verify()


def test_tracked_source_drift_refuses_before_native_compilation(sources):
    header = sources["public_checkout"] / "accessor.h"
    header.write_text(header.read_text() + "\n// changed")
    with pytest.raises(RtlIntakeRefusal, match="tracked modifications"):
        A.issue_independent_accessor_intake(**sources)
    assert not sources["output_root"].exists()


def test_mismatching_selected_sdk_header_refuses_before_native_compilation(sources):
    header = sources["include_root"] / "accessor.h"
    header.write_text(header.read_text().replace("SELECTED_CONSTANT 57", "SELECTED_CONSTANT 59"))
    with pytest.raises(RtlIntakeRefusal, match="before compilation"):
        A.issue_independent_accessor_intake(**sources)
    assert not sources["output_root"].exists()


def test_forbidden_path_refuses_without_reading_spec(sources, tmp_path):
    forbidden = tmp_path / "not-admitted"
    forbidden.mkdir()
    sources["reviewed_spec"] = forbidden / "does-not-exist.json"
    sources["forbidden_roots"] = (forbidden,)
    with pytest.raises(RtlIntakeRefusal, match="protected implementation"):
        A.issue_independent_accessor_intake(**sources)


def test_spec_cannot_supply_instruction_values_or_arbitrary_native_code(sources):
    path = sources["reviewed_spec"]
    spec = json.loads(path.read_text())
    spec["opcode"] = 57
    path.write_text(json.dumps(spec))
    with pytest.raises(RtlIntakeRefusal, match="exactly"):
        A.issue_independent_accessor_intake(**sources)
    spec.pop("opcode")
    spec["fields"] = ["first; return 0"]
    path.write_text(json.dumps(spec))
    with pytest.raises(RtlIntakeRefusal, match="expressions or code"):
        A.issue_independent_accessor_intake(**sources)


def test_transformed_native_word_constructor_cannot_become_bitfield_authority(sources):
    checkout = sources["public_checkout"]
    header = checkout / "accessor.h"
    header.write_text(header.read_text().replace("word(w)", "word(w ^ 1)"))
    subprocess.run(["git", "-C", str(checkout), "commit", "-qam", "different source semantics"], check=True)
    sources["commit"] = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    shutil.copyfile(header, sources["include_root"] / "accessor.h")
    with pytest.raises(RtlIntakeRefusal, match="zero-based bit projection"):
        A.issue_independent_accessor_intake(**sources)


def test_decode_cannot_write_to_public_or_forbidden_source(sources, tmp_path):
    forbidden = tmp_path / "private"
    forbidden.mkdir()
    sources["forbidden_roots"] = (forbidden,)
    authority = A.issue_independent_accessor_intake(**sources)
    with pytest.raises(RtlIntakeRefusal, match="protected implementation"):
        authority.decode_words([0], forbidden / "calls")
    with pytest.raises(RtlIntakeRefusal, match="public source"):
        authority.decode_words([0], sources["public_checkout"] / "calls")

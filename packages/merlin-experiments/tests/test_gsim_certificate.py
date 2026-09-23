"""The installed producer identifies FIRRTL headers without textual regex guessing."""

import pytest
from merlin_experiments.phase2 import gsim_certificate as producer


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("circuit Main :\n", "Main"),
        (" \tcircuit\t_Main9$ : @[source 1:2]\n", "_Main9$"),
        ("circuit A: trailing metadata", "A"),
        ("\u2003circuit\u2003A\u2003:\n", "A"),
        ("circuit 9Main :", None),
        ("circuit $Main :", None),
        ("circuit Máin :", None),
        ("circuit Main-1 :", None),
        ("circuit Main Other :", None),
        ("circuitMain :", None),
        ("circuit Main", None),
        ("circuit :", None),
        ("; circuit Main :", None),
        ("module Main :", None),
        ("circuit", None),
        ("", None),
    ],
)
def test_circuit_header_grammar(tmp_path, line, expected):
    source = tmp_path / "model.fir"
    source.write_text(line)
    assert producer._firrtl_circuit(source) == expected


@pytest.mark.parametrize(("line_number", "expected"), [(32, "Last"), (33, None)])
def test_circuit_scan_stops_after_32_lines(tmp_path, line_number, expected):
    source = tmp_path / "model.fir"
    source.write_text("; comment\n" * (line_number - 1) + "circuit Last :\n")
    assert producer._firrtl_circuit(source) == expected


def test_circuit_returns_first_valid_header_after_invalid_input(tmp_path):
    source = tmp_path / "model.fir"
    source.write_bytes(b"\xffinvalid\ncircuit 4Bad :\ncircuit First :\ncircuit Second :\n")
    assert producer._firrtl_circuit(source) == "First"


def test_installed_cli_seals_only_explicit_model_sources(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "Top.h").write_text("// generated header\n")
    (model / "unselected.h").write_text("// not part of this model\n")
    output = tmp_path / "model_manifest.json"
    assert (
        producer.main(["model-manifest", "--model-root", str(model), "--file", "Top.h", "--output", str(output)]) == 0
    )
    manifest = producer.validate_model_manifest(output)
    assert [row["path"] for row in manifest["files"]] == ["Top.h"]

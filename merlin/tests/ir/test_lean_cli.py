"""The public Lean command must replay actual proofs and distinguish abstention from success."""

import json

import pytest

from merlin.verify.cli import main

PROGRAM = """func.func @subject(%a: i32, %b: i32) -> i32 {
  %r = arith.addi %a, %b : i32
  return %r : i32
}
"""


def test_cli_checks_and_replays_chain(tmp_path, lean_tools, capsys):
    stage = tmp_path / "stage"
    stage.mkdir()
    before = stage / "before.mlir"
    before.write_text(PROGRAM)
    (stage / "after.mlir").write_text(PROGRAM.replace("addi %a, %b", "addi %b, %a"))
    (stage / "meta.json").write_text('{"boundary": "integer_commutation"}')
    compiler = ["--compiler-python", str(lean_tools.m2m_python)]

    def run(arguments):
        status = main(["lean", *arguments, *compiler])
        return status, json.loads(capsys.readouterr().out)

    # Both universal and typed entrypoints must invoke the real Lean checker.
    for command in ("stage", "typed-stage"):
        status, receipt = run([command, str(stage)])
        assert status == 0 and receipt["verdict"] == "verified"
        assert ("index_bits" in receipt) == (command == "typed-stage")
    for command in ("support", "definedness"):
        status, receipt = run(
            [
                command,
                str(before),
                "--function",
                "subject",
                "--work",
                str(tmp_path / command),
            ]
        )
        assert status == 0 and receipt["verdict"] == "verified"
    chain = [
        "chain",
        str(stage),
        "--function",
        "subject",
        "--support",
        str(tmp_path / "support/receipt.json"),
        "--output",
        str(tmp_path / "chain.json"),
    ]
    status, conditional = run(chain)
    assert status == 2 and conditional["verdict"] != "verified"
    assert conditional["refinement_verdict"] == "verified"
    chain += ["--definedness", str(tmp_path / "definedness/receipt.json")]
    status, defined = run(chain)
    assert status == 0 and defined["verdict"] == "verified"

    # A saved green receipt must not qualify different live compiler output.
    (stage / "after.mlir").write_text(PROGRAM.replace("arith.addi", "arith.subi"))
    status, stale = run(chain)
    assert status == 2 and stale["verdict"] != "verified"
    assert json.loads((tmp_path / "chain.json").read_text()) == stale


def test_cli_missing_tool_replaces_old_success(tmp_path, capsys):
    (tmp_path / "receipt.json").write_text('{"verdict": "verified"}')
    status = main(
        [
            "lean",
            "support",
            str(tmp_path / "missing.mlir"),
            "--function",
            "subject",
            "--work",
            str(tmp_path),
            "--compiler-python",
            str(tmp_path / "missing-python"),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert status == 2 and result["verdict"] == "unmeasured"
    assert json.loads((tmp_path / "receipt.json").read_text()) == result


@pytest.mark.parametrize("width", ["0", "-1", "abc"])
def test_cli_rejects_invalid_domain_before_checking(width):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "lean",
                "typed-stage",
                "missing-stage",
                "--compiler-python",
                "missing-python",
                "--index-bits",
                width,
            ]
        )
    assert error.value.code == 2

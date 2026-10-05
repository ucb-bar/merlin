"""The source census detects disagreements without certifying instruction legality."""

import json
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.cli import main as targetgen_main
from merlin.targetgen.isa_census import (
    _model_classes,
    _patterns,
    derive_source_census,
)

_RTL_REVISION = "0079c0541111197741a231c002e3843fa6f545b2"
_CONTROLS = ", ".join(["N"] * 17)


def _pattern(*, opcode: int, funct7: int | None = None, funct3: int | None = None, vi_mode: int | None = None) -> str:
    mask = 0x7F
    value = opcode
    if funct7 is not None:
        mask |= 0x7F << 25
        value |= funct7 << 25
    if funct3 is not None:
        mask |= 0x7 << 12
        value |= funct3 << 12
    if vi_mode is not None:
        mask |= 0x7 << 13
        value |= vi_mode << 13
    return "".join(str((value >> bit) & 1) if (mask >> bit) & 1 else "?" for bit in reversed(range(32)))


def _write_sources(
    tmp_path: Path, patterns: dict[str, str], model_source: str, decoded: tuple[str, ...] | None = None
) -> dict[str, Path]:
    pattern_file = tmp_path / "Instructions.scala"
    decoder_file = tmp_path / "IDecode.scala"
    model_isa_file = tmp_path / "isa_definition.py"
    pattern_file.write_text("\n".join(f'  def {name} = BitPat("b{bits}")' for name, bits in patterns.items()))
    decoder_file.write_text(
        "val table:\n" + "\n".join(f"  {name} -> List({_CONTROLS})," for name in (decoded or tuple(patterns)))
    )
    model_isa_file.write_text(model_source)
    return {"pattern_file": pattern_file, "decoder_file": decoder_file, "model_isa_file": model_isa_file}


def test_vi_mode_uses_the_vi_encoding_not_rv32_funct3(tmp_path: Path) -> None:
    files = _write_sources(
        tmp_path,
        {"VLI_ROW": _pattern(opcode=0x5F, vi_mode=1)},
        "class VLI_ROW(VIType, opcode=0b1011111, funct3=0b001):\n    def exec(self): pass\n",
    )
    census = derive_source_census(**files, rtl_revision=_RTL_REVISION)
    row = census["rows"][0]
    assert row["model_candidates"][0]["name"] == "VLI_ROW"
    assert row["model_candidates"][0]["relation"] == "keyword_encoding_implies_pattern"
    assert census["summary"]["model_classes_without_compatible_pattern"] == []


def test_census_preserves_decode_gaps_overlaps_and_dma_kind_conflicts(tmp_path: Path) -> None:
    files = _write_sources(
        tmp_path,
        {
            "DMA_CONFIG_ANY": _pattern(opcode=0x7F, funct7=0),
            "DMA_WAIT_ANY": _pattern(opcode=0x7F, funct7=1),
            "ALIAS": _pattern(opcode=0x7F, funct7=0),
        },
        "class DMA_CONFIG_CH0(DMAType, opcode=0x7f, funct7=1): pass\n",
        decoded=("DMA_CONFIG_ANY", "DMA_WAIT_ANY", "UNDEFINED"),
    )
    census = derive_source_census(**files, rtl_revision=_RTL_REVISION)
    summary = census["summary"]
    assert summary["patterns_not_decoded"] == ["ALIAS"]
    assert summary["decoder_rows_without_pattern"] == ["UNDEFINED"]
    assert summary["overlapping_patterns"] == [["DMA_CONFIG_ANY", "ALIAS"]]
    assert summary["dma_kind_conflicts"] == [["DMA_WAIT_ANY", "DMA_CONFIG_CH0"]]
    assert census["rows"][0]["model_candidates"] == []
    assert census["rows"][1]["model_candidates"][0]["name"] == "DMA_CONFIG_CH0"
    assert census["rows"][0]["qualification"]["execution"] == "unreviewed"


def test_census_rejects_unpinned_or_malformed_source(tmp_path: Path) -> None:
    files = _write_sources(
        tmp_path,
        {"X": _pattern(opcode=1)},
        "class X(RType, opcode=1): pass\n",
    )
    with pytest.raises(ValueError, match="exact selected RTL commit"):
        derive_source_census(**files, rtl_revision="main")
    files["pattern_file"].write_text('def X = BitPat("b101")\n')
    with pytest.raises(ValueError, match="invalid or duplicate 32-bit"):
        derive_source_census(**files, rtl_revision=_RTL_REVISION)
    files["pattern_file"].write_text(f'def X = BitPat("b{_pattern(opcode=1)}")\n')
    files["pattern_file"].write_text(files["pattern_file"].read_text() + 'def MALFORMED = BitPat("b101"\n')
    with pytest.raises(ValueError, match="unparsed BitPat definition"):
        derive_source_census(**files, rtl_revision=_RTL_REVISION)
    files["pattern_file"].write_text(f'def X = BitPat("b{_pattern(opcode=1)}")\n')
    files["decoder_file"].write_text("val table:\nX -> List(A, B)\n")
    with pytest.raises(ValueError, match="17-column"):
        derive_source_census(**files, rtl_revision=_RTL_REVISION)
    files["decoder_file"].write_text(f"val table:\nX -> List({_CONTROLS}),\nY -> List(\n")
    with pytest.raises(ValueError, match="unparsed or duplicate row"):
        derive_source_census(**files, rtl_revision=_RTL_REVISION)


def test_source_revision_verification_binds_both_git_objects(tmp_path: Path) -> None:
    rtl = tmp_path / "rtl"
    model = tmp_path / "model"
    rtl.mkdir()
    model.mkdir()
    files = _write_sources(
        rtl, {"X": _pattern(opcode=1)}, "class X(RType, opcode=1): pass\n",
    )
    model_file = model / "isa_definition.py"
    model_file.write_bytes(files["model_isa_file"].read_bytes())
    files["model_isa_file"].unlink()
    files["model_isa_file"] = model_file

    def commit(root: Path) -> str:
        for args in (["init", "-q"], ["add", "."], [
            "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "selected source",
        ]):
            subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    rtl_revision = commit(rtl)
    model_revision = commit(model)
    census = derive_source_census(
        **files, rtl_revision=rtl_revision, model_revision=model_revision,
        verify_revisions=True,
    )
    assert census["source_revision_verification"]["status"] == "verified"
    files["decoder_file"].write_text(files["decoder_file"].read_text() + "// changed\n")
    with pytest.raises(ValueError, match="bytes differ"):
        derive_source_census(
            **files, rtl_revision=rtl_revision, model_revision=model_revision,
            verify_revisions=True,
        )
    files["decoder_file"].write_bytes(subprocess.run(
        ["git", "-C", str(rtl), "show", f"{rtl_revision}:IDecode.scala"],
        check=True, capture_output=True,
    ).stdout)
    with pytest.raises(ValueError, match="HEAD differs"):
        derive_source_census(
            **files, rtl_revision="a" * 40, model_revision=model_revision,
            verify_revisions=True,
        )


def test_installed_targetgen_audit_writes_status_and_removes_stale_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = _write_sources(tmp_path, {"X": _pattern(opcode=1)}, "class X(RType, opcode=1): pass\n")
    output = tmp_path / "audit" / "census.json"
    args = [
        "audit-isa",
        "--patterns",
        str(files["pattern_file"]),
        "--decoder",
        str(files["decoder_file"]),
        "--model-isa",
        str(files["model_isa_file"]),
        "--rtl-revision",
        _RTL_REVISION,
        "--out",
        str(output),
    ]
    assert targetgen_main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "SOURCE_CROSSWALK_ONLY"
    assert json.loads(output.read_text())["summary"]["patterns"] == 1
    files["model_isa_file"].write_text("class Y(RType, opcode=2): pass\n")
    assert targetgen_main(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "SOURCE_DISCREPANCIES"
    assert json.loads(output.read_text())["summary"]["model_classes_without_compatible_pattern"] == ["Y"]
    files["pattern_file"].write_text('def X = BitPat("bnotbits")\n')
    assert targetgen_main(args) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"
    assert not output.exists()


def test_curated_source_keeps_known_rtl_model_disagreements_visible() -> None:
    contract = repo_root() / "examples/atlas/phase1/contracts/hwbringup_atlas_v0"
    patterns = _patterns((contract / "rtl/atlas/scalar/Instructions.scala").read_text())
    models = _model_classes((contract / "isa_include/isa_definition.py").read_text())
    assert len(patterns) == 99
    assert len(models) == 127
    for name in ("CSRRCI", "VSQUARE_BF16", "VCUBE_BF16"):
        pattern, model = patterns[name], models[name]
        assert (pattern["value"] ^ model["value"]) & pattern["mask"] & model["mask"]
    assert models["VLI_ROW"]["value"] & (0x7 << 13) == 1 << 13

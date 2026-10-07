"""The source census detects disagreements without certifying instruction legality."""

import json
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import ext_path, repo_root
from merlin.targetgen.cli import main as targetgen_main
from merlin.targetgen.isa_census import (
    UNKNOWN,
    _model_classes,
    _patterns,
    derive_format_layouts,
    derive_source_census,
)

_RTL_REVISION = "0079c0541111197741a231c002e3843fa6f545b2"
_CONTROLS = ", ".join(["N"] * 17)

#: A synthetic model's instruction-format module: each format's ENCODER is what places its keyword
#: fields, and the census reads the positions from it (test data, not a real target's layout).
_FORMATS = """
def _mask(val, bits):
    return val & ((1 << bits) - 1)

class Instruction:
    def __init_subclass__(cls, **kw):
        pass

class RType(Instruction):
    def to_bytecode(self):
        funct7_b = _mask(self.funct7, 7)
        rs2_b = _mask(self.rs2, 5)
        funct3_b = _mask(self.funct3, 3)
        opcode_b = _mask(self.opcode, 7)
        return (funct7_b << 25) | (rs2_b << 20) | (funct3_b << 12) | opcode_b

class DMAType(RType):
    pass

class VIType(Instruction):
    def to_bytecode(self):
        imm_b = _mask(self.imm, 16)
        funct3_b = _mask(self.funct3, 3)
        opcode_b = _mask(self.opcode, 7)
        return (imm_b << 16) | (funct3_b << 13) | opcode_b
"""


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
    format_file = tmp_path / "formats.py"
    pattern_file.write_text("\n".join(f'  def {name} = BitPat("b{bits}")' for name, bits in patterns.items()))
    decoder_file.write_text(
        "val table:\n" + "\n".join(f"  {name} -> List({_CONTROLS})," for name in (decoded or tuple(patterns)))
    )
    model_isa_file.write_text(model_source)
    format_file.write_text(_FORMATS)
    return {
        "pattern_file": pattern_file,
        "decoder_file": decoder_file,
        "model_isa_file": model_isa_file,
        "format_file": format_file,
    }


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


def test_field_positions_are_read_from_each_formats_own_encoder() -> None:
    layouts = derive_format_layouts(_FORMATS)
    assert layouts["RType"]["funct7"] == (25, 7) and layouts["RType"]["opcode"] == (0, 7)
    assert layouts["VIType"]["funct3"] == (13, 3)  # the same keyword, where THIS format's encoder puts it
    assert layouts["DMAType"] == layouts["RType"]  # no encoder of its own: it encodes as the format it extends
    ambiguous = derive_format_layouts(
        "class Twice:\n    def enc(self):\n        a = m(self.f, 2)\n        return (a << 3) | (a << 9)\n"
    )
    assert ambiguous == {}  # a field placed at two positions is dropped, never guessed between


def test_a_class_whose_format_cannot_be_derived_is_unknown_and_matched_against_nothing(tmp_path: Path) -> None:
    """MUTATION: remove the format module. No assumed RISC-V layout stands in for it."""
    files = _write_sources(tmp_path, {"X": _pattern(opcode=1)}, "class X(RType, opcode=1): pass\n")
    files["format_file"].unlink()
    files.pop("format_file")
    census = derive_source_census(**files, rtl_revision=_RTL_REVISION)
    assert census["summary"]["model_classes_without_derived_layout"] == ["X"]
    assert census["summary"]["model_classes_without_compatible_pattern"] == []
    assert census["rows"][0]["model_candidates"] == []
    assert census["sources"]["formats"]["status"] == UNKNOWN
    unknown_format = _model_classes("class Y(Mixin, opcode=3): pass\n", derive_format_layouts(_FORMATS))
    assert unknown_format["Y"]["layout"] == UNKNOWN and unknown_format["Y"]["mask"] is None


def test_the_format_module_is_found_beside_the_model_from_its_own_import(tmp_path: Path) -> None:
    (tmp_path / "tree" / "configs").mkdir(parents=True)
    files = _write_sources(
        tmp_path / "tree" / "configs",
        {"X": _pattern(opcode=0x33, funct7=1, funct3=2)},
        "from fmtpkg.formats import RType\n\nclass X(RType, opcode=0x33, funct3=2, funct7=1): pass\n",
    )
    files.pop("format_file").unlink()
    (tmp_path / "tree" / "fmtpkg").mkdir()
    (tmp_path / "tree" / "fmtpkg" / "formats.py").write_text(_FORMATS)
    census = derive_source_census(**files, rtl_revision=_RTL_REVISION)
    assert census["rows"][0]["model_candidates"][0]["relation"] == "keyword_encoding_implies_pattern"
    assert census["sources"]["formats"][0]["path"] == str(tmp_path / "tree" / "fmtpkg" / "formats.py")


def test_source_revision_verification_binds_both_git_objects(tmp_path: Path) -> None:
    rtl = tmp_path / "rtl"
    model = tmp_path / "model"
    rtl.mkdir()
    model.mkdir()
    files = _write_sources(
        rtl,
        {"X": _pattern(opcode=1)},
        "class X(RType, opcode=1): pass\n",
    )
    # The model ISA and the format module it is encoded with are both the model's own sources.
    for key, name in (("model_isa_file", "isa_definition.py"), ("format_file", "formats.py")):
        moved = model / name
        moved.write_bytes(files[key].read_bytes())
        files[key].unlink()
        files[key] = moved

    def commit(root: Path) -> str:
        for args in (
            ["init", "-q"],
            ["add", "."],
            [
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "selected source",
            ],
        ):
            subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    rtl_revision = commit(rtl)
    model_revision = commit(model)
    census = derive_source_census(
        **files,
        rtl_revision=rtl_revision,
        model_revision=model_revision,
        verify_revisions=True,
    )
    verification = census["source_revision_verification"]
    assert verification["status"] == "verified"
    assert [row["path_at_revision"] for row in verification["formats"]] == ["formats.py"]
    files["format_file"].write_text(files["format_file"].read_text() + "# changed\n")
    with pytest.raises(ValueError, match="bytes differ"):
        derive_source_census(
            **files,
            rtl_revision=rtl_revision,
            model_revision=model_revision,
            verify_revisions=True,
        )
    files["format_file"].write_bytes(
        subprocess.run(
            ["git", "-C", str(model), "show", f"{model_revision}:formats.py"],
            check=True,
            capture_output=True,
        ).stdout
    )
    files["decoder_file"].write_text(files["decoder_file"].read_text() + "// changed\n")
    with pytest.raises(ValueError, match="bytes differ"):
        derive_source_census(
            **files,
            rtl_revision=rtl_revision,
            model_revision=model_revision,
            verify_revisions=True,
        )
    files["decoder_file"].write_bytes(
        subprocess.run(
            ["git", "-C", str(rtl), "show", f"{rtl_revision}:IDecode.scala"],
            check=True,
            capture_output=True,
        ).stdout
    )
    with pytest.raises(ValueError, match="HEAD differs"):
        derive_source_census(
            **files,
            rtl_revision="a" * 40,
            model_revision=model_revision,
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
        "--formats",
        str(files["format_file"]),
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
    model_text = (contract / "isa_include/isa_definition.py").read_text()
    assert len(patterns) == 99
    # Without the model's format module every class is still enumerated, and none is positioned.
    unplaced = _model_classes(model_text, {})
    assert len(unplaced) == 127 and all(m["layout"] == UNKNOWN for m in unplaced.values())
    try:
        formats = ext_path("npu_model") / "npu_model" / "isa.py"
    except KeyError:
        pytest.skip("the model's instruction-format module is not available on this host")
    if not formats.is_file():
        pytest.skip(f"the model's instruction-format module is not at {formats}")
    models = _model_classes(model_text, derive_format_layouts(formats.read_text()))
    assert len(models) == 127 and all(m["layout"] == "derived" for m in models.values())
    for name in ("CSRRCI", "VSQUARE_BF16", "VCUBE_BF16"):
        pattern, model = patterns[name], models[name]
        assert (pattern["value"] ^ model["value"]) & pattern["mask"] & model["mask"]
    assert models["VLI_ROW"]["value"] & (0x7 << 13) == 1 << 13

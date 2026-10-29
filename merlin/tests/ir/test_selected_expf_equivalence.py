"""Default-off callee selection, live byte checks and actual link ownership guards."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from merlin.llvmlower import selected_expf_equivalence as E


@pytest.fixture
def grant(tmp_path, monkeypatch):
    library = tmp_path / "libm.a"
    library.write_bytes(b"reviewed archive fixture")
    import hashlib

    monkeypatch.setattr(E, "SELECTED_LIBM_SHA256", hashlib.sha256(library.read_bytes()).hexdigest())
    return E.select_expf_underflow_equivalence(library)


def test_unselected_and_nested_scope(grant):
    assert E.current_expf_underflow_equivalence() is None
    with E.selected_expf_underflow_scope(grant):
        assert E.current_expf_underflow_equivalence() is grant
        with E.selected_expf_underflow_scope(grant):
            assert E.current_expf_underflow_equivalence() is grant
        assert E.current_expf_underflow_equivalence() is grant
    assert E.current_expf_underflow_equivalence() is None
    assert (grant.minimum_input_bits, grant.maximum_input_bits) == (0xC2CFF1B6, 0xFF7FFFFF)
    assert (grant.result_bits, grant.raised_fflags, grant.errno_value) == (0, 3, 34)
    with pytest.raises(FrozenInstanceError):
        grant.library_path = Path("elsewhere")


def test_live_archive_mutation_refuses_and_scope_cleans_up(grant):
    with pytest.raises(E.ExpfEquivalenceError, match="bytes changed"):
        with E.selected_expf_underflow_scope(grant):
            grant.library_path.write_bytes(b"changed")
            E.current_expf_underflow_equivalence()
    assert E.current_expf_underflow_equivalence() is None
    with pytest.raises(E.ExpfEquivalenceError):
        E.bind_expf_underflow_link(grant, [], map_path=Path("map"))


def test_foreign_and_wrong_library_refuse(tmp_path):
    library = tmp_path / "libm.a"
    library.write_bytes(b"unsupported")
    with pytest.raises(E.ExpfEquivalenceError):
        E.select_expf_underflow_equivalence(library)
    with pytest.raises(E.ExpfEquivalenceError):
        with E.selected_expf_underflow_scope({"library_sha256": E.SELECTED_LIBM_SHA256}):
            pass


def test_link_binds_exact_archive(grant, tmp_path):
    original = ["riscv-gcc", "-march=rv64gc", "-mabi=lp64d", "input.o", "-lm", "-lgcc", "-o", "out.elf"]
    linked = E.bind_expf_underflow_link(grant, original, map_path=tmp_path / "link.map")
    assert "-lm" in original and "-lm" not in linked
    assert str(grant.library_path) in linked
    assert linked[-1] == "-Wl,-Map=" + str(tmp_path / "link.map")


@pytest.mark.parametrize(
    "extra",
    [
        "-lm",
        "other/libm.a",
        "-Wl,--wrap=expf",
        "-Wl,--defsym=expf=foreign",
        "-Wl,-Map=old",
        "@response",
        "-shared",
        "-lmath",
    ],
)
def test_link_override_refuses(grant, tmp_path, extra):
    with pytest.raises(E.ExpfEquivalenceError):
        E.bind_expf_underflow_link(
            grant, ["gcc", "-march=rv64gc", "-mabi=lp64d", "-lm", extra], map_path=tmp_path / "new.map"
        )


def test_existing_map_and_wrong_abi_refuse(grant, tmp_path):
    p = tmp_path / "old.map"
    p.write_text("stale")
    for args, path in [(["gcc", "-lm"], tmp_path / "new.map"), (["gcc", "-march=rv64gc", "-mabi=lp64d", "-lm"], p)]:
        with pytest.raises(E.ExpfEquivalenceError):
            E.bind_expf_underflow_link(grant, args, map_path=path)


def _evidence(grant, tmp_path, *, foreign=None):
    elf = tmp_path / "out.elf"
    elf.write_bytes(b"\x7fELF\x02\x01" + bytes(12) + b"\xf3\x00")
    rows = []
    for index, (section, member) in enumerate(E._SECTIONS.items()):
        owner = f"{grant.library_path}({member})" if section != foreign else "other.o"
        # Exercise both one-line and wrapped GNU section formats.
        rows.append(f" {section}" + ("\n" if index % 2 else " ") + f" 0x80000000 0x10 {owner}")
    map_path = tmp_path / "out.map"
    map_path.write_text("\n".join(rows))
    return elf, map_path


def test_link_map_joins_all_callee_sections(grant, tmp_path):
    elf, path = _evidence(grant, tmp_path)
    result = E.verify_expf_underflow_link(grant, elf=elf, map_path=path)
    assert len(result["sections"]) == 6
    assert result["qualification"] == "UNQUALIFIED"


@pytest.mark.parametrize("section", list(E._SECTIONS))
def test_override_of_each_actual_callee_section_refuses(grant, tmp_path, section):
    elf, path = _evidence(grant, tmp_path, foreign=section)
    with pytest.raises(E.ExpfEquivalenceError, match="foreign owner"):
        E.verify_expf_underflow_link(grant, elf=elf, map_path=path)


def test_scope_exit_reopens_archive(grant):
    with pytest.raises(E.ExpfEquivalenceError, match="bytes changed"):
        with E.selected_expf_underflow_scope(grant):
            grant.library_path.write_bytes(b"changed at scope exit")
    assert E.current_expf_underflow_equivalence() is None


def test_conflicting_target_flag_refuses(grant, tmp_path):
    with pytest.raises(E.ExpfEquivalenceError):
        E.bind_expf_underflow_link(
            grant, ["gcc", "-march=rv64gc", "-march=rv64imac", "-mabi=lp64d", "-lm"], map_path=tmp_path / "new.map"
        )

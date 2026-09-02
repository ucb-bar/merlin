"""The harvester's machine-independence normalisation.

These objdump fixtures are TRACKED test data and the EXPERT side of every CCA comparison is lifted
from them, so a diff on one is supposed to mean the search target moved. objdump names the object by
its full path -- which lives in a temp dir -- so without normalisation every harvest rewrites all of
them with a new TMPDIR and zero instruction change.
"""

import importlib.util

from merlin.common.paths import repo_root

_spec = importlib.util.spec_from_file_location(
    "_harvest", repo_root() / "build_tools" / "scripts" / "harvest_xnnpack_fixtures.py")
_h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h)


def _header(path: str) -> str:
    return f"\n{path}:\tfile format elf64-littleriscv\n\nDisassembly of section .text:\n"


def test_the_temp_path_is_replaced_by_the_fixture_name():
    out = _h._machine_independent(_header("/scratch/agustin/tmp/merlin_harvest_ab12/x_rvv.o"), "x_rvv.objdump")
    assert "x_rvv.objdump:\tfile format elf64-littleriscv" in out
    assert "merlin_harvest_ab12" not in out, "the temp dir must not reach a tracked fixture"


def test_normalisation_does_not_depend_on_the_object_extension():
    """The regression this file exists for.

    The gate used to be ``sep and head.endswith(".o")``. Adding the link step to the harvester moved
    the object from ``.o`` to ``.so`` and silently turned the whole normalisation off -- no error, no
    skipped-fixture message, just churn again. Any extension objdump can be pointed at must normalise.
    """
    for ext in (".o", ".so", ".elf", ""):
        out = _h._machine_independent(_header(f"/scratch/agustin/tmp/t_9/x_rvv{ext}"), "x_rvv.objdump")
        assert "/scratch" not in out, f"normalisation silently skipped for a {ext or '(none)'} object"


def test_instruction_lines_are_untouched():
    text = _header("/tmp/q/x.so") + "  1c:\t02b7f0d7          \tvfmacc.vf\tv1, fa5, v11\n"
    out = _h._machine_independent(text, "x.objdump")
    assert "vfmacc.vf\tv1, fa5, v11" in out
    assert out.count("file format") == 1


def test_normalisation_is_idempotent():
    once = _h._machine_independent(_header("/tmp/q/x.so"), "x.objdump")
    assert _h._machine_independent(once, "x.objdump") == once

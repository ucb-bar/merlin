"""Link independent generated-style C modules with colliding private helpers."""

import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.compile.object_exports import localize_object_symbols


def test_local_definitions_link_without_changing_code_or_dependencies(tmp_path):
    tools = {name: shutil.which("llvm-" + name) for name in ("nm", "objcopy")}
    if not all(tools.values()):
        pytest.skip("select LLVM nm and objcopy")
    objects = []
    for name, addend in (("first", 7), ("second", 11)):
        source, original, localized = (tmp_path / (name + suffix) for suffix in (".c", ".o", "-local.o"))
        source.write_text(
            'extern void observe(int);\n'
            f'int region_000_bias={addend};\n'
            'int region_000(int x) { observe(x); return x+region_000_bias; }\n'
            f'int {name}_run(int x) {{ return region_000(x); }}\n'
        )
        subprocess.run([shutil.which("cc"), "-O0", "-c", str(source), "-o", str(original)], check=True)
        record = localize_object_symbols(original, localized, (name + "_run",),
                                         nm=Path(tools["nm"]), objcopy=Path(tools["objcopy"]))
        assert record["public_symbols"] == [name + "_run"]
        assert record["localized_symbols"] == ["region_000", "region_000_bias"]
        assert record["unresolved_symbols"] == ["observe"]
        for section in (".text", ".data"):
            for object_path in (original, localized):
                dump = object_path.with_suffix(section)
                subprocess.run([tools["objcopy"], "--dump-section=" + section + "=" + str(dump),
                                str(object_path)], check=True)
            assert original.with_suffix(section).read_bytes() == localized.with_suffix(section).read_bytes()
        objects.append(localized)
    fixture = tmp_path / "main.c"
    fixture.write_text("""#include <assert.h>
      int first_run(int); int second_run(int);
      static int calls;
      void observe(int x) { assert(x==3 || x==5); ++calls; }
      int main(void) { assert(first_run(3)==10 && second_run(5)==16 && calls==2); return 0; }
    """)
    failed = subprocess.run([shutil.which("cc"), str(fixture), str(tmp_path / "first.o"),
                             str(tmp_path / "second.o"), "-o", str(tmp_path / "bad")], capture_output=True)
    assert failed.returncode != 0
    subprocess.run([shutil.which("cc"), str(fixture), *map(str, objects), "-o", str(tmp_path / "run")], check=True)
    subprocess.run([str(tmp_path / "run")], check=True)
    with pytest.raises(ValueError, match="public symbol is absent"):
        localize_object_symbols(tmp_path / "first.o", tmp_path / "missing.o", ("absent",),
                                nm=Path(tools["nm"]), objcopy=Path(tools["objcopy"]))
    assert not (tmp_path / "missing.o").exists()

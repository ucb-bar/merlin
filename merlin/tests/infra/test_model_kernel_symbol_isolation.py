"""Separate kernel objects retain their own helpers and share only undefined runtime imports."""

import json
import shutil
import subprocess

import pytest

from merlin.model_check import cache


@pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("cc", "objcopy", "nm")),
    reason="requires a C compiler and object tools",
)
@pytest.mark.parametrize("tool_family", ["gnu", "selected"])
@pytest.mark.parametrize("builder", ["replay-cache", "whole-model"])
def test_auxiliary_definitions_do_not_collide_or_capture_runtime_imports(tmp_path, tool_family, builder):
    from merlin.llvmlower.device_build import _objcopy

    objcopy = shutil.which("objcopy") if tool_family == "gnu" else _objcopy()
    if objcopy is None:
        pytest.skip("selected object tool is unavailable")
    objects = []
    symbols = []
    for index in range(2):
        source = tmp_path / f"kernel{index}.c"
        source.write_text(
            "extern int shared_runtime_value(void);\n"
            f"int helper_state = {index + 1};\n"
            "int kernel_entry(void) { return helper_state; }\n"
            "void kernel_complete(void) {}\n"
            "int selected(void) { kernel_complete(); return kernel_entry() + helper_state + shared_runtime_value(); }\n"
        )
        raw = source.with_suffix(".o")
        subprocess.run(["cc", "-c", str(source), "-o", str(raw)], check=True)
        directory = tmp_path / f"stored{index}"
        if builder == "replay-cache":
            cache._store_object(directory, raw, "selected", objcopy, {"interface_sha256": f"input{index}"})
            record = json.loads((directory / "entry.json").read_text())
        else:
            from merlin.llvmlower.device_build import rename_object_command

            directory.mkdir()
            record = {"kernel_symbol": f"selected_{index}"}
            subprocess.run(rename_object_command(objcopy, "selected", record["kernel_symbol"],
                                                raw, directory / "kernel.o"), check=True)
        obj = directory / "kernel.o"
        exported = subprocess.run(
            ["nm", "--defined-only", "--extern-only", str(obj)], check=True, text=True, capture_output=True
        ).stdout
        assert {line.split()[-1] for line in exported.splitlines() if line.strip()} == {record["kernel_symbol"]}
        undefined = subprocess.run(["nm", "--undefined-only", str(obj)], check=True, text=True, capture_output=True).stdout
        assert "shared_runtime_value" in undefined
        objects.append(str(obj))
        symbols.append(record["kernel_symbol"])
    driver = tmp_path / "main.c"
    driver.write_text(
        "int shared_runtime_value(void) { return 7; }\n"
        + "\n".join(f"extern int {symbol}(void);" for symbol in symbols)
        + f"\nint main(void) {{ return {symbols[0]}() == 9 && {symbols[1]}() == 11 ? 0 : 1; }}\n"
    )
    image = tmp_path / "linked"
    subprocess.run(["cc", str(driver), *objects, "-o", str(image)], check=True)
    subprocess.run([str(image)], check=True)

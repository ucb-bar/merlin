"""Constant sharing preserves values and refuses observable identity or mutation."""

import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.llvmlower.immutable_constant_pool import ACCESS, PoolOperand, plan_immutable_constant_pool


def operand(tmp_path, group, symbol, raw=b"\x03\x07\x0b", **changes):
    path = tmp_path / (group + symbol + ".bin")
    path.write_bytes(raw)
    return replace(PoolOperand(group, symbol, path, 8, "unsigned_bytes", ACCESS), **changes)


def plan(values):
    return plan_immutable_constant_pool(values, symbol_prefix="shared_", reserved_symbols={"shared_0"})


def test_different_symbols_share_complete_values_and_stronger_alignment(tmp_path):
    a = operand(tmp_path, "first", "left")
    b = operand(tmp_path, "second", "right", alignment=64)
    result = plan([a, b])
    assert len(result.blobs) == 1 and result.blobs[0].alignment == 64
    assert result.symbol_maps() == {"first": {"left": "shared_1"}, "second": {"right": "shared_1"}}
    assert result.receipt()["bindings"][1]["alignment"] == 64


def test_equal_samples_do_not_merge_different_complete_arrays_or_representations(tmp_path):
    a = operand(tmp_path, "a", "first", b"\x03\x07\x0b")
    b = operand(tmp_path, "b", "second", b"\x03\x07\x0d")
    c = operand(tmp_path, "c", "third", representation="signed_bytes")
    assert len(plan([a, b, c]).blobs) == 3


@pytest.mark.parametrize("access", ["", "readonly_bytes", "identity_observed", "writable"])
def test_unknown_or_observable_address_contracts_refuse(tmp_path, access):
    with pytest.raises(ValueError, match="unsafe operand contracts"):
        plan([operand(tmp_path, "stage", "constant", access=access)])


def test_duplicate_selected_binding_refuses_even_with_same_contents(tmp_path):
    a = operand(tmp_path, "stage", "data")
    with pytest.raises(ValueError, match="duplicate"):
        plan([a, a])


def test_changed_duplicate_is_reopened_not_trusted_from_representative(tmp_path):
    a, b = operand(tmp_path, "a", "one"), operand(tmp_path, "b", "two")
    result = plan([a, b])
    b.path.write_bytes(b"\x03\x07\x0c")
    with pytest.raises(ValueError, match="binding changed"):
        result.recheck()


def test_replacing_duplicate_with_symlink_refuses(tmp_path):
    a, b = operand(tmp_path, "a", "one"), operand(tmp_path, "b", "two")
    result = plan([a, b])
    b.path.unlink()
    b.path.symlink_to(a.path)
    with pytest.raises(ValueError, match="regular"):
        result.recheck()


def test_hash_collision_does_not_authorize_storage_alias(tmp_path, monkeypatch):
    from merlin.llvmlower import immutable_constant_pool

    a, b = operand(tmp_path, "a", "one"), operand(tmp_path, "b", "two", raw=b"\x03\x07\x0c")
    monkeypatch.setattr(immutable_constant_pool, "_digest", lambda path: "collision")
    with pytest.raises(ValueError, match="byte equality"):
        plan([a, b])


def test_native_link_keeps_two_callers_and_private_tail_values(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native C compiler unavailable")
    values = [operand(tmp_path, "alpha", "x"), operand(tmp_path, "beta", "x"),
              operand(tmp_path, "beta", "tail", b"\x05\x09")]
    result = plan(values)
    assembly = [".section .rodata"]
    for blob in result.blobs:
        assembly.extend([f".balign {blob.alignment}", f".global {blob.symbol}", f"{blob.symbol}:",
                         f'.incbin "{blob.path}"'])
    source = tmp_path / "pool.S"
    source.write_text("\n".join(assembly) + "\n")
    objects = []
    for group, mapping in result.symbol_maps().items():
        declarations = "\n".join(f"extern const unsigned char {name}[];" for name in mapping.values())
        x = mapping["x"]
        tail = f" + {mapping['tail']}[0] + {mapping['tail']}[1]" if "tail" in mapping else ""
        path = tmp_path / (group + ".c")
        path.write_text(declarations + f"\nint {group}(void){{return {x}[0]+{x}[1]+{x}[2]{tail};}}\n")
        obj = path.with_suffix(".o")
        subprocess.run([cc, "-c", str(path), "-o", str(obj)], check=True, capture_output=True)
        objects.append(obj)
    main = tmp_path / "main.c"
    main.write_text("int alpha(void);int beta(void);int main(void){return alpha()!=21 || beta()!=35;}\n")
    exe = tmp_path / "check"
    subprocess.run([cc, str(source), *map(str, objects), str(main), "-o", str(exe)], check=True, capture_output=True)
    result.recheck()
    subprocess.run([str(exe)], check=True, capture_output=True)
    assert len(result.blobs) == 2

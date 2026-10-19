"""Namespace storage selection checks real object symbols, including references."""

import os
import shutil
import subprocess

import pytest

from merlin.llvmlower.object_namespace import namespace_object


@pytest.fixture
def objects(tmp_path):
    tools = {name: shutil.which(name) for name in ("cc", "nm", "objcopy")}
    if not all(tools.values()):
        pytest.skip("native object tools unavailable")
    source = tmp_path / "input.c"
    source.write_text("extern int external; static volatile int local;\nint entry(void){return external+local;}\n")
    obj = tmp_path / "input.o"
    subprocess.run([tools["cc"], "-c", str(source), "-o", str(obj)], check=True)

    def inspect(path):
        result = subprocess.run([tools["nm"], "--format=posix", str(path)], check=True, capture_output=True, text=True)
        return tuple(line.split()[0] for line in result.stdout.splitlines() if line.strip())

    calls = []

    def rewrite(source, destination, renames):
        calls.append(dict(renames))
        flags = ["--redefine-sym=" + key + "=" + value for key, value in renames.items()]
        subprocess.run([tools["objcopy"], *flags, str(source), str(destination)], check=True)

    return obj, tmp_path / "output.o", inspect, rewrite, calls


def test_no_matching_symbol_shares_actual_inode_and_never_rewrites(objects):
    source, output, inspect, rewrite, calls = objects
    before = source.read_bytes()
    receipt = namespace_object(source, output, {"other": "namespaced_other"}, inspect_symbols=inspect, rewrite=rewrite)
    assert not calls and os.path.samefile(source, output) and output.read_bytes() == before
    assert receipt["method"] == "shared_unchanged_object" and receipt["shared_inode"]


@pytest.mark.parametrize("symbol", ["entry", "external", "local"])
def test_global_undefined_and_local_names_each_require_actual_rewrite(objects, symbol):
    source, output, inspect, rewrite, calls = objects
    before = source.read_bytes()
    receipt = namespace_object(source, output, {symbol: "private_" + symbol}, inspect_symbols=inspect, rewrite=rewrite)
    assert len(calls) == 1 and not os.path.samefile(source, output)
    assert symbol not in inspect(output) and "private_" + symbol in inspect(output)
    assert source.read_bytes() == before and receipt["method"] == "provider_symbol_rewrite"


def test_wrong_rewriter_or_symbol_collision_cannot_satisfy_namespace(objects):
    source, output, inspect, rewrite, _ = objects
    with pytest.raises(ValueError, match="collides"):
        namespace_object(source, output, {"entry": "external"}, inspect_symbols=inspect, rewrite=rewrite)
    assert not output.exists()
    with pytest.raises(ValueError, match="output roster"):
        namespace_object(
            source,
            output,
            {"entry": "private_entry"},
            inspect_symbols=inspect,
            rewrite=lambda s, d, _: shutil.copyfile(s, d),
        )


def test_changed_source_during_inspection_refuses_before_output(objects):
    source, output, inspect, rewrite, _ = objects

    def mutate(path):
        names = inspect(path)
        with path.open("ab") as stream:
            stream.write(b"changed")
        return names

    with pytest.raises(ValueError, match="changed during"):
        namespace_object(source, output, {}, inspect_symbols=mutate, rewrite=rewrite)
    assert not output.exists()


def test_existing_destination_keeps_its_contents(objects):
    source, output, inspect, rewrite, _ = objects
    output.write_bytes(b"keep")
    with pytest.raises(ValueError, match="fresh"):
        namespace_object(source, output, {}, inspect_symbols=inspect, rewrite=rewrite)
    assert output.read_bytes() == b"keep"

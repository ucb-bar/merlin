"""Independent conversion and source-binding controls for the graph exporters."""

from merlin.common.digest import sha256_bytes

import hashlib
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from merlin.common.paths import repo_root


ROOT = repo_root() / "examples/atlas/comparisons/refresh"
with patch.object(sys, "path", [str(ROOT), *sys.path]):
    spec = importlib.util.spec_from_file_location("resnet_export_provider_controls", ROOT / "resnet_provider.py")
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)


def test_bf16_rounding_ties_before_delegating_target_encoding():
    observed = []
    scalar = SimpleNamespace(
        tensor_scale_exponent=lambda maxima: 0,
        pack_bf16=lambda bits, scale: observed.append((bits, scale)) or (bits & 255),
    )
    values = np.array([0x3F808000, 0x3F818000, 0x80000000, 0x3F807FFF], dtype=np.uint32).view(np.float32)
    result, scale = provider.Provider(None, scalar, np).encode(values)
    assert scale == 0
    assert result.tolist() == [0x80, 0x82, 0, 0x80]
    assert set(observed) == {(0x3F80, 127), (0x3F82, 127), (0x8000, 127)}


@pytest.mark.parametrize(
    "value",
    [
        np.array([np.inf], dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.array([1], dtype=np.float64),
        np.array([0x7F7FFFFF], dtype=np.uint32).view(np.float32),
    ],
)
def test_nonfinite_wrong_dtype_and_bf16_overflow_refused_before_codec(value):
    def forbidden(*args):
        raise AssertionError("Codec must not see inadmissible values")

    with pytest.raises(ValueError):
        provider.Provider(None, SimpleNamespace(pack_bf16=forbidden), np).encode(value)


def test_provider_digest_admitted_before_import_and_actual_module_bound():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "codec.py"
        path.write_text("answer = 17\n")
        with pytest.raises(ValueError, match="digest"):
            provider.load_provider("bad_export_provider", path, "0" * 64)
        digest = sha256_bytes(path.read_bytes())
        module, identity = provider.load_provider("good_export_provider", path, digest)
        assert module.answer == 17
        assert identity == {"path": str(path), "sha256": digest}


def test_source_bookend_detects_post_admission_change():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "source.py"
        path.write_text("value = 1\n")
        before = {str(path): {"path": str(path), "sha256": sha256_bytes(path.read_bytes())}}
        assert provider.finish_bookend(before)["unchanged"]
        path.write_text("value = 2\n")
        with pytest.raises(ValueError, match="changed"):
            provider.finish_bookend(before)


def test_initial_bookend_preserves_admitted_digest(tmp_path):
    path = tmp_path / "source.json"
    path.write_bytes(b"replacement")
    old = {"path": str(path), "sha256": "0" * 64}
    source = {"checkpoint": old, "input_stream": old}
    with pytest.raises(ValueError, match="before export"):
        provider.source_bookend(SimpleNamespace(), [old], source)


@pytest.mark.parametrize("cached", provider.PACKING_FILES[1:])
def test_cached_target_from_another_checkout_refused_before_import(tmp_path, cached):
    from atlas_provider_files import PACKING_FILES, LAYOUT_HEADER

    for name in PACKING_FILES:
        (tmp_path / name).write_text("raise AssertionError('must not import')\n")
    (tmp_path.parent / "runtime").mkdir(exist_ok=True)
    (tmp_path.parent / "runtime" / LAYOUT_HEADER).write_text("/* bound layout */\n")
    args = SimpleNamespace(atlas_tools=tmp_path, packing_provider_sha256="0" * 64)
    with patch.dict(sys.modules, {cached[:-3]: SimpleNamespace(__file__="/other/provider/" + cached)}):
        with pytest.raises(ValueError, match="cached"):
            provider.load_packing(args)


@pytest.mark.parametrize("field", ["source_manifest", "checkpoint", "input_stream", "providers"])
def test_same_shape_plan_with_other_source_or_provider_refused(field):
    identity = {"path": "/fixture", "sha256": "a" * 64}
    source = {"checkpoint": identity, "input_stream": identity}
    plan = {
        "status": "exported_compile_plan_only",
        "source_manifest": identity,
        **source,
        "providers": [identity, identity],
    }
    provider.validate_plan(plan, source, identity, [identity, identity, identity])
    wrong = {"path": "/fixture", "sha256": "b" * 64}
    plan[field] = [wrong, identity] if field == "providers" else wrong
    with pytest.raises(ValueError):
        provider.validate_plan(plan, source, identity, [identity, identity, identity])


def test_optimized_python_refused_before_source_read():
    with patch.object(sys, "flags", SimpleNamespace(optimize=1)):
        with pytest.raises(ValueError, match="assertions"):
            provider.read_source(SimpleNamespace(relocate=[]), None, None, np)


@pytest.mark.parametrize("mutate_layout", [False, True])
def test_packing_binds_transitive_imports_and_layout_header(tmp_path, mutate_layout):
    tools = tmp_path / "tools"
    tools.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    layout = runtime / "atlas_host_layout.h"
    layout.write_text("/* original layout */\n")
    for name in provider.PACKING_FILES[1:]:
        (tools / name).write_text("# synthetic provider helper\n")
    source = "\n".join("import " + name[:-3] for name in provider.PACKING_FILES[1:]) + "\n"
    if mutate_layout:
        source += "from pathlib import Path\nPath(" + repr(str(layout)) + ").write_text('changed')\n"
    entry = tools / "compile_workload_inputs.py"
    entry.write_text(source)
    args = SimpleNamespace(atlas_tools=tools, packing_provider_sha256=sha256_bytes(entry.read_bytes()))
    clean = {key: value for key, value in sys.modules.items() if key not in
             {name[:-3] for name in provider.PACKING_FILES}}
    with patch.dict(sys.modules, clean, clear=True), patch.object(sys, "path", list(sys.path)):
        if mutate_layout:
            with pytest.raises(ValueError, match="changed during import"):
                provider.load_packing(args)
        else:
            _, _, identities = provider.load_packing(args)
            assert {Path(item["path"]) for item in identities} == {
                *(tools / name for name in provider.PACKING_FILES), layout,
            }

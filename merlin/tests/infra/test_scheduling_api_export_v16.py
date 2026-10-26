"""Prospective API discovery closes imports without changing old source grants."""

import hashlib
import json
import subprocess
import sys

import pytest

from merlin.common.paths import repo_root
from merlin.compile.scheduling.export import public_modules
from merlin.compile.scheduling.library_sources import library_modules
from merlin.targetgen.compiler_library import CompilerLibraryError, freeze_compiler_library
from merlin.targetgen.tool_profile import (
    COMMON_SUBSTRATE_V9,
    attach_common_substrate,
    tool_profile_prompt,
    verify_tool_profile,
)
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release, reviewed_shared_manifest


def test_v6_through_v15_rosters_are_exactly_preserved():
    rows = {}
    for version in range(6, 16):
        rows[str(version)] = {
            "owner": hashlib.sha256(json.dumps(library_modules(version), separators=(",", ":")).encode()).hexdigest()
        }
        if version != 9:
            rows[str(version)]["public"] = hashlib.sha256(
                json.dumps(public_modules(version), separators=(",", ":")).encode()
            ).hexdigest()
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == "67438f48b24e9467afff8bab48973654b22322b56608e64c8e41623dcc2de24c"
    assert library_modules() == library_modules(6)
    assert public_modules() == public_modules(6)
    additions = set(public_modules(16)) - set(public_modules(15))
    assert additions == {
        "merlin.compile.scheduling." + name
        for name in (
            "dense_segmented_pipeline",
            "dense_array_retention",
            "tile_load_coalescing",
            "panel_tile_loads",
            "api_catalog",
        )
    }


def test_normal_v16_release_review_profile_and_isolated_public_symbol_imports(tmp_path):
    from merlin_experiments.phase2.common_transforms import supports_check

    root = tmp_path / "prospective"
    document = prepare_shared_source_release(
        root,
        python_source_root=repo_root() / "src",
        license_directory=repo_root(),
        version=16,
        source_binding=True,
    )
    assert reviewed_shared_manifest(root, version=16) == document
    # The historical roster remains fixed. New HEAD source contains a new
    # explicit owner dependency and must be refreshed under V18, never silently
    # added to a frozen V16 source selection.
    with pytest.raises(CompilerLibraryError, match="unreviewed library dependency"):
        freeze_compiler_library(
            root,
            review_id="synthetic-prospective-api-v16",
            public_modules=tuple(document["public_modules"]),
            sources=tuple((m["path"], m["module"]) for m in document["members"])
            + tuple((name, None) for name in document["review_resources"]),
        )
    bundle = dict(arm="raw_baseline", tools=[], allowed=[], denied=[], host_inputs=[])
    attach_common_substrate(bundle, version=COMMON_SUBSTRATE_V9, python_source_root=root, host_index_bits=64)
    profile = verify_tool_profile(bundle)
    assert profile["shared_scheduling_library"] == document
    assert supports_check(profile)
    assert not supports_check({"common_substrate": "merlin.common_compiler_substrate.v16"})
    with pytest.raises(ValueError, match="unknown common compiler substrate"):
        attach_common_substrate(
            dict(arm="raw_baseline", tools=[], allowed=[], denied=[], host_inputs=[]),
            version="merlin.common_compiler_substrate.v16",
            python_source_root=root,
            host_index_bits=64,
        )
    assert "scheduling_capabilities()" in tool_profile_prompt(bundle)
    assert document["scheduling_capabilities"]["inventory_version"] == 16
    for row in document["scheduling_capabilities"]["capabilities"]:
        assert row["usage"] and row["requires"]
        for symbol in row["symbols"]:
            module, _ = symbol.rsplit(".", 1)
            assert module in document["public_modules"]
    assert "merlin.runtime.reference" not in document["public_modules"]
    code = "\n".join(
        [
            "import sys,importlib,json,pathlib",
            "root=pathlib.Path(" + repr(str(root)) + ")",
            "sys.path.insert(0,str(root))",
            "from merlin.compile.scheduling.api_catalog import scheduling_capabilities",
            "manifest=json.loads((root/'shared-library.json').read_text())",
            "catalog=scheduling_capabilities()",
            "assert catalog==manifest['scheduling_capabilities']",
            "members={m['module']:m for m in manifest['members'] if m['module'] is not None}",
            "for row in catalog['capabilities']:",
            " for symbol in row['symbols']:",
            "  module,name=symbol.rsplit('.',1);owner=importlib.import_module(module)",
            "  assert pathlib.Path(owner.__file__)==root/members[module]['path']",
            "  assert callable(getattr(owner,name)),symbol",
            "print(json.dumps({'public_symbols':sum(len(r['symbols']) for r in catalog['capabilities'])}))",
        ]
    )
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["public_symbols"] >= 20
    help_result = subprocess.run(
        [sys.executable, "-B", "-m", "merlin.targetgen.generate_bundles", "--help"], text=True, capture_output=True
    )
    assert help_result.returncode == 0, help_result.stderr
    assert COMMON_SUBSTRATE_V9 in help_result.stdout

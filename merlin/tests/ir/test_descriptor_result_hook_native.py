"""Reuse the independent complete execution oracle via the actual new API.

No source is manually rewritten: the old fixture's explicit rewrite is replaced
by original identity issuance, and the ordinary file entry consumes the roster.
"""

import json
from pathlib import Path

import pytest
import test_descriptor_result_identity_native as execution
from test_descriptor_result_identity import selection

from merlin.common.digest import sha256_file
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import lower as L
from merlin.llvmlower.target_data_layout import _query
from merlin.llvmlower.toolchain import clang


@pytest.mark.parametrize("rows,cols", [(2, 7), (3, 5)])
def test_new_ordinary_file_hook_executes_complete_native_state(tmp_path, monkeypatch, rows, cols):
    seen = []
    # This only adapts the already independent execution harness. Production's
    # rewrite remains untouched and must run from inside lower_model_file.
    monkeypatch.setattr(execution, "rewrite_descriptor_result_identities", lambda *args: [{"calls": 1}])

    def ordinary(prepared, work, *, targets, features):
        path = Path(work).parent / "original.mlir"
        path.write_text(prepared)
        roster = (selection(parse_mlir_text(prepared)),) if features else ()
        result = L.lower_model_file(
            path,
            work,
            targets=targets,
            descriptor_result_identities=roster,
            **({"index_bits": 64, "data_layout": _query(clang(), ("-O2", "-fPIC"))} if roster else {}),
        )
        report = result.stats.get("descriptor_result_identity")
        if roster:
            assert report["rewrites"] == 1
            assert report["report"]["sha256"] == sha256_file(Path(report["report"]["path"]))
            recipe = json.loads((Path(work) / "lowering_recipe.json").read_text())
            assert recipe["descriptor_result_identity"]["report"] == report["report"]
            assert "descriptor_result_identity" in recipe["features"]
            assert report["selected_source"] != report["original_source"]
        else:
            assert report is None
        seen.append(bool(roster))
        return result

    monkeypatch.setattr(execution, "lower_model", ordinary)
    execution.test_real_native_result_identity_preserves_all_outputs_and_carried_state(tmp_path, rows, cols)
    assert seen == [False, True]
    pair = json.loads((tmp_path / "pair.json").read_text())
    assert pair["identity"]["malloc_sites"] == 0 < pair["original"]["malloc_sites"]

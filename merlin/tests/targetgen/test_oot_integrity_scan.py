"""The integrity scan must detect REAL harness imports, never prose.

Regression for a substring false-positive: the scan flagged ``"from merlin" in text`` — which matched a
docstring ("Lowering from merlin_iface command buffer …") or a comment — as a forbidden harness import,
failing a legitimately self-contained package before any capsule graded. Real ``import merlin`` /
``from merlin[.…] import`` is detected structurally (AST); the reference/oracle dotted paths stay
substring-matched (they name the actual surface, not a common word).
"""
from __future__ import annotations

from merlin.targetgen.oot_runner import _py_imports_merlin


def test_docstring_and_comment_mentions_do_not_count():
    # the exact shape that false-flagged GLM's package
    assert _py_imports_merlin('"""Lowering from merlin_iface command buffer to gemmini dialect."""\n') is None
    assert _py_imports_merlin("# derived from merlin_iface grammar\nx = 1\n") is None
    assert _py_imports_merlin("s = 'we do not import merlin here'\n") is None
    assert _py_imports_merlin("import merlin_iface_helper\n") is None  # unrelated module, not the harness


def test_real_harness_imports_are_caught():
    assert _py_imports_merlin("import merlin\n") == "merlin"
    assert _py_imports_merlin("import merlin.runtime.reference as r\n") == "merlin.runtime.reference"
    assert _py_imports_merlin("from merlin.targetgen.capsule_golden import golden\n") \
        == "merlin.targetgen.capsule_golden"
    assert _py_imports_merlin("from merlin import runtime\n") == "merlin"


def test_self_contained_package_passes():
    # a package importing only stdlib / its own top-level staged modules is clean
    assert _py_imports_merlin("import json, struct\nfrom oot_starterkit import parse_interface\n") is None
    assert _py_imports_merlin("from mlir_oot.lowering import lower\n") is None


def test_unparseable_source_is_not_a_false_positive():
    # a syntax error is the build gate's job, not integrity's — do not flag it as an import
    assert _py_imports_merlin("def (:\n  from merlin import x\n") is None

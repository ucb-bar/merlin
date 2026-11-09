"""Installed CLI routing retains explicit unsupported provider and linkage outcomes."""
import json
from types import SimpleNamespace

import pytest

from merlin.compiler.build.linkage import merge_linkage_requirements
from merlin.targetgen.cli import build_parser


def test_cli_application_refuses_provider_without_adapter_before_publication(tmp_path, monkeypatch):
    from merlin.semantic_compiler import snapshot, target_binding

    profile = SimpleNamespace(digest=lambda: 'profile', target_identity='synthetic')
    monkeypatch.setattr(snapshot, 'open_native_snapshot', lambda path: SimpleNamespace(profile=profile))
    monkeypatch.setattr(target_binding, 'load_native_target_binding',
                        lambda name: SimpleNamespace(profile=lambda: profile))
    output, status = tmp_path/'application', tmp_path/'status.json'
    args = build_parser().parse_args([
        'native-compile', '--application', '--engine', 'merlin_native', '--support', 'synthetic',
        '--snapshot', str(tmp_path/'snapshot'), '--compile-inputs', str(tmp_path/'inputs'),
        '--abi', 'auto', '--target-source', str(tmp_path), '--mode', 'diagnostic',
        '--out', str(output), '--status-file', str(status), '--application-memory-bytes', '1024',
    ])
    assert args.func(args) == 2
    report = json.loads(status.read_text())
    assert report['status'] == 'unsupported_semantics' and 'adapter' in report['reason']
    assert not output.exists()


def test_link_requirements_preserve_selected_runtime_recipes_and_refuse_conflicts():
    archive = {'schema':'merlin.host_linkage_contract.v1', 'supplier':'selected_math_archive',
               'archive_sha256':'a'*64, 'symbols':['sinf']}
    runtime = {'schema':'merlin.host_runtime_linkage_contract.v1',
               'supplier':'selected_host_math_runtime_object', 'symbols':['cosf'],
               'recipes':[{'policy':'explicit', 'source_sha256':'b'*64,
                           'compile_flags':['-fno-fast-math'], 'link_flags':['-Wl,--wrap=cosf']}]}
    merged = merge_linkage_requirements([archive, runtime, archive])
    assert len(merged) == 2 and runtime in merged and archive in merged
    with pytest.raises(ValueError, match='different suppliers'):
        merge_linkage_requirements([archive, {**archive, 'archive_sha256':'b'*64}])

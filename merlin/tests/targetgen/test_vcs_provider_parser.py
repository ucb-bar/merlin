"""Selected OOT parser routing; compilation and simulator execution are synthetic."""

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("scenario", ["routing", "missing_sim", "timeout", "nonzero", "missing_done", "parse_failure"])
def test_vcs_selected_provider_parser(tmp_path, scenario):
    providers = []
    for target, value in (("synthetic_alpha", 11), ("synthetic_beta", 22)):
        provider = tmp_path / target
        (provider / "contracts").mkdir(parents=True)
        (provider / "provider.yaml").write_text(
            f"schema: merlin.provider.v1\nid: {target}\ntarget: {target}\nrole: support\n"
        )
        (provider / "contracts/target_contract.yaml").write_text(f"name: {target}\nplugin:\n  backend: backend.py\n")
        (provider / "backend.py").write_text(
            "from merlin.runtime.backends.base import register, BackendInfo, BackendKind, TargetClass\n"
            f"register(BackendInfo({target!r}, TargetClass.CPU, BackendKind.KERNEL, __name__))\n"
            "calls = []\n"
            "def parse_output(console):\n"
            "    calls.append(console)\n"
            "    if console == 'bad parser': raise ValueError('selected parser failed')\n"
            f"    return {{'result': [[{value}]]}}, {{'cycles': {value}}}\n"
        )
        providers.append(str(provider))
    script = textwrap.dedent("""
        import sys, subprocess
        from pathlib import Path
        from unittest.mock import patch
        from merlin.targetgen import heavy_oracles as H
        from merlin.runtime.backends import base
        scenario, root = sys.argv[1], Path(sys.argv[2])
        alpha, beta = 'synthetic_alpha', 'synthetic_beta'
        compiled = []
        def compile(cb, llvm, workdir, *, target):
            compiled.append(target)
            return root / 'synthetic.elf'
        def execute(argv, **kwargs):
            assert argv == [str(root / 'simv'), str(root / 'synthetic.elf')]
            if scenario == 'timeout': raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
            console = {'missing_done': 'no marker', 'parse_failure': 'bad parser'}.get(scenario, 'DONE')
            return subprocess.CompletedProcess(argv, 2 if scenario == 'nonzero' else 0, console, '')
        with patch.object(H, 'vcs_simv', return_value=None if scenario == 'missing_sim' else root / 'simv'), \
             patch.object(H.oot_compile, 'compile_lowered_to_elf', side_effect=compile), \
             patch.object(subprocess, 'run', side_effect=execute):
            if scenario == 'routing':
                for target, value in ((alpha, 11), (beta, 22), (alpha, 11)):
                    result = H.vcs_adapter(target)({}, 'synthetic', root, 3)
                    assert result['outputs'] == {'result': [[value]]}
                    assert result['cycles'] == value
                    assert result['oracle'] == {'kind': 'rtl_vcs', 'derived_from_rtl': True}
                assert compiled == [alpha, beta, alpha]
                assert len(base.get_backend(alpha).calls) == 2
                assert len(base.get_backend(beta).calls) == 1
            else:
                expected = ValueError if scenario == 'parse_failure' else H.OracleUnavailable
                target = 'unregistered_without_sim' if scenario == 'missing_sim' else beta
                try:
                    H.vcs_adapter(target)({}, 'synthetic', root, 3)
                except expected:
                    pass
                else:
                    raise AssertionError('unavailable or malformed output became a verdict')
                assert compiled == ([] if scenario == 'missing_sim' else [beta])
        print('verified')
    """)
    env = {
        **os.environ,
        "MERLIN_TARGET_PATH": os.pathsep.join(providers),
        "MERLIN_TARGETS_DIR": str(tmp_path / "no-native"),
    }
    result = subprocess.run(
        [sys.executable, "-P", "-c", script, scenario, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "verified"

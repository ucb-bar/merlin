"""Phase-2 contract owners must not pull native experiment scripts into imports."""

import json
import subprocess
import sys
import sysconfig

from merlin.common.paths import module_source_path, python_source_dir


def test_phase2_contracts_cold_import_without_native_scripts(tmp_path):
    roots = [str(python_source_dir()), str(module_source_path("merlin_experiments").parent.parent)]
    roots.extend({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
    program = """
import importlib.abc, json, sys
sys.path[:0] = json.loads(sys.argv[1])
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split('.')[0] not in {
            'perf_campaign', 'perf_prompt', 'perf_gsim_gate', 'perf_agent_stage',
            'run_global_perf_experiment', '_pbcommon', 'perf_pk_claim', 'perf_pr_claim',
            'perf_affine_claim', 'perf_paired_claim', 'perf_claim_dispatch'
        }, fullname
sys.meta_path.insert(0, NoNative())
from merlin_experiments.phase2.campaign import completion_counts, CampaignGateError
from merlin_experiments.phase2.prompt import PerfCell
from merlin_experiments.phase2.gsim_gate import canonical_workload, GsimGateError
from merlin_experiments.phase2.claims import dispatch, pk
from merlin_experiments.phase2.broker import Broker, BrokerAction
from merlin_experiments.phase2.broker_policy import BrokerServices
from merlin_experiments.phase2.corpus_feedback import CorpusFeedbackPolicy
from merlin_experiments.phase2.whole_model import WholeModelPolicy
assert Broker.__module__ == 'merlin_experiments.phase2.broker'
assert CorpusFeedbackPolicy.__module__ == 'merlin_experiments.phase2.corpus_feedback'
assert WholeModelPolicy.__module__ == 'merlin_experiments.phase2.whole_model'
assert BrokerAction('inspect', ('tool',), (), 'inspect', False).available
assert BrokerServices().global_analysis_view is None
declared = pk._ACCEPTANCE_BASE['analyzer']
resolved = dispatch.resolve([{'performance': {'acceptance': {'analyzer': declared}}}])
assert resolved.module is pk and resolved.identity.declared == declared
try:
    canonical_workload({})
except GsimGateError:
    pass
else:
    raise AssertionError('missing workload evidence must refuse')
cell = PerfCell('fixture', 'case', 'gsim', 'r000')
cell.validate()
assert cell.label == 'fixture/case/gsim/r000'
try:
    completion_counts([], [])
except CampaignGateError:
    pass
else:
    raise AssertionError('empty completion must refuse')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", program, json.dumps(roots)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

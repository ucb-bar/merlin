from pathlib import Path
import os
import pytest
from merlin_experiments.phase0.evidence import load_exported_evidence
from merlin_experiments.phase0.program_admission import account_interface_text, summarize
from merlin.targetgen.target_registry import observed_contract
from merlin.targetgen.rtl.facts import observed_facts
from merlin.targetgen.contract.interface_emit import parse_interface_mlir, InterfaceGrammarError
from merlin.targetgen.interface_observations import command_rows

@pytest.fixture(scope="module")
def evidence():
    if 'P0' not in os.environ or 'RTL_FACTS' not in os.environ:
        pytest.skip('requires a generated Phase 0 export: set P0 and RTL_FACTS')
    e=load_exported_evidence(Path(os.environ['P0'])/'phase0')
    with observed_contract('gemmini',e.contract), observed_facts('gemmini',e.refreshed_facts,Path(os.environ['RTL_FACTS'])):
        yield e

@pytest.mark.parametrize('family',['elementwise_map','movement','reduction'])
def test_actual_rejected_program_is_admitted_from_observed_types(family,evidence):
    text=(Path(__file__).parent/'fixtures'/f'{family}.mlir').read_text()
    screen=summarize(account_interface_text(text,target='gemmini',evidence=evidence),scope='regression')
    assert screen['status']=='admitted', screen['reason']
    assert screen['placements'].get('accelerator',0)>0

def test_incompatible_tensor_shapes_do_not_acquire_no_broadcast_credit():
    text=(Path(__file__).parent/'fixtures/elementwise_map.mlir').read_text()
    cb=parse_interface_mlir(text)
    cb['tensors']['X1']['shape']=[1,16]
    row=command_rows(cb)[0]
    assert 'broadcasting' not in row

def test_commit_result_is_checked_against_its_explicit_tensor_type():
    text=(Path(__file__).parent/'fixtures/reduction.mlir').read_text()
    assert 'Y0' not in parse_interface_mlir(text)['tensors']
    parsed=parse_interface_mlir(text,include_results=True)
    assert parsed['tensors']['Y0']['shape']==[4,16]
    text=text.replace('-> tensor<4x16xi8>','-> !merlin_iface.acc<i32>')
    with pytest.raises(InterfaceGrammarError):parse_interface_mlir(text,include_results=True)

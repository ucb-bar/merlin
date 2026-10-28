"""Independent SDK-free checks of the actual Q report validator via extracted AST.

No gate acceptance, project imports, Lean or runtime launch. Historical source/report
data are controls; mutations are JSON-field controls, not parser-valid MLIR faults.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent / 'handler-controls-v1'
PATH = ROOT/'src/merlin/verify/bf16_policy_gate.py'
OLD = Path('/home/jeremy/Projects/merlin/out/artifacts/verification/merlin-passes/frontend/bf16-f32acc')


def main():
    OUT.mkdir(exist_ok=False)
    data = PATH.read_bytes()
    (OUT/'bf16_policy_gate.py.snapshot').write_bytes(data)
    names = {'_attr','_children','_half_contraction','source_sites','validate_report'}
    tree = ast.parse(data)
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    errors = ast.parse((ROOT/'src/merlin/verify/gate_errors.py').read_text())
    classes = [n for n in errors.body if isinstance(n,ast.ClassDef)]
    ns = {'json':json,'policy':SimpleNamespace(CHECKER='MerlinVerify.Check.Bf16F32Acc.check',
          THEOREM='MerlinVerify.Check.Bf16F32Acc.check_sound'),
          'EXACT':'MerlinVerify.Check.Bf16F32AccExact.check_sound',
          'SITES':'MerlinVerify.Check.Bf16F32AccSites.check_sound'}
    exec(compile(ast.Module(body=classes+nodes,type_ignores=[]),str(PATH),'exec',dont_inherit=True),ns)
    receipt=json.loads((OLD/'SY_model_smolvla/receipt.json').read_text())
    doc=json.loads(Path(receipt['exports']['before']['path']).read_text())
    report=json.loads(Path(receipt['cert']['path']).read_text())
    controls=[]
    def run(label, mutate, expected):
        r=copy.deepcopy(report)
        mutate(r)
        try:
            count=len(ns['validate_report'](r,doc)); outcome='consistent'
        except Exception as exc:
            count=None; outcome=type(exc).__name__
        controls.append({'control':label,'outcome':outcome,'site_count':count,'expected':expected,'passed':outcome==expected})
    run('actual historical report data',lambda r:None,'consistent')
    for key in ('checked','semantic_match','site_qualification_passed'):
        run('false '+key,lambda r,k=key:r['pass_boundary'].__setitem__(k,False),'GateRejected')
        run('integer '+key,lambda r,k=key:r['pass_boundary'].__setitem__(k,1),'GateUnavailable')
    run('conditional-only',lambda r:r.pop('pass_boundary'),'GateUnavailable')
    run('wrong root',lambda r:r['pass_boundary'].__setitem__('semantic_match_theorem','other'),'GateUnavailable')
    run('drop last site',lambda r:r['pass_boundary']['site_obligations'].pop(),'GateRejected')
    run('duplicate first site',lambda r:r['pass_boundary']['site_obligations'].append(r['pass_boundary']['site_obligations'][0]),'GateRejected')
    run('reorder sites',lambda r:r['pass_boundary']['site_obligations'].reverse(),'GateRejected')
    run('change result id',lambda r:r['pass_boundary']['site_obligations'][0]['source_results'].__setitem__(0,999),'GateRejected')
    run('bool covered count',lambda r:r['pass_boundary'].__setitem__('covered_sites',True),'GateRejected')
    run('extra flag',lambda r:r['sites'][0].__setitem__('mul','fast'),'GateRejected')
    run('report self approves',lambda r:r.__setitem__('policy_reviewed',True),'GateUnavailable')
    run('whole capture false is separate',lambda r:r.__setitem__('qualification_passed',False),'consistent')
    result={'kind':'SDK-free actual extracted validator controls; no live gate/checker outcome',
            'source_sha256':hashlib.sha256(data).hexdigest(),'source_path':str(PATH.relative_to(ROOT)),
            'raw_source_export':receipt['exports']['before'],'historical_report':receipt['cert'],
            'controls':controls,'passed':all(c['passed'] for c in controls),
            'native_executed':False,'approval_granted':False,'semantic_negatives_observed':False}
    (OUT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'controls':len(controls),'passed':result['passed']}))
    if not result['passed']: raise SystemExit(1)


if __name__=='__main__':main()

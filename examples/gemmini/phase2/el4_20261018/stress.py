"""Independent functional stress guards, separate from the frozen performance objective."""
import json
from pathlib import Path

import common as C
import audit


def main():
    from merlin.perf.derived_bound import machine_from_facts
    from merlin.perf.movement_peak import movement_peak
    facts = json.loads(Path(C.environment()['MERLIN_RTL_FACTS']).read_text())
    machine = machine_from_facts('gemmini', facts=facts, measure_fill=False)
    ports = movement_peak(facts)
    rows = []
    tests = [('aligned',32,32,48),('m_tail',33,32,48),('n_tail',32,32,47),
             ('k_tail',32,33,48),('all_tails',33,31,47),('short_m_control',1,32,48),
             ('rhs_fit',17,512,496),('rhs_over_capacity',17,512,512),
             ('large_projection_prefill',32,2048,256),('large_projection_decode',1,2048,256),
             ('deep_reduction_panel',33,3280,80)]
    for name,m,k,n in tests:
        entry={'name':name,'kind':'layer','op':'matmul','source_role':'derived_sweep',
               'source_reference':'independent shape/tail/residency guard, not a performance objective',
               'M':m,'K':k,'N':n,'epilogue':[]}
        descriptor, text = audit.build(entry,audit.binding())
        source = C.OUT/'stress-sources'/name/'capsule.interface.mlir'
        source.parent.mkdir(parents=True,exist_ok=True)
        source.write_text(text)
        rows.append(audit.inventory_row('stress/'+name,source,descriptor,'independent_functional_guard',machine,ports))
    # A connected chain keeps one explicit RHS live while a temporary RHS occupies
    # later scratchpad rows, then reuses the original handle after both commits.
    source=C.OUT/'stress-sources/live_resident_chain/capsule.interface.mlir'
    source.parent.mkdir(parents=True,exist_ok=True)
    source.write_text('''module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<17x17xi8>
  %V = merlin_iface.tensor {name = "V", role = "weight"} : tensor<17x17xi8>
  %A = merlin_iface.tensor {name = "A", role = "input"} : tensor<33x17xi8>
  %R = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<17x17xi8>) -> !merlin_iface.resident
  %T = merlin_iface.resident_pack %V {layout = "packed_rhs"} : (tensor<17x17xi8>) -> !merlin_iface.resident
  %a0 = merlin_iface.matmul %A, %R : (tensor<33x17xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %a0 {name = "Y0", epilogue = ["acc_scale", "relu"], acc_scale = 0.25 : f32, output_dtype = "i8"} : (!merlin_iface.acc<i32>) -> tensor<33x17xi8>
  %a1 = merlin_iface.matmul %Y0, %T : (tensor<33x17xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y1 = merlin_iface.commit %a1 {name = "Y1", epilogue = ["acc_scale"], acc_scale = 0.25 : f32, output_dtype = "i8"} : (!merlin_iface.acc<i32>) -> tensor<33x17xi8>
  %a2 = merlin_iface.matmul %Y1, %R : (tensor<33x17xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y2 = merlin_iface.commit %a2 {name = "Y2", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<33x17xi32>
  merlin_iface.evict %T : (!merlin_iface.resident) -> ()
  merlin_iface.evict %R : (!merlin_iface.resident) -> ()
}
''')
    rows.append(audit.inventory_row('stress/live_resident_chain',source,{'purpose':'live explicit RHS survives temporary and readout state changes'},'independent_functional_guard',machine,ports))
    C.save(C.EVIDENCE/'stress-manifest.json',{'schema':'phase2.exploratory.stress_manifest.v1',
           'performance_objective':False,'members':rows,'members_sha256':C.digest(rows)})
    print(json.dumps({'members':len(rows),'refusals':[(x['identity'],x['baseline']) for x in rows if x['baseline']['status']!='not_executed']}))


if __name__=='__main__':
    main()

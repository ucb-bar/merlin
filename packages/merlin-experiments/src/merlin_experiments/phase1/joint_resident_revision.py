"""Review a false positive reuse obligation on a single output-tile control."""
from __future__ import annotations
import copy
from pathlib import Path
from .feedback import joint_performance as PERF
from ..phase2.contracts import document_sha256


def verify(evidence, *, facts_pin, members):
    from .joint_source_revision import _capsule
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir
    from merlin.llvmlower.device_shim import logical_buffers
    if (set(evidence) != {'schema','name','before','after','facts','array'}
            or evidence['schema'] != 'merlin.single_use_resident_review.v1'
            or evidence['facts'] != facts_pin):
        raise ValueError('single-use review needs exact selected hardware facts')
    old, old_files = _capsule(evidence['before'])
    new, new_files = _capsule(evidence['after'])
    name=evidence['name']
    expected=copy.deepcopy(old)
    if (name not in members or old.get('name') != name or new.get('name') != name
            or old_files != new_files or not old_files
            or members[name]['capsule_sha256'] != document_sha256(old)
            or expected.get('expected',{}).get('modes',{}).get('resident_reuse') is not True):
        raise ValueError('single-use correction has changed source or stale identity')
    expected['expected']['modes']['resident_reuse']=False
    if expected != new:
        raise ValueError('single-use correction changes another source obligation')
    source=old.get('interface_mlir');screen=old.get('software_screen',{})
    if (source not in old_files or screen.get('status') != 'admitted'
            or screen.get('program_sha256') != old_files[source]):
        raise ValueError('single-use correction requires admitted unchanged source')
    text=(Path(evidence['before']['path']).parent/source).read_text()
    commands=parse_interface_mlir(text)['commands']
    if [c['opcode'] for c in commands] != ['RES_PACK','MATMUL_RESIDENT','COMMIT','EVICT']:
        raise ValueError('single-use correction needs exactly one resident contraction and commit')
    pack,mul,commit,evict=commands
    if (mul['operands']['rhs'] != pack['operands']['dst']
            or commit['operands']['src'] != mul['operands']['dst']
            or evict['operands']['handle'] != pack['operands']['dst']):
        raise ValueError('single-use correction has ambiguous resident consumers')
    buffers,reason=logical_buffers(text)
    if not buffers or len(buffers)!=3:
        raise ValueError('single-use correction requires three exact tensor interfaces')
    byname={b.name:b for b in buffers}
    a=byname[mul['operands']['lhs']];b=byname[pack['operands']['src']];y=byname[commit['operands']['dst']]
    arrays=PERF._read(facts_pin)['facts']['arrays']
    arrays=[a for a in arrays if a['name']==evidence['array'] and a.get('corroborated') is True]
    if len(arrays)!=1:
        raise ValueError('single-use correction needs one corroborated array geometry')
    rows,cols=arrays[0]['rows'],arrays[0]['cols']
    if (type(rows) is not int or type(cols) is not int or min(rows,cols)<=0
            or len(a.shape)!=2 or len(b.shape)!=2 or len(y.shape)!=2
            or a.shape[1]!=b.shape[0] or y.shape!=(a.shape[0],b.shape[1])
            or not 0<a.shape[0]<=rows or not 0<b.shape[1]<=cols
            or a.dtype!='i8' or b.dtype!='i8' or y.dtype!='i32'):
        raise ValueError('single-use correction cannot remove a multi-output-tile reuse obligation')
    return name,document_sha256(new)

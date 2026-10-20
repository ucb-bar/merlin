"""Adversarial checks for the exploratory source/effect join, using a real emission."""
import copy
import json

import common as C


def main():
    manifest=json.loads((C.EVIDENCE/'stress-manifest.json').read_text())
    member=next(x for x in manifest['members'] if x['identity']=='stress/mixed_projection')
    source=json.loads(__import__('pathlib').Path(member['command_buffer_path']).read_text())
    prior=json.loads((C.EVIDENCE/'projection-check.json').read_text())
    emitted_path=__import__('pathlib').Path(prior['real_emission'])
    emitted=json.loads(emitted_path.read_text())
    assert source!=emitted  # Actual LLVM tagging populated ISA positions.
    expected=C.logical_command_buffer(source)
    assert expected==C.logical_command_buffer(emitted)
    corruptions=[]
    for label, mutate in [
        ('dropped_work',lambda v:v['commands'].pop()),
        ('readout_change',lambda v:v['commands'][-1]['attributes'].update(acc_scale=.5)),
        ('placement_change',lambda v:v['params']['lane_placement'][-1].update(lane='host')),
        ('source_identity',lambda v:v['params']['global_program_plan'].update(source_sha256='0'*64)),
        ('missing_dependency',lambda v:v['params']['global_program_plan']['tasks'][-1].update(reads=[])),
        ('abi_change',lambda v:v['kernel_abi']['args'].pop()),
    ]:
        bad=copy.deepcopy(emitted);mutate(bad)
        assert expected!=C.logical_command_buffer(bad),label
        corruptions.append(label)
    C.save(C.EVIDENCE/'projection-check.json',{'real_emission':str(emitted_path),
           'source_and_emission_semantics_equal':True,'rejected_mutations':corruptions,
           'only_ignored_fields':['final ISA schedule count','prologue/epilogue ISA ranges','task ISA start/end']})
    print('real mixed-source join and six corruption controls passed')


if __name__=='__main__':
    main()

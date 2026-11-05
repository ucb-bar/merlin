"""Explicit additive revision does not weaken ordinary observer relocation."""
import copy
import hashlib
import json

import pytest

from merlin.runtime.backends.rocc_selection import relocate_operand_protocol
from merlin_experiments.corpus import operand_revision as R
from test_selected_operand_protocol import selected


def fixture(tmp_path):
    _, facts, blob, sidecar, _, _ = selected(tmp_path)
    def pin(name, raw):
        path=tmp_path/name;path.write_bytes(raw)
        return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
    fact_pin=pin('facts.json',blob)
    old_pin=pin('before.json',json.dumps(sidecar).encode())
    new=copy.deepcopy(sidecar);fields=[{'name':'source_kind','const':'ordinary'}]
    new['protocol']['roles']['classes']['MVIN']['fields'] += fields
    new_pin=pin('after.json',json.dumps(new).encode())
    before={'original_facts':fact_pin,'sidecar':old_pin}
    after={'original_facts':fact_pin,'sidecar':new_pin}
    contract={'name':'array','rtl_checks':['decode_clean','legal_funct'],
              'rocc_operand_roles':{'version':1,'word_bits':64,'instructions':[{'funct':4,'class':'INPUT','operands':{}}]},
              'rocc_operand_selection':before}
    review={'schema':R.SCHEMA,'previous_sidecar_sha256':old_pin['sha256'],
            'selected_sidecar_sha256':new_pin['sha256'],'original_facts_sha256':fact_pin['sha256'],
            'field_additions':{'MVIN':fields},'source_pin_replacements':[]}
    return contract,after,review,pin,new


def test_explicit_revision_binds_but_relocation_still_refuses(tmp_path):
    contract,after,review,pin,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='cannot change'):
        relocate_operand_protocol(contract,after)
    updated=R.select(contract,after,pin('review.json',json.dumps(review).encode()))
    assert updated['rocc_operand_selection']==after
    assert {k:v for k,v in updated.items() if k!='rocc_operand_selection'}=={k:v for k,v in contract.items() if k!='rocc_operand_selection'}


@pytest.mark.parametrize('change',['existing_field','layout','new_class','unreviewed_addition','wrong_parent'])
def test_review_cannot_change_unlisted_semantics(tmp_path,change):
    contract,after,review,pin,new=fixture(tmp_path)
    if change=='existing_field':new['protocol']['roles']['classes']['MVIN']['fields'][0]['name']='other'
    elif change=='layout':new['layouts']['fields'][0]['width']=16
    elif change=='new_class':new['protocol']['roles']['classes']['EXTRA']={'fields':[]}
    elif change=='unreviewed_addition':new['protocol']['roles']['classes']['MVIN']['fields'].append({'name':'more','const':1})
    else:review['previous_sidecar_sha256']='0'*64
    after['sidecar']=pin('altered.json',json.dumps(new).encode());review['selected_sidecar_sha256']=after['sidecar']['sha256']
    with pytest.raises(ValueError):R.select(contract,after,pin('review.json',json.dumps(review).encode()))

"""Accuracy is absolute, identity scoped, public and descriptive only."""
import copy

import pytest
from merlin_experiments.phase1.feedback import model_accuracy as A


def pair(n,*,model=100,actual=100,group=1,macs=4096,split='unsplit'):
    binding={k:f'{n:064x}' for k in A.BINDINGS}
    binding['source_group_sha256']=f'{group:064x}'
    binding['physical_domain_sha256']='d'*64
    return dict(model=dict(binding=binding,public_source=True,timing_complete=True,unresolved=[],
            model_sha256='a'*64,comparison_domain_sha256='b'*64,cycles=model),
        physical=dict(binding=dict(binding),public_source=True,engine='gsim',numerical_status='pass',
                      structural_status='pass',cycles=actual),
        family='contraction',useful_macs=macs,split=split)


def test_signed_cancellation_is_not_accuracy_and_tiny_is_not_large():
    records=[pair(1,model=80),pair(2,model=120)]
    card=A.card(records);summary=card['partitions'][0]['summary']
    assert summary['median_absolute_relative_error']==pytest.approx(.2)
    assert summary['pairs_within_5_percent']==0
    assert summary['independent_source_groups']==1
    assert card['partitions'][0]['missing_size_buckets']==['large','medium']
    assert card['broad_under_5_percent_claim'] is False and card['qualification_issued'] is False


@pytest.mark.parametrize('key',A.BINDINGS)
def test_binding_mutations_never_join(key):
    record=pair(1);record['physical']['binding'][key]='f'*64
    card=A.card([record])
    assert card['status']=='UNKNOWN' and card['usable_program_pairs']==0


def test_medium_large_domains_duplicates_and_unknown_are_kept_distinct():
    small=pair(1);medium=pair(2,group=2,macs=64*256*256);large=pair(3,group=3,macs=512**3)
    large['model']['model_sha256']='c'*64
    unknown=pair(4);unknown['model']['timing_complete']=False
    card=A.card([small,small,medium,large,unknown])
    assert len(card['partitions'])==2 and card['usable_program_pairs']==3
    assert card['excluded']['incomplete_model_or_unqualified_physical_product']==1
    assert card['partitions'][0]['covered_size_buckets']==['medium','small']
    assert card['partitions'][1]['covered_size_buckets']==['large']


def test_conflicting_duplicates_and_source_split_leak_are_excluded():
    a=pair(1);b=copy.deepcopy(a);b['model']['cycles']=101
    assert A.card([a,b])['excluded']=={'contradictory_duplicate_pair':1}
    card=A.card([pair(1,split='train'),pair(2,split='held')])
    assert card['usable_program_pairs']==0
    assert card['excluded']['source_group_crosses_train_held']==2


@pytest.mark.parametrize('change',['private','numeric','trace','cycles','model','scope'])
def test_uncertified_or_incomplete_evidence_has_no_accuracy(change):
    row=pair(1)
    if change=='private':row['model']['public_source']=False
    elif change=='numeric':row['physical']['numerical_status']='UNKNOWN'
    elif change=='trace':row['physical']['structural_status']='fail'
    elif change=='cycles':row['model']['cycles']=float('nan')
    elif change=='model':row['model'].pop('model_sha256')
    else:row['physical']['binding'].pop('scope_sha256')
    assert A.card([row])['usable_program_pairs']==0


def test_legacy_ratio_card_never_qualifies_or_invents_bindings():
    card=A.legacy_card([{'test_case':'same','execution_digest':'d','ratio':1.0}]*80)
    assert card['status']=='UNKNOWN' and card['usable_program_pairs']==0
    assert card['legacy_distinct_capsules']==1 and card['legacy_program_versions']==80

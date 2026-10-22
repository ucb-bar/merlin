from merlin_experiments.phase1.joint_calibration_priority import summarize


def inputs():
    return dict(profile={'parameters':[{'path':['target','latency']}],
                'members':{'small':{'source_group_sha256':'group'}}},
        selection={'members':{'small':{'capsule_sha256':'source'}}},
        runtime={'model_sha256':'model'},
        denial={'deny':[{'model_sha256':'model','price_paths':[['target','latency']],
                         'reason':'structural residual'}]},
        questions={'schema':'merlin.public_model_coverage_questions.v1','questions':[
            {'id':'gap','priority':0,'evidence_domain':'public','observed_features':['cpu.divide']}]} )


def row(**changes):
    r=dict(name='small',source='source',compiler='compiler',elf='elf',model='model',scope='scope',
        numeric='pass',timing_complete=False,trace_status='unknown',unresolved=['missing price'],
        entry={},report_pin={},retained_l3=True,features={'cpu.divide':1})
    r.update(changes)
    return r


def test_denial_and_numerical_success_never_authorize_dispatch():
    r=summarize(rows=[row()],**inputs())
    assert r['automatic_fit_status']=='denied_structural_error'
    assert not r['fit_authorized'] and not r['dispatch_authorized']
    assert r['observations'][0]['retained_l3_reuse']=='requires_exact_ELF_input_scope_engine_config_join'
    assert r['priorities'][0]['unresolved_observations']==1
    assert r['actual_model_blockers'][0]['reason']=='missing price'
    assert r['actual_model_blockers'][0]['capsules']==['small']


def test_duplicate_compiler_or_elf_cannot_fill_independent_variant_quota():
    r=summarize(rows=[row(),row(),row(compiler='other')],**inputs())
    assert len(r['observations'])==2
    assert r['independent_variants']['source_groups_with_two_distinct_compilers_and_ELFs']==0
    r=summarize(rows=[row(),row(compiler='other',elf='other')],**inputs())
    assert r['independent_variants']['source_groups_with_two_distinct_compilers_and_ELFs']==1
    assert not r['fit_authorized']


def test_stale_sources_and_unselected_workloads_never_enter_current_groups():
    r=summarize(rows=[row(source='old'),row(name='private_reference')],**inputs())
    assert len(r['observations'])==1
    assert not r['observations'][0]['source_current']
    assert r['independent_variants']['current_source_groups']==0


def test_owner_refresh_hook_preserves_selected_review_and_never_blocks_native(tmp_path, monkeypatch):
    import json
    from merlin_experiments.phase1 import joint_calibration_owner as owner
    from merlin_experiments.phase1 import joint_calibration_priority as priority
    from merlin_experiments.phase1.joint_calibration_admission import ENV
    seen=[]
    def call(**kwargs): seen.append(kwargs)
    monkeypatch.setattr(priority,'refresh',call)
    denial={'path':'review.json','sha256':'d'*64}
    selection={'run_dir':'native-run','environment':{ENV:json.dumps(denial)}}
    state={'policy':{'native_calibration':{'path':'profile','sha256':'p'*64}}}
    owner.refresh_model_priority(selection,state,tmp_path,{'path':'questions','sha256':'q'*64})
    assert seen[0]['state'] is state and seen[0]['denial_pin']==denial
    assert seen[0]['output']==tmp_path/'model_priority.json'
    def fail(**kwargs): raise ValueError('stale descriptive evidence')
    monkeypatch.setattr(priority,'refresh',fail)
    owner.refresh_model_priority(selection,state,tmp_path)
    report=json.loads((tmp_path/'model_priority.json').read_text())
    assert report['status']=='blocked_coverage_review'
    assert not report['dispatch_authorized'] and not report['fit_authorized']

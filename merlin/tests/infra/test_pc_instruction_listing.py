import pytest

from merlin.perf.pc_instruction_listing import instruction_bytes, inventory, listing_regions, read_listing


def listing(text):
    return read_listing(text.strip().splitlines())


def test_call_body_includes_tail_blocks_after_ret():
    rows = listing('''
0x100 0x000000ef jal 200
0x200 0x00008082 ret
0x202 0x00000001 nop
0x204 0x0000bff5 j 200
0x104 0x00008082 ret
''')
    regions = listing_regions(rows)
    assert len(regions) == 2
    assert regions[1].indices == (1, 2, 3)
    assert regions[1].closed and regions[0].closed
    assert inventory(rows)["regions"][1]["backedges"] == [
        {"from_pc": 0x204, "to_pc": 0x200, "conditional": False,
         "trip_count": None, "basis": "listed_branch_target"}]
    assert instruction_bytes(rows[1:3]) == bytes.fromhex('82800100')


def test_nested_call_and_missing_return_remain_open():
    rows = listing('''
0x100 0x000000ef jal 200
0x200 0x000000ef jal 300
0x300 0x00008082 ret
0x204 0x00008082 ret
''')
    regions = listing_regions(rows)
    assert regions[2].closed
    assert not regions[1].closed and not regions[0].closed


def test_tail_jump_not_a_call_and_bad_rows_fail_closed():
    assert len(listing_regions(listing('0x100 0x0000006f j 100'))) == 1
    for text in ('not an instruction', '0x100 0x00010001 nop', '0x100 0x0000001f bad'):
        with pytest.raises(ValueError, match='row 1'):
            listing(text)


def test_selected_decoder_retains_unclaimed_custom_sites_and_unknown_values():
    rows = listing('''
0x100 0x0200307b .insn r 0x7b,3,1,x0,x0,x0
0x104 0x0400307b .insn r 0x7b,3,2,x0,x0,x0
0x108 0x00008082 ret
''')
    result = inventory(rows, table={"custom_opcode": 123, "legal_funct": [1], "names": {"1": "load"}})
    audit = result['regions'][0]['selected_decode']
    assert audit['endpoint_insns'] == 1
    assert len(audit['unaccounted']) == 1
    assert result['count_basis'] == 'listed_instruction_sites'
    assert 'register_values' in result['unknown']
    assert 'dynamic_loop_multiplicity' in result['unknown']


def test_cfg_proves_loop_and_keeps_unreachable_tail_separate():
    from merlin.perf.pc_instruction_listing import control_flow
    rows = listing('''
0x100 0x00000001 nop
0x102 0x00000001 nop
0x104 0x00000001 bnez a0,102
0x106 0x00008082 ret
0x108 0x00000001 j 102
''')
    cfg = control_flow(rows)
    assert len(cfg['natural_loops']) == 1
    assert cfg['natural_loops'][0]['header_pc'] == 0x102
    assert cfg['natural_loops'][0]['trip_count'] is None
    assert next(block for block in cfg['blocks'] if block['pc'] == 0x108)['reachable_from_entry'] is False


def test_body_match_requires_full_bytes_preserves_multiple_histories():
    from merlin.perf.pc_instruction_listing import match_symbol_bodies
    rows = listing('0x100 0x00000001 nop\n0x102 0x00008082 ret')
    same = listing('0x200 0x00000001 nop\n0x202 0x00008082 ret')
    different = listing('0x200 0x00000001 nop\n0x202 0x00000001 nop\n0x204 0x00008082 ret')
    result = match_symbol_bodies(rows, listing_regions(rows),
                                symbols={'same': same, 'also_same': same, 'longer': different},
                                candidate_sha256='a' * 64)
    assert {row['symbol'] for row in result[0]['matches']} == {'same', 'also_same'}
    assert result[0]['whole_elf_identity'] == 'unknown'


def test_pc_relative_constant_does_not_invent_loaded_memory():
    from merlin.perf.pc_instruction_listing import pc_relative_constants
    rows = listing('''
0x1000 0xfffff717 auipc a4,0xfffff
0x1004 0x80070713 addi a4,a4,-2048
0x1008 0x00073703 ld a4,0(a4)
''')
    facts = pc_relative_constants(rows)
    assert len(facts) == 1
    assert facts[0]['value'] == (1 << 64) - 2048
    assert facts[0]['pointee_value'] is None


def test_canonical_returns_and_discontiguous_blocks():
    from merlin.perf.pc_instruction_listing import control_flow
    rows = listing('''
0x100 0x00000001 c.nop
0x104 0x00008067 jalr zero,0(ra)
''')
    assert listing_regions(rows)[0].closed
    cfg = control_flow(rows)
    assert len(cfg['blocks']) == 2
    assert cfg['blocks'][0]['successors'] == []
    assert not cfg['blocks'][1]['reachable_from_entry']


def test_omitted_callee_is_unknown_instead_of_an_empty_closed_body():
    rows = listing('0x100 0x000000ef jal 200\n0x104 0x00008082 ret')
    regions = listing_regions(rows)
    assert regions[1].entry_pc is None
    assert regions[1].indices == () and not regions[1].closed


def test_static_schedule_and_cca_do_not_promote_site_counts_to_executions():
    from merlin.kernels.kernel_evidence import ArtifactPin, KernelEvidence, cca_projection
    from merlin.perf.pc_instruction_listing import lift_schedule
    rows = listing('0x100 0x0200307b .insn r 0x7b,3,1,x0,x0,x0')
    evidence = KernelEvidence('selected-body', artifacts=(ArtifactPin('disassembly','body','a'*64),
                              ArtifactPin('facts','facts','b'*64)), capture_kind='static_disassembly')
    model = lift_schedule(rows, table={"custom_opcode":123,"legal_funct":[1],"names":{"1":"load"}},
                          evidence=evidence)
    assert model.command_count == 1 and not model.complete
    assert model.source_kind == 'static_disassembly'
    assert 'register_and_memory_values_not_supplied' in model.unknowns
    cca = cca_projection(model, op='unknown')
    assert cca.dispatch is None
    from dataclasses import asdict
    assert cca.memory is None
    assert all(value is None for value in asdict(cca.compute).values())

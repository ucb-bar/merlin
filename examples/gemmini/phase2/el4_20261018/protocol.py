"""Bind linked counter windows and target command sites to the selected RTL protocol."""
import collections
import json
from pathlib import Path
import subprocess

import common as C
C.configure()
from evaluate import OBJDUMP


def main():
    facts=json.loads(Path(C.environment()['MERLIN_RTL_FACTS']).read_text())
    opcodes={int(x['custom_opcode']) for x in facts['facts']['interfaces'] if 'custom_opcode' in x}
    assert len(opcodes)==1
    opcode=opcodes.pop()
    sources={
        'rocket_fence':C.P1/'.tmp/chipyard-fire-aligned/generators/rocket-chip/src/main/scala/rocket/RocketCore.scala',
        'gemmini_busy':C.P1/'.tmp/chipyard-fire-aligned/generators/gemmini/src/main/scala/gemmini/Controller.scala',
        'reservation_hazards':C.P1/'.tmp/chipyard-fire-aligned/generators/gemmini/src/main/scala/gemmini/ReservationStation.scala',
        'load_state':C.P1/'.tmp/chipyard-fire-aligned/generators/gemmini/src/main/scala/gemmini/LoadController.scala',
        'store_state':C.P1/'.tmp/chipyard-fire-aligned/generators/gemmini/src/main/scala/gemmini/StoreController.scala',
        'target_firrtl':C.P1/'merlin/out/artifacts/probes/gemmini/fire-aligned-30mhz/firesim-target/elaboration/firechip.chip.FireSim.FireSimGemminiRocketConfig.fir',
        'simulation_firrtl':C.P1/'.tmp/chipyard-fire-aligned/sims/verilator/generated-src/chipyard.harness.TestHarness.GemminiGsimFireSimAlignedConfig/chipyard.harness.TestHarness.GemminiGsimFireSimAlignedConfig.fir'}
    pins={}
    for name,p in sources.items():
        terms = ['id_rocc_busy','id_do_fence'] if name in {'rocket_fence','target_firrtl','simulation_firrtl'} else ['overlaps','deps_ld','deps_ex','deps_st'] if name=='reservation_hazards' else ['io.busy :='] if name=='gemmini_busy' else ['config_stride','config_activation','DoConfig']
        matches=[]
        for index,line in enumerate(p.read_text().splitlines(),1):
            if any(term in line for term in terms):
                matches.append({'line':index,'text':line.strip()})
        pins[name]={'path':str(p),'sha256':C.file_hash(p),'excerpts':matches[:45]}
    assert pins['target_firrtl']['sha256']=='b7d9826b1c9f16fd00bc757b4334a7f330a842c6a494e0c28608dc351b305bf0'
    rows=[]
    for receipt_path in sorted((C.OUT/'build').glob('*/*/*/build-receipt.json')):
        receipt=json.loads(receipt_path.read_text());elf=Path(receipt['elf'])
        assert C.file_hash(elf)==receipt['elf_sha256']
        listing=subprocess.run([str(OBJDUMP),'-d',str(elf)],capture_output=True,text=True,check=True).stdout
        symbol=None;sites=collections.Counter();functions=collections.defaultdict(collections.Counter)
        for line in listing.splitlines():
            if line.endswith('>:') and '<' in line:
                symbol=line.split('<',1)[1][:-2]
            if ':' not in line:
                continue
            tokens=line.split(':',1)[1].split()
            if not tokens or len(tokens[0])!=8:
                continue
            try:
                word=int(tokens[0],16)
            except ValueError:
                continue
            if word & 0x7f == opcode:
                funct=(word>>25)&0x7f
                sites[str(funct)]+=1
                functions[symbol][str(funct)]+=1
        audit=json.loads((elf.parent/'counter-protocol-audit.json').read_text())
        assert audit['begin_pc']<audit['entry_call_pc']<audit['completion_call_pc']<audit['end_pc']
        assert sites
        # Primitive wrapper must not execute a second cloned command stream.
        duplicated_wrapper = bool(functions.get('kernel_entry')) and bool(functions.get('gemmini_kernel'))
        rows.append({'elf_sha256':receipt['elf_sha256'],'context':receipt['context'],
                     'counter_audit_sha256':C.file_hash(elf.parent/'counter-protocol-audit.json'),
                     'custom_opcode':opcode,'static_instruction_sites_by_funct':dict(sites),
                     'functions':{k:dict(v) for k,v in functions.items()},
                     'duplicated_primitive_wrapper':duplicated_wrapper,
                     'dynamic_multiplicity':'UNKNOWN: static sites do not count loop executions',
                     'physical_traffic_or_utilization':'UNKNOWN: instruction sites are not DMA traffic or busy cycles'})
    C.save(C.EVIDENCE/'protocol-audit.json',{'schema':'phase2.exploratory.protocol_audit.v1',
           'selected_facts_sha256':C.file_hash(C.environment()['MERLIN_RTL_FACTS']), 'source_pins':pins,
           'inference':'selected Rocket fence stalls on RoCC busy; Gemmini busy includes reservation and scratchpad activity; reservation station encodes local RAW/WAR/WAW and per-class order',
           'limit':'static selected-source and final-ELF evidence plus fresh complete-output tests; not a complete formal protocol certificate',
           'executables':rows,'duplicated_wrappers':[x['elf_sha256'] for x in rows if x['duplicated_primitive_wrapper']]})
    print(json.dumps({'executables':len(rows),'duplicated_wrappers':sum(x['duplicated_primitive_wrapper'] for x in rows)}))


if __name__=='__main__':
    main()

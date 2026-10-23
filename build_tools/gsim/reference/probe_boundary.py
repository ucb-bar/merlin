#!/usr/bin/env python3
"""Bounded read/write canary for scalar FASED timing-core drivers.

The functional store is a prototype: aligned INCR full-beat requests, one
transaction outstanding, explicit byte strobes, and egress-ID selection. It is
not the production gSIM memory interface. The reference supplies all stimulus;
the candidate replays it exactly, never independently rescheduling requests.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import selectors
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run(reference, candidate, ports_path, settings_path, output, *, build_receipt, maximum_cycles=50000):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    receipt_path = Path(build_receipt)
    receipt = json.loads(receipt_path.read_text())
    receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    for path, expected in receipt['outputs'].items():
        if hashlib.sha256((receipt_path.parent/path).read_bytes()).hexdigest() != expected:
            raise ValueError('native build output differs from receipt: '+path)
    expected_binaries = [receipt['outputs']['verilator/VFasedTimingStandalone'], receipt['outputs']['gsim-driver/run']]
    if [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (reference,candidate)] != expected_binaries:
        raise ValueError('probe binaries differ from build receipt')
    ports = json.loads(Path(ports_path).read_text())['ports']
    inputs = dict.fromkeys(ports['inputs'], 0)
    if ports['inputs'].get('io_tNasti_w_bits_data') != 64 or ports['inputs'].get('io_tNasti_w_bits_strb') != 8:
        raise ValueError('this canary requires a 64-bit full-strobe boundary')
    if ports['inputs'].get('io_tNasti_ar_bits_id',0) < 4:
        raise ValueError('canary requires at least four ID bits')
    requested_settings, effective_settings = {}, {}
    for token in Path(settings_path).read_text().split():
        key, _, value = token.partition('=')
        if not key.startswith('+mm_') or not key.endswith('_0'):
            raise ValueError('explicit model-0 settings required')
        port = 'io_mmReg_'+key[4:-2]
        if key in requested_settings: raise ValueError('duplicate runtime setting')
        requested_settings[key] = int(value)
        # Match bridge register assignment width, and disclose truncation.
        effective_settings[port] = int(value) & ((1 << ports['inputs'][port])-1)
    expected_settings = {p for p in inputs if p.startswith('io_mmReg_')}
    if set(effective_settings) != expected_settings:
        raise ValueError('runtime settings do not cover all configuration ports')
    inputs.update(effective_settings)
    operations = [
        ('read', 0x1000, 8, 1), ('write', 0x1000, 8, 2), ('read', 0x1000, 8, 3),
        ('read', 0x2000, 1, 4), ('write', 0x2000, 1, 5), ('read', 0x2000, 1, 6),
        ('read', 0x40000, 8, 7), ('read', 0x80000, 8, 8),
        ('read', 0x1000, 8, 9), ('write', 0x1040, 8, 10), ('read', 0x1040, 8, 11),
    ]
    errors = (output/'reference.stderr').open('w')
    proc = subprocess.Popen([reference], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=errors, text=True, bufsize=1)
    selector = selectors.DefaultSelector(); selector.register(proc.stdout,selectors.EVENT_READ)
    deadline = time.monotonic()+60
    store, traces, completed = {}, [], []
    op_index, accepted_addr, sent_beats = 0, False, 0
    active_read, active_write = None, None
    start_cycle = None
    def value(addr): return sum(store.get(addr+i, 0) << (8*i) for i in range(8))
    try:
        for cycle in range(maximum_cycles):
            inputs['reset'] = int(cycle < 8)
            inputs['io_tNasti_r_ready'] = int(cycle % 7 not in (2, 3))
            inputs['io_tNasti_b_ready'] = int(cycle % 11 not in (1, 2, 3))
            for channel in ('ar','aw','w'):
                inputs['io_tNasti_'+channel+'_valid'] = 0
            if active_read:
                ident, addr, beat, length = active_read
                inputs.update(io_egressResp_rBits_id=ident, io_egressResp_rBits_data=value(addr+beat*8),
                              io_egressResp_rBits_last=int(beat+1 == length))
            if active_write is not None:
                inputs['io_egressResp_bBits_id'] = active_write
            op = operations[op_index] if op_index < len(operations) else None
            if cycle >= 16 and op:
                kind, address, beats, ident = op
                if start_cycle is None: start_cycle = cycle
                channel = 'ar' if kind == 'read' else 'aw'
                for field, val in {'addr':address,'id':ident,'len':beats-1,'size':3,'burst':1}.items():
                    inputs['io_tNasti_'+channel+'_bits_'+field] = val
                inputs['io_tNasti_'+channel+'_valid'] = int(not accepted_addr)
                if kind == 'write' and accepted_addr and sent_beats < beats:
                    inputs.update(io_tNasti_w_valid=1, io_tNasti_w_bits_id=ident,
                        io_tNasti_w_bits_last=int(sent_beats+1 == beats), io_tNasti_w_bits_strb=255,
                        io_tNasti_w_bits_data=(0x1234567800000000+address+sent_beats))
            proc.stdin.write(' '.join(str(inputs[name]) for name in ports['inputs'])+'\n'); proc.stdin.flush()
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise TimeoutError('native reference wall deadline exceeded')
            line = proc.stdout.readline()
            if not line: raise RuntimeError('reference terminated: '+(output/'reference.stderr').read_text()[-1500:])
            values = list(map(int,line.split()))
            if len(values) != len(ports['outputs']): raise ValueError('bad scalar output row')
            outputs = dict(zip(ports['outputs'], values))
            traces.append({'cycle':cycle,'inputs':dict(inputs),'outputs':outputs})
            if not op or cycle < 16: continue
            kind, address, beats, ident = op
            channel = 'ar' if kind == 'read' else 'aw'
            if inputs['io_tNasti_'+channel+'_valid'] and outputs['io_tNasti_'+channel+'_ready']:
                accepted_addr = True
            if inputs['io_tNasti_w_valid'] and outputs['io_tNasti_w_ready']:
                word = inputs['io_tNasti_w_bits_data']
                for byte in range(8): store[address+sent_beats*8+byte]=(word>>(8*byte))&255
                sent_beats += 1
            done = False
            if outputs['io_tNasti_r_valid'] and inputs['io_tNasti_r_ready']:
                if not active_read: raise AssertionError('read without selected functional egress')
                rid, addr, beat, length = active_read
                assert outputs['io_tNasti_r_bits_id'] == rid
                assert outputs['io_tNasti_r_bits_data'] == value(addr+beat*8)
                assert outputs['io_tNasti_r_bits_last'] == int(beat+1 == length)
                active_read[2] += 1
                if beat+1 == length: active_read = None; done = True
            if outputs['io_tNasti_b_valid'] and inputs['io_tNasti_b_ready']:
                assert active_write == ident == outputs['io_tNasti_b_bits_id']
                assert sent_beats == beats
                active_write = None; done = True
            if outputs['io_egressReq_r_valid']:
                assert outputs['io_egressReq_r_bits'] == ident
                active_read = [ident,address,0,beats]
            if outputs['io_egressReq_b_valid']:
                assert outputs['io_egressReq_b_bits'] == ident
                active_write = ident
            if done:
                completed.append({'operation':list(op),'start_cycle':start_cycle,'completed_cycle':cycle})
                op_index += 1; accepted_addr = False; sent_beats = 0; start_cycle = None
                if op_index == len(operations): break
        if len(completed) != len(operations): raise TimeoutError(f'{len(completed)}/{len(operations)} transactions completed')
    finally:
        proc.stdin.close()
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        selector.close()
        errors.close()
    if proc.returncode != 0: raise RuntimeError('reference exited nonzero: '+str(proc.returncode))
    stimulus = ''.join(' '.join(str(frame['inputs'][name]) for name in ports['inputs'])+'\n' for frame in traces)
    (output/'stimulus.txt').write_text(stimulus)
    replay = subprocess.run([candidate],input=stimulus,text=True,capture_output=True,timeout=60)
    (output/'candidate.stderr').write_text(replay.stderr)
    replay.check_returncode()
    rows = replay.stdout.splitlines()
    if len(rows) != len(traces): raise AssertionError('candidate produced different number of cycles')
    if any(len(row.split()) != len(ports['outputs']) for row in rows):
        raise AssertionError('candidate produced malformed output row')
    trace_ports = {d:{n:{'width':w,'valid_when':None} for n,w in ps.items()} for d,ps in ports.items()}
    for name,spec in trace_ports['outputs'].items():
        if '_bits' in name:
            valid = name.partition('_bits')[0]+'_valid'
            if valid in trace_ports['outputs']: spec['valid_when']=valid
    common = dict(schema='merlin.memory_boundary_trace.v1', memory_reference_sha256=digest({'build_receipt_sha256':receipt_sha256,'effective_settings':effective_settings}),
        initial_state_sha256=digest({'reset_cycles':8,'functional_store':'zero','cold_native_init':True}),
        scope_sha256=digest({'scope':'standalone_timing_core','operations':operations}),
        stimulus_sha256=hashlib.sha256(stimulus.encode()).hexdigest(), ports=trace_ports)
    # Reset first, then use contiguous target-cycle indices for comparison.
    for name, binary in [('reference',reference),('candidate',candidate)]:
        with (output/(name+'.jsonl')).open('w') as stream:
            stream.write(json.dumps(dict(common,engine_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest()))+'\n')
            for index,frame in enumerate(traces[8:]):
                result = dict(frame,cycle=index)
                if name == 'candidate':
                    result['outputs']=dict(zip(ports['outputs'],map(int,rows[index+8].split())))
                stream.write(json.dumps(result)+'\n')
    report = {'schema':'merlin.fased_boundary_probe.v1','cycles':len(traces),'completed':completed,'build_receipt_sha256':receipt_sha256,
        'requested_settings':requested_settings,'effective_settings':effective_settings,
        'deployment_equivalence':'NOT_ESTABLISHED','production_adapter':False,
        'limitations':['one outstanding transaction','aligned INCR full-beat requests with full byte strobes','no whole-SoC integration'],
        'reference_binary_sha256':hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
        'candidate_binary_sha256':hashlib.sha256(Path(candidate).read_bytes()).hexdigest()}
    (output/'probe.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','ports','settings','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--build-receipt',required=True)
    a=p.parse_args(); run(a.reference,a.candidate,a.ports,a.settings,a.output,build_receipt=a.build_receipt)


if __name__ == '__main__': main()

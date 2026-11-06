"""Build a checksum-bound full Tiny host ELF after exact C arithmetic replay."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).absolute().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/host_iss")]
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked, lexical_absolute_path, map_metadata_paths
from atlas_tooling import sha256
from full_bundle import KernelAssets, FullBundle
from elf_image import Image
from tiny_full_contract import contract, plan, validate_bundle
from chipyard import arena
from chipyard import result_transport
from chipyard.firesim_target import package as package_firesim


def pin(pins, path, expected=None):
    path = checked(path)
    digest = sha256(path)
    if ((expected is not None and digest != expected)
            or (str(path) in pins and pins[str(path)] != digest)):
        raise ValueError("conflicting or changed build input: " + str(path))
    pins[str(path)] = digest
    return path


def helper_pins(pins):
    for name in ('full_bundle','bundle','block_bundle','tiny_full_contract','tiny_contract','elf_image',
                 'atlas_path_guard','atlas_tooling','atlas_host_layout'):
        relative = 'tools/host_iss/' if name in ('full_bundle','bundle','block_bundle',
                    'tiny_full_contract','tiny_contract','elf_image') else 'tools/'
        expected = ROOT / relative / (name+'.py') if relative != 'tools/' else ROOT / kernel_library.relative_path(name+'.py')
        module = sys.modules.get(name)
        if module is None or checked(module.__file__) != expected:
            raise ValueError('build helper imported from another source: '+name)
        pin(pins,expected)
    layout = sys.modules['atlas_host_layout']
    if checked(layout.HEADER) != ROOT/'runtime/atlas_host_layout.h':
        raise ValueError('build layout header differs')
    layout.recheck()
    pin(pins,layout.HEADER)
    if checked(arena.__file__) != ROOT/'tools/chipyard/arena.py':
        raise ValueError('build placement helper differs')
    pin(pins,arena.__file__)
    pin(pins,ROOT/'tools/chipyard/firesim_target.py')
    pin(pins,ROOT/'tools/host_iss/placement.py')
    pin(pins,ROOT/'tools/chipyard/result_transport.py')


QUALIFIED = ('tiny_full_model.c','tiny_full_model.h','smol_host_ops.c','smol_host_ops.h',
    'smol_host_bits.h','llama_host_attention.c','llama_host_attention.h','llama_host_layout.c',
    'llama_host_layout.h','llama_rotary_table.c','llama_rotary_table.h','llama_host_bmm.c',
    'llama_host_bmm.h','smol_host_softmax.c','smol_host_softmax.h','tiny_block.h')


def qualified_sources(pins,sources,gate):
    """Bind the math actually compiled to the complete C gate's exact bytes."""
    for name in QUALIFIED:
        identities={digest for path,digest in gate['source_sha256'].items() if Path(path).name==name}
        if len(identities)!=1:
            raise ValueError('missing or ambiguous gate-qualified source: '+name)
        pin(pins,sources[name],next(iter(identities)))


def gate_receipt(path, digest):
    path = checked(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("C gate receipt changed")
    gate = json.loads(raw)
    if (gate.get("status") != "PASS_FULL_TINY_C_ARITHMETIC_REPLY_REPLAY"
            or any(gate.get(name) is not True for name in
                   ("all_actual_predecessors_exact", "all22_KV_all_steps_exact", "source_bookends_verified"))
            or not isinstance(gate.get("source_sha256"), dict) or not gate["source_sha256"]
            or not isinstance(gate.get("greedy_tokens"), list)
            or not 2 <= len(gate['greedy_tokens']) <= 33
            or len(gate.get("ledger", [])) != 155 * len(gate['greedy_tokens'])):
        raise ValueError("complete exact full Tiny C gate required")
    calls = [(row['step'],row['site']) for row in plan(len(gate['greedy_tokens'])-1)
             if row['columns'][0] == 0]
    for index,(row,(step,site)) in enumerate(zip(gate['ledger'],calls)):
        if (type(row.get('index')) is not int or row['index'] != index
                or type(row.get('step')) is not int or row['step'] != step or row.get('site') != site
                or any(not isinstance(row.get(name),str) or len(row[name]) != 64
                       or any(c not in '0123456789abcdef' for c in row[name])
                       for name in ('actual_input_sha256','reply_sha256'))):
            raise ValueError('C gate projection ledger differs')
    return gate


def staged_contract(bundle):
    full = json.loads(bundle.read('provenance.json'))
    if bundle.provenance != dict(full,schema='atlas.full_kernel_assets.v1'):
        raise ValueError('staged/full model provenance differs')
    return contract(SimpleNamespace(provenance=full,read=bundle.read))


def emit_assets(bundle, data, host, segment=False):
    """Materialize immutable blobs and actual writable buffers with fixed ABI."""
    declarations, assembly, symbols, locations = [], [], {}, {}
    def blob(name, member):
        if name in symbols:
            if symbols[name] != member:
                raise ValueError("conflicting ELF symbol")
            return name
        symbols[name] = member
        raw = bundle.read(member)
        local = "blob-" + hashlib.sha256(raw).hexdigest() + ".bin"
        destination = host / local
        if not destination.exists():
            destination.write_bytes(raw)
        declarations.append("extern const uint8_t " + name + "[];")
        if local in locations:
            assembly.extend(['.global '+name, '.type '+name+',@object',
                             '.set '+name+','+locations[local], '.size '+name+','+str(len(raw))])
            return name
        locations[local] = name
        assembly.extend(['.section .rodata,"a",@progbits', '.balign 8', '.global ' + name,
                         '.type ' + name + ',@object', name + ':', '.incbin "' + local + '"',
                         '.size ' + name + ',.-' + name])
        return name
    kernels = data['kernels']
    for item in data['host_parameters'].values():
        blob(item['symbol'], item['file'])
    table, expected = [], []
    geometry = plan(data['decode_count'])
    for index, (item, shape) in enumerate(zip(kernels, geometry)):
        weights = blob('tiny_full_weights_' + str(index), item['weights'])
        program = blob('tiny_full_program_' + str(index), item['original'])
        expected.append(blob('tiny_full_panel_ref_' + str(index), item['expected']))
        m, k, n = item['physical_mkn']
        fields = [item['logical_rows'], m, k, n, item['columns'][0], shape['width'], shape['logical'], item['A_scale']]
        fields += [weights, k*n, '(const uint32_t*)'+program, len(bundle.read(item['original']))//4,
                   item['output_offset'], item['total_dram_bytes'], item['requested_dma']['load_bytes'],
                   item['requested_dma']['store_bytes']]
        table.append('{' + ','.join(map(str, fields)) + '}')
    refs = data['reference_buffers']
    actual_declarations, cases = [], []
    for item in refs:
        name, size = item['actual_symbol'], item['bytes']
        blob(item['expected_symbol'], item['file'])
        dtype = {'uint16':'uint16_t', 'uint32':'uint32_t', 'float32':'float'}[item['dtype']]
        count = size // {'uint16':2, 'uint32':4, 'float32':4}[item['dtype']]
        actual_declarations.append(dtype + ' ' + name + '[' + str(count) + '];')
    for step in range(data['decode_count']+1):
        lines = ['case '+str(step)+':']
        for layer in range(22):
            for field in ('keys','values'):
                name=f'tiny_full_cache_{step:02d}_{layer:02d}_{field}'
                size = 4 * (128 + step) * 64 * 2
                lines.append(f'memcpy({name},tiny_full_state_global.{field}[tiny_full_state_global.bank]'
                             f'+{layer}*TINY_FULL_CACHE_WORDS,{size});')
        lines.append('for(size_t i=0;i<32000;++i){uint32_t bits=(uint32_t)logits[i]<<16;'
                     f'memcpy(tiny_full_logits_{step:02d}+i,&bits,4);}}')
        for item in refs:
            name = item['actual_symbol']
            if name.startswith(f'tiny_full_cache_{step:02d}_') or name == f'tiny_full_logits_{step:02d}':
                lines.append(f'if(!equal({name},{item["expected_symbol"]},{item["bytes"]}))return 1;')
        lines.append(f'if(tiny_full_greedy_tokens[{step}]!='
                     f'((const uint32_t*)tiny_full_greedy_tokens_expected)[{step}])return 1;')
        if step:
            lines.append(f'if(tiny_full_decode_inputs[{step-1}]!='
                         f'((const uint32_t*)tiny_full_decode_inputs_expected)[{step-1}])return 1;')
        lines.append('break;');cases.extend(lines)
    parameters = data['host_parameters']
    word = lambda key: '(const uint16_t*)' + parameters[key]['symbol']
    norms = lambda suffix: '{'+','.join(word(f'model.layers.{i}.{suffix}.weight') for i in range(22))+'}'
    parameter_init = '{'+','.join((word('model.embed_tokens.weight'), norms('input_layernorm'),
        norms('post_attention_layernorm'), word('model.norm.weight'),word('rotary_cosine'),word('rotary_sine'),
        '0x1.4f8b588e368f1p-17f', '0', '0', '0', '0'))+'}'
    header = '''#ifndef TINY_FULL_ASSETS_H
#define TINY_FULL_ASSETS_H
#include "tiny_full_model.h"
#include "tiny_projection.h"
'''
    header += (f'#define TINY_FULL_KERNELS {len(kernels)}\n#define TINY_FULL_STEPS {data["decode_count"]+1}\n'
               f'#define TINY_FULL_DRAM_BYTES {max(k["total_dram_bytes"] for k in kernels)}\n')
    header += f'#define TINY_FULL_ARENA_BASE UINT64_C({kernels[0]["dram_base"]})\n'
    header += '''extern volatile uint64_t tohost,fromhost;
extern const tiny_panel tiny_full_panels[TINY_FULL_KERNELS];
extern const uint8_t *const tiny_full_panel_expected[TINY_FULL_KERNELS];
extern const tiny_full_parameters tiny_full_parameters_global;
extern tiny_full_state tiny_full_state_global;
extern uint32_t tiny_full_greedy_tokens[TINY_FULL_STEPS],tiny_full_decode_inputs[TINY_FULL_STEPS-1];
extern const int64_t *const tiny_full_prefill_tokens;
int tiny_full_save_check_step(unsigned,const uint16_t *);
#endif
'''
    source = '#include "tiny_full_assets.h"\n#include <string.h>\n'+'\n'.join(declarations)+'\n'
    source += '\n'.join(actual_declarations)+'\n'
    source += 'const tiny_panel tiny_full_panels[TINY_FULL_KERNELS]={\n'+',\n'.join(table)+'\n};\n'
    source += 'const uint8_t *const tiny_full_panel_expected[TINY_FULL_KERNELS]={'+','.join(expected)+'};\n'
    source += 'const tiny_full_parameters tiny_full_parameters_global='+parameter_init+';\n'
    source += 'static uint16_t keys[2][22*TINY_FULL_CACHE_WORDS],values[2][22*TINY_FULL_CACHE_WORDS];\n'
    source += 'tiny_full_state tiny_full_state_global={{keys[0],keys[1]},{values[0],values[1]},0,0,0};\n'
    source += 'const int64_t *const tiny_full_prefill_tokens=(const int64_t*)'+parameters['prefill_tokens']['symbol']+';\n'
    source += ('static int equal(const void*a,const void*b,size_t n){const uint8_t*x=a,*y=b;'
               'for(size_t i=0;i<n;++i)if(x[i]!=y[i])return 0;return 1;}\n')
    source += ('int tiny_full_save_check_step(unsigned step,const uint16_t*logits){switch(step){\n'
               +'\n'.join(cases)+'\ndefault:return 1;}return 0;}\n')
    if segment:
        source = source.replace('static uint16_t keys[2][22*TINY_FULL_CACHE_WORDS],values[2][22*TINY_FULL_CACHE_WORDS];',
                                'uint16_t tiny_full_keys[2][22*TINY_FULL_CACHE_WORDS],tiny_full_values[2][22*TINY_FULL_CACHE_WORDS];')
        source = source.replace('{keys[0],keys[1]},{values[0],values[1]}',
                                '{tiny_full_keys[0],tiny_full_keys[1]},{tiny_full_values[0],tiny_full_values[1]}')
    (host/'tiny_full_assets.h').write_text(header)
    (host/'tiny_full_assets.c').write_text(source)
    (host/'images.S').write_text('\n'.join(assembly)+'\n.section .note.GNU-stack,"",@progbits\n')


def place_bundle(out, arena_base, dram_base, dram_bytes):
    """Relocate copied programs and bind the new physical placement metadata."""
    metadata = {name: json.loads((out/name).read_bytes()) for name in
                ('kernels.json', 'provenance.json', 'contract.json')}
    kernels = metadata['kernels.json']['kernels']
    size = max(k['total_dram_bytes'] for k in kernels)
    arena.window(arena_base, size)
    if (dram_base < 0 or dram_bytes <= 0 or dram_base + dram_bytes > 1 << 64
            or not dram_base <= arena_base <= dram_base + dram_bytes - size):
        raise ValueError('DMA arena outside physical DRAM range')
    sums = {}
    for line in (out/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1); sums[name] = digest
    if arena_base != arena.DEFAULT_ARENA:
        for item in kernels:
            for key, suffix, transform in (
                ('original', '.bin', lambda raw: arena.relocate(raw, item['dram_base'], arena_base)),
                ('source', '.S', lambda raw: arena.relocate_assembly(raw.decode(), item['dram_base'], arena_base).encode())):
                raw = transform((out/item[key]).read_bytes())
                digest = hashlib.sha256(raw).hexdigest()
                name = 'programs/' + digest + suffix
                (out/name).write_bytes(raw); sums[name] = digest; item[key] = name
            item['dram_base'] = arena_base
    placement = dict(arena_base=arena_base, dram_base=dram_base, dram_bytes=dram_bytes)
    # Default builds retain the original provenance and program bytes.
    if placement != dict(arena_base=arena.DEFAULT_ARENA, dram_base=0x80000000, dram_bytes=64 << 30):
        for document in metadata.values():
            document['placement'] = placement
            document['kernels'] = kernels
        raw = json.dumps(metadata['contract.json'], indent=2).encode() + b'\n'
        (out/'contract.json').write_bytes(raw); sums['contract.json'] = hashlib.sha256(raw).hexdigest()
        for name in ('kernels.json', 'provenance.json'):
            metadata[name]['contract_sha256'] = sums['contract.json']
            raw = json.dumps(metadata[name], indent=2).encode() + b'\n'
            (out/name).write_bytes(raw); sums[name] = hashlib.sha256(raw).hexdigest()
        (out/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(sums.items())))
    return placement


def build(args):
    segment_root = getattr(args, 'segment_boundaries', None)
    reference = reference_metadata = assembler = model_before = None
    if segment_root is not None:
        if not getattr(args, 'segment_sha256', None) or not getattr(args, 'segment_model_root', None):
            raise ValueError('segment build requires reference checksum pin and pinned model root')
        import segment_assets
        from run_model_tile import configure_model, model_binding
        reference, reference_metadata = segment_assets.prepare(segment_root, args.segment_sha256)
        model_before = model_binding(args.segment_model_root)
        _, _, assembler, _ = configure_model(args.segment_model_root)
    out = checked(args.out)
    if out.exists():
        raise ValueError("fresh complete full bundle required")
    gate = gate_receipt(args.gate,args.gate_sha256)
    mappings = []
    for value in args.relocate:
        old, separator, new = value.partition('=')
        if not separator:
            raise ValueError('source map requires OLD=NEW')
        mappings.append((lexical_absolute_path(old),lexical_absolute_path(new)))
    def mapped(path):
        matches = [(old,new) for old,new in mappings if path == old or old in path.parents]
        if not matches:
            return path
        old,new = max(matches,key=lambda pair:len(pair[0].parts))
        return new/path.relative_to(old)
    gate = map_metadata_paths(gate,mapped)
    pins = {}
    pin(pins,args.gate,args.gate_sha256)
    for path,digest in gate['source_sha256'].items():pin(pins,path,digest)
    helper_pins(pins)
    bundle = KernelAssets(args.assets,'tiny-full')
    data = staged_contract(bundle)
    if bundle.provenance.get('program_policy','original') == 'traffic-reuse-v1':
        from kernel_library import code_sources
        for name,digest in code_sources(traffic_reuse=True).items():
            pin(pins,ROOT/kernel_library.relative_path(name),digest)
        for path,digest in bundle.provenance['reuse']['generator_source_sha256'].items():
            pin(pins,path,digest)
    if (bundle.provenance['capture']['sha256'] != gate['capture_sha256']
            or data['greedy_tokens'] != gate['greedy_tokens']):
        raise ValueError("kernel capture/C gate extent or greedy identity differs")
    provider = checked(args.merlin)
    names = ('smol_host_ops.c','smol_host_ops.h','smol_host_bits.h','llama_host_attention.c',
        'llama_host_attention.h','llama_host_layout.c','llama_host_layout.h','llama_rotary_table.c',
        'llama_rotary_table.h','llama_host_bmm.c','llama_host_bmm.h','smol_host_softmax.c',
        'smol_host_softmax.h','smol_host_sleef_LICENSE.txt','tiny_block.h','host_matrix_fp8pack.c','host_matrix_fp8pack.h')
    sources = {name:provider/name for name in names}
    for name in ('tiny_full_model.c','tiny_full_model.h','tiny_full_tiles.h','tiny_full_projection.c','tiny_full_projection.h',
                 'tiny_full_target.c','tiny_full_start.S','tiny_full_link.ld'):
        sources[name]=Path(__file__).with_name(name)
    for name in ('tiny_projection.h','atlas_host.c','atlas_host.h','atlas_host_layout.h','atlas_result.h',
                 'atlas_resnet_graph_math.c','atlas_numeric.h','tiny_attention.c','tiny_attention.h'):
        sources[name]=ROOT/'runtime'/name
    support=checked(args.support)
    for name in ('rv64_support.c','expf_bf16_values.inc'):
        sources[name]=support/name
    for name in ('string.h','math.h','fenv.h'):
        sources['include/'+name]=support/'include'/name
    qualified_sources(pins,sources,gate)
    for path in sources.values():pin(pins,path)
    pin(pins,__file__)
    if reference is not None:
        pin(pins,Path(__file__).with_name('segment_assets.py'))
        pin(pins,ROOT/'tools/host_iss/tiny_segment_contract.py')
        pin(pins,ROOT/'tools/host_iss/segment_boundary.py')
        sources['tiny_full_segment_target.c'] = Path(__file__).with_name('tiny_full_segment_target.c')
        pin(pins,sources['tiny_full_segment_target.c'])
    compiler,linker=pin(pins,args.cc),pin(pins,args.ld)
    resource=checked(args.resource_include)
    out.mkdir(parents=True)
    for name in bundle.sums:
        path=checked(out/name);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(bundle.read(name))
    (out/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(bundle.sums.items())))
    arena_base = args.arena_base if args.arena_base is not None else arena.DEFAULT_ARENA
    placement = place_bundle(out, arena_base, args.dram_base, args.dram_bytes)
    copied = KernelAssets(out, 'tiny-full')
    data = staged_contract(copied)
    host=out/'host';host.mkdir()
    for name,path in sources.items():
        dest=host/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(checked(path).read_bytes())
    emit_assets(copied,data,host,segment=reference is not None)
    transport_record = result_transport.write_config(host,args)
    commands=[]
    def run(command,label):
        result=subprocess.run(command,cwd=host,capture_output=True,text=True,timeout=180)
        (host/(label+'.log')).write_text(result.stdout+result.stderr)
        commands.append({'argv':command,'returncode':result.returncode})
        if result.returncode:raise ValueError(label+' failed')
    flags=['--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany','-std=c11',
        '-O2','-Wall','-Wextra','-Werror','-ffp-contract=off','-fno-fast-math','-ffreestanding',
        '-nostdinc','-isystem',str(resource),'-I','include','-I','.']
    units=['tiny_full_model.c','tiny_full_projection.c','tiny_full_target.c','tiny_full_assets.c','tiny_attention.c',
        'llama_host_attention.c','llama_host_layout.c','llama_rotary_table.c','llama_host_bmm.c',
        'host_matrix_fp8pack.c','smol_host_ops.c','smol_host_softmax.c','rv64_support.c','atlas_host.c',
        'images.S','tiny_full_start.S']
    if reference is not None:
        units[units.index('tiny_full_target.c')] = 'tiny_full_segment_target.c'
    for path in host.iterdir():
        if path.is_file():pin(pins,path)
    for name in ('string.h','math.h','fenv.h'):pin(pins,host/'include'/name)
    objects=[]
    for index,name in enumerate(units):
        obj=f'object-{index}.o';objects.append(obj)
        run([str(compiler),*flags,'-M','-MF',obj+'.before.d',name],'dependencies-'+str(index))
        dep=(host/(obj+'.before.d')).read_text().replace('\\\n',' ').split(':',1)[1].split()
        before=set()
        for value in dep:
            path=pin(pins,host/value if not Path(value).is_absolute() else value);before.add(str(path))
        run([str(compiler),*flags,'-MD','-MF',obj+'.d','-c',name,'-o',obj],'compile-'+str(index))
        after=(host/(obj+'.d')).read_text().replace('\\\n',' ').split(':',1)[1].split()
        for value in after:
            path=checked(host/value if not Path(value).is_absolute() else value)
            if str(path) not in before:raise ValueError('new dependency after compile')
            pin(pins,path)
        pin(pins,host/obj)
    if any(sha256(checked(path))!=digest for path,digest in pins.items()):raise ValueError('prelink source changed')
    default_placement = dict(arena_base=arena.DEFAULT_ARENA, dram_base=0x80000000, dram_bytes=64 << 30)
    link_placement = [] if placement == default_placement else [
        '--defsym=TINY_FULL_RAM_BASE='+str(args.dram_base),
        '--defsym=TINY_FULL_RAM_BYTES='+str(args.dram_bytes),
        '--defsym=TINY_FULL_DEVICE_BASE='+str(arena_base),
        '--defsym=TINY_FULL_DEVICE_BYTES='+str(max(k['total_dram_bytes'] for k in data['kernels']))]
    if args.rodata_after_arena:
        end = arena_base + max(k['total_dram_bytes'] for k in data['kernels'])
        link_placement.append('--defsym=TINY_FULL_RODATA_BASE='+str(end))
    run([str(linker),'-flavor','gnu','--threads=1','-m','elf64lriscv','-static','--no-undefined',*link_placement,
         '-T','tiny_full_link.ld','-Map','host.map',*objects,'-o','host.elf'],'link')
    bundle.verify_bookends()
    if any(sha256(checked(path))!=digest for path,digest in pins.items()):raise ValueError('build source bookend changed')
    placement_audit = arena.audit(host,args.dram_base,args.dram_bytes)
    package_record = (package_firesim(host,'atlas-tiny-full',args.dram_base,args.dram_bytes,uart=transport_record)
                      if args.target == 'firesim' else None)
    record={'schema':'atlas.tiny_full_host_build.v1','actual_returncode':0,'source_bookends_verified':True,
        'elf':{'sha256':sha256(host/'host.elf')},'contract_sha256':copied.provenance['contract_sha256'],
        'source_sha256':pins,'commands':commands,'fresh_instruction_execution':False,
        'gate_sha256':args.gate_sha256,'capture_sha256':gate['capture_sha256'],
        'placement':placement,'placement_audit':placement_audit,
        'program_policy':copied.provenance.get('program_policy','original'),
        'rodata_after_arena':args.rodata_after_arena,
        'target':args.target,'firesim_package':package_record,
        'result_transport':transport_record,'uart':transport_record,'result_name':'tiny-full',
        'result_progress_launches': [index+1 for index,shape in enumerate(plan(data['decode_count']))
                                    if shape['logical'] in (6,7) and shape['columns'][1] == shape['width']],
        'source_path_relocations': [{'old':str(old),'new':str(new)} for old,new in mappings]}
    if reference is not None:
        record.update(segment_mode=True,result_name='tiny-segment',result_progress_launches=[],
                      reference_sha256sums_sha256=reference.checksums_sha256)
    sums=dict(copied.sums);sums['host/host.elf']=record['elf']['sha256']
    for name in ('atlas_result_transport.h','result-transport.json'):
        sums['host/'+name]=sha256(host/name)
    if package_record is not None:
        for name in ('firesim-contract.json','marshal.json'):
            sums['host/'+name]=sha256(host/name)
    if reference is not None:
        for name in [*sources, 'tiny_full_assets.c', 'tiny_full_assets.h', 'images.S']:
            sums['host/'+name] = sha256(host/name)
        record['segment_chain_sha256'] = segment_assets.bind(out,reference,reference_metadata,
            Image((host/'host.elf').read_bytes()),data,copied.provenance,sums)
    def receipt():
        raw=json.dumps(record,indent=2).encode()+b'\n';(host/'build.json').write_bytes(raw)
        sums['host/build.json']=hashlib.sha256(raw).hexdigest()
        (out/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(sums.items())))
    receipt()
    try:
        complete=FullBundle(out,'tiny-full')
        image = Image(complete.read('host/host.elf'))
        validate_bundle(complete,image)
        if reference is not None:
            from tiny_segment_contract import validate_bundle as validate_segments
            model_after = model_binding(args.segment_model_root)
            if model_after['revision'] != model_before['revision'] or model_after['curve_sha256'] != model_before['curve_sha256']:
                raise ValueError('segment model profile changed')
            profile = dict(revision=model_after['revision'],backend='curve',curve_sha256=model_after['curve_sha256'])
            validate_segments(complete,image,assembler,profile)
        if any(sha256(checked(path))!=digest for path,digest in pins.items()):
            raise ValueError('admission source bookend changed')
    except BaseException:
        record.update(actual_returncode=1,elf_admission_verified=False)
        receipt()
        raise
    record.update(status='PASS_PREPARED_TINY_FULL_HOST_ELF_UNEXECUTED',elf_admission_verified=True)
    receipt()
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','gate','merlin','support','cc','ld','resource-include','out'):
        parser.add_argument('--'+name,type=checked,required=True)
    parser.add_argument('--gate-sha256',required=True)
    parser.add_argument('--segment-boundaries',type=checked)
    parser.add_argument('--segment-sha256')
    parser.add_argument('--segment-model-root',type=checked)
    parser.add_argument('--relocate',action='append',default=[],metavar='OLD=NEW',
                        help='explicit physical-path rebinding for retained gate inputs')
    arena.add_arguments(parser)
    parser.add_argument('--target',choices=('sim','firesim'),default='sim')
    parser.add_argument('--rodata-after-arena',action='store_true',
                        help='place immutable host data after the DMA arena, leaving a load-segment gap')
    result_transport.add_arguments(parser)
    build(parser.parse_args())

if __name__=='__main__':main()

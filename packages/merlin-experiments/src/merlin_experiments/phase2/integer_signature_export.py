"""Exact bounded output export for ordinary same-ELF integer observations.

Only absolute signature symbols are added through stock HTIF symbol metadata.
The primary ELF, instructions, loaded bytes and selected counter window remain
unchanged. Source-owned expected values stay in the host, outside the namespace.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.runtime import _native_files as F

from . import retained_memory_observation as M

SCHEMA = "merlin.integer_signature_export.v1"


def _pin(path):
    return F.pin(F.canonical(Path(path)), max_bytes=64 << 20)


def _output_region(elf_bytes, cb):
    """Export only an exact contiguous union of independently sized outputs."""
    from merlin.runtime.harness_render import _roster, _tensor
    from ..phase1.feedback.native_output_readback import _elf_output_symbols

    normalized, slots = _roster(cb)
    names = normalized['kernel_abi']['outputs']
    if len(names)==1:
        return M.output_region(elf_bytes,cb)
    if (not 2<=len(names)<=64 or elf_bytes[16:18]!=b'\x02\x00'
            or b'begin_signature\0'in elf_bytes or b'end_signature\0'in elf_bytes):
        raise ValueError('integer export needs bounded original outputs without existing signature metadata')
    expected=cb.get('_oracle_expected_outputs',{})
    if set(expected)!=set(names):
        raise ValueError('integer export omits the complete original output oracle')
    fields=[]
    for name in names:
        spec=normalized['tensors'][name];shape,dtype,extent=_tensor(spec)
        if spec['dtype']not in {'i8','i16','i32','i64'}or not 0<extent<=M.MAX_BYTES:
            raise ValueError('integer output export has unsupported or excessive storage')
        fields.append(dict(tensor=name,symbol=f'tensor_{list(slots).index(name)}',bytes=extent,
            shape=list(shape),dtype=spec['dtype'],width=dtype.element_bits//8))
    if sum(f['bytes']for f in fields)>M.MAX_BYTES:
        raise ValueError('integer output export exceeds its complete byte budget')
    addresses,_=_elf_output_symbols(elf_bytes,{f['symbol']:f['bytes']for f in fields})
    for field in fields:field['base']=addresses[field['symbol']]
    ordered=sorted(fields,key=lambda f:f['base'])
    if any(a['base']+a['bytes']!=b['base']for a,b in zip(ordered,ordered[1:])):
        raise ValueError('integer output export refuses gaps, overlaps or unrelated objects')
    blocks=[]
    for field in ordered:
        values=list(M._flatten(expected[field['tensor']]))
        if len(values)*field['width']!=field['bytes']:
            raise ValueError('integer output export oracle extent differs')
        try:blocks.append(b''.join(v.to_bytes(field['width'],'little',signed=True)for v in values))
        except OverflowError as error:raise ValueError('integer output export value exceeds its declared type')from error
    return dict(base=ordered[0]['base'],bytes=sum(f['bytes']for f in ordered),fields=ordered),b''.join(blocks)


@dataclass(frozen=True)
class IntegerSignatureExport:
    elf: tuple[str, str]
    command_buffer: tuple[str, str]

    @classmethod
    def prepare(cls, *, elf, command_buffer):
        ep, cp = _pin(elf), _pin(command_buffer)
        value = cls((ep['path'], ep['sha256']), (cp['path'], cp['sha256']))
        value.materialize(elf)
        return value

    def materialize(self, elf):
        if type(self) is not IntegerSignatureExport:
            raise ValueError('output export requires its exact bounded integer plan')
        original, cb = dict(zip(('path','sha256'), self.elf)), dict(zip(('path','sha256'), self.command_buffer))
        if _pin(original['path']) != original or _pin(cb['path']) != cb or _pin(elf)['sha256'] != original['sha256']:
            raise ValueError('output export source or original executable changed')
        region, expected = _output_region(Path(original['path']).read_bytes(), json.loads(Path(cb['path']).read_bytes()))
        return region, expected, M.signature_symbols(region['base'], region['bytes'])

    def record(self):
        return dict(schema=SCHEMA, elf=dict(zip(('path','sha256'),self.elf)),
                    command_buffer=dict(zip(('path','sha256'),self.command_buffer)))

    @classmethod
    def reopen(cls, record):
        if record.get('schema') != SCHEMA or set(record) != {'schema','elf','command_buffer'}:
            raise ValueError('unsupported ordinary output-export selection')
        value=cls(tuple(record['elf'][k]for k in ('path','sha256')),
                  tuple(record['command_buffer'][k]for k in ('path','sha256')))
        value.materialize(Path(value.elf[0]))
        return value

    def commands(self, argv, *, elf, supplement, signature):
        self.materialize(elf)
        if (not argv or argv[-1] != str(elf)
                or any(arg.startswith(('+signature=', '+signature-granularity=', '+symbol-elf=')) for arg in argv)):
            raise ValueError('ordinary command has ambiguous ELF or preexisting output-export flags')
        return [*argv[:-1], '+symbol-elf='+str(supplement), '+signature='+str(signature),
                '+signature-granularity=1', argv[-1]]

    def values(self, signature):
        region, expected, _ = self.materialize(Path(self.elf[0]))
        path=F.canonical(Path(signature));raw=path.read_bytes()
        data=M.signature_bytes(raw,region['bytes'])
        if data != expected or path.read_bytes() != raw:
            raise ValueError('ordinary same-ELF complete output differs from its numerical source')
        result={}
        for field in region.get('fields',[region]):
            width=int(field['dtype'][1:])//8
            offset=field['base']-region['base']
            block=data[offset:offset+field['bytes']]
            values=[int.from_bytes(block[i:i+width],'little',signed=True)for i in range(0,len(block),width)]
            cols=field['shape'][-1]
            result[field['tensor']]=[values[i:i+cols]for i in range(0,len(values),cols)]
        return result


def reopen_values(observation, process):
    """Reopen the actual same-ELF signature producer; a saved match bit is insufficient."""
    export=observation.get('output_export')
    if not isinstance(export,dict):
        raise ValueError('ordinary replay has no complete output export')
    plan=IntegerSignatureExport.reopen(export['plan'])
    row=I.verify(Path(process['path']))
    if _pin(process['path'])['sha256'] != process['sha256'] or row.get('stage') != 'archive_screen_count':
        raise ValueError('ordinary output export lost its actual observation producer')
    signature=process['full_output_signature'];supplement=export['supplement']
    if _pin(signature['path']) != signature or _pin(supplement['path']) != supplement:
        raise ValueError('ordinary output-export product changed')
    _,_,symbols=plan.materialize(Path(plan.elf[0]))
    if (Path(supplement['path']).read_bytes()!=symbols or supplement not in row['inputs']
            or signature not in row['outputs']
            or not any(p['sha256']==plan.elf[1]for p in row['inputs'])):
        raise ValueError('ordinary output-export ELF/supplement/signature producer join differs')
    args=row['argv']
    for option in ['+symbol-elf='+supplement['path'], '+signature='+signature['path'], '+signature-granularity=1']:
        if args.count(option)!=1:
            raise ValueError('ordinary output-export command differs')
    if sum(arg.startswith('+symbol-elf=')for arg in args)!=1 or sum(arg.startswith('+signature=')for arg in args)!=1:
        raise ValueError('ordinary output-export command is ambiguous')
    return plan.values(Path(signature['path']))

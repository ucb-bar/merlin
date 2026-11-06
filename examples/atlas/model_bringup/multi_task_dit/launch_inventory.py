"""Inventory an existing full forward export and capture actual kernel boundaries.

No export, graph rewrite, quantization, model download, or device execution is
performed. Family placement is a geometric candidate, never FP32 support.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re


def checked(path):
    path = Path(path).absolute()
    for candidate in (path, path.resolve()):
        if any(re.search('hammer|vlsi', part, re.I) for part in candidate.parts):
            raise ValueError('restricted path')
    return path


def identity(path):
    path = checked(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def load_exporter(library):
    spec = importlib.util.spec_from_file_location('atlas_export_ops', checked(library) / 'coverage/export_ops.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def phase_records(records):
    """Use each predictor's first time-MLP projection as an iteration boundary."""
    step = -1
    for row in records:
        module = row['module']
        if module.endswith('noise_predictor.time_mlp.1') and row['op'] == 'aten.linear.default':
            step += 1
        if 'observation_encoder.vision_encoder' in module:
            phase = 'vision'
        elif 'observation_encoder.text_encoder' in module:
            phase = 'text'
        elif 'noise_predictor' in module:
            phase = 'denoise'
        elif step >= 0:
            phase = 'scheduler'
        else:
            phase = 'conditioning'
        row.update(phase=phase, step=step if step >= 0 else 0)


def classify(row):
    op = row['op']
    if op == 'aten.linear.default':
        return ['projection_gemm'], 'FP32 input/weight are unsupported; FP8 staging, padding, full-K paneling, bias, and numerical qualification required'
    if op.startswith('aten.conv2d'):
        return ['projection_gemm'], 'patch projection requires explicit im2col/gather and FP8 staging; current ResNet 3x3 gather does not implement CLIP 16x16 stride-16'
    if op == 'aten.scaled_dot_product_attention.default':
        dim = row['inputs'][0][0][-1]
        return (['attention'] if dim in (64, 128, 256) else []), 'FP32 Q/K/V unsupported; mask/scale and output boundary require validation; no whole-model attention execution established'
    if 'layer_norm' in op:
        return [], 'LayerNorm mean/variance and learned bias are not RMSNorm; no complete matching family'
    if 'gelu' in op:
        return ['smol_fusion'], 'standalone GELU and CLIP QuickGELU are not the qualified fused bias/GELU(tanh)/FP8 boundary'
    if 'silu' in op:
        return ['epilogue'], 'standalone SiLU is not the qualified gate/up multiply FP8 epilogue'
    name = op.split('.')[1] if op.startswith('aten.') else ''
    if name in {'view', 'reshape', 'transpose', 'permute', 'unsqueeze', 'squeeze', 'expand', 'slice', 'select', 'cat', 'stack', 'split', 'chunk', 'flatten', 'to', 'embedding', 'index', 'arange', 'full', 'new_ones', 'alias'}:
        return [], 'layout, indexing, or constant/control operation; no generated model host dispatcher'
    return [], 'no complete Atlas family for this FP32 operation in the model dataflow'


def projection_shapes(records):
    groups = OrderedDict()
    for row in records:
        if row['op'] != 'aten.linear.default':
            continue
        a, weight = row['inputs'][:2]
        m, k, n = math.prod(a[0][:-1]), a[0][-1], weight[0][0]
        key = f'{m}x{k}x{n}'
        group = groups.setdefault(key, {'id': key, 'm': m, 'k': k, 'n': n, 'count': 0, 'semantic_flops_each': 2*m*k*n, 'node_indices': [], 'modules': []})
        group['count'] += 1
        group['node_indices'].append(row['index'])
        if row['module'] not in group['modules']:
            group['modules'].append(row['module'])
    return list(groups.values())


def document(program, exporter, source, receipt=None):
    records = []
    exporter.walk(program.graph_module, '', records)
    phase_records(records)
    for row in records:
        family, reason = classify(row)
        row.update(candidate_families=family, direct_fp32_support=False, uncovered_reason=reason)
    ops = exporter.aggregate(records)
    for row in ops:
        row.update(candidate_families=classify(row)[0], direct_fp32_support=False, uncovered_reason=classify(row)[1])
    kinds = Counter(row['op'] for row in records)
    candidate_flops = sum(row['flops'] - row.get('bias_flops', 0) for row in records if row['op'] == 'aten.linear.default')
    total = sum(row['flops'] for row in records)
    raw = Counter(str(n.target) for n in program.graph.nodes if n.op == 'call_function')
    doc = {'schema': 'atlas.coverage.inventory.v2', 'model': 'multi_task_dit',
           'scope': 'All operations in the supplied complete action-chunk export, including every denoising iteration; dataset processing and queue control outside export ABI are excluded.',
           'sources': {'export': identity(source), 'export_ops': identity(exporter.__file__)}, 'node_count':len(records), 'method':'torch.export.load existing full forward artifact', 'environment':exporter.environment(), 'ops': ops, 'calls': exporter.calls(records),
           'attention': exporter.attention(records), 'projection_shapes': projection_shapes(records),
           'counting': {'semantic_flops': 'atlas-kernels coverage/export_ops.py: multiply-add=2, linear/conv bias=1/output, SDPA=4*B*H*Q*K*D+5*B*H*Q*K; other compute approximated per output/reduction element; layout=0. Not cycles or model accuracy.',
                        'raw_call_function_occurrences': sum(raw.values()), 'inventoried_occurrences': len(records),
                        'excluded_noncompute_counts': {k:v for k,v in raw.items() if k in exporter.SKIP}},
           'summary': {'op_kind_count': len(kinds), 'op_counts': dict(sorted(kinds.items())), 'semantic_flops': total,
                       'projection_calls': len(exporter.calls(records)), 'linear_calls': sum(r['op']=='aten.linear.default' for r in records),
                       'attention_calls': len(exporter.attention(records)), 'denoising_iterations': 1+max((r['step'] for r in records if r['phase']=='denoise'), default=-1),
                       'linear_candidate_flops_excluding_bias': candidate_flops,
                       'uncovered_fp32_flops': total, 'qualified_fp32_flops': 0},
           'mapping_policy': {'status': 'CANDIDATE_ONLY', 'quantization': 'none performed', 'full_model_execution': 'PENDING', 'bias': 'separate uncovered component',
                              'family_manifests': {family: identity(checked(exporter.__file__).parent.parent / 'kernels' / family / 'kernel.yaml') for family in ('projection_gemm', 'attention', 'epilogue', 'smol_fusion')}},
           'capture_crosscheck': {'status': 'NOT_RUN'}}
    if receipt:
        doc['sources']['capture_receipt'] = identity(receipt)
        doc['session'] = json.loads(checked(receipt).read_text()).get('scope', {})
    return doc, records


def capture_boundaries(program, inputs_path, out, records, expected_path=None):
    import numpy as np
    import torch
    from torch.fx import Interpreter
    out.mkdir(parents=True, exist_ok=True)
    groups = {r['id']: r for r in projection_shapes(records)}
    seen, captures, attention_seen, attention_captures = set(), [], set(), []
    attention_counts = Counter(tuple(tuple(x[0]) for x in row['inputs'][:3]) for row in records if row['op'] == 'aten.scaled_dot_product_attention.default')
    module = program.module()
    metadata = {n.name:n for n in program.graph.nodes}

    class Recorder(Interpreter):
        def run_node(self, node):
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            result = super().run_node(node)
            if node.op != 'call_function':
                return result
            if str(node.target) == 'aten.linear.default':
                a, w = args[:2]
                m, k, n = a.numel()//a.shape[-1], a.shape[-1], w.shape[0]
                key = f'{m}x{k}x{n}'
                if key not in seen:
                    seen.add(key)
                    A, W = a.reshape(m,k).detach().contiguous(), w.T.detach().contiguous()
                    values = {'A': A.numpy(), 'W': W.numpy(), 'expected_without_bias': torch.mm(A,W).numpy(), 'original_output': result.reshape(m,n).numpy()}
                    if len(args)>2 and args[2] is not None:
                        values['bias'] = args[2].numpy()
                    path = out / f'projection-{key}.npz'
                    np.savez(path, **values)
                    original = metadata[node.name]
                    stack = original.meta.get('nn_module_stack') or {}
                    captures.append({**groups[key], 'artifact': identity(path), 'node': node.name, 'module': list(stack.values())[-1][0] if stack else '',
                                     'source_parameter': str(original.args[1].target), 'keys': {name:list(value.shape) for name,value in values.items()},
                                     'array_sha256': {name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in values.items()},
                                     'scope': 'first actual activation per full logical M/K/N; multiplicity is inventory count, other calls have different values'})
            elif str(node.target) == 'aten.scaled_dot_product_attention.default':
                q,k,v = args[:3]
                key = '_'.join('x'.join(map(str,x.shape)) for x in (q,k,v))
                if key not in attention_seen:
                    attention_seen.add(key)
                    values = {'Q': q.contiguous().numpy(), 'K': k.contiguous().numpy(), 'V': v.contiguous().numpy(), 'original_output': result.contiguous().numpy()}
                    if len(args)>3 and isinstance(args[3], torch.Tensor):
                        values['mask'] = args[3].contiguous().numpy()
                    path = out / f'attention-{len(attention_captures):02d}.npz'
                    np.savez(path, **values)
                    mask = args[3] if len(args)>3 and isinstance(args[3],torch.Tensor) else None
                    causal = bool(kwargs.get('is_causal', args[5] if len(args)>5 else False))
                    mask_kind = 'causal' if causal else 'none'
                    if mask is not None:
                        visible = mask if mask.dtype == torch.bool else mask >= 0
                        triangle = torch.ones((q.shape[-2],k.shape[-2]),dtype=torch.bool).tril()
                        mask_kind = 'none' if bool(visible.all()) else 'causal' if bool(torch.all(visible == triangle)) else 'arbitrary'
                    original = metadata[node.name]
                    stack = original.meta.get('nn_module_stack') or {}
                    attention_captures.append({'family':'attention', 'node':node.name, 'module':list(stack.values())[-1][0] if stack else '',
                                               'batch':math.prod(q.shape[:-3]), 'heads':q.shape[-3], 'kv_heads':k.shape[-3], 'rows':q.shape[-2], 'keys':k.shape[-2], 'head_dim':q.shape[-1],
                                               'mask':mask_kind, 'count':attention_counts[tuple(tuple(x.shape) for x in (q,k,v))],
                                               'artifact':identity(path), 'array_shapes':{name:list(value.shape) for name,value in values.items()},
                                               'scalar_kwargs': {name:value for name,value in kwargs.items() if isinstance(value,(int,float,bool))},
                                               'scalar_args': {str(i):value for i,value in enumerate(args[3:],3) if isinstance(value,(int,float,bool))}})
            return result

    with np.load(checked(inputs_path), allow_pickle=False) as archive:
        names = list(archive.files)
        inputs = tuple(torch.from_numpy(archive[name].copy()) for name in names)
    with torch.inference_mode():
        actual = Recorder(module).run(*inputs)
    if isinstance(actual,(list,tuple)) and len(actual)==1:
        actual=actual[0]
    check = {'status':'PENDING', 'reason':'no eager output supplied'}
    if expected_path:
        expected = np.load(checked(expected_path), allow_pickle=False)
        value = actual.numpy()
        check = {'status':'PASS' if np.array_equal(value,expected) else 'FAIL', 'exact':bool(np.array_equal(value,expected)), 'max_abs':float(np.max(np.abs(value-expected))), 'shape':list(value.shape)}
        if check['status']=='FAIL':
            raise RuntimeError(f'Boundary interpreter output mismatch: {check}')
    if seen != set(groups):
        raise RuntimeError(f'Missing projection geometry: {set(groups)-seen}')
    return {'schema':'merlin.multi_task_dit.boundaries.v1', 'status':'PASS', 'source_inputs':identity(inputs_path), 'input_order':names,
            'coverage':'one first occurrence per distinct full linear M/K/N and per distinct Q/K/V geometry', 'projection_count':len(captures),
            'projection_logical_occurrences':sum(c['count'] for c in captures), 'projections':captures, 'attention':attention_captures,
            'full_forward_eager_check':check, 'qualification':'PENDING; captured FP32 values only, no quantization or Atlas execution'}


def capture_live_boundaries(loader_path, out, expected_path=None):
    """Hook a full eager policy without waiting for export/compiler processing."""
    import numpy as np
    import torch
    import torch.nn.functional as functional
    loader_path = checked(loader_path)
    spec = importlib.util.spec_from_file_location('atlas_boundary_loader', loader_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    model, inputs = loader.get_model_and_inputs()
    model.eval().requires_grad_(False)
    out = checked(out)
    out.mkdir(parents=True, exist_ok=True)
    groups, attention, stack, handles = OrderedDict(), OrderedDict(), [], []

    def record_linear(module, args, output):
        a, w = args[0], module.weight
        m, k, n = a.numel()//a.shape[-1], a.shape[-1], w.shape[0]
        key = f'{m}x{k}x{n}'
        if key in groups:
            groups[key]['count'] += 1
            return
        A, W = a.reshape(m,k).detach().contiguous(), w.T.detach().contiguous()
        values = {'A':A.numpy(), 'W':W.numpy(), 'expected_without_bias':torch.mm(A,W).numpy(), 'original_output':output.reshape(m,n).numpy()}
        if module.bias is not None:
            values['bias'] = module.bias.numpy()
        path = out / f'projection-{key}.npz'
        np.savez(path, **values)
        groups[key] = {'id':key,'m':m,'k':k,'n':n,'count':1,'semantic_flops_each':2*m*k*n,'artifact':identity(path),
                       'node':stack[-1], 'module':stack[-1], 'source_parameter':stack[-1]+'.weight',
                       'keys':{name:list(value.shape) for name,value in values.items()},
                       'array_sha256':{name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in values.items()},
                       'scope':'first actual activation per full logical M/K/N; count covers all calls of this geometry'}

    original_sdpa = functional.scaled_dot_product_attention

    def record_sdpa(q,k,v,*args,**kwargs):
        output = original_sdpa(q,k,v,*args,**kwargs)
        mask = kwargs.get('attn_mask',args[0] if args else None)
        causal = bool(kwargs.get('is_causal',args[2] if len(args)>2 else False))
        mask_kind = 'causal' if causal else 'none'
        if mask is not None:
            visible = mask if mask.dtype == torch.bool else mask >= 0
            triangle = torch.ones((q.shape[-2],k.shape[-2]),dtype=torch.bool).tril()
            mask_kind = 'none' if bool(visible.all()) else 'causal' if bool(torch.all(visible == triangle)) else 'arbitrary'
        key = (tuple(q.shape),tuple(k.shape),tuple(v.shape),mask_kind)
        if key in attention:
            attention[key]['count'] += 1
            return output
        values = {'Q':q.contiguous().numpy(),'K':k.contiguous().numpy(),'V':v.contiguous().numpy(),'original_output':output.contiguous().numpy()}
        if mask is not None:
            values['mask'] = mask.contiguous().numpy()
        path = out / f'attention-{len(attention):02d}.npz'
        np.savez(path,**values)
        attention[key] = {'family':'attention','node':stack[-1],'module':stack[-1], 'batch':math.prod(q.shape[:-3]),'heads':q.shape[-3],
                          'kv_heads':k.shape[-3],'rows':q.shape[-2],'keys':k.shape[-2],'head_dim':q.shape[-1],'mask':mask_kind,'count':1,
                          'artifact':identity(path),'array_shapes':{name:list(value.shape) for name,value in values.items()},
                          'scalar_kwargs':{name:value for name,value in kwargs.items() if isinstance(value,(int,float,bool))},
                          'scalar_args':{str(i+3):value for i,value in enumerate(args) if isinstance(value,(int,float,bool))}}
        return output

    def leave(module,args,output):
        stack.pop()

    for name,module in model.named_modules():
        handles.append(module.register_forward_pre_hook(lambda module,args,name=name:stack.append(name)))
        if isinstance(module,torch.nn.Linear):
            handles.append(module.register_forward_hook(record_linear))
        handles.append(module.register_forward_hook(leave))
    functional.scaled_dot_product_attention = record_sdpa
    try:
        with torch.inference_mode():
            actual = model(*inputs).numpy()
    finally:
        functional.scaled_dot_product_attention = original_sdpa
        for handle in handles:
            handle.remove()
    np.save(out/'eager.npy',actual)
    check = {'status':'PENDING','reason':'no independent eager output supplied'}
    if expected_path:
        expected = np.load(checked(expected_path),allow_pickle=False)
        check = {'status':'PASS' if np.array_equal(actual,expected) else 'FAIL','exact':bool(np.array_equal(actual,expected)),
                 'max_abs':float(np.max(np.abs(actual-expected))),'shape':list(actual.shape)}
        if check['status']=='FAIL':
            raise RuntimeError(f'Live capture differs from independent eager output: {check}')
    scope = loader.describe(model,inputs) if hasattr(loader,'describe') else {}
    result = {'schema':'merlin.multi_task_dit.boundaries.v1','status':'PASS','source_loader':identity(loader_path), 'session':scope,
              'coverage':'full eager forward hooks: every nn.Linear and functional SDPA; first actual boundary per full geometry, all invocation multiplicities counted; full ATen census requires export inventory',
              'projection_count':len(groups),'projection_logical_occurrences':sum(row['count'] for row in groups.values()),
              'attention_logical_occurrences':sum(row['count'] for row in attention.values()), 'projections':list(groups.values()),'attention':list(attention.values()),
              'full_forward_eager_check':check,'qualification':'PENDING; captured FP32 values only, no quantization or Atlas execution'}
    (out.parent/'boundaries.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',type=Path)
    parser.add_argument('--loader',type=Path,help='Live eager boundary capture; --export is unnecessary')
    parser.add_argument('--threads',type=int,default=4,choices=(1,2,3,4))
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--receipt',type=Path)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--expected',type=Path)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    if args.loader:
        result=capture_live_boundaries(args.loader,checked(args.out)/'boundaries',args.expected)
        print(json.dumps({key:result[key] for key in ('projection_count','projection_logical_occurrences','attention_logical_occurrences','full_forward_eager_check')},indent=2),flush=True)
        return
    if not args.export:
        parser.error('--export or --loader is required')
    program=torch.export.load(checked(args.export))
    exporter=load_exporter(args.library)
    out=checked(args.out)
    out.mkdir(parents=True,exist_ok=True)
    receipt=args.receipt
    if receipt:
        snapshot=out/'source-capture-receipt.json'
        snapshot.write_bytes(checked(receipt).read_bytes())
        receipt=snapshot
    doc,records=document(program,exporter,args.export,receipt)
    (out/'nodes.json').write_text(json.dumps(records)+'\n')
    (out/'inventory.json').write_text(json.dumps(doc,indent=2)+'\n')
    mapping = {'schema':'atlas.coverage.mapping.v1', 'kernels':{'projection_gemm':'candidate geometry only, FP8 staging required', 'attention':'candidate geometry only, FP8 staging required'}, 'builds':{},
               'bench':{'multi_task_dit':{'panel_rows':128,'panel_columns':[{'n':128}]}},
               'rules':{'multi_task_dit':[{'match':{'op':'aten.linear.default'},'where':'atlas:projection_gemm',
                                        'note':'CANDIDATE ONLY: full logical M/K/N, FP32 unsupported, bias separately unmapped; numerical execution qualification required.',
                                        'cite':['atlas-kernels:kernels/projection_gemm/kernel.yaml']},
                                       {'match':{'op':'aten.scaled_dot_product_attention.default'},'where':'atlas:attention',
                                        'note':'CANDIDATE ONLY: actual mask and scale retained in boundaries.json; this mapping is not a qualified FP32 implementation.',
                                        'cite':['atlas-kernels:kernels/attention/kernel.yaml']}]}}
    (out/'mapping.yaml').write_text(json.dumps(mapping,indent=2)+'\n')
    if args.inputs:
        boundaries=capture_boundaries(program,args.inputs,out/'boundaries',records,args.expected)
        boundaries.update(source_export=doc['sources']['export'],session=doc.get('session',{}))
        (out/'boundaries.json').write_text(json.dumps(boundaries,indent=2)+'\n')
    print(json.dumps(doc['summary'],indent=2),flush=True)


if __name__=='__main__':
    main()

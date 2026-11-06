"""Desugar scalar affine expressions into the published modeled vocabulary.

The frozen common attribute schema reserves activation for an array. AFFINE_MAP
requires a scalar activation token in its semantic declaration, so it cannot be
persisted with that token under this schema. This pass uses explicit modeled
fills, exact integer sums, and accumulator readouts instead of changing a schema
or serializing a different expression than the LLVM lowering implements.
"""
from copy import deepcopy
from fractions import Fraction
from math import prod


def _fraction(value):
    return Fraction(value) if isinstance(value, float) else Fraction(str(value))


def _effective_scale(attrs):
    if 'scale_numerator' in attrs:
        return Fraction(int(attrs['scale_numerator']), int(attrs['scale_denominator']))
    return _fraction(attrs.get('readout_scale', attrs.get('acc_scale', 1)))


def _readout_attrs(attrs, dtype, *, divisor=1, relu=False):
    result = {'output_dtype': dtype, 'epilogue': ['relu'] if relu else []}
    if attrs.get('readout_semantics') == 'f32_product_rne':
        result['readout_semantics'] = 'f32_product_rne'
        if divisor != 1:
            if 'acc_scale_tensor' in attrs:
                raise ValueError('runtime readout scale divided by an average count needs a retained scale formation')
            if 'acc_scale_by_column' in attrs:
                result['acc_scale_by_column'] = [float(_fraction(s) / divisor) for s in attrs['acc_scale_by_column']]
            else:
                result['acc_scale'] = float(_fraction(attrs.get('acc_scale', attrs.get('readout_scale', 1))) / divisor)
        elif 'acc_scale_by_column' in attrs:
            result['acc_scale_by_column'] = list(attrs['acc_scale_by_column'])
        elif 'acc_scale_tensor' in attrs:
            result['acc_scale_tensor'] = attrs['acc_scale_tensor']
        else:
            result['acc_scale'] = attrs.get('acc_scale', attrs.get('readout_scale', 1))
        result['epilogue'].append('acc_scale')
    else:
        scale = _effective_scale(attrs) / divisor
        result.update({'scale_numerator': str(scale.numerator), 'scale_denominator': str(scale.denominator)})
        for key in ('zero_point', 'quant_min', 'quant_max', 'rounding'):
            if key in attrs:
                result[key] = attrs[key]
    return result


class _Normalizer:
    def __init__(self, cb):
        self.cb = cb
        self.tensors = cb.setdefault('tensors', {})
        self.commands = []
        self.serial = 0
        self.intervals = {}
        self.source = None

    def temporary(self, shape, dtype, purpose):
        while True:
            name = '_normalized_' + purpose + '_' + str(self.serial)
            self.serial += 1
            if name not in self.tensors:
                break
        self.tensors[name] = {'shape': list(shape), 'dtype': dtype, 'role': 'intermediate'}
        return name

    def emit(self, opcode, operands=None, attrs=None):
        command = {k: deepcopy(v) for k, v in self.source.items() if k.startswith('_')}
        command['opcode'] = opcode
        if operands:
            command['operands'] = dict(operands)
        if attrs:
            command['attributes'] = dict(attrs)
        self.commands.append(command)
        return command

    def fence(self):
        self.emit('FENCE')

    def interval(self, tensor):
        if tensor in self.intervals:
            return self.intervals[tensor]
        dtype = self.tensors[tensor]['dtype']
        bits = int(''.join(c for c in dtype if c.isdigit()) or '32')
        return (0, 2**bits-1) if dtype.startswith('u') else (-2**(bits-1), 2**(bits-1)-1)

    def residual(self, lhs, rhs, weights, dst, activation=False):
        left, right = self.interval(lhs), self.interval(rhs)
        ranges = [(min(w*lo, w*hi), max(w*lo, w*hi)) for (lo,hi),w in zip((left,right),weights)]
        low, high = sum(x[0] for x in ranges), sum(x[1] for x in ranges)
        if activation:
            low, high = max(low, 0), max(high, 0)
        self.intervals[dst] = low, high
        self.emit('RESIDUAL_ADD', {'lhs': lhs, 'rhs': rhs, 'dst': dst},
                  {'lhs_scale': float(weights[0]), 'rhs_scale': float(weights[1]),
                   'bound_lsb': 0, 'epilogue': ['relu'] if activation else [],
                   'output_dtype': self.tensors[dst]['dtype'],
                   'lhs_bound': max(abs(x) for x in left), 'rhs_bound': max(abs(x) for x in right),
                   'accumulator_min': low, 'accumulator_max': high,
                   'input_bound': max(abs(low), abs(high))})

    def commit(self, src, dst, attrs):
        low, high = self.interval(src)
        attrs = dict(attrs)
        attrs.update({'input_bound': max(abs(low),abs(high)), 'accumulator_min': low, 'accumulator_max': high})
        self.fence()
        self.emit('COMMIT', {'src': src, 'dst': dst}, attrs)
        if self.tensors[dst]['dtype'] in ('i8', 'int8'):
            self.intervals[dst] = (-128, 127)

    def move(self, src, dst, form='reshape'):
        self.fence()
        self.emit('MOVEMENT', {'src':src,'dst':dst}, {'form':form,'output_dtype':self.tensors[dst]['dtype']})
        self.intervals[dst]=self.interval(src)

    def concatenate(self, sources, axis, purpose):
        level=0
        while len(sources)>1:
            groups=[]
            for at in range(0,len(sources),8):
                chunk=sources[at:at+8]
                if len(chunk)==1:
                    groups.append(chunk[0]);continue
                shape=list(self.tensors[chunk[0]]['shape'])
                shape[axis]=sum(self.tensors[n]['shape'][axis] for n in chunk)
                target=self.temporary(shape,'i8',purpose+'_'+str(level))
                self.fence()
                self.emit('CONCAT',{**{'src'+str(i):name for i,name in enumerate(chunk)},'dst':target},
                          {'axis':axis,'output_dtype':'i8'})
                self.intervals[target]=(min(self.interval(n)[0] for n in chunk),max(self.interval(n)[1] for n in chunk))
                groups.append(target)
            sources=groups;level+=1
        return sources[0]

    def matrix_accumulator(self, sources, weights, shape, constant=0):
        """Return a real modeled MATMUL accumulator and optional bias tensor."""
        columns=shape[-1] if shape else 1
        rows=prod(shape)//columns
        segments=[]
        coefficients=[]
        ranges=[]
        for source,coefficient in zip(sources,weights):
            lo,hi=self.interval(source)
            ranges.append((min(lo*coefficient,hi*coefficient),max(lo*coefficient,hi*coefficient)))
            if not coefficient:
                continue
            if self.tensors[source]['dtype'] not in ('i8','int8'):
                raise ValueError('narrow affine contraction requires signed-byte source tensors')
            current=source
            if list(self.tensors[current]['shape'])!=list(shape):
                expanded=self.temporary(shape,'i8','broadcast_operand')
                self.move(current,expanded,'broadcast');current=expanded
            if self.tensors[current]['shape']!=[rows,columns]:
                matrix=self.temporary([rows,columns],'i8','matrix_operand')
                self.move(current,matrix);current=matrix
            remaining=coefficient
            while remaining:
                part=min(remaining,127) if remaining>0 else max(remaining,-128)
                remaining-=part
                segments.append(current);coefficients.append(part)
        if not segments:
            source=sources[0]
            if self.tensors[source]['dtype'] not in ('i8','int8'):
                raise ValueError('zero affine contraction requires a signed-byte carrier')
            current=source
            if list(self.tensors[current]['shape'])!=list(shape):
                expanded=self.temporary(shape,'i8','broadcast_operand')
                self.move(current,expanded,'broadcast');current=expanded
            if self.tensors[current]['shape']!=[rows,columns]:
                matrix=self.temporary([rows,columns],'i8','matrix_operand')
                self.move(current,matrix);current=matrix
            segments=[current];coefficients=[0]
        matrix_source=self.concatenate(segments,1,'source_panel')
        inner=len(segments)*columns
        flat=self.temporary([inner*columns],'i8','coefficient_zero')
        self.emit('FILL',{'dst':flat},{'value_code':0,'output_dtype':'i8'})
        for index,coefficient in enumerate(coefficients):
            if coefficient==0:continue
            diagonal=self.temporary([columns],'i8','coefficient_diagonal')
            self.emit('FILL',{'dst':diagonal},{'value_code':coefficient,'output_dtype':'i8'})
            updated=self.temporary([inner*columns],'i8','coefficient_insert')
            self.fence()
            self.emit('INSERT_SLICE',{'dest':flat,'src':diagonal,'dst':updated},
                      {'offsets':[index*columns*columns],'strides':[columns+1],'output_dtype':'i8'})
            flat=updated
        matrix_weight=self.temporary([inner,columns],'i8','coefficient_matrix')
        self.move(flat,matrix_weight)
        accumulator='_normalized_accumulator_'+str(self.serial)
        self.serial+=1
        self.fence()
        self.emit('MATMUL',{'lhs':matrix_source,'rhs':matrix_weight,'dst':accumulator})
        low=constant+sum(r[0] for r in ranges)
        high=constant+sum(r[1] for r in ranges)
        if low<-(1<<31) or high>(1<<31)-1:
            raise ValueError('affine full declared integer domain exceeds signed i32')
        self.intervals[accumulator]=(low,high)
        bias=None
        if constant:
            bias=self.temporary([columns],'i32','affine_bias')
            self.emit('FILL',{'dst':bias},{'value_code':constant,'output_dtype':'i32'})
            self.intervals[bias]=(constant,constant)
        return accumulator,bias,rows,columns

    def matrix_readout(self, sources, weights, dst, attrs, *, constant=0, relu=False):
        shape=self.tensors[dst]['shape']
        dtype=self.tensors[dst]['dtype']
        attributes=dict(attrs)
        if not any(k in attributes for k in ('acc_scale_by_column','acc_scale_tensor')):
            scale=attributes.get('acc_scale',attributes.get('readout_scale',1)) if attributes.get('readout_semantics')=='f32_product_rne' else _effective_scale(attributes)
            if scale==0 and attributes.get('zero_point',0)==0:
                self.emit('FILL',{'dst':dst},{'value_code':0,'output_dtype':dtype})
                self.intervals[dst]=(0,0)
                return
        if attributes.get('readout_semantics')!='f32_product_rne':
            scale=_effective_scale(attributes)
            if scale<0:
                if relu:
                    raise ValueError('negative affine scale after activation needs its original exact construction')
                weights=[-w for w in weights]
                constant=-constant
                attributes['scale_numerator']=str(-scale.numerator)
                attributes['scale_denominator']=str(scale.denominator)
        accumulator,bias,rows,columns=self.matrix_accumulator(sources,weights,shape,constant)
        outattrs=_readout_attrs(attributes,'i8' if dtype in ('i32','int32') and relu else dtype,relu=relu)
        if attributes.get('readout_semantics')!='f32_product_rne':
            scale=_effective_scale(attributes)
            if attributes.get('zero_point',0)!=0 or attributes.get('quant_min',-128)!=-128 or attributes.get('quant_max',127)!=127:
                raise ValueError('nonstandard affine byte quantizer has no modeled device readout')
            from affine_reduction import _prove_readout
            bound=max(abs(v) for v in self.interval(accumulator))
            if not _prove_readout(scale,bound):
                raise ValueError('affine exact-rational readout lacks full-interval f32 equivalence')
            outattrs={'output_dtype':outattrs['output_dtype'],'epilogue':['relu'] if relu else [],'acc_scale':float(scale)}
            if scale!=1:outattrs['epilogue'].append('acc_scale')
        if bias:
            outattrs['bias']=bias
            outattrs['epilogue'].insert(0,'bias_add')
        target=dst
        widened=dtype in ('i32','int32') and relu
        if widened:
            if self.interval(accumulator)[1]>127:
                raise ValueError('rectify-through-byte proof requires complete positive bound at most 127')
            target=self.temporary([rows,columns],'i8','rectified_bytes')
        elif shape!=[rows,columns]:
            target=self.temporary([rows,columns],dtype,'matrix_readout')
        self.commit(accumulator,target,outattrs)
        if target!=dst:
            self.move(target,dst)

    def affine(self, c):
        o,a=c['operands'],c.get('attributes',{})
        dst=o['dst']
        shape,dtype=self.tensors[dst]['shape'],self.tensors[dst]['dtype']
        sources=[o[k] for k in ('x','y') if k in o]
        weights=[int(w) for w in a['weights']]
        if not sources or len(weights)!=len(sources) or len(sources)>2:
            raise ValueError('affine normalization requires one integer weight per source and at most two sources')
        activation=a.get('activation','none')
        if activation not in ('none','relu'):
            raise ValueError('unsupported affine activation')
        if a.get('operand_scales') is not None:
            if not a.get('per_operand_rounding') or len(a['operand_scales'])!=len(sources):
                raise ValueError('independent operand scales require per-operand rounding')
            updated=[]
            for source,scale in zip(sources,a['operand_scales']):
                narrow=self.temporary(self.tensors[source]['shape'],'i8','operand_readout')
                self.matrix_readout([source],[1],narrow,
                                    {'acc_scale':scale,'readout_semantics':'f32_product_rne'})
                updated.append(narrow)
            sources=updated
        constant=int(a.get('constant',0))
        if dtype not in ('i32','int32') or activation=='relu':
            self.matrix_readout(sources,weights,dst,a,constant=constant,relu=activation=='relu')
            return
        if _effective_scale(a)!=1 or a.get('zero_point',0)!=0:
            raise ValueError('full-width affine result requires unit scale and zero zero-point')
        if len(sources)==1:
            sources.append(sources[0]);weights.append(0)
        if constant:
            partial=self.temporary(shape,'i32','weighted_sum')
            self.residual(sources[0],sources[1],weights,partial)
            literal=self.temporary(shape,'i32','constant')
            self.emit('FILL',{'dst':literal},{'value_code':constant,'output_dtype':'i32'})
            self.intervals[literal]=(constant,constant)
            self.fence()
            self.residual(partial,literal,[1,1],dst)
        else:
            self.residual(sources[0],sources[1],weights,dst)

    def window(self, c):
        a = dict(c.get('attributes',{}))
        activation = a.pop('activation', 'none')
        if activation not in ('none','relu'):
            raise ValueError('unsupported window reduction activation')
        split = activation == 'relu' or a.get('readout_semantics') == 'f32_product_rne'
        if not split:
            output=deepcopy(c)
            output['attributes']=a
            self.commands.append(output)
            return
        # Captured composition reductions have a fixed reduced domain. Native
        # pooling capsules with no activation keep their native modeled command.
        if a.get('mode') != 'reduce':
            raise ValueError('activated non-reduce window readout requires an explicitly represented raw sum')
        o = c['operands']
        src,dst = o['src'],o['dst']
        source_shape = self.tensors[src]['shape']
        dims = [int(d)%len(source_shape) for d in a.get('dims',range(len(source_shape)))]
        count = prod(source_shape[d] for d in set(dims))
        raw = self.temporary(self.tensors[dst]['shape'],'i32','window_accumulator')
        rawattrs = {k:deepcopy(v) for k,v in a.items() if k not in (
            'readout_scale','scale_numerator','scale_denominator','zero_point','quant_min','quant_max','rounding',
            'acc_scale','acc_scale_by_column','acc_scale_tensor','readout_semantics','epilogue','output_dtype')}
        rawattrs.update({'average':False,'readout_scale':1.0,'output_dtype':'i32'})
        self.emit('WINDOW_SUM',{'src':src,'dst':raw},rawattrs)
        lo,hi = self.interval(src)
        initial = int(a.get('initial',0))
        self.intervals[raw] = lo*count+initial,hi*count+initial
        self.commit(raw,dst,_readout_attrs(a,self.tensors[dst]['dtype'],
                                         divisor=count if a.get('average',False) else 1,relu=activation=='relu'))

    def run(self):
        originals = self.cb['commands']
        for c in originals:
            if c['opcode']=='WINDOW_SUM' and c.get('attributes',{}).get('initial')==0:
                c=deepcopy(c)
                c['attributes'].pop('initial',None)
            self.source = c
            op = c['opcode']
            if op == 'AFFINE_MAP':
                self.affine(c)
            elif op == 'WINDOW_SUM' and isinstance(c.get('attributes',{}).get('activation'),str):
                self.window(c)
            else:
                command = deepcopy(c)
                a = command.get('attributes',{})
                if op in ('COMMIT','RESIDUAL_ADD') and isinstance(a.get('activation'),str):
                    activation = a.pop('activation')
                    if activation == 'relu':
                        stages = list(a.get('epilogue',[]))
                        if 'relu' not in stages:
                            stages.insert(0,'relu')
                        a['epilogue'] = stages
                    elif activation != 'none':
                        raise ValueError('unsupported readout activation')
                self.commands.append(command)
                if op == 'FILL':
                    value = int(a['value_code'])
                    self.intervals[c['operands']['dst']] = value,value
                elif op == 'MOVEMENT' and c.get('operands',{}).get('dst') in self.tensors:
                    self.intervals[c['operands']['dst']] = self.interval(c['operands']['src'])
        self.cb['commands'] = self.commands
        return self.cb


def normalize(cb):
    """Return a copy with explicit schema-compatible, semantically modeled stages."""
    # Preserve the frontend's typed source-object identities. Only the mutable
    # persisted program is copied; source inventories remain the same objects.
    result = dict(cb)
    result['commands'] = deepcopy(cb.get('commands', []))
    result['tensors'] = deepcopy(cb.get('tensors', {}))
    result['params'] = deepcopy(cb.get('params', {}))
    if result.get('declined'):
        return result
    if 'outputs' not in result:
        result['outputs'] = [n for n,t in result.get('tensors',{}).items() if t.get('role')=='output']
    try:
        return _Normalizer(result).run()
    except (ValueError,KeyError,TypeError,ZeroDivisionError) as exc:
        result['commands'] = []
        result['declined'] = {'op':'AFFINE_MAP','reason':'Exact affine normalization: '+str(exc)}
        return result

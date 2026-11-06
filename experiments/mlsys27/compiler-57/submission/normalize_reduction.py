"""Represent activated byte reductions with a modeled ones contraction."""
from math import prod
from fractions import Fraction


def normalize(cb):
    if cb.get('declined'):return cb
    tensors=cb['tensors'];out=[];serial=0
    for command in cb['commands']:
        if command['opcode']!='WINDOW_SUM' or command.get('attributes',{}).get('mode')!='reduce' or command.get('attributes',{}).get('integer_wrap'):
            out.append(command);continue
        a=dict(command.get('attributes',{}));initial=int(a.pop('initial',0))
        a.setdefault('readout_scale',1.0)
        relu=a.pop('activation','none')=='relu' or 'relu' in a.get('epilogue',[])
        o=command['operands'];src,dst=o['src'],o['dst'];ss=tensors[src]['shape']
        if not relu and not initial and a.get('readout_semantics')!='f32_product_rne':
            out.append(dict(command,attributes=a));continue
        if tensors[src]['dtype']!='i8':
            cb['commands']=[];cb['declined']={'op':'reduction','reason':'Activated or initialized wide-storage reduction requires a separately proved representation'};return cb
        metadata={k:v for k,v in command.items() if k.startswith('_')}
        def emit(op,operands,attrs=None):
            c=dict(metadata,opcode=op,operands=operands)
            if attrs:c['attributes']=attrs
            out.append(c)
        def temp(shape,dtype='i8'):
            nonlocal serial
            name='_reduce_lowered_'+str(serial);serial+=1
            while name in tensors:name='_reduce_lowered_'+str(serial);serial+=1
            tensors[name]={'shape':list(shape),'dtype':dtype,'role':'intermediate'}
            return name
        axes=sorted({int(d)%len(ss) for d in a.get('dims',range(len(ss)))})
        keep=[d for d in range(len(ss)) if d not in axes]
        perm=keep+axes;r=prod(ss[d] for d in axes);p=prod(ss[d] for d in keep)
        if p!=prod(tensors[dst]['shape']) or 128*r+abs(initial)>(1<<31)-1:
            cb['commands']=[];cb['declined']={'op':'reduction','reason':'Reduction output layout or full-domain accumulator bound is not valid'};return cb
        ordered=src
        if perm!=list(range(len(ss))):
            ordered=temp([ss[d] for d in perm]);emit('MOVEMENT',{'src':src,'dst':ordered},{'form':'permute','perm':perm,'output_dtype':'i8'})
        matrix=ordered
        if tensors[ordered]['shape']!=[p,r]:
            matrix=temp([p,r]);emit('MOVEMENT',{'src':ordered,'dst':matrix},{'form':'reshape','output_dtype':'i8'})
        weight=temp([r,1]);emit('FILL',{'dst':weight},{'value_code':1,'output_dtype':'i8'})
        acc='_reduce_acc_'+str(serial);serial+=1
        emit('MATMUL',{'lhs':matrix,'rhs':weight,'dst':acc})
        result=dst if tensors[dst]['shape']==[p,1] else temp([p,1],tensors[dst]['dtype'])
        attrs={'output_dtype':tensors[dst]['dtype'],'epilogue':['relu'] if relu else []}
        divisor=r if a.get('average',False) else 1
        if a.get('readout_semantics')=='f32_product_rne':
            attrs['readout_semantics']='f32_product_rne'
            attrs['acc_scale']=float(a.get('acc_scale',a.get('readout_scale',1)))/divisor
            attrs['epilogue'].append('acc_scale')
            if 'acc_scale_by_column' in a:raise ValueError('Reduction readout partition must follow its flattened output layout')
        elif tensors[dst]['dtype']=='i8':
            scale=Fraction(int(a['scale_numerator']),int(a['scale_denominator'])) if 'scale_numerator' in a else Fraction(str(a.get('readout_scale',1)))
            scale/=divisor;attrs.update(scale_numerator=str(scale.numerator),scale_denominator=str(scale.denominator))
        if initial:
            bias=temp([1],'i32');emit('FILL',{'dst':bias},{'value_code':initial,'output_dtype':'i32'})
            attrs['bias']=bias;attrs['epilogue'].insert(0,'bias_add')
        emit('COMMIT',{'src':acc,'dst':result},attrs)
        if result!=dst:emit('MOVEMENT',{'src':result,'dst':dst},{'form':'reshape','output_dtype':tensors[dst]['dtype']})
    cb['commands']=out
    return cb

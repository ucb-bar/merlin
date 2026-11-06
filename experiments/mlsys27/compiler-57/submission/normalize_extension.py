"""Represent an integer residual extension in a contraction's actual K domain.

A byte residual c*R added before readout is an ordinary contraction against
signed-byte diagonal coefficients. The coefficients are materialized by modeled
device fills and strided inserts; no runtime tensor values are inspected.
"""
from copy import deepcopy
from fractions import Fraction


def _unit_scale(attrs):
    if 'scale_numerator' in attrs:
        try:
            scale=Fraction(int(attrs['scale_numerator']),int(attrs['scale_denominator']))
        except (KeyError,TypeError,ValueError,ZeroDivisionError):
            return False
        return scale==1
    return attrs.get('readout_scale',attrs.get('acc_scale',1))==1


def _is_plain_seed(command,tensors):
    if command.get('opcode')!='AFFINE_MAP':return False
    operands=command.get('operands',{});attrs=command.get('attributes',{})
    weights=attrs.get('weights')
    if set(operands)!=set(('x','dst')) or not isinstance(weights,list) or len(weights)!=1:
        return False
    if type(weights[0]) is not int or abs(weights[0])>381:
        return False
    if attrs.get('output_dtype','i32') not in ('i32','int32'):
        return False
    if attrs.get('constant',0)!=0 or attrs.get('activation','none')!='none' or not _unit_scale(attrs):
        return False
    if attrs.get('zero_point',0)!=0 or any(k in attrs for k in (
        'operand_scales','per_operand_rounding','acc_scale_by_column','acc_scale_tensor')):
        return False
    source=tensors.get(operands['x'],{});dest=tensors.get(operands['dst'],{})
    return source.get('dtype') in ('i8','int8') and dest.get('dtype') in ('i32','int32') and source.get('shape')==dest.get('shape')


def _read_names(command):
    operands=command.get('operands',{})
    names=[value for key,value in operands.items() if key!='dst' and isinstance(value,str)]
    attrs=command.get('attributes',{})
    names.extend(attrs[key] for key in ('accumulator_seed','bias','acc_scale_tensor')
                 if isinstance(attrs.get(key),str))
    return names


class _ExtensionNormalizer:
    def __init__(self,cb):
        self.cb=cb
        self.tensors=cb.setdefault('tensors',{})
        self.commands=cb.get('commands',[])
        self.serial=0
        self.used_names=set(self.tensors)
        for command in self.commands:
            self.used_names.update(command.get('operands',{}).values())
        self.owner=None

    def temporary(self,shape,purpose):
        while True:
            name='_extension_'+purpose+'_'+str(self.serial)
            self.serial+=1
            if name not in self.used_names:break
        self.used_names.add(name)
        self.tensors[name]={'shape':list(shape),'dtype':'i8','role':'intermediate'}
        index=self.owner.get('_source_op_index')
        if type(index) is int:
            self.cb.setdefault('_temporary_sources',{})[name]={
                'source_op_index':index,'purpose':'source_integer_coefficient_k_extension'}
        return name

    def emit(self,out,opcode,operands=None,attributes=None,owner=None):
        source=self.owner if owner is None else owner
        command={key:deepcopy(value) for key,value in source.items() if key.startswith('_')}
        command['opcode']=opcode
        if operands:command['operands']=dict(operands)
        if attributes:command['attributes']=dict(attributes)
        out.append(command)
        return command

    def concatenate(self,out,sources,axis,purpose):
        sources=list(sources)
        while len(sources)>1:
            next_level=[]
            for at in range(0,len(sources),8):
                group=sources[at:at+8]
                if len(group)==1:
                    next_level.append(group[0]);continue
                shape=list(self.tensors[group[0]]['shape'])
                shape[axis]=sum(self.tensors[name]['shape'][axis] for name in group)
                result=self.temporary(shape,purpose)
                self.emit(out,'FENCE')
                self.emit(out,'CONCAT',
                          {**{'src'+str(i):name for i,name in enumerate(group)},'dst':result},
                          {'axis':axis,'output_dtype':'i8'})
                next_level.append(result)
            sources=next_level
        return sources[0]

    def rewrite(self,command,seed_command,resolved_rhs):
        operands=command['operands'];attrs=command.get('attributes',{})
        src=seed_command['operands']['x']
        lhs=operands['lhs'];rhs=resolved_rhs
        a=self.tensors.get(lhs,{});w=self.tensors.get(rhs,{});r=self.tensors[src]
        if any(t.get('dtype') not in ('i8','int8') for t in (a,w,r)):
            return None
        ashape,wshape,rshape=a.get('shape',[]),w.get('shape',[]),r.get('shape',[])
        if any(len(s)!=2 for s in (ashape,wshape,rshape)):
            return None
        m,k=ashape;n=wshape[-1]
        if wshape[0]!=k or rshape!=[m,n] or any(type(d) is not int or d<=0 for d in (m,k,n)):
            return None
        # This construction uses the actual dense RHS orientation. Independent
        # batches and transposed carriers need their own source-layout rewrite.
        if attrs.get('transpose_rhs') or attrs.get('transpose_b') or attrs.get('transpose_lhs'):
            return None
        coefficient=seed_command['attributes']['weights'][0]
        parts=[];remaining=coefficient
        while remaining:
            part=min(remaining,127) if remaining>0 else max(remaining,-127)
            remaining-=part;parts.append(part)
        if not parts:parts=[0]
        out=[]
        self.owner=seed_command
        extended_lhs=lhs;extended_rhs=rhs
        if parts:
            extended_lhs=self.concatenate(out,[lhs]+[src]*len(parts),1,'operand_panel')
            length=len(parts)*n*n
            current=self.temporary([length],'coefficient_zero')
            self.emit(out,'FILL',{'dst':current},{'value_code':0,'output_dtype':'i8'})
            for segment,part in enumerate(parts):
                values=self.temporary([n],'coefficient_values')
                self.emit(out,'FILL',{'dst':values},{'value_code':part,'output_dtype':'i8'})
                inserted=self.temporary([length],'coefficient_insert')
                self.emit(out,'FENCE')
                self.emit(out,'INSERT_SLICE',{'dest':current,'src':values,'dst':inserted},
                          {'offsets':[segment*n*n],'strides':[n+1],'output_dtype':'i8'})
                current=inserted
            coefficients=self.temporary([len(parts)*n,n],'coefficient_matrix')
            self.emit(out,'FENCE')
            self.emit(out,'MOVEMENT',{'src':current,'dst':coefficients},
                      {'form':'reshape','output_dtype':'i8'})
            extended_rhs=self.concatenate(out,[rhs,coefficients],0,'weight_panel')
        updated_attrs={key:deepcopy(value) for key,value in attrs.items() if key!='accumulator_seed'}
        self.emit(out,'FENCE')
        # The contraction remains attributed to its actual original producer;
        # the source extension owns the introduced coefficient/movement work.
        fused=self.emit(out,'MATMUL',{'lhs':extended_lhs,'rhs':extended_rhs,'dst':operands['dst']},
                        updated_attrs,owner=command)
        indices=[command.get('_source_op_index'),seed_command.get('_source_op_index')]
        if all(type(i) is int for i in indices):
            fused['_fused_source_op_indices']=list(dict.fromkeys(indices))
        return out

    def run(self):
        resident={};resident_epochs={};producers={};consumers={}
        for index,command in enumerate(self.commands):
            dst=command.get('operands',{}).get('dst')
            if dst in self.tensors:producers.setdefault(dst,[]).append(index)
            for name in _read_names(command):consumers.setdefault(name,[]).append(index)
        rewritten={};remove=set();remove_tensors=set()
        for index,command in enumerate(self.commands):
            operands=command.get('operands',{})
            if command['opcode']=='RES_PACK':
                source=operands.get('src');seen=set();capture=index
                while source in resident and source not in seen:
                    seen.add(source);capture=resident_epochs[source];source=resident[source]
                resident[operands['dst']]=source
                resident_epochs[operands['dst']]=capture
                continue
            if command['opcode']=='EVICT':
                resident.pop(operands.get('handle'),None)
                resident_epochs.pop(operands.get('handle'),None);continue
            if command['opcode'] not in ('MATMUL','MATMUL_RESIDENT'):
                continue
            seed=command.get('attributes',{}).get('accumulator_seed')
            seed_producers=producers.get(seed,[])
            if len(seed_producers)!=1 or seed_producers[0]>=index:
                continue
            seed_index=seed_producers[0];seed_command=self.commands[seed_index]
            if not _is_plain_seed(seed_command,self.tensors):
                continue
            rhs_name=operands.get('rhs')
            rhs=resident.get(rhs_name,rhs_name)
            if rhs_name in resident and any(resident_epochs[rhs_name]<writer<index for writer in producers.get(rhs,[])):
                continue
            replacement=self.rewrite(command,seed_command,rhs)
            if replacement is None:
                continue
            rewritten[index]=replacement
            # Keep a seed whose result is observable or read by another source
            # operation. Its independent materialization still has that owner.
            if consumers.get(seed)==[index] and seed not in self.cb.get('outputs',[]) and self.tensors[seed].get('role')=='intermediate':
                remove.add(seed_index);remove_tensors.add(seed)
        output=[]
        for index,command in enumerate(self.commands):
            if index in remove:continue
            output.extend(rewritten.get(index,[command]))
        for seed in remove_tensors:
            self.tensors.pop(seed,None)
            self.cb.get('_temporary_sources',{}).pop(seed,None)
        self.cb['commands']=output
        return self.cb


def normalize(cb):
    """Rewrite recognized static rank-2 byte residual extensions in place.

    Other accumulator initializers remain untouched, with their original typed
    declarations, rather than being silently treated as this construction.
    """
    if cb.get('declined'):
        return cb
    return _ExtensionNormalizer(cb).run()

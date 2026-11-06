"""Realize a prefix reduction with ordinary modeled movement and contraction.

For every line, multiplying by the structural upper-triangular 0/1 matrix
sums input positions through the corresponding output position. The matrix is
produced by device fills and slices; it contains no input-dependent answers.
"""
from math import prod


def normalize_prefix(cb):
    if cb.get('declined'):
        return cb
    tensors=cb['tensors']
    commands=[]
    for source_command in cb['commands']:
        attrs=source_command.get('attributes',{})
        if source_command['opcode']!='WINDOW_SUM' or attrs.get('mode')!='prefix' or attrs.get('integer_wrap'):
            commands.append(source_command)
            continue
        src=source_command['operands']['src'];dst=source_command['operands']['dst']
        shape=list(tensors[src]['shape'])
        axis=int(attrs['axis'])
        if axis<0:axis+=len(shape)
        initial=int(attrs.get('initial',0))
        if not 0<=axis<len(shape) or tensors[src]['dtype']!='i8' or tensors[dst]['dtype']!='i32' or tensors[dst]['shape']!=shape:
            cb['declined']={'op':'prefix_sum','reason':'Prefix construction requires signed-byte input and same-shaped signed-i32 output'}
            cb['commands']=[]
            return cb
        length=shape[axis]
        if length<=0 or 128*length+abs(initial)>2147483647:
            cb['declined']={'op':'prefix_sum','reason':'Prefix full declared byte-domain sum exceeds the exact signed-i32 accumulator bound'}
            cb['commands']=[]
            return cb
        lines=prod(shape)//length
        source_index=source_command.get('_source_op_index')
        stem=dst+'_prefix'
        serial=0
        def tensor(label,dims,dtype='i8'):
            nonlocal serial
            name=stem+'_'+label
            while name in tensors:
                serial+=1;name=stem+'_'+label+'_'+str(serial)
            tensors[name]={'shape':list(dims),'dtype':dtype,'role':'intermediate'}
            return name
        def emit(opcode,operands,attributes=None):
            command={'opcode':opcode,'operands':operands}
            if attributes:command['attributes']=attributes
            if source_index is not None:command['_source_op_index']=source_index
            commands.append(command)
        def move(a,b,attributes):
            emit('MOVEMENT',{'src':a,'dst':b},{'output_dtype':tensors[b]['dtype'],**attributes})
        perm=[i for i in range(len(shape)) if i!=axis]+[axis]
        ordered_shape=[shape[i] for i in perm]
        ordered_src=src
        if perm!=list(range(len(shape))):
            ordered_src=tensor('ordered',ordered_shape)
            move(src,ordered_src,{'form':'permute','perm':perm})
        matrix_src=ordered_src
        if ordered_shape!=[lines,length]:
            matrix_src=tensor('input',[lines,length])
            move(ordered_src,matrix_src,{'form':'reshape'})
        zero_row=tensor('zero_row',[1,length])
        emit('FILL',{'dst':zero_row},{'value_code':0,'output_dtype':'i8'})
        rows=[]
        for row in range(length):
            suffix=tensor('ones_'+str(row),[1,length-row])
            emit('FILL',{'dst':suffix},{'value_code':1,'output_dtype':'i8'})
            if row==0:
                rows.append(suffix)
            else:
                result=tensor('row_'+str(row),[1,length])
                emit('INSERT_SLICE',{'dest':zero_row,'src':suffix,'dst':result},{'offsets':[0,row],'strides':[1,1],'output_dtype':'i8'})
                rows.append(result)
        level=0
        while len(rows)>1:
            joined=[]
            for first in range(0,len(rows),8):
                group=rows[first:first+8]
                if len(group)==1:
                    joined.append(group[0]);continue
                result=tensor('cat_'+str(level)+'_'+str(first),[sum(tensors[n]['shape'][0] for n in group),length])
                emit('CONCAT',{**{'src'+str(i):n for i,n in enumerate(group)},'dst':result},{'axis':0,'output_dtype':'i8'})
                joined.append(result)
            rows=joined;level+=1
        weight=rows[0]
        accumulator=stem+'_acc'
        matrix_dst=dst if ordered_shape==[lines,length] and perm==list(range(len(shape))) else tensor('output',[lines,length],'i32')
        emit('MATMUL',{'lhs':matrix_src,'rhs':weight,'dst':accumulator})
        commit_attrs={'output_dtype':'i32','epilogue':[]}
        if initial:
            seed=tensor('seed',[length],'i32')
            emit('FILL',{'dst':seed},{'value_code':initial,'output_dtype':'i32'})
            commit_attrs.update({'bias':seed,'epilogue':['bias_add']})
        emit('COMMIT',{'src':accumulator,'dst':matrix_dst},commit_attrs)
        ordered_dst=matrix_dst
        if ordered_shape!=[lines,length]:
            ordered_dst=dst if perm==list(range(len(shape))) else tensor('ordered_output',ordered_shape,'i32')
            move(matrix_dst,ordered_dst,{'form':'reshape'})
        if perm!=list(range(len(shape))):
            inverse=[perm.index(i) for i in range(len(shape))]
            move(ordered_dst,dst,{'form':'permute','perm':inverse})
    cb['commands']=commands
    return cb

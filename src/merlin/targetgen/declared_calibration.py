"""Initialize static observers from a declared domain, never evaluation payloads."""


def initialize_observers(prepared, declaration, observer_type):
    import math
    import torch

    if declaration.get('schema') != 'merlin.declared_observer_domain.v1':
        raise ValueError('unknown declared calibration contract')
    lower, upper = declaration['minimum'], declaration['maximum']
    if not all(type(v) in (int,float) and math.isfinite(v) for v in (lower,upper)) or not lower < upper:
        raise ValueError('declared calibration endpoints must be finite and ordered')
    if declaration.get('evaluation_inputs') != 'never_read':
        raise ValueError('declared calibration must exclude evaluation inputs')
    count = 0
    for node in prepared.graph.nodes:
        if node.op != 'call_module':
            continue
        observer = prepared.get_submodule(node.target)
        if not isinstance(observer, observer_type):
            continue
        axis = getattr(observer, 'ch_axis', None)
        shape = [2]
        varying_axis = 0
        if axis is not None:
            metadata = node.args[0].meta.get('val')
            if metadata is None:
                raise ValueError('per-channel declared range needs static observer input metadata')
            axis = axis % len(metadata.shape)
            shape = [1] * max(2, len(metadata.shape))
            shape[axis] = int(metadata.shape[axis])
            varying_axis = next(i for i in range(len(shape)) if i != axis)
            shape[varying_axis] = 2
        sample_shape = [1] * len(shape)
        sample_shape[varying_axis] = 2
        sample = torch.tensor([lower,upper],dtype=torch.float32).reshape(sample_shape).expand(shape)
        observer(sample)
        if declaration.get('scale_policy') == 'enclosing_power_of_two':
            from types import MethodType

            original = observer.calculate_qparams

            def qparams(self, original=original):
                scales, zero = original()
                def enclose(value):
                    fraction, exponent = math.frexp(value)
                    return value if fraction == .5 else math.ldexp(1., exponent)
                rounded = torch.tensor([enclose(v) for v in scales.reshape(-1).tolist()],
                                       dtype=scales.dtype,device=scales.device).reshape(scales.shape)
                return rounded, zero

            observer.calculate_qparams = MethodType(qparams,observer)
        count += 1
    return count


def normalize_quantizer_nans(model, policy):
    """Make the declared byte-quantizer NaN rule explicit before frontend Q.

    The selected frontend's m2m_q CPU evaluator refuses NaN. Its input can
    instead select a representable point in storage code zero's rounding cell.
    For the declared binary scales and byte zero points, -zero_point*scale is
    exact at f32 (and the declared half/bfloat16 storage range). This selector
    belongs to quantization only; source arithmetic sees the resulting codes.
    """
    if policy != 'byte_storage_zero_nan_saturating_infinity':
        return 0
    import torch
    count=0
    def quantized_source(value):
        pending, seen = [value], set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            if current.op == 'call_function' and str(current.target).startswith('quantized_decomposed.dequantize_'):
                return True
            pending.extend(current.all_input_nodes)
        return False
    for node in list(model.graph.nodes):
        if node.op != 'call_function' or str(node.target) != 'quantized_decomposed.quantize_per_tensor.default':
            continue
        source,scale,zero,low,high=node.args[:5]
        if quantized_source(source):
            # Internal Q edges retain their direct recipe producer. After an
            # external byte quantizer, rational arithmetic is finite; unknown
            # nonfinite source arithmetic still refuses its separate proof.
            continue
        if not low <= 0 <= high:
            raise ValueError('storage-zero NaN requires zero in the declared code range')
        # Literal qparams are required by the static recipe. No payload read.
        replacement=-int(zero)*float(scale)
        from fractions import Fraction
        declared_dtype=source.meta.get('val').dtype
        encoded=torch.tensor(replacement,dtype=declared_dtype).item()
        if Fraction(encoded) != -int(zero)*Fraction(float(scale)):
            raise ValueError('NaN storage-zero selector lacks an exact declared-dtype representative')
        with model.graph.inserting_before(node):
            mask=model.graph.call_function(torch.ops.aten.isnan.default,(source,))
            fill=model.graph.call_function(torch.ops.aten.full_like.default,(source,replacement))
            selected=model.graph.call_function(torch.ops.aten.where.self,(mask,fill,source))
        for added in (mask,fill,selected):
            added.meta.update({key:source.meta[key] for key in ('val','tensor_meta','stack_trace','nn_module_stack','source_fn_stack') if key in source.meta})
        if 'val' in mask.meta:
            mask.meta['val']=dtype_metadata(mask.meta['val'],torch.bool)
        node.args=(selected,*node.args[1:])
        count+=1
    model.graph.lint()
    model.recompile()
    return count


def dtype_metadata(value,dtype):
    """Change observed metadata without reading even a real metadata payload."""
    from torch._subclasses.fake_tensor import FakeTensor,FakeTensorMode
    if not isinstance(value,FakeTensor):value=FakeTensorMode().from_tensor(value)
    return value.to(dtype=dtype)


def declared_integer_recipes(model, domain):
    """Refuse an unproved affine domain before any payload agreement gate."""
    from merlin.targetgen._recipe_integer import integer_recipes

    rows = integer_recipes(model, declared_domain=domain)
    refused = [row for row in rows if row.get('declared_affine_domain', {}).get('status') == 'refused']
    if refused:
        from merlin.targetgen._recipe_quantizer import NoEligibleRecipeOperations

        error = NoEligibleRecipeOperations('retained affine intermediates lack a full declared-domain exactness proof')
        error.code = 'declared_affine_domain_unproved'
        error.details = dict(integer_recipes=refused)
        raise error
    return rows

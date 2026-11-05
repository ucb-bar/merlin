"""Carry actual bounded-RNE scalar emissions across native ownership checking.

This joins an original callback and independently checks its effect footprint.
The legalizer still owns the numerical law and assembler/native-oracle checks;
neither a report nor this footprint check issues borrowed implementation effects.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import CodeType, FunctionType

from merlin.common.digest import sha256_file

from .host_transform_chain import HostLLVMArtifact

_SEAL = object()


def _functions():
    from . import late_quant_rne as L

    return tuple((name, value, value.__code__) for name, value in sorted(vars(L).items())
                 if type(value) is FunctionType and value.__module__ == L.__name__)


def _closure(callback):
    return tuple((name, id(cell), id(cell.cell_contents), cell.cell_contents)
                 for name, cell in zip(callback.__code__.co_freevars, callback.__closure__ or (), strict=True))


def is_original_callback(callback):
    """Recognize the existing callback; this alone supplies no effect witness."""
    from . import late_quant_rne as L

    nested = tuple(c for c in L.merlin_host_llvm_transform.__code__.co_consts
                   if type(c) is CodeType and c.co_name == 'transform')
    return (type(callback) is FunctionType and len(nested) == 1
            and callback.__code__ is nested[0] and callback.__globals__ is vars(L))


def select_custody(callback, *, descriptor_identity, intermediate_storage, chain=None):
    """Keep strict identity selection and carry optional borrowed storage too.

    Other storage callbacks retain their ordinary conservative behavior. An
    actual original callback still needs complete emission footprint checking.
    """
    if type(descriptor_identity) is not bool or type(intermediate_storage) is not bool:
        raise TypeError('bounded RNE selection requires explicit identity/storage choices')
    if descriptor_identity and (callback is not None or chain is not None):
        if chain is not None:
            raise ValueError('checked descriptor identity does not support a later host LLVM transform chain')
        return BoundedRNECustody(callback)
    if intermediate_storage and chain is None and is_original_callback(callback):
        return BoundedRNECustody(callback)
    return None


def _scalar_conversion_changes(before, after, document, *, host_isa, combine_clamp, temporary_prefix):
    """Linear independent footprint check, with every other source byte retained.

    A scalar fptosi or the final integer add of the original bounded chain may
    become the exact two-instruction pure conversion sequence. No pointer,
    memory, foreign call, CFG, signature or attribute edit
    is admitted. The original numerical matcher is not replayed here.
    """
    import hashlib

    from .late_quant_rne import _tokens

    if (document.get('source_sha256') != hashlib.sha256(before.encode()).hexdigest()
            or document.get('rewritten_sha256') != hashlib.sha256(after.encode()).hexdigest()
            or document.get('host_isa') != host_isa or document.get('combine_clamp') is not combine_clamp):
        raise ValueError('bounded RNE footprint differs from the actual original emission')
    original, selected = before.splitlines(keepends=True), after.splitlines(keepends=True)
    locals_before = {t.text for t in _tokens(before) if t.text.startswith('%')}
    introduced, changes = set(), []
    i = j = 0
    routes = document['routes']
    while i < len(original):
        if j < len(selected) and original[i] == selected[j]:
            if original[i].startswith('define '):
                introduced.clear()
            i += 1; j += 1
            continue
        if j + 1 >= len(selected) or len(changes) >= len(routes):
            raise ValueError('bounded RNE changed nonconversion source bytes')
        old = _tokens(original[i])
        conversion = (len(old) == 7 and [t.text for t in old[1:4]] == ['=', 'fptosi', 'float']
                      and old[5].text == 'to')
        final_add = (len(old) == 7 and [t.text for t in old[1:3]] == ['=', 'add'] and old[5].text == ',')
        if not (conversion or final_add) or not old[0].text.startswith('%'):
            raise ValueError('bounded RNE changed a pointer, effect, control or unrelated instruction')
        name, dtype = old[0].text, old[6].text if conversion else old[3].text
        row = routes[len(changes)]
        if (row.get('result') != name or row.get('integer_dtype') != dtype
                or not dtype.startswith('i') or not dtype[1:].isdigit() or not 1 <= int(dtype[1:]) <= 24):
            raise ValueError('bounded RNE conversion differs from its original ordered result')
        prefix, suffix = original[i][:old[0].start], original[i][old[-1].end:]
        first = _tokens(selected[j])
        temporary = first[0].text if first else ''
        if (not temporary.startswith('%' + temporary_prefix + '.')
                or not temporary[len(temporary_prefix) + 2:].isdigit()
                or temporary in locals_before or temporary in introduced):
            raise ValueError('bounded RNE temporary is not a fresh scalar')
        if host_isa == 'portable':
            conversion = f'{temporary} = call float @llvm.roundeven.f32(float {row["clamped_input"]})'
            result = f'{name} = fptosi float {temporary} to {dtype}'
        elif combine_clamp:
            lower, upper = -(1 << (int(dtype[1:]) - 1)), (1 << (int(dtype[1:]) - 1)) - 1
            if row.get('bounds') != [lower, upper]:
                raise ValueError('bounded RNE clamp bounds differ from the original integer domain')
            conversion = (f'{temporary} = call i32 asm "fmax.s ft0, $1, $2\\0Afmin.s ft0, ft0, $3\\0A'
                          f'fcvt.w.s $0, ft0, rne", "=r,f,f,f,~{{ft0}}"'
                          f'(float {row["raw_input"]}, float {lower:.6e}, float {upper:.6e})')
            result = f'{name} = trunc i32 {temporary} to {dtype}'
        else:
            conversion = (f'{temporary} = call i32 asm "fcvt.w.s $0, $1, rne", "=r,f"'
                          f'(float {row["clamped_input"]})')
            result = f'{name} = trunc i32 {temporary} to {dtype}'
        if (selected[j] != prefix + conversion + '\n'
                or selected[j + 1] != prefix + result + suffix):
            raise ValueError('bounded RNE added an unsupported scalar or memory effect')
        introduced.add(temporary); changes.append(name)
        i += 1; j += 2
    trailing = ''.join(selected[j:])
    if trailing and not (host_isa == 'portable' and trailing == '\ndeclare float @llvm.roundeven.f32(float)\n'
                         and '@llvm.roundeven.f32' not in before):
        raise ValueError('bounded RNE appended unrelated declarations or instructions')
    identity = (not routes and before == after
                and document.get('identity') == 'no_matching_bounded_conversions'
                and 'refusal' not in document)
    if len(changes) != len(routes) or not changes and not identity:
        raise ValueError('bounded RNE did not consume every original scalar conversion')
    return tuple(changes)


class BoundedRNECustody:
    """One original live callback, frozen before other package callbacks run."""

    def __init__(self, callback):
        from . import late_quant_rne as L

        if not is_original_callback(callback):
            raise ValueError('borrowed lifetime requires the original bounded RNE callback')
        closure = _closure(callback)
        values = {name: value for name, _, _, value in closure}
        if (set(values) != {'llvm_bin', 'host_isa', 'combine_clamp', 'temporary_prefix'}
                or not isinstance(values['llvm_bin'], Path) or not values['llvm_bin'].is_absolute()
                or values['host_isa'] not in ('rv64gc', 'portable')
                or type(values['combine_clamp']) is not bool
                or values['host_isa'] == 'portable' and values['combine_clamp']):
            raise ValueError('bounded RNE requires the original explicit compiler and scalar policy')
        self.callback, self._closure, self._code = callback, closure, callback.__code__
        self.policy = {key: values[key] for key in ('host_isa', 'combine_clamp', 'temporary_prefix')}
        self._policy = tuple(self.policy.items())
        self._functions = _functions()
        self._checker = tuple((f, f.__code__) for f in _implementation_functions())
        self._owners = tuple(HostLLVMArtifact.capture(p) for p in (
            Path(__file__), Path(L.__file__), Path(__file__).with_name('host_transform_dispatch.py'),
            values['llvm_bin'] / 'llvm-as'))
        self._input = None
        self._observed = self._actual = None
        self._consumed = False

    def revalidate(self):
        if (self.callback.__code__ is not self._code or _closure(self.callback) != self._closure
                or tuple(self.policy.items()) != self._policy or _functions() != self._functions):
            raise ValueError('original bounded RNE callback or policy changed')
        if any(f is not old or f.__code__ is not code for f, (old, code) in
               zip(_implementation_functions(), self._checker, strict=True)):
            raise ValueError('original bounded RNE footprint implementation changed')
        for product in (*self._owners, *((self._input,) if self._input is not None else ())):
            product.validate()

    def bind_input(self, source):
        self.revalidate()
        if self._input is not None:
            raise ValueError('bounded RNE custody requires one actual input')
        self._input = HostLLVMArtifact.capture(source)
        self.revalidate()

    def consume(self, selected, work, observation, compilation):
        from .host_bounded_rne import require_report

        self.revalidate()
        if (self._input is None or self._consumed or observation is not self._observed
                or self._actual != (Path(selected).resolve(), Path(work).resolve())):
            raise ValueError('bounded RNE custody requires its actual ordinary callback invocation')
        selected, work = Path(selected).resolve(), Path(work).resolve()
        if selected != work / 'model.ll':
            raise ValueError('bounded RNE output is outside the actual ordinary callback')
        pins = tuple(HostLLVMArtifact.capture(work / name) for name in
                     ('model.ll', 'receipt.json', 'source.bc', 'model.bc', 'model.native.ll', 'model.native.bc'))
        document = json.loads((work / 'receipt.json').read_text())
        require_report(document)
        if observation != {'source_path': str(self._input.path), 'source_sha256': self._input.sha256,
                           'selected_path': str(selected), 'selected_sha256': pins[0].sha256}:
            raise ValueError('bounded RNE custody differs from the actual ordinary transform result')
        if (document.get('native_oracle_sha256') != sha256_file(work / 'model.native.ll')
                or document.get('llvm_as_sha256') != self._owners[-1].sha256):
            raise ValueError('bounded RNE assembler or paired native oracle changed')
        before, after = self._input.path.read_text(), selected.read_text()
        results = _scalar_conversion_changes(before, after, document, **self.policy)
        self.revalidate()
        emission = BoundedRNEEmission(_SEAL, self, pins, before, after, results)
        for index, pin in enumerate((*self._owners, self._input, *pins)):
            compilation.bind_preparation('bounded_rne_lifetime_' + str(index), pin.path)
        self._consumed = True
        return emission

    def apply(self, source, work, compilation):
        """Invoke the ordinary dispatcher once; saved output data supplies no token."""
        from .host_transform_dispatch import _transform_host_ir

        self.bind_input(source)
        selected, observation = _transform_host_ir(Path(source), Path(work), self.callback)
        self.revalidate()
        self._observed = observation
        self._actual = (Path(selected).resolve(), Path(work).resolve())
        try:
            emission = self.consume(selected, work, observation, compilation)
        finally:
            self._observed = None
        return selected, observation, emission


class BoundedRNEEmission:
    """Actual checked footprint; cannot be restored from receipt fields."""

    def __init__(self, seal, owner, pins, before, after, results):
        if seal is not _SEAL or type(owner) is not BoundedRNECustody:
            raise TypeError('bounded RNE emission requires its actual original callback')
        self._owner, self._pins = owner, pins
        self.before, self.after, self.results = before, after, results
        original_lines = set(before.splitlines())
        self.scalar_calls = tuple(line.strip() for line in after.splitlines()
                                  if line not in original_lines and ' = call ' in line)
        self._state = tuple(vars(self).items())

    def revalidate(self):
        if any(getattr(self, name) is not value for name, value in self._state):
            raise ValueError('original bounded RNE emission changed')
        self._owner.revalidate()
        for pin in self._pins:
            pin.validate()

    def record(self):
        self.revalidate()
        return {'schema': 'merlin.bounded_rne_lifetime.v1', 'results': list(self.results),
                'source': self._owner._input.to_dict(), 'selected': self._pins[0].to_dict(),
                'numerical_report': self._pins[1].to_dict(),
                'noncapturing_scalar_calls': list(self.scalar_calls),
                'pointer_memory_control_and_other_calls': 'byte_identical'}


def _implementation_functions():
    from .host_transform_dispatch import _transform_host_ir

    return (_functions, _closure, is_original_callback, select_custody, _scalar_conversion_changes, _implementation_functions,
            BoundedRNECustody.__init__, BoundedRNECustody.revalidate,
            BoundedRNECustody.bind_input, BoundedRNECustody.consume, BoundedRNECustody.apply,
            BoundedRNEEmission.__init__, BoundedRNEEmission.revalidate, BoundedRNEEmission.record, _transform_host_ir)

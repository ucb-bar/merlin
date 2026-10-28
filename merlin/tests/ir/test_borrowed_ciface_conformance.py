"""Only the actual ABI annotation can transport native lifetime custody."""

from copy import deepcopy

import pytest

from merlin.llvmlower.borrowed_arena import _ciface_annotation_changes
from merlin.llvmlower.ciface_inline import inline_wide_ciface_implementations


def fixture():
    arguments = ', '.join(f'i64 %unused{i}' for i in range(257))
    before = f'''define void @forward({arguments}) {{
  %buffer = call ptr @malloc(i64 64)
  store i32 3, ptr %buffer
  call void @borrowed(ptr %buffer)
  call void @free(ptr %buffer)
  ret void
}}
define void @_mlir_ciface_forward(ptr %descriptor) {{
  ret void
}}
declare ptr @malloc(i64)
declare void @free(ptr)
declare void @borrowed(ptr)
'''
    after, report = inline_wide_ciface_implementations(before)
    return before, after, report


def test_independent_annotation_conformance_preserves_exact_calls_memory_and_signatures():
    before, after, report = fixture()
    assert _ciface_annotation_changes(before, after, report) == ('forward',)
    assert before.splitlines()[1:] == after.splitlines()[1:]


@pytest.mark.parametrize('old,new', [
    ('store i32 3', 'store i32 4'),
    ('@borrowed(ptr %buffer)', '@other(ptr %buffer)'),
    ('call void @free(ptr %buffer)', 'call void @borrowed(ptr %buffer)'),
    ('i64 %unused0', 'ptr %unused0'),
    ('alwaysinline', 'alwaysinline nounwind'),
    ('ret void\n}', 'ret void\n  ret void\n}'),
])
def test_annotation_receipt_does_not_permit_another_change(old, new):
    before, after, report = fixture()
    assert old in after
    assert _ciface_annotation_changes(before, after.replace(old, new), report) is None


def test_forged_annotation_report_does_not_supply_permission():
    before, after, report = fixture()
    forged = deepcopy(report)
    forged['inlined'][0]['implementation'] = 'other'
    assert _ciface_annotation_changes(before, after, forged) is None
    assert _ciface_annotation_changes(before, after, {}) is None

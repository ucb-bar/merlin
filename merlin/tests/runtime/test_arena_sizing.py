"""What the lowered model still asks ``malloc`` for, read off the IR that will actually run.

Once the static arena plan has seated what it can, the rest of the model's buffers -- the HEAP RESIDUE --
still go through the allocator, and the region must leave room for them. They are priced at their SUM:
that is the bound that holds whatever the allocator does with frees, and an under-count is the direction
that produces an image which dies in its tail rather than a wasteful one.
"""

from __future__ import annotations

from merlin.runtime.backends import zephyr_memory as zmem
from merlin.runtime.backends import zephyr_model as zm


def _ll(tmp_path, body: str):
    f = tmp_path / "model.ll"
    f.write_text(body)
    return f


def test_every_spelling_of_an_allocation_is_counted(tmp_path):
    """The IR spells the same call several ways. A pattern narrow enough to read misses one of them, and a
    missed allocation under-reports the requirement -- which is the direction that produces a broken image
    rather than a wasteful one."""
    total, dynamic = zm.allocation_bytes(
        _ll(
            tmp_path,
            """
  %1 = call ptr @malloc(i64 1024)
  %2 = tail call noalias ptr @malloc(i64 2048) #3
  %3 = call noalias align 16 ptr @malloc(i64 noundef 4096)
  %4 = call ptr @llvm.memcpy(ptr %a, ptr %b, i64 8)
""",
        )
    )
    assert (total, dynamic) == (1024 + 2048 + 4096, 0)


def test_a_computed_size_is_reported_not_assumed_zero(tmp_path):
    """Its bytes are genuinely unknown at build time. Counting it as nothing would make an incomplete
    answer look like a complete one."""
    total, dynamic = zm.allocation_bytes(
        _ll(
            tmp_path,
            """
  %1 = call ptr @malloc(i64 512)
  %2 = call ptr @malloc(i64 %n)
""",
        )
    )
    assert total == 512
    assert dynamic == 1


def test_the_declaration_of_malloc_is_not_an_allocation(tmp_path):
    """Every lowered module declares ``malloc``; reading the declaration as a call reported one "dynamic"
    allocation on every model, an unknown that was never there."""
    total, dynamic = zm.allocation_bytes(_ll(tmp_path, "declare ptr @malloc(i64)\n  %1 = call ptr @malloc(i64 64)\n"))
    assert (total, dynamic) == (64, 0)


RESIDUE = """declare ptr @malloc(i64)

define void @forward(ptr %0) {
  %a = call ptr @malloc(i64 1000)
  %b = call ptr @malloc(i64 %n)
  ret void
}

define internal void @forward..omp_par(ptr %0) {
omp.par.entry:
  %c = call ptr @malloc(i64 300)
  br label %loop

loop:
  %d = call ptr @malloc(i64 20)
  br i1 %x, label %loop, label %out

out:
  ret void
}
"""


def test_the_residue_counts_an_outlined_region_once_per_hart():
    """A buffer allocated inside an outlined parallel region exists once per hart at the same time, so a
    multicore image's residue is not its single-core one. The region's body opens with its own entry
    label, as `mlir-translate` spells it; reading that as an empty unnamed block refused every 2-hart build."""
    one = zmem.heap_residue(RESIDUE, harts=1)
    two = zmem.heap_residue(RESIDUE, harts=2)
    assert one.bytes == 1000 + 300 + 20 and one.allocations == 3
    assert two.bytes == 1000 + 2 * (300 + 20) and two.allocations == 5
    assert one.dynamic == 1 and one.in_loop == 1

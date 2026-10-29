"""Prepare a whole-entry operational proof for current empty/full unrolling."""


def prepare_query(before, after, *, factor, index_bits=64):
    from .compiler_unroll_query import prepare_query as prepare

    return prepare(before, after, factor=factor, index_bits=index_bits, mode="inline")

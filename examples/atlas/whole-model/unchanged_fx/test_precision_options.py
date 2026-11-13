"""Explicit domain selection and invocation budgets must survive the CLI."""
import argparse
import sys
from unittest.mock import patch

import numpy as np
import pytest

from atlas_mlir_native.reduction import (
    NONNEGATIVE_NORMAL_REDUCTION_POLICY, REDUCTION_POLICY, input_spec,
)
from merlin.semantic_compiler.buffer_allocation import current_allocation_wall_budget_ms
from precision_options import add_reduction_policy_argument, verify_reduction_policy
import compile_capsule
from merlin.compile.c_bf16_linear import SOURCE_POLICY


OLD_HEAD_POLICY = "bf16-product-rne-tree32-f32-rne-bf16-rne-tiles-ltr-chop-v1"


def parse_policy(*args):
    parser = argparse.ArgumentParser()
    add_reduction_policy_argument(parser)
    return parser.parse_args(args).reduction_policy


def test_default_domain_still_refuses_the_small_normal_rms_word():
    selected = parse_policy()
    assert selected == REDUCTION_POLICY
    value = np.array([0x1ff50000], dtype=np.uint32).view(np.float32)
    with pytest.raises(ValueError):
        input_spec(reduction_policy=selected).encode(value)


def test_explicit_nonnegative_domain_admits_the_exact_rms_word():
    selected = parse_policy('--reduction-policy', NONNEGATIVE_NORMAL_REDUCTION_POLICY)
    value = np.array([0, 0x1ff50000, 0x3f800000], dtype=np.uint32).view(np.float32)
    assert input_spec(reduction_policy=selected).encode(value).tolist() == [0, 0x1ff5, 0x3f80]


@pytest.mark.parametrize('word', [0x80000000, 0xbf800000, 0x00010000, 0x7f800000, 0x7fc00000])
def test_selected_nonnegative_domain_still_refuses_negative_subnormal_and_nonfinite(word):
    with pytest.raises(ValueError):
        input_spec(reduction_policy=NONNEGATIVE_NORMAL_REDUCTION_POLICY).encode(
            np.array([word], dtype=np.uint32).view(np.float32))


def test_unknown_policy_refuses_at_cli():
    with pytest.raises(SystemExit):
        parse_policy('--reduction-policy', 'unqualified-domain')


@pytest.mark.parametrize('saved', [REDUCTION_POLICY, NONNEGATIVE_NORMAL_REDUCTION_POLICY])
def test_reuse_requires_precision_and_compiled_selection_to_agree(saved):
    precision = {'reduction': saved}
    verify_reduction_policy('layer0', precision, {'reduction_policy': saved}, saved)
    verify_reduction_policy('layer0', precision, {}, saved)
    other = NONNEGATIVE_NORMAL_REDUCTION_POLICY if saved == REDUCTION_POLICY else REDUCTION_POLICY
    with pytest.raises(ValueError, match='different selected reduction policy'):
        verify_reduction_policy('layer0', precision, {'reduction_policy': saved}, other)
    with pytest.raises(ValueError, match='different selected reduction policy'):
        verify_reduction_policy('layer0', precision, {'reduction_policy': other}, saved)


@pytest.mark.parametrize('policy', [None, OLD_HEAD_POLICY, 'unknown'])
def test_head_reuse_refuses_missing_or_obsolete_precision_policy(policy):
    precision = {'reduction': REDUCTION_POLICY}
    if policy is not None:
        precision['host_bf16_linear'] = policy
    compilation = {'regions': [{'description': {
        'arithmetic': 'selected-host-bf16-matmul', 'numeric_semantics': SOURCE_POLICY}}]}
    with pytest.raises(ValueError, match='different selected host BF16 linear policy'):
        verify_reduction_policy('head', precision, compilation, REDUCTION_POLICY)


@pytest.mark.parametrize('policy', [None, OLD_HEAD_POLICY, 'unknown'])
def test_head_reuse_refuses_inconsistent_or_missing_compiled_arithmetic(policy):
    precision = {'reduction': REDUCTION_POLICY, 'host_bf16_linear': SOURCE_POLICY}
    description = {'arithmetic': 'selected-host-bf16-matmul'}
    if policy is not None:
        description['numeric_semantics'] = policy
    compilation = {'regions': [
        {'description': {'arithmetic': 'selected-host-bf16-matmul', 'numeric_semantics': SOURCE_POLICY}},
        {'description': description}]}
    with pytest.raises(ValueError, match='different selected host BF16 linear policy'):
        verify_reduction_policy('head', precision, compilation, REDUCTION_POLICY)


@pytest.mark.parametrize('recorded', [False, True])
def test_head_reuse_accepts_current_policy_and_preserves_precision_only_receipts(recorded):
    precision = {'reduction': REDUCTION_POLICY, 'host_bf16_linear': SOURCE_POLICY}
    compilation = {'reduction_policy': REDUCTION_POLICY}
    if recorded:
        compilation['regions'] = [
            {'description': {'arithmetic': 'byte-preserving'}},
            {'description': {'arithmetic': 'selected-host-bf16-matmul', 'numeric_semantics': SOURCE_POLICY}}]
    verify_reduction_policy('head', precision, compilation, REDUCTION_POLICY)


@pytest.mark.parametrize('override,expected', [(None, 1000), (2300, 2300)])
def test_compile_cli_records_selected_domain_and_uses_default_or_explicit_wall_budget(override, expected):
    command = ['compile_capsule.py', '--capsule', '.', '--output', 'unused-output',
               '--reduction-policy', NONNEGATIVE_NORMAL_REDUCTION_POLICY]
    if override is not None:
        command += ['--allocation-timeout-ms', str(override)]
    observed = []
    def compile_selected(args):
        observed.append((args.reduction_policy, args.allocation_timeout_ms, current_allocation_wall_budget_ms()))
    with patch.object(sys, 'argv', command), patch.object(compile_capsule, 'compile_capsule', compile_selected):
        compile_capsule.main()
    assert observed == [(NONNEGATIVE_NORMAL_REDUCTION_POLICY, expected, expected)]
    assert current_allocation_wall_budget_ms() == 1000


@pytest.mark.parametrize('budget', ['0', '-1'])
def test_compile_cli_refuses_nonpositive_allocation_budget(budget):
    with patch.object(sys, 'argv', ['compile_capsule.py', '--capsule', '.', '--output', 'unused-output',
                                  '--allocation-timeout-ms', budget]), pytest.raises(SystemExit):
        compile_capsule.main()


@pytest.mark.parametrize('module_name', ['composition', 'cache_composition'])
@pytest.mark.parametrize('override,expected', [(None, 1000), (2300, 2300)])
def test_driver_clis_select_default_or_explicit_budget_and_domain(module_name, override, expected):
    import importlib

    driver = importlib.import_module(module_name)
    command = [module_name + '.py']
    for option in ('regions', 'compiled', 'execution', 'atlas-root', 'model-python', 'model-root'):
        command += ['--' + option, 'unused']
    command += ['--reduction-policy', NONNEGATIVE_NORMAL_REDUCTION_POLICY]
    if override is not None:
        command += ['--allocation-timeout-ms', str(override)]
    original_parse = argparse.ArgumentParser.parse_args
    observed = []
    class Parsed(Exception):
        pass
    def parse_and_stop(parser, *args, **kwargs):
        selected = original_parse(parser, *args, **kwargs)
        observed.append((selected.allocation_timeout_ms, selected.reduction_policy))
        raise Parsed
    with patch.object(sys, 'argv', command), patch.object(argparse.ArgumentParser, 'parse_args', parse_and_stop):
        with pytest.raises(Parsed):
            driver.main()
    assert observed == [(expected, NONNEGATIVE_NORMAL_REDUCTION_POLICY)]

"""Informational logit quality against distinct BF16 and FP32 native references."""
from merlin.common.jsonio import write_pretty_json
import argparse
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from capture_tinyllama_native import safe_path, identity, array_identity, TOKEN_SHA


def read_session(receipt_path, status, fields, session_key='session'):
    record = identity(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != status or receipt.get('source_bookends_verified') is not True:
        raise ValueError('missing admitted exact session gate')
    item = receipt[session_key]
    session = identity(item['path'], item['sha256'])
    path = safe_path(session['path'])
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != len(fields) or {x.filename for x in members} != {x + '.npy' for x in fields} or sum(
            x.file_size for x in members) > 16 * 1024 ** 2:
            raise ValueError('unexpected bounded session archive')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in fields}
    tokens = arrays['tokens']
    if tokens.dtype != np.int64 or tokens.shape !=(1, 160) or array_identity(tokens)['sha256'] != TOKEN_SHA:
        raise ValueError('frozen 128+32 tokens differ')
    identity(record['path'], record['sha256'])
    identity(session['path'], session['sha256'])
    return receipt, arrays, [record, session]


def logits(array, bf16):
    dtype = np.uint16 if bf16 else np.float32
    if array.dtype != dtype or array.shape !=(33, 32000):
        raise ValueError('logit shape/precision differs')
    value =(array.astype(np.uint32) << 16).view(np.float32) if bf16 else array
    if not np.isfinite(value).all():
        raise ValueError('nonfinite logits')
    return value


def score(values, tokens):
    values = values[:32].astype(np.float64)
    labels = tokens[0, 128:160]
    maxima = values.max(axis=1, keepdims=True)
    shift = values - maxima
    logprob = shift - np.log(np.exp(shift).sum(axis=1, keepdims=True))
    loss = -logprob[np.arange(32), labels]
    predicted = np.argmax(values, axis=1)
    return {'policy': 'independent NumPy FP64 scoring, distinct from original Torch FP32 NLL receipt',
            'mean_nll': float(loss.mean()), 'perplexity': float(np.exp(loss.mean())),
            'argmax_correct': int(np.count_nonzero(predicted == labels)), 'argmax_ids': predicted.tolist(), 'nll': loss.tolist()}


def comparison(actual, reference):
    a = actual[:32].astype(np.float64)
    r = reference[:32].astype(np.float64)
    delta = a - r
    def logprob(x):
        x = x - x.max(axis=1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=1, keepdims=True))
    la, lr = logprob(a), logprob(r)
    kl = np.sum(np.exp(lr) *(lr - la), axis=1)
    tolerance = 1e-4 + 1e-4 * np.abs(r)
    return {'scored_logit_values': int(a.size), 'scored_tokens': 32,
            'relative_l2': float(np.linalg.norm(delta) / np.linalg.norm(r)),
            'max_abs': float(np.abs(delta).max()), 'mean_abs': float(np.abs(delta).mean()),
            'argmax_agreement': int(np.count_nonzero(np.argmax(a, axis=1) == np.argmax(r, axis=1))),
            'unchanged_original_rtol_atol_1e4_passed': bool(np.all(np.abs(delta) <= tolerance)),
            'violating_original_1e4_values': int(np.count_nonzero(np.abs(delta) > tolerance)),
            'mean_KL_reference_to_actual': float(kl.mean()), 'per_token_KL_reference_to_actual': kl.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'fp32', 'out'):
        p.add_argument('--' + name, type=safe_path, required=True)
    p.add_argument('--candidate', type=safe_path)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('fresh quality output required')
    native, n, records = read_session(a.native, 'passed_frozen_native_BF16_session', {'logits_bf16', 'tokens', 'entry_bf16'})
    fp32, f, more = read_session(a.fp32, 'passed_separate_FP32_reference', {'logits', 'tokens'})
    records += more
    b = logits(n['logits_bf16'], True)
    r = logits(f['logits'], False)
    if not np.array_equal(n['tokens'], f['tokens']):
        raise ValueError('references use different tokens')
    report = {'status': 'reported_informational_quality_original_gates_unchanged', 'command': sys.argv,
            'scope': (
                'same frozen raw-text 128-prefill + 32-teacher-forced inputs; 32 scored distributions; '
                'final 33rd distribution unscored'
            ),
            'reference_policies': {'native': 'unchanged original BF16 full model',
                                  'FP32': (
                                      'all same immutable BF16 checkpoint weights exactly cast FP32 and complete model '
                                      'arithmetic FP32; not a recovered FP32 training checkpoint'
                                  )},
            'original_BF16_vs_FP32': comparison(b, r), 'original_BF16_scoring': score(b, n['tokens']),
            'FP32_scoring': score(r, n['tokens']), 'original_Torch_FP32_NLL': {'BF16': native['metrics']['mean_nll'],
        'FP32': fp32[
        'metrics']['mean_nll']},
            'quality_is_explicit_relaxation': True, 'canonical_language_quality_claim': False,
        'new_prompt_or_chat_template': False}
    if a.candidate:
        candidate, c, more = read_session(a.candidate, 'passed_exact_target_model_block_with_informational_native_quality', {
            'original_bf16', 'candidate_bf16', 'tokens'}, 'session_logits')
        records += more
        if (candidate.get('prefill_device_logical_ops') != 7 or candidate.get('prefill_device_programs') != 12
                or candidate.get('parameters_buffers_tokens_preserved') is not True or candidate.get(
            'native_restoration_exact') is not True
                or len(candidate.get('target_programs', [])) != 12 or not np.array_equal(c['tokens'], n[
            'tokens']) or not np.array_equal(c['original_bf16'], n['logits_bf16'])):
            raise ValueError('incomplete/wrong native+target firstblock session')
        values = logits(c['candidate_bf16'], True)
        report.update(actual_target_firstblock_vs_original_BF16=comparison(values, b),
            actual_target_firstblock_vs_FP32=comparison(
            values, r),
                      actual_target_firstblock_scoring=score(values, n['tokens']), candidate_original_Torch_FP32_NLL=candidate[
            'candidate_metrics']['mean_nll'],
                      target_execution_scope=(
                          'seven firstblock logical projections,12 device programs, unchanged host/native '
                          'full suffix and 32 decode steps'
                      ))
        for call in candidate['target_programs']:
            rec = call['child']['receipt']
            records.append(identity(rec['path'], rec['sha256']))
    records.append(identity(__file__))
    for rec in records:
        identity(rec['path'], rec['sha256'])
    report['input_records'] = records
    report['source_bookends_verified'] = True
    write_pretty_json(a.out, report, allow_nan=False, sort_keys=False)
    print(json.dumps({'status': report['status'], 'original_BF16_vs_FP32': report['original_BF16_vs_FP32']}))
if __name__ == '__main__':
    main()

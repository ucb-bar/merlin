"""Informational logit quality against distinct BF16 and FP32 native references."""
import json
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

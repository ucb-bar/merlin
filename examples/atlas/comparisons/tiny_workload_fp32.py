"""Separate FP32 arithmetic reference for the same immutable BF16 checkpoint."""
import argparse
from pathlib import Path
import random
import sys
import numpy as np
import tiny_workload_native as native
from tiny_workload_block import guarded_session
from capture_tinyllama_native import safe_path, identity


def main():
    from workload_cli import run_report
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical', type=safe_path, required=True)
    p.add_argument('--out', type=safe_path, required=True)
    a = p.parse_args()
    report = {'status': 'failed', 'command': sys.argv,
        'reference_policy': (
            'Every immutable BF16 checkpoint weight exactly cast to FP32; complete native '
            'operators run FP32. This is separate from the unchanged original BF16 reference, '
            'not a recovered FP32 training checkpoint.'
        )}
    def execute(a, report):
        import torch
        rng = torch.get_rng_state().clone()
        nrng = np.random.get_state()
        prng = random.getstate()
        try:
            records =[identity(__file__), identity(native.__file__), identity(Path(__file__).with_name('tiny_workload_block.py'))]
            model, tokens, old, torch, cache_utils, modeling_llama, q = native.prepare(a.historical, records, precision='float32')
            before = native.state(model, torch)
            caller_tokens = tokens.clone()
            first, tails = guarded_session(model, tokens, torch, cache_utils, modeling_llama)
            repeated, tails2 = guarded_session(model, tokens, torch, cache_utils, modeling_llama)
            native.require(native.summary(first, torch) == native.summary(repeated, torch) and tails == tails2,
                'FP32 session repeat differs')
            native.require(before == native.state(model, torch) and torch.equal(tokens, caller_tokens),
                'FP32 caller state changed')
            np.savez(a.out / 'session.npz', logits=first['logits'].detach().numpy(), tokens=tokens.numpy())
            for r in records:
                identity(r['path'], r['sha256'])
            report.update(status='passed_separate_FP32_reference', source_records=records, source_bookends_verified=True,
                          full_parameters=before[0], buffers=before[1], native_repeat_exact=True,
                all_22_cache_prefix_lengths_verified=True,
                          all_33_layer0_actual_KV_tails_verified=True, full_model_executed=True, Atlas_execution=False,
                          metrics=native.score(first['logits'], tokens, torch), summary=native.summary(first, torch),
                session=identity(
                a.out / 'session.npz'))
        finally:
            torch.set_rng_state(rng)
            np.random.set_state(nrng)
            random.setstate(prng)
            report['caller_torch_rng_restored'] = torch.equal(rng, torch.get_rng_state())
    raise SystemExit(run_report(a, execute,
        filename='receipt.json', output=a.out, report=report, summary=('status', 'wall_seconds', 'peak_rss_kib')))
if __name__ == '__main__':
    main()

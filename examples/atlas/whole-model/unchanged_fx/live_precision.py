"""Select the example's source-bound precision policy for a live capture."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from precision_options import add_reduction_policy_argument, build_precision_policy


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--kind', choices=('layer0', 'head', 'norm'), default='layer0')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--softmax-mask-policy', choices=(
        'bf16-minfinite-zero-bounded32-nonempty-v1',
        'bf16-minfinite-zero-rowspan80-nonempty-v1'),
        default='bf16-minfinite-zero-rowspan80-nonempty-v1')
    add_reduction_policy_argument(parser)
    args = parser.parse_args()
    policy = build_precision_policy(args.kind,
        sha256((args.capture / 'model.mlir').read_bytes()).hexdigest(),
        reduction_policy=args.reduction_policy, softmax_mask_policy=args.softmax_mask_policy)
    with args.out.open('x') as stream:
        stream.write(json.dumps(policy, indent=2) + '\n')


if __name__ == '__main__':
    main()

"""Run multi-source baseline transfer model."""

import argparse
import os
import sys
import numpy as np

sys.path.insert(1, os.path.dirname(__file__))
import wang_eisner_udv1  # noqa: ignore=E402

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=str, required=True,
                    help='Parent directory to write job logs and models to.')
parser.add_argument('--main', type=str, default='train_baseline.py',
                    help='Script to run (e.g. train_baseline.py)')
parser.add_argument('--udv1-dir', type=str, default='data/ud_v1.2',
                    help='Path to UD version 1.2 directory (e.g. data/ud_v1.2)')
parser.add_argument('--split', type=str, choices={'cv', 'full'},
                    default='full',
                    help='Full=test set, CV=5-fold cross-validation on train.')
parser.add_argument('--seeds', type=int, default=None,
                    help='Number of parallel runs with different random seeds.')
parser.add_argument('--args', nargs=argparse.REMAINDER,
                    help='Arguments for the main script (e.g. --pos-dim 50).')


def run(args, src, tgt, name):
    src_patterns = [os.path.join(args.udv1_dir, l) for l in src]
    tgt_patterns = [os.path.join(args.udv1_dir, l) for l in tgt]
    cmd = ['python', args.main,
           '--src-patterns', ' '.join(src_patterns),
           '--tgt-patterns', ' '.join(tgt_patterns),
           '--experiment-dir', os.path.join(args.output_dir, name),
           ' '.join(args.args) if args.args else '']
    if args.seeds is not None:
        for _ in range(args.seeds):
            seed = np.random.randint(10000)
            print(' '.join(cmd + ['--seed', str(seed)]))
    else:
        print(' '.join(cmd))


def main(args):
    if args.split == 'cv':
        for i in range(len(wang_eisner_udv1.FOLDS)):
            src = [l for j, fold in enumerate(wang_eisner_udv1.FOLDS)
                   for l in fold if j != i]
            tgt = wang_eisner_udv1.FOLDS[i]
            run(args, src, tgt, 'fold-%d' % i)
    elif args.split == 'full':
        src = wang_eisner_udv1.TRAIN
        tgt = wang_eisner_udv1.TEST
        run(args, src, tgt, 'full')
    else:
        raise RuntimeError('Unknown split.')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

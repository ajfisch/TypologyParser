"""Compute paired permutation test.

If multiple prediction files are provided per model (i.e. random seeds),
they are assumed to be independent, and significance is measured over
average scores per datum.

Paired permutation test is done over tree-level UAS for a single langauge.
If p < 0.05, then we consider model X signficantly better than model Y.
"""

import argparse
import numpy as np
from clparse.utils import read_conllu

parser = argparse.ArgumentParser()
parser.add_argument("--conllx", type=str, nargs='+', default=None)
parser.add_argument("--conlly", type=str, nargs='+', default=None)
parser.add_argument("--gold", type=str, default=None)


def mc_paired_perm_test(xs, ys, nmc=10000):
    n, k = len(xs), 0
    zs = xs - ys
    diff = np.abs(np.mean(zs))
    for j in range(nmc):
        signs = np.random.randint(0, 2, n) * 2 - 1
        k += diff <= np.abs(np.mean(signs * zs))
    return k / float(nmc)


def eval_per_sent(pred, gold, ignore_punct=True):
    """Evaluate UAS for each sentence."""
    uas = []
    for tpred, tgold in zip(read_conllu(pred), read_conllu(gold)):
        assert len(tpred) == len(tgold)
        n = 0
        n_correct = 0
        for i in range(len(tgold)):
            if ignore_punct:
                if tgold[i]['upostag'] == "PUNCT":
                    continue
            n += 1
            if tpred[i]['head'] == tgold[i]['head']:
                n_correct += 1
        uas.append(float(n_correct) / n if n != 0 else 1.0)
    return np.asarray(uas)


def main(args):
    uas_x = np.mean(
        [eval_per_sent(x, args.gold) for x in args.conllx], dim=1)
    uas_y = np.mean(
        [eval_per_sent(y, args.gold) for y in args.conlly], dim=1)
    pvalue = mc_paired_perm_test(uas_x, uas_y)
    print(f"p-value: {pvalue}")


if __name__ == "__main__":
    main(parser.parse_args())

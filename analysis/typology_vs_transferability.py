""" Compare typological similarity to parser transferability through k-nearest neighbors
"""

import numpy as np
import os
import pickle
import operator
import argparse
import gzip
from statistics import mean
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--feature-path", type=str, required=True,
                    help="Path to the typology vectors")
parser.add_argument("--transfer-acc", type=str, default="../typologies/pairwise_transfer_acc.pkl.gz",
                    help="Path to the pairwise transfer results")
parser.add_argument("--ks", type=int, nargs='+', default=[1, 3, 5, 10],
                    help="Precision@k")


def knn(features, lang, support_langs, k):
    dists = {}
    for l in support_langs:
        dists[l] = np.linalg.norm(np.asarray(features[lang]) - np.asarray(features[l]))
    sorted_dists = sorted(dists.items(), key=operator.itemgetter(1))
    return sorted_dists[:k]


def main(args):
    # transfer_acc[tgt][src] is the transfer accuracy from src to tgt.
    with gzip.open(args.transfer_acc, "rb") as fr:
        transfer_acc = pickle.load(fr)
    with gzip.open(args.feature_path, "rb") as fr:
        typology_vec = pickle.load(fr)

    precision_ks = defaultdict(dict)
    for target, accs in transfer_acc.items():
        source_langs = list(accs.keys())
        best_source_ref = sorted(accs.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        topk_source_typ = list(zip(*knn(typology_vec, target, source_langs, max(args.ks))))[0]
        for k in args.ks:
            if best_source_ref in topk_source_typ[:k]:
                precision_ks[k][target] = 1
            else:
                precision_ks[k][target] = 0

    for k in args.ks:
        print("Precision@{}: {:.4f}".format(k, mean(precision_ks[k].values())))


if __name__ == "__main__":
    main(parser.parse_args())


""" Compute meta-typological features through mahalanobis metric learning """

import pickle
import argparse
import gzip
from statistics import mean, stdev
import numpy as np
from metric_learn import MMC_Supervised, MMC

parser = argparse.ArgumentParser()
parser.add_argument("--transfer-acc", type=str, required=True,
                    help="Path to the pairwise transfer results")
parser.add_argument("--feature-path", type=str, required=True,
                    help="Path to the typology vectors")
parser.add_argument("--output-file", type=str, required=True,
                    help="Path to the transformed typology vectors")


def main(args):
    print("Deriving similar/dissimilar constraints for metric learning.")
    with gzip.open(args.transfer_acc, "rb") as fr:
        # transer_acc[tgt][src]: accuracy of src->tgt
        transfer_acc = pickle.load(fr)
    _mean = {l: mean(list(transfer_acc[l].values())) for l in transfer_acc.keys()}
    _std = {l: stdev(list(transfer_acc[l].values())) for l in transfer_acc.keys()}

    alpha = 0.5
    sim_pairs = []
    dissim_pairs = []

    meta_langs = list(transfer_acc.keys())
    for i in range(len(meta_langs)):
        for j in range(i+1, len(meta_langs)):
            l1 = meta_langs[i]
            l2 = meta_langs[j]
            if transfer_acc[l1][l2] > _mean[l1] + alpha * _std[l1] and \
               transfer_acc[l2][l1] > _mean[l2] + alpha * _std[l2]:
                sim_pairs.append([l1, l2])
            elif transfer_acc[l1][l2] < _mean[l1] - alpha * _std[l1] and \
                 transfer_acc[l2][l1] < _mean[l2] - alpha * _std[l2]:
                dissim_pairs.append([l1, l2])

    # constraints: [simA, simB, dissimA, dissimB]
    constraints = list(zip(*sim_pairs)) + list(zip(*dissim_pairs))
    constraints = [list(map(lambda l: meta_langs.index(l), lst)) for lst in constraints]
    constraints = [np.array(x) for x in constraints]

    print("Mahalanobis metric learning.")
    with gzip.open(args.feature_path, "rb") as fr:
        typology_vec = pickle.load(fr)
    meta_X = np.array([typology_vec[l] for l in meta_langs])
    mmc = MMC()
    mmc.fit(meta_X, constraints)

    print("Apply the learned metric to the full typology vector space.")
    all_langs = list(typology_vec.keys())
    X = np.array([typology_vec[l] for l in all_langs])
    X = mmc.transform(X).tolist()
    typology_vec_transformed = {all_langs[i]: X[i] for i in range(len(all_langs))}

    with gzip.open(args.output_file, "wb") as fw:
        pickle.dump(typology_vec_transformed, fw)

if __name__ == "__main__":
    main(parser.parse_args())


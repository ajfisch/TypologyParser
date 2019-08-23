"""Test if typology is predictable from encoder hidden state."""

import argparse
import gzip
import pickle

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch

from clparse.features.wals_from_corpus import unflatten

parser = argparse.ArgumentParser()
parser.add_argument('embeddings', type=str,)
parser.add_argument('--typology', type=str,
                    default='typologies/wals_udv1_from_corpus.pkl.gz')
parser.add_argument('--seed', type=int, default=1234)


def main(args):
    print('Loading data...')
    data = torch.load(args.embeddings)

    # X is the final layer embeddings.  L is the language.
    X_train, X_test = data['X_train'], data['X_test']
    L_train, L_test = data['Y_train'], data['Y_test']
    print('Num train: %d' % len(X_train))
    print('Num test: %d' % len(X_test))

    # Load typology.
    with gzip.open(args.typology, 'rb') as f:
        typology = pickle.load(f)
        if not isinstance(next(iter(typology.values())), dict):
            typology = {k: unflatten(v) for k, v in typology.items()}
    features = set([k for v in typology.values() for k in v.keys()])
    features = sorted(features)
    print('Typology features: %s' % features)

    # Try to predict each feature separately.
    all_logreg = []
    all_majority = []
    for f in features:
        # Get the unique values that are to be predicted.
        values = list(set([v[f] for v in typology.values()]))

        # Create Y labels by matching language to typology feature index.
        Y_train = np.array([values.index(typology[l][f]) for l in L_train])
        Y_test = np.array([values.index(typology[l][f]) for l in L_test])

        # Train logreg classifier.
        clf = LogisticRegression(
            random_state=0,
            solver='lbfgs',
            multi_class='multinomial',
            max_iter=10000,
            C=1)
        clf.fit(X_train, Y_train)

        # Compute majority baseline.
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_train, Y_train)

        # Score.
        logreg_score = clf.score(X_test, Y_test) * 100
        all_logreg.append(logreg_score)
        majority_score = dummy.score(X_test, Y_test) * 100
        all_majority.append(majority_score)
        print(f, 'LogReg: %2.2f\tMajority: %2.2f' %
              (logreg_score, majority_score))

    print('-' * 39)
    print('Avg', 'LogReg: %2.2f\tMajority: %2.2f' %
          (np.mean(all_logreg), np.mean(all_majority)))


if __name__ == '__main__':
    main(parser.parse_args())

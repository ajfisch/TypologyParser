"""Language cluster identifier."""

import argparse
import gzip
import pickle

from sklearn.cluster import KMeans
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--feature-path', required=True, type=str)
parser.add_argument('--output-file', required=True, type=str)
parser.add_argument('--k', type=int, default=6)
parser.add_argument('--seed', type=int, default=1234)


def main(args):
    print('Loading features...')
    with gzip.open(args.feature_path, 'rb') as f:
        features = pickle.load(f)

    langs, vecs = list(zip(*list(features.items())))
    vecs = np.array(vecs)

    print('Computing clusters...')
    kmeans = KMeans(n_clusters=args.k, random_state=args.seed, n_jobs=-1)
    kmeans.fit(vecs)
    print('Iterations: %d/%d' % (kmeans.n_iter_, kmeans.max_iter))
    clusters = dict(list(zip(langs, list(kmeans.labels_))))

    print('Saving.')
    with gzip.open(args.output_file, 'wb') as f:
        pickle.dump(clusters, f)


if __name__ == '__main__':
    main(parser.parse_args())

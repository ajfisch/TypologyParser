"""Bulk compute typology features from conllu files."""

import argparse
import glob
import gzip
import multiprocessing
import os
import pickle

import tqdm

from clparse.utils import get_name_from_file
from clparse.utils import init_dir
from clparse.features import liu_from_corpus
from clparse.features import wang_eisner_from_corpus
from clparse.features import wals_from_corpus

parser = argparse.ArgumentParser()
parser.add_argument('--input-patterns', type=str, nargs='+', default=None)
parser.add_argument('--output-file', type=str, default='out')
parser.add_argument('--feature', type=str,
                    choices={'wals', 'wang_eisner', 'liu'},
                    default='wals')
parser.add_argument('--num-workers', type=int, default=None)


def main(args):
    inputs = [f for pattern in args.input_patterns for f in glob.glob(pattern)]
    langs = [get_name_from_file(f) for f in inputs]
    if args.feature == 'wals':
        map_fn = wals_from_corpus.compute_features
    elif args.feature == 'wang_eisner':
        map_fn = wang_eisner_from_corpus.compute_features
    elif args.feature == 'liu':
        map_fn = liu_from_corpus.compute_features
    else:
        raise ValueError('Unsupported feature type: %s' % args.feature)

    print('Processing %d datasets.' % len(inputs))
    processes = args.num_workers or multiprocessing.cpu_count() - 1
    workers = multiprocessing.Pool(processes)
    features = []
    with tqdm.tqdm(total=len(inputs)) as pbar:
        for vec in workers.imap(map_fn, inputs):
            features.append(vec)
            pbar.update()
    feature_dict = {k: v for k, v in zip(langs, features)}
    init_dir(os.path.dirname(args.output_file))
    with gzip.open(args.output_file, 'wb') as f:
        pickle.dump(feature_dict, f)


if __name__ == '__main__':
    main(parser.parse_args())

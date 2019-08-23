"""Bulk serialize GD conllu datasets into hdf5 files."""

import argparse
import functools
import h5py
import multiprocessing
import os
import pickle
import shutil

import numpy as np
import tqdm

from clparse.data.iso import convert_code
from clparse.data.treebank import Treebank
from clparse.utils import bool_flag
from clparse.utils import init_dir
from clparse.utils import get_name_from_file

parser = argparse.ArgumentParser()
parser.add_argument('--langs', type=str, nargs='+', default=None)
parser.add_argument('--input-dir', type=str, default=None)
parser.add_argument('--output-dir', type=str, default='out')
parser.add_argument('--min-len', type=int, default=2)
parser.add_argument('--max-len', type=int, default=100)
parser.add_argument('--max-arcs', type=int, default=500000)
parser.add_argument('--even-sampling', type=bool_flag, default=True)
parser.add_argument('--num-workers', type=int, default=None)


def serialize_dataset(input_file, args):
    lang = 'gd_' + get_name_from_file(input_file)
    dirname = os.path.join(args.output_dir, lang)
    init_dir(dirname)
    output_file = os.path.join(dirname, 'train.hdf5')
    params = dict(min_tree_len=args.min_len,
                  max_tree_len=args.max_len,
                  subsample=args.max_arcs,
                  subsampling_key='arcs',
                  selective_sharing_feature_loader=None)
    treebank = Treebank(lang, input_file, **params)
    serialized = np.array([pickle.dumps(i) for i in treebank])
    with h5py.File(output_file, 'w') as f:
        f.attrs['lang'] = lang
        f.attrs['args'] = np.void(pickle.dumps(params))
        f.create_dataset('examples', data=serialized)
    shutil.copyfile(input_file, os.path.join(dirname, 'train.conllu'))


def main(args):
    datasets = []
    for lang in args.langs:
        for n_lang in ['%s@N' % l for l in args.langs] + [None]:
            for v_lang in ['%s@V' % l for l in args.langs] + [None]:
                name = '~'.join(filter(None, [lang, n_lang, v_lang]))
                name = os.path.join(name, name + '-gd-train.conllu')
                name = os.path.join(
                    args.input_dir, 'GD_' + convert_code(lang), name)
                assert os.path.exists(name)
                datasets.append(name)
    print('Processing %d datasets.' % len(datasets))
    processes = args.num_workers or multiprocessing.cpu_count() - 1
    workers = multiprocessing.Pool(processes)
    map_fn = functools.partial(serialize_dataset, args=args)
    with tqdm.tqdm(total=len(datasets)) as pbar:
        for _ in workers.imap_unordered(map_fn, datasets):
            pbar.update()


if __name__ == '__main__':
    main(parser.parse_args())

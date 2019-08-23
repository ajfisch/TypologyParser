"""Universal Dependency treebank utilities for data loading."""

import bisect
import collections
import functools
import glob
import gzip
import h5py
import logging
import multiprocessing
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from clparse.data.dictionary import Dictionary
from clparse.features.selective_sharing import selective_sharing_features
from clparse.utils import get_name_from_dir
from clparse.utils import read_conllu
from clparse.utils import retriable

logger = logging.getLogger()


# ------------------------------------------------------------------------------
#
# Treebank dataset classes.
#
# ------------------------------------------------------------------------------


class Treebank(Dataset):
    """A dataset that represents a set of Universal Dependency trees."""

    def __init__(self, lang, filename, **kwargs):
        """Initialize Treebank dataset.

        Args:
          lang: Treebank language identifier, e.g. "en".
          filename: Path to treebank, of type ".conllu" or ".hdf5".
          kwargs: Extra arguments for loading conllu file.
        """
        self.lang = lang
        if filename.endswith('.conllu'):
            self.load_from_conllu(filename, **kwargs)
            self.deserialize = False
        elif filename.endswith('.hdf5'):
            self.load_from_hdf5(filename, **kwargs)
            self.deserialize = True
        else:
            raise ValueError('Unsupported file type.')

    def load_from_hdf5(self, filename, **kwargs):
        """Load pre-parsed HDF5 file."""
        self.filename = filename
        with retriable(h5py.File, filename, 'r') as dataset:
            assert dataset.attrs['lang'] == self.lang, \
                'Language does not match.'
            saved_args = pickle.loads(dataset.attrs['args'].tostring())
            for k, v in kwargs.items():
                assert saved_args[k] == v, 'Saved arguments do not match.'
            self.num_examples = len(dataset['examples'])

    def load_from_conllu(
            self,
            filename,
            min_tree_len=1,
            max_tree_len=None,
            subsample=None,
            subsampling_key='arcs',
            selective_sharing_feature_loader=None):
        """Initialize treebank from conllu file.

        Args:
          filename: Path to treebank in conllu format.
          min_tree_len: Skip trees with less than min_tree_len tokens.
          max_tree_len: Skip trees with more than max_tree_len tokens.
          subsample: Take a random sample of N examples.
          subsampling_key: Subsample by either "trees" or "arcs".
          selective_sharing_feature_loader: Callback to compute selective
            sharing features. First two arguments must be language and POS tags.
        """
        # Parse conllu file.
        self.examples = []
        arcs = 0
        trees = 0
        for i, ex in enumerate(read_conllu(filename)):
            # Throwout trees that don't meet length requirements.
            if min_tree_len and len(ex) < min_tree_len:
                continue
            if max_tree_len and len(ex) > max_tree_len:
                continue

            # Store the index of the tree as "tid" so that we can
            # recover the original order. (This is necessary to
            # compare to the gold annotations file.)
            ex = dict(tid=i,
                      lang=self.lang,
                      words=[t['form'] for t in ex],
                      pos=[t['upostag'].lower() for t in ex],
                      deprels=[t['deprel'].lower() for t in ex],
                      heads=[t['head'] for t in ex])
            self.examples.append(ex)
            arcs += len(ex['words'])
            trees += 1

        # If subsample is set, subsample up to N arcs or trees.
        if subsampling_key not in ['arcs', 'trees']:
            raise ValueError('Unknown subsampling key %s' % subsampling_key)
        total = arcs if subsampling_key == 'arcs' else trees
        if subsample and total > subsample:
            kept = []
            arcs = 0
            trees = 0
            np.random.shuffle(self.examples)
            for ex in self.examples:
                total = arcs if subsampling_key == 'arcs' else trees
                if total >= subsample:
                    break
                kept.append(ex)
                arcs += len(ex['words'])
                trees += 1
            self.examples = kept

        self.num_examples = len(self.examples)
        logger.info('Loaded %d arcs and %d trees from %s' %
                    (arcs, self.num_examples, filename))

        # Compute selective sharing features, if function provided.
        # This is done using multiprocessing for speed.
        if selective_sharing_feature_loader is not None:
            logger.info('Computing selective sharing features...')
            workers = multiprocessing.Pool(
                max(1, multiprocessing.cpu_count() - 1))
            selective_sharing_feature_loader = functools.partial(
                selective_sharing_feature_loader, self.lang)
            features = workers.map(selective_sharing_feature_loader,
                                   (ex['pos'] for ex in self.examples))
            for f, ex in zip(features, self.examples):
                ex['selective'] = torch.from_numpy(f)

        self.num_examples = len(self.examples)

    def get_lang(self, index):
        return self.lang

    def get_length(self, index):
        if self.deserialize:
            # It is too expensive to deserialize, so we just return a
            # reasonable average number.
            return 30
        return len(self[index]['pos'])

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        if self.deserialize:
            with retriable(h5py.File, self.filename, 'r') as dataset:
                ex = pickle.loads(dataset['examples'][index])
        else:
            ex = self.examples[index]
        return ex


class ConcatTreebank(ConcatDataset):
    """Wrapper for concatenating multiple Treebanks."""

    def resolve(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not "
                                 "exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_lang(self, idx):
        dataset_idx, sample_idx = self.resolve(idx)
        return self.datasets[dataset_idx].get_lang(sample_idx)

    def get_length(self, idx):
        dataset_idx, sample_idx = self.resolve(idx)
        return self.datasets[dataset_idx].get_length(sample_idx)


class TreebankBatch:
    """Container for a batch of Univeral Dependency trees."""

    def __init__(self, dicos, examples):
        """Initialize batch.

        Args:
          dicos: Dict of Dictionary objects.
            - lang: Dictionary of languages.
            - deprel: Dictionary of dependency relations.
            - pos: Dictionary of part-of-speech tags.
          examples: Batch of trees.
        """
        # Sort by length.
        examples = sorted(examples, key=lambda x: -len(x['pos']))

        # Store batch size.
        self.batch_size = len(examples)

        # Get batch languages.
        lang = [dicos['lang'][ex['lang']] for ex in examples]
        assert len(set(lang)) == 1, 'Multiple languages in batch.'

        # Get batch sequences with padding.
        pos, deprels, heads, mask = [], [], [], []
        max_len = max([len(ex['pos']) for ex in examples])
        for ex in examples:
            # POS tags are padded with 0 for PAD.
            # Also use PAD for unknown tokens in dictionaries with no UNK.
            padding = [0] * (max_len - len(ex['pos']))
            pos.append([max(dicos['pos'][p], 0) for p in ex['pos']] + padding)

            # Dependency relations are padded with -1 for IGNORE.
            padding = [-1] * (max_len - len(ex['pos']))
            deprels.append([dicos['deprel'][r]
                            for r in ex['deprels']] + padding)
            
            # Head indices are also padded with -1 for IGNORE.
            heads.append([h for h in ex['heads']] + padding)

            # mask[j] = 1 if pos[j] = PAD, and 0 otherwise.
            padding = [1] * (max_len - len(ex['pos']))
            mask.append([0] * len(ex['pos']) + padding)

            # Untensorized data.
            metadata = dict(words=[ex['words'] for ex in examples],
                            pos=[ex['pos'] for ex in examples],
                            tids=[ex['tid'] for ex in examples],
                            lang=examples[0]['lang'])

        # Add selective sharing features.
        if 'selective' in examples[0]:
            selective = []
            for ex in examples:
                # Pad to N x N x feature_space_dim (F.pad is reversed order).
                pad = max_len - len(ex['pos'])
                selective.append(F.pad(ex['selective'], (0, 0, 0, pad, 0, pad)))
            selective = torch.stack(selective, dim=0)
        else:
            selective = None

        # Convert to torch tensors.
        self.data = dict(lang=torch.LongTensor(lang),
                         pos=torch.LongTensor(pos),
                         deprels=torch.LongTensor(deprels),
                         heads=torch.LongTensor(heads),
                         mask=torch.ByteTensor(mask),
                         metadata=metadata,
                         selective=selective)

    def __len__(self):
        return self.batch_size

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def pin_memory(self):
        """Pin batch memory for quicker GPU transfer."""
        for k, v in self.data.items():
            if torch.is_tensor(v):
                self.data[k] = v.pin_memory()
        return self

    def to(self, device):
        """Transfer batch to specified device."""
        for k, v in self.data.items():
            if torch.is_tensor(v):
                self.data[k] = v.to(device, non_blocking=True)
        return self


class TreebankSampler(Sampler):
    """Batch sampler for Universal Dependency treebanks."""

    def __init__(
            self,
            dataset,
            batch_size=1,
            loop=False,
            even_sampling=False):
        """Initialize sampler.

        Args:
          dataset: Instance of Treebank.
          batch_size: Number of tokens per batch.
          loop: If True, infinitely loop over data.
          even_sampling: If True, sample evenly by language. Only relevant if
            looping behavior is enabled.
        """
        self.even_sampling = even_sampling
        self.loop = loop

        # Sort examples by:
        # 1) Language.
        # 2) Descending length.
        # 3) A random tie breaker.
        def map_fn(i):
            length = dataset.get_length(i)
            lang = dataset.get_lang(i)
            return (np.random.random(), -length, lang)
        
        # Last key is the primary key for np.lexsort.
        keys = [map_fn(i) for i in range(len(dataset))]
        indices = np.lexsort(tuple(zip(*keys)))

        # Break off into monolingual batches.
        # Target batch_size *tokens* per batch.
        self.batches = []
        curr_batch = []
        curr_tokens = 0
        for i in indices:
            # Target size met: break off.
            if curr_tokens >= batch_size:
                self.batches.append(curr_batch)
                curr_batch = []
                curr_tokens = 0
            # Language changed: break off.
            if len(curr_batch) > 0 and keys[curr_batch[-1]][-1] != keys[i][-1]:
                self.batches.append(curr_batch)
                curr_batch = []
                curr_tokens = 0
            # Append to batch, increment size count.
            # Note: keys[i][-2] is negative sentence length.
            curr_batch.append(i)
            curr_tokens -= keys[i][-2]
        # Add the remainder.
        if len(curr_batch) > 0:
            self.batches.append(curr_batch)

        logger.info('Average sentences per batch: %d' %
                    np.mean([len(b) for b in self.batches]))

        # Split out batches into language buckets if doing even sampling.
        # language: [[idx_1, ..., idx_n], ..., [idx_1, ..., idx_n]]
        if even_sampling:
            self.by_lang = collections.defaultdict(list)
            self.counts = collections.defaultdict(int)
            for batch in self.batches:
                lang = keys[batch[0]][-1]
                self.by_lang[lang].append(batch)
                self.counts[lang] += len(batch)
            self.i2l = [lang for lang in self.by_lang.keys()]

            # Compute sampling proportions similar to MBERT
            # https://github.com/google-research/bert/blob/master/multilingual.md
            totals = np.array([self.counts[l] for l in self.i2l])
            probs = totals / totals.sum()
            exp_probs = np.power(probs, 0.7)
            self.exp_probs = exp_probs / exp_probs.sum()

    def __iter__(self):
        # Run one epoch.
        if not self.loop:
            np.random.shuffle(self.batches)
            for batch in self.batches:
                yield batch

        # Iterate over data "forever."
        else:
            while True:
                if self.even_sampling:
                    # Choose a language sampled from the exponential probs.
                    lang = np.random.choose(self.i2l, p=self.exp_probs)
                    # Choose a  batch from that language uniformly at random.
                    idx = np.random.randint(len(self.by_lang[lang]))
                    batch = self.by_lang[lang][idx]
                    yield batch
                else:
                    # Randomly iterate all batches once, and then repeat.
                    np.random.shuffle(self.batches)
                    for batch in self.batches:
                        yield batch

    def __len__(self):
        if self.loop:
            return None
        return len(self.batches)


# ------------------------------------------------------------------------------
#
# Loading functions
#
# ------------------------------------------------------------------------------


def load_treebanks(
        filename_map,
        min_tree_len=None,
        max_tree_len=None,
        subsample=None,
        subsampling_key='arcs',
        selective_sharing_feature_loader=None):
    """Load Treebank objects from a list of files.

    Args:
      filename_map: Dict of language to filename, e.g. {"en": "en.conllu"}.
      min_tree_len: Skip trees with less than min_tree_len tokens.
      max_tree_len: Skip trees with more than max_tree_len tokens.
      subsample: Take a random sample of N examples.
      subsampling_key: Subsample by either "trees" or "arcs".
      selective_sharing_feature_loader: Callback to compute selective
        sharing features. First two arguments must be language and POS tags.

    Returns:
      A concatenated Treebank dataset over all provided treebanks.
    """
    def map_fn(args):
        lang, filename = args
        return Treebank(
            lang=lang,
            filename=filename,
            min_tree_len=min_tree_len,
            max_tree_len=max_tree_len,
            subsample=subsample,
            subsampling_key=subsampling_key,
            selective_sharing_feature_loader=selective_sharing_feature_loader)

    # Load treebanks (could be trivially parallelized).
    treebanks = list(map(map_fn, filename_map.items()))

    # Exit if treebanks are empty.
    if len(treebanks) == 0:
        logger.warn('Empty treebank!')

    # Combine treebanks together.
    joint_treebank = ConcatTreebank(treebanks)
    return joint_treebank


def build_dicos(*treebanks):
    """Build dictionaries from provided Treebanks."""
    dicos = {'lang': Dictionary(),
             'pos': Dictionary(pad=True, unk=True),
             'deprel': Dictionary()}
    for treebank in treebanks:
        for ex in treebank:
            dicos['lang'].add(ex['lang'])
            dicos['pos'].update(ex['pos'])
            dicos['deprel'].update(ex['deprels'])
    return dicos


def get_iterator(
        treebank,
        dicos,
        batch_size=1,
        loop=False,
        even_sampling=False,
        num_workers=0):
    """Get an iterator over Treebanks that yields TreebankBatch."""

    # Custom collate function.
    def collate_fn(batch):
        return TreebankBatch(dicos, batch)

    # Exit if the treebank is empty.
    if len(treebank) == 0:
        logger.warn('Empty treebank!')

    sampler = TreebankSampler(
        dataset=treebank,
        batch_size=batch_size,
        loop=loop,
        even_sampling=even_sampling)
    iterator = DataLoader(dataset=treebank,
                          batch_sampler=sampler,
                          num_workers=num_workers,
                          collate_fn=collate_fn)
    return iterator


# ------------------------------------------------------------------------------
#
# Setting specific loading.
#
# ------------------------------------------------------------------------------


def load_multi_source_data(args, dicos=None):
    """Load data for multi-source transfer experiments."""
    # Set source treebanks.
    src_treebanks = [(get_name_from_dir(f), f)
                     for pattern in args.src_patterns
                     for f in glob.glob(pattern)]
    train_map = {t: os.path.join(f, 'train.conllu') for t, f in src_treebanks}
    valid_map = {t: os.path.join(f, 'valid.conllu') for t, f in src_treebanks}
    test_map = {t: os.path.join(f, 'test.conllu') for t, f in src_treebanks}

    # Set GD treebanks. These are precomputed hdf5 files. Train only.
    if getattr(args, 'gd_patterns', False):
        gd_treebanks = [(get_name_from_dir(f), f)
                        for pattern in args.gd_patterns
                        for f in glob.glob(pattern)]
        gd_map = {t: os.path.join(f, 'train.hdf5') for t, f in gd_treebanks}
    else:
        gd_map = {}

    # Set target treebanks for eval. We use these treebanks to initialize the
    # dictionaries at the start so that they don't have to be expanded at
    # test time. We load all splits, as we might run on any of them.
    prepare_target = getattr(args, 'tgt_patterns', False)
    if prepare_target:
        tgt_treebanks = [(get_name_from_dir(f), f)
                         for pattern in args.tgt_patterns
                         for f in glob.glob(pattern)]
        tgt_train_map = {t: os.path.join(f, 'train.conllu')
                         for t, f in tgt_treebanks}
        tgt_valid_map = {t: os.path.join(f, 'valid.conllu')
                         for t, f in tgt_treebanks}
        tgt_test_map = {t: os.path.join(f, 'test.conllu')
                        for t, f in tgt_treebanks}

    # Load selective sharing feature resources and preprocessor.
    if getattr(args, 'use_selective_sharing', False):
        with gzip.open(args.typology, 'rb') as f:
            typology = pickle.load(f)
        with gzip.open(args.selection_template, 'rb') as f:
            template = pickle.load(f)
            args.selective_feature_dim = len(template)
        selective_sharing_feature_loader = functools.partial(
            selective_sharing_features,
            lang2wals=typology,
            feature_space=template)
    else:
        selective_sharing_feature_loader = None

    # Load treebanks and build dictionaries
    _load_treebanks = functools.partial(
        load_treebanks,
        selective_sharing_feature_loader=selective_sharing_feature_loader)
    train_trees = _load_treebanks(
        train_map,
        min_tree_len=args.min_len,
        max_tree_len=args.max_len,
        subsample=args.max_arcs)
    if gd_map:
        gd_trees = _load_treebanks(
            gd_map,
            min_tree_len=args.min_len,
            max_tree_len=args.max_len,
            subsample=args.max_arcs)
    else:
        gd_trees = []
    valid_trees = _load_treebanks(valid_map)
    test_trees = _load_treebanks(test_map)

    # Add target.
    if prepare_target:
        tgt_train_trees = _load_treebanks(tgt_train_map)
        tgt_valid_trees = _load_treebanks(tgt_valid_map)
        tgt_test_trees = _load_treebanks(tgt_test_map)

    # Build dictionaries.
    if dicos is None:
        trees = [train_trees, valid_trees, test_trees]
        if prepare_target:
            trees.extend([tgt_train_trees, tgt_valid_trees, tgt_test_trees])
        dicos = build_dicos(*trees)

    # Add GD languages to languages dictionary.
    if gd_map:
        for lang in gd_map.keys():
            dicos['lang'].add(lang)

    args.num_pos = len(dicos['pos'])
    args.num_lang = len(dicos['lang'])
    args.num_deprel = len(dicos['deprel'])

    logger.info('=' * 50)
    logger.info('Num langs: %d' % args.num_lang)
    logger.info('Num pos: %d' % args.num_pos)
    logger.info('Num deprel: %d' % args.num_deprel)
    logger.info('Num train trees: %d' % len(train_trees))
    logger.info('Num synthetic train trees: %d' % len(gd_trees))
    logger.info('Num valid trees: %d' % len(valid_trees))
    logger.info('Num test trees: %d' % len(test_trees))
    logger.info('=' * 50)

    # Make iterators.
    iterator = functools.partial(get_iterator,
                                 dicos=dicos,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers)
    train_iter = iterator(train_trees,
                          even_sampling=args.even_sampling,
                          loop=True)
    if gd_trees:
        gd_iter = iterator(gd_trees,
                           even_sampling=args.even_sampling,
                           loop=True)
    else:
        None
    valid_iter = iterator(valid_trees)
    test_iter = iterator(test_trees)
    if prepare_target:
        tgt_train_iter = iterator(tgt_train_trees)
        tgt_valid_iter = iterator(tgt_valid_trees)
        tgt_test_iter = iterator(tgt_test_trees)

    # Store files + iterators.
    files = dict(train=train_map, gd=gd_map, valid=valid_map, test=test_map)
    iterators = dict(train=train_iter,
                     gd=gd_iter,
                     valid=valid_iter,
                     test=test_iter)
    if prepare_target:
        files.update(dict(tgt_train=tgt_train_map,
                          tgt_valid=tgt_valid_map,
                          tgt_test=tgt_test_map))
        iterators.update(dict(tgt_train=tgt_train_iter,
                              tgt_valid=tgt_valid_iter,
                              tgt_test=tgt_test_iter))

    return files, iterators, dicos

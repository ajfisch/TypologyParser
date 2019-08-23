"""Dump baseline encoder states to disk."""

import argparse
import os

import torch
import tqdm

from clparse import utils
from clparse.data.treebank import get_iterator
from clparse.data.treebank import load_treebanks
from clparse.parsers.parser import Parser

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    help='Path to model directory.')
parser.add_argument('--treebanks', type=str, default='data/ud_v1.2',
                    help='Path to treebank data.')
parser.add_argument('--output', type=str, default=None,
                    help='Path to write results to.')
parser.add_argument('--batch-size', type=int, default=1000)
parser.add_argument('--cuda', type=utils.bool_flag, default=True)
parser.add_argument('--seed', type=int, default=1234)
args = parser.parse_args()


def encode(model, device, iterator):
    examples = []
    langs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(iterator, ncols=130):
            batch = batch.to(device)
            lang = batch['lang'][0].view(1, 1)
            embs = model.embed(lang, batch['pos'], batch['mask'])
            encs = model.encode(lang, embs, batch['mask']).cpu()
            max_pool = encs.max(dim=1)[0]
            examples.append(max_pool)
            langs.extend([batch['metadata']['lang']] * len(batch))
    return torch.cat(examples, dim=0), langs


def main(args):
    utils.set_seed(args.seed)
    device = utils.init_device(args.cuda)
    logger = utils.init_logger()

    logger.info('Loading model...')
    model = Parser.load(args.model).to(device)
    model.eval()
    dicos = model.dicos

    # Data is in-domain (loaded from source datasets).
    logger.info('Loading data...')
    train_file_map = {}
    valid_file_map = {}
    test_file_map = {}
    for f in model.args.src_treebanks:
        lang = utils.get_name_from_dir(f)
        basedir = os.path.join(args.treebanks, lang)
        f_train = basedir + '/train.conllu'
        train_file_map[lang] = f_train
        f_valid = basedir + '/valid.conllu'
        valid_file_map[lang] = f_valid
        f_test = basedir + '/test.conllu'
        test_file_map[lang] = f_test

    # Load up to 2k trees from each language for training.
    train_trees = load_treebanks(
        train_file_map,
        subsample=2000,
        subsampling_key='trees')
    train_iterator = get_iterator(train_trees, dicos, args.batch_size)

    # Load up to 500 trees from each language for validation.
    valid_trees = load_treebanks(
        valid_file_map,
        subsample=500,
        subsampling_key='trees')
    valid_iterator = get_iterator(valid_trees, dicos, args.batch_size)

    # Load up to 500 trees from each language for testing.
    test_trees = load_treebanks(
        test_file_map,
        subsample=500,
        subsampling_key='trees')
    test_iterator = get_iterator(test_trees, dicos, args.batch_size)

    logger.info('=' * 50)
    logger.info("Encoding...")

    train_examples, train_langs = encode(model, device, train_iterator)
    valid_examples, valid_langs = encode(model, device, valid_iterator)
    test_examples, test_langs = encode(model, device, test_iterator)
    data = {'X_train': train_examples.numpy(), 'Y_train': train_langs,
            'X_valid': valid_examples.numpy(), 'Y_valid': valid_langs,
            'X_test': test_examples.numpy(), 'Y_test': test_langs}

    # Save.
    dirname = os.path.dirname(args.output)
    utils.init_dir(dirname)
    torch.save(data, args.output)


if __name__ == '__main__':
    main(parser.parse_args())

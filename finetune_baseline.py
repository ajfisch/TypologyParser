"""Finetune trained model on target treebank."""

import argparse
import json
import os
import tqdm

from clparse import utils
from clparse.data.treebank import get_iterator
from clparse.data.treebank import load_treebanks
from clparse.evaluator import Evaluator
from clparse.parsers.parser import Parser
from clparse.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    help='Path to model directory.')
parser.add_argument('--treebank', type=str, required=True,
                    help='Path to treebank directory to finetune.')
parser.add_argument('--output', type=str, default=None,
                    help='Directory to write results to.')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--N', type=int, default=10)
parser.add_argument('--optimizer', default='adam,lr=0.001')
parser.add_argument('--cuda', type=utils.bool_flag, default=True)
parser.add_argument('--seed', type=int, default=1234)


def main(args):
    utils.set_seed(args.seed)
    device = utils.init_device(args.cuda)
    logger = utils.init_logger()

    logger.info('=' * 50)
    logger.info('Fine-tuning Script')
    logger.info('=' * 50)

    logger.info('Loading model...')
    model = Parser.load(args.model)
    model = model.to(device)
    dicos = model.dicos
    model.args.optimizer = args.optimizer

    args.num_lang = len(args.treebank)
    trainer = Trainer(model.args, model, device)
    evaluator = Evaluator(args, model, device)
    logger.info('=' * 50)

    logger.info('Loading data...')
    name = utils.get_name_from_dir(args.treebank)

    # We use the validation split for sourcing training sentences, as the
    # true training set is used for testing.
    train_file_map = {name: os.path.join(args.treebank, 'valid.conllu')}

    # Train data is the test data.
    test_file_map = {name: os.path.join(args.treebank, 'train.conllu')}
    train_trees = load_treebanks(
        train_file_map,
        subsample=args.N,
        subsampling_key='trees')
    test_trees = load_treebanks(test_file_map)
    train_iterator = get_iterator(
        treebank=train_trees,
        dicos=dicos,
        batch_size=args.batch_size,
        loop=True)
    test_iterator = get_iterator(
        treebank=test_trees,
        dicos=dicos,
        batch_size=500)

    logger.info('=' * 50)
    logger.info("Fine-tuning...")
    for step, inputs in tqdm.tqdm(enumerate(train_iterator, 1),
                                  total=args.steps, ncols=130):
        trainer.step(inputs)
        if step == args.steps:
            break

    logger.info("Evaluating...")
    metrics = evaluator.run_official(test_iterator, test_file_map)

    if args.output:
        utils.init_dir(args.output)
        model.save(os.path.join(args.output, 'model'))
        with open(os.path.join(args.output, 'test-metrics.json'), 'w') as f:
            json.dump(metrics, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

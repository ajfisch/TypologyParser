"""Make conllu formatted predictions on target treebanks."""

import argparse
import os
import json

from clparse import utils
from clparse.data.treebank import get_iterator
from clparse.data.treebank import load_treebanks
from clparse.evaluator import Evaluator
from clparse.evaluator import TaFEvaluator
from clparse.parsers.parser import Parser
from clparse.parsers.taf_parser import TaFParser
from clparse.parsers.tass_parser import TaSSParser

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    help='Path to model directory.')
parser.add_argument('--output', type=str, required=True,
                    help='Path to write results to.')
parser.add_argument('--model-type', type=str,
                    choices={'baseline', 'taf', 'tass'},
                    default='baseline')
parser.add_argument('--treebanks', type=str, nargs='+', required=True,
                    help='Paths to treebank *files* to evaluate.')
parser.add_argument('--batch-size', type=int, default=1000)
parser.add_argument('--cuda', type=utils.bool_flag, default=True)


def main(args):
    device = utils.init_device(args.cuda)
    logger = utils.init_logger()

    logger.info('=' * 50)
    logger.info('Prediction Script')
    logger.info('=' * 50)

    logger.info('Loading model...')
    if args.model_type == 'baseline':
        parser_cls = Parser
        evaluator_cls = Evaluator
    elif args.model_type == 'taf':
        parser_cls = TaFParser
        evaluator_cls = TaFEvaluator
    elif args.model_type == 'tass':
        parser_cls = TaSSParser
        evaluator_cls = Evaluator
    else:
        raise ValueError('Unkown model type.')

    model = parser_cls.load(args.model).to(device)
    dicos = model.dicos

    args.num_lang = len(args.treebanks)
    evaluator = evaluator_cls(args, model, device)

    logger.info('=' * 50)
    logger.info('Loading data...')
    file_map = {utils.get_name_from_file(f): f for f in args.treebanks}
    trees = load_treebanks(file_map)
    iterator = get_iterator(trees, dicos, args.batch_size)
    logger.info('=' * 50)

    logger.info("Predicting...")
    utils.init_dir(args.output)
    evaluator.dump_predictions(iterator, args.output)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

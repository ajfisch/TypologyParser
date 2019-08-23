"""Evaluate trained model on target treebanks."""

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
parser.add_argument('--model-type', type=str,
                    choices={'baseline', 'taf', 'tass'},
                    default='baseline')
parser.add_argument('--treebanks', type=str, nargs='+', required=True,
                    help='Paths to treebank *files* to evaluate.')
parser.add_argument('--output', type=str, default=None,
                    help='Path to write results to.')
parser.add_argument('--batch-size', type=int, default=1000)
parser.add_argument('--cuda', type=utils.bool_flag, default=True)
parser.add_argument('--as-lang', type=str, default=None)
parser.add_argument('--strict', type=utils.bool_flag, default=True)


def main(args):
    device = utils.init_device(args.cuda)
    logger = utils.init_logger()

    logger.info('=' * 50)
    logger.info('Evaluation Script')
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

    # Make dictionary access non-strict?
    for d in dicos.values():
        d.strict = args.strict

    args.num_lang = len(args.treebanks)
    evaluator = evaluator_cls(args, model, device)

    logger.info('=' * 50)
    logger.info('Loading data...')
    file_map = {utils.get_name_from_file(f): f
                for f in args.treebanks}

    # Check that languages are in the dictionary. If not, add "as_lang".
    if args.as_lang:
        for l in file_map.keys():
            if l not in dicos['lang']:
                dicos['lang'].t2i[l] = dicos['lang'].t2i[args.as_lang]

    trees = load_treebanks(file_map)
    iterator = get_iterator(trees, dicos, args.batch_size)
    logger.info('=' * 50)

    logger.info("Evaluating...")
    metrics = evaluator.run_official(iterator, file_map)

    if args.output:
        logger.info('Saving results to %s' % args.output)
        dirname = os.path.dirname(args.output)
        utils.init_dir(dirname)
        with open(args.output, 'w') as f:
            json.dump(metrics, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

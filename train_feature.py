"""Train the multi-source transfer model with typology input features."""

import argparse
import json

from clparse import experiment
from clparse.data.treebank import load_multi_source_data
from clparse.evaluator import TaFEvaluator
from clparse.parsers.taf_parser import TaFParser
from clparse.trainer import Trainer

parser = argparse.ArgumentParser()
experiment.add_base_arguments(parser)
experiment.add_typology_arguments(parser)


def main(args):
    if args.config:
        with open(args.config) as f:
            for k, v in json.load(f).items():
                setattr(args, k, v)

    args, logger, device = experiment.init_experiment(args)

    if args.pretrained:
        logger.info('Loading pretrained model...')
        model = TaFParser.load(args.pretrained)
        dicos = model.dicos
        model = model.to(device)
        logger.info("Model: %s" % model)
    else:
        dicos = None

    logger.info('Loading data...')
    files, iterators, dicos = load_multi_source_data(args, dicos)

    if not args.pretrained:
        logger.info('Initializing model...')
        model = TaFParser(args, dicos)
        model = model.to(device)
        logger.info("Model: %s" % model)

    model = experiment.train_with_validation(
        args=args,
        device=device,
        model=model,
        iterators=iterators,
        trainer_cls=Trainer,
        evaluator_cls=TaFEvaluator)

    experiment.evaluate(
        args=args,
        device=device,
        model=model,
        iterators=iterators,
        files=files,
        evaluator_cls=TaFEvaluator)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

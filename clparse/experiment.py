"""Common training and evaluation routines."""

import json
import logging
import os
import sys
import time
import uuid
import numpy as np

from clparse import utils

logger = logging.getLogger()


def add_base_arguments(parser):
    """Add base experiment arguments to parser."""

    # Main experiment params
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON file with configuration options.')
    parser.add_argument('--experiment-dir', type=str, default=None,
                        help='Parent directory where experiments are saved.')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help=('Child directory for experiment: '
                              '<experiment_dir>/<experiment_name>.'))
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to warm-start model.')
    parser.add_argument('--seed', type=int, default=1013,
                        help='Random seed for experiment.')
    parser.add_argument('--cuda', type=utils.bool_flag, default=True,
                        help='Enable GPU runtime.')
    parser.add_argument('--multigpu', type=utils.bool_flag, default=False,
                        help='Enable multi-gpu data parallel runtime.')

    # Data config
    parser.add_argument('--src-patterns', type=str, nargs='+',
                        default=['data/ud_v1.2/en'],
                        help='Glob patterns for source language directories.')
    parser.add_argument('--tgt-patterns', type=str, nargs='+', default=None,
                        help='Glob patterns for target langauge directories.')
    parser.add_argument('--gd-patterns', type=str, nargs='+', default=None,
                        help='Glob patterns for GD language directories.')
    parser.add_argument('--min-len', type=int, default=2,
                        help='Minimum number of tokens per train tree.')
    parser.add_argument('--max-len', type=int, default=100,
                        help='Maximum number of tokens per train tree.')
    parser.add_argument('--max-arcs', type=int, default=500000,
                        help='Maximum number of tokens/arcs per train source.')
    parser.add_argument('--even-sampling', type=utils.bool_flag, default=False,
                        help='Use Multi-BERT style balanced sampling.')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of processes for data loading.')

    # Training
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Batch size in tokens.')
    parser.add_argument('--steps', type=int, default=200000,
                        help='Maximum number of training steps.')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='Log every N steps.')
    parser.add_argument('--eval-interval', type=int, default=2000,
                        help='Validate every N steps.')
    parser.add_argument('--checkpoint-interval', type=int, default=-1,
                        help='Save current model every N steps.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Terminate after N evals without improvement.')
    parser.add_argument('--patience-margin', type=float, default=0.1,
                        help='Margin to judge if eval metric has improved.')

    # Optimization
    parser.add_argument('--beta', type=float, default=0.2,
                        help='Sample real languages with probability beta.')
    parser.add_argument('--valid-metric', type=str, default='UAS_AVG',
                        help='Metric to use for early stopping/patience.')
    parser.add_argument('--metric-sign', type=int, default=1,
                        help='+1 if metric is maximized, -1 if minimized.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use: "name,param=value,..."')
    parser.add_argument('--clip-grad-norm', type=float, default=1,
                        help='Gradient clipping threshold.')

    # Model config
    parser.add_argument('--pos-dim', type=int, default=50,
                        help='Part-of-speech embedding dimension.')
    parser.add_argument('--hidden-dim', type=int, default=400,
                        help='LSTM hidden dimension.')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of LSTM layers.')
    parser.add_argument('--arc-dim', type=int, default=500,
                        help='Deep arc state dimension.')
    parser.add_argument('--deprel-dim', type=int, default=100,
                        help='Deep dependency label state dimension')
    parser.add_argument('--dropout', type=float, default=0.33,
                        help='Input dropout throughout network.')
    parser.add_argument('--variational-dropout', type=float, default=0.33,
                        help='State dropout used in LSTM encoder.')


def add_typology_arguments(parser):
    """Add typology related arguments to parser."""

    # Typology config
    parser.add_argument('--use-cluster-features', type=utils.bool_flag,
                        default=False,
                        help='Use one-hot clustering (quantized) features.')
    parser.add_argument('--typology-clusters', type=str, default=None,
                        help='Path to cluster assignment map.')
    parser.add_argument('--use-hand-features', type=utils.bool_flag,
                        default=False,
                        help='Use hand-defined language features, e.g. WALS.')
    parser.add_argument('--hand-typology', type=str,
                        default='typologies/wals_udv1.pkl.gz',
                        help='Path to feature vector assignment map.')
    parser.add_argument('--hand-typology-dim', type=int, default=128,
                        help='Dimension of typology feature embedding.')
    parser.add_argument('--hand-typology-layers', type=int, default=1,
                        help='Number of FFNN layers for encoding typology.')
    parser.add_argument('--hand-typology-activation', type=str, default='tanh',
                        help='Activation of FFNN layers for encoding typology.')
    parser.add_argument('--hand-typology-dropout', type=float, default=0,
                        help='FFNN dropout to apply for encoding typology.')

    parser.add_argument('--use-neural-features', type=utils.bool_flag,
                        default=False,
                        help='Use neural features derived from the corpus.')
    parser.add_argument('--neural-typology-rnn-dim', type=int, default=128,
                        help='Dimension of RNN typology extractor.')
    parser.add_argument('--neural-typology-rnn-layers', type=int, default=1,
                        help='Number of RNN typology extractor layers.')
    parser.add_argument('--neural-typology-dim', type=int, default=64,
                        help='Dimension of neural typology feature embedding.')
    parser.add_argument('--neural-typology-layers', type=int, default=1,
                        help='Number of FFNN layers for encoding RNN output.')
    parser.add_argument('--neural-typology-activation', type=str,
                        default='sigmoid',
                        help='Activation of FFNN layers for encoding output.')
    parser.add_argument('--neural-typology-dropout', type=float, default=0,
                        help='FFNN dropout to apply for encoding RNN output.')


def add_selective_sharing_arguments(parser):
    """Add selective sharing related arguments to parser."""

    # Selective sharing config
    parser.add_argument('--use-selective-sharing', type=utils.bool_flag,
                        default=True,
                        help='Use hand-crafted selective sharing features')
    parser.add_argument('--typology', type=str,
                        default='typologies/wals_mapping_udv1.pkl.gz',
                        help=('Path to mapping containing *original* typology '
                              'definitions, e.g. "SV".'))
    parser.add_argument('--selection-template', type=str,
                        default='typologies/ss_wals_template.pkl.gz',
                        help=('Path to templates for defining selective '
                              'sharing rules.'))


def init_experiment(args):
    """Initialize experiment."""
    # Check for CUDA
    device = utils.init_device(args.cuda)

    # Set random seed for reproducibility
    if getattr(args, 'seed', -1) >= 0:
        utils.set_seed(args.seed)

    # Setup experiment directory
    if not args.experiment_dir:
        args.experiment_dir = os.path.join('results/adhoc',
                                           time.strftime('%Y%m%d'))
    if not args.experiment_name:
        args.experiment_name = str(uuid.uuid4())[:8]

    args.path = os.path.join(args.experiment_dir, args.experiment_name)
    utils.init_dir(args.path)

    # Filenames inside experiment directory
    args.model_dir = os.path.join(args.path, 'model')
    args.checkpoint_dir = os.path.join(args.path, 'checkpoint')
    args.log_file = os.path.join(args.path, 'log.txt')

    # Initialize logger
    logger = utils.init_logger(args.log_file)
    logger.info('============ Initialized logger ============')
    logger.info('Command run: %s' % ' '.join(sys.argv))
    logger.info('The experiment will be stored in %s' % args.path)

    # Log config
    logger.info('Config: %s' % json.dumps(vars(args), indent=2, sort_keys=True))

    return args, logger, device


def train_with_validation(
        args,
        device,
        model,
        iterators,
        trainer_cls,
        evaluator_cls):
    """Run training with validation and early stopping.

    Args:
      args: Experiment params.
      device: Torch device to run models on (CPU or GPU #).
      model: Parser model to train.
      iterators: Set of iterator over batches of data.
        - train: Training data.
        - valid: Validation data.
      trainer_cls: Trainer class to instantiate.
      evaluator_cls: Evaluator class to instantiate.

    Returns:
      Best model checkpoint.
    """
    trainer = trainer_cls(args, model, device)
    evaluator = evaluator_cls(args, model, device)

    logger.info('============ Begin training ============')
    best_valid = -float('inf')
    past_valids = []

    train_iterator = iter(iterators['train'])
    if iterators['gd'] is not None:
        gd_iterator = iter(iterators['gd'])
    else:
        gd_iterator = None

    for step in range(1, args.steps + 1):
        # If synthetic GD data is provided then with probability 1 - beta we
        # sample a batch from it. Otherwise we sample a batch of real data.
        if gd_iterator is not None and np.random.random() > args.beta:
            inputs = next(gd_iterator)
        else:
            inputs = next(train_iterator)

        trainer.step(inputs)

        if step % args.log_interval == 0:
            trainer.log(step)

        if args.checkpoint_interval > 0 and \
           step % args.checkpoint_interval == 0:
            model.save(os.path.join(args.checkpoint_dir, str(step)))

        if step % args.eval_interval == 0:
            logger.info('============ Validating ============')
            metrics = evaluator.run_unofficial(iterators['valid'])
            metric = args.metric_sign * metrics[args.valid_metric]

            # Save if best model.
            if metric > best_valid:
                best_valid = metric
                logger.info('Best validation %s so far: %2.2f' %
                            (args.valid_metric, best_valid))
                model.save(args.model_dir)

            # Stop if past improvement has been small.
            if len(past_valids) < args.patience:
                past_valids.append(metric)
            else:
                past_valids = past_valids[1:] + [metric]
                if max(past_valids) - past_valids[0] <= args.patience_margin:
                    logger.info('Improvement less than or equal to %2.2f for '
                                '%d iterations. Stopping.' %
                                (args.patience_margin, args.patience))
                    break
            logger.info('====================================')

    logger.info('Finished training.')

    logger.info('Reloading best model.')
    model = model.load(args.model_dir).to(device)
    return model


def evaluate(
        args,
        device,
        model,
        iterators,
        files,
        evaluator_cls):
    """Run full evaluation on available splits.

    Args:
      args: Experiment params.
      device: Torch device to run models on (CPU or GPU #).
      model: Parser model to evaluate.
      iterators: Set of iterator over batches of data.
        - valid: (Optional) Source validation data.
        - test: (Optional) Source test data.
        - tgt_train: (Optional) Target train data.
        - tgt_valid: (Optional) Target validation data.
        - tgt_test: (Optional) Target test data.
      files: Set of conllu files with the same fields as iterators.
      evaluator_cls: Evaluator class to instantiate.
    """
    logger.info('============ Official Evaluation ============')
    evaluator = evaluator_cls(args, model, device)
    for key in ['valid', 'test', 'tgt_train', 'tgt_valid', 'tgt_test']:
        if iterators.get(key):
            logger.info('Evaluating %s' % key)
            metrics = evaluator.run_official(iterators[key], files[key])
            metrics_filename = os.path.join(args.path, '%s_metrics.json' % key)
            with open(metrics_filename, 'w') as f:
                json.dump(metrics, f)

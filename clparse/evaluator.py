"""Evaluation utilities."""

import collections
import json
import logging
import multiprocessing
import os
import re
import subprocess
import tempfile

import numpy as np
import tqdm

from clparse.meters import AverageMeter
from clparse.utils import clean_conllu_file
from clparse.utils import dump_to_conllu

logger = logging.getLogger()


# ------------------------------------------------------------------------------
#
# Functional interface to accessing official eval.pl.
#
# ------------------------------------------------------------------------------


def evaluate_official(inputs):
    """Run the eval.pl script using predicted vs. gold annotations."""
    lang, outputs, gold_file = inputs
    pred_file = tempfile.NamedTemporaryFile(delete=False).name
    dump_to_conllu(outputs, pred_file)

    tids = set([o[0] for o in outputs])
    gold_clean = tempfile.NamedTemporaryFile(delete=False).name
    clean_conllu_file(gold_file, gold_clean, tids)

    score_file = tempfile.NamedTemporaryFile(delete=False).name
    this_dir = os.path.dirname(os.path.realpath(__file__))
    binary = os.path.join(this_dir, 'eval.pl')
    with open(score_file, 'w') as f:
        cmd = ['perl', binary, '-g', gold_clean, '-s', pred_file]
        subprocess.call(cmd, stdout=f)

    with open(score_file) as f:
        las_line = f.readline()
        las = float(re.findall(r'(\d+\.\d+) %', las_line)[0])
        uas_line = f.readline()
        uas = float(re.findall(r'(\d+\.\d+) %', uas_line)[0])

    for fname in (pred_file, gold_clean, score_file):
        os.remove(fname)

    return lang, las, uas


# ------------------------------------------------------------------------------
#
# Parser evaluator classes.
#
# ------------------------------------------------------------------------------


class Evaluator:
    """Evaluator class for the basic parser."""

    def __init__(self, args, model, device):
        """Initialize Evaluator.

        Args:
          args: Evaluation config.
          model: Model to evaluate.
          device: Torch device to run on (CPU vs GPU number).
        """
        self.device = device
        self.args = args
        if args.num_lang > 1:
            workers = multiprocessing.Pool(
                min(multiprocessing.cpu_count() - 1, args.num_lang))
            self.map = workers.map
        else:
            self.map = map
        self.M = model
        self.keys = model.KEYS_FOR_PREDICT

    def pre(self, iterator):
        """Hook called *before* testing on the data in the iterator."""
        pass

    def post(self):
        """Hook called *after* testing on the data in the iterator."""
        pass

    def predict(self, inputs):
        """Run wrapped model in inference mode over inputs."""
        self.M.eval()
        args = {k: inputs[k] for k in self.keys}
        return self.M.predict(**args)

    def log(self, metrics):
        metrics = {k: round(v, 2) for k, v in metrics.items()}
        logger.info('Metrics: %s' %
                    json.dumps(metrics, indent=2, sort_keys=True))

    def dump_predictions(self, iterator, dirname):
        """Dump predictions on examples drawn from iterator to dirname."""
        self.pre(iterator)
        outputs = collections.defaultdict(list)
        for inputs in tqdm.tqdm(iterator, ncols=130):
            inputs = inputs.to(self.device)
            pred_heads, pred_deprel = self.predict(inputs)
            dicos = self.M.dicos
            for i in range(len(inputs)):
                N = len(inputs['metadata']['words'][i])
                lang = inputs['metadata']['lang']
                tid = inputs['metadata']['tids'][i]
                words = inputs['metadata']['words'][i]
                pos = inputs['metadata']['pos'][i]
                heads = pred_heads[i, :N].tolist()
                deprels = [dicos['deprel'][r]
                           for r in pred_deprel[i, :N].tolist()]
                outputs[lang].append((tid, words, pos, heads, deprels))
        self.post()
        for lang, output in outputs.items():
            filename = os.path.join(dirname, lang)
            dump_to_conllu(output, filename)

    def run_unofficial(self, iterator):
        """Run unofficial (faster) evaluation using torch ops."""
        uas_meter = collections.defaultdict(AverageMeter)
        las_meter = collections.defaultdict(AverageMeter)
        self.pre(iterator)
        for inputs in tqdm.tqdm(iterator, ncols=130):
            inputs = inputs.to(self.device)
            mask = inputs['mask'].eq(0).float()
            lang = inputs['metadata']['lang']
            items = mask.sum().item()
            pred_heads, pred_deprel = self.predict(inputs)
            gold_heads, gold_deprel = inputs['heads'], inputs['deprels']
            attached = pred_heads.eq(gold_heads).float() * mask
            labelled = pred_deprel.eq(gold_deprel).float() * mask
            uas = attached.sum().item()
            las = (attached * labelled).sum().item()
            uas_meter[lang].update(uas * 100, items)
            las_meter[lang].update(las * 100, items)
        self.post()

        metrics = {}
        for lang, meter in uas_meter.items():
            uas = meter.evaluate()
            metrics['UAS_%s' % lang] = uas
        for lang, meter in las_meter.items():
            las = meter.evaluate()
            metrics['LAS_%s' % lang] = las
        metrics['UAS_AVG'] = np.mean([m.evaluate() for m in uas_meter.values()])
        metrics['LAS_AVG'] = np.mean([m.evaluate() for m in las_meter.values()])

        self.log(metrics)

        return metrics

    def run_official(self, iterator, gold_files):
        """Run official (slower) evaluation using standard eval.pl script."""
        outputs = collections.defaultdict(list)
        self.pre(iterator)
        for inputs in tqdm.tqdm(iterator, ncols=130):
            inputs = inputs.to(self.device)
            pred_heads, pred_deprel = self.predict(inputs)
            dicos = self.M.dicos
            for i in range(len(inputs)):
                lang = inputs['metadata']['lang']
                N = len(inputs['metadata']['words'][i])
                tid = inputs['metadata']['tids'][i]
                words = inputs['metadata']['words'][i]
                pos = inputs['metadata']['pos'][i]
                heads = pred_heads[i, :N].tolist()
                deprels = [dicos['deprel'][r]
                           for r in pred_deprel[i, :N].tolist()]
                outputs[lang].append((tid, words, pos, heads, deprels))
        self.post()

        metrics = {}
        args = []
        for lang, filename in gold_files.items():
            args.append((lang, outputs[lang], filename))

        for lang, las, uas in self.map(evaluate_official, args):
            metrics['LAS_%s' % lang] = las
            metrics['UAS_%s' % lang] = uas

        metrics['LAS_AVG'] = np.mean([metrics['LAS_%s' % lang]
                                      for lang in gold_files.keys()])
        metrics['UAS_AVG'] = np.mean([metrics['UAS_%s' % lang]
                                      for lang in gold_files.keys()])

        self.log(metrics)
        return metrics


class TaFEvaluator(Evaluator):
    """Evaluator class for Typology as Feature parsers."""

    def pre(self, iterator, **kwargs):
        """Initialize on-the-fly corpus statistics cache."""
        if self.M.args.use_neural_features:
            self.M.neural_typology.set_cache(iterator)

    def post(self):
        """Clear cached corpus statistics."""
        if self.M.args.use_neural_features:
            self.M.neural_typology.clear_cache()

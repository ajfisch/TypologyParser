"""Experiment utilities."""

import argparse
import datetime
import filecmp
import logging
import os
import pathlib
import random
import shutil
import time

import conllu
import dependency_decoding
import numpy as np
import torch

from clparse.data.iso import convert_lang

INF = 1e8


def put(x, device, **kwargs):
    """Recursively transfer object x to torch device (CPU or GPU #)."""
    if isinstance(x, dict):
        x = {k: put(item, device, **kwargs) for k, item in x.items()}
    elif isinstance(x, list) or isinstance(x, tuple):
        x = [put(item, device, **kwargs) for item in x]
    elif torch.is_tensor(x):
        x = x.to(device, **kwargs)
    return x


def retriable(fn, *args, tries=10, time_to_sleep=1, **kwargs):
    """Retry fn(*args, **kwargs) repeatedly."""
    i = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            i += 1
            if i < tries:
                time.sleep(time_to_sleep)
            else:
                raise e


def export_resource(src, dest):
    """Copy source to dest if it does not exists or they are different."""
    if not os.path.exists(dest) or not filecmp.cmp(src, dest):
        shutil.copyfile(src, dest)


def mst_decode(scores):
    """Decode arc-factored parse using maximum spanning tree."""
    device = scores.device
    scores = scores.cpu().double().numpy()
    heads, _ = dependency_decoding.chu_liu_edmonds(scores)
    heads[0] = 0  # Set root to itself
    return torch.LongTensor(heads).to(device)


def init_dir(dirname):
    """Ensure directory exists."""
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    """Set random seeds for used packages."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_name_from_dir(dirname, map_to_iso=False):
    """Derive language from directory name: /path/to/lang."""
    filename = dirname.strip('/')
    lang = os.path.basename(filename)
    if map_to_iso:
        lang = convert_lang(lang)
    return lang


def get_name_from_file(filename, map_to_iso=False):
    """Derive language from filename: /path/to/lang/filename."""
    dirname = os.path.dirname(filename)
    return get_name_from_dir(dirname, map_to_iso)


def init_device(cuda):
    """Choose device based on config and system capabilities."""
    cuda = cuda and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    return device


def bool_flag(s):
    """Parse boolean arguments from the command line."""
    if s.lower() in ['off', 'false', '0', 'n', 'no', 'f']:
        return False
    if s.lower() in ['on', 'true', '1', 'y', 'yes', 't']:
        return True
    raise argparse.ArgumentTypeError("Invalid value for a boolean flag")


def read_conllu(filename):
    """Iterate over trees in a conllu formatted file."""
    with open(filename) as f:
        for tree in conllu.parse_incr(f):
            tree = [t for t in tree if isinstance(t['id'], int)]
            yield tree


def dump_to_conllu(outputs, filename):
    """Dump outputs to a conllu formatted file."""
    N = len(outputs)
    ordered = {}
    for i in range(N):
        tid, words, pos, heads, deprels = outputs[i]
        idx = range(1, len(words) + 1)
        output = []
        for (j, w, p, h, dr) in zip(idx, words, pos, heads, deprels):
            w = w if w != '_' else 'x'  # Hack for eval.pl script.
            output.append('\t'.join(
                [str(j), w, '_', p, p, '_', str(h), dr, '_', '_']))
        output = '\n'.join(output)
        ordered[tid] = output

    with open(filename, 'w') as f:
        for i in sorted(ordered.keys()):
            ex = ordered[i]
            f.write(ex + '\n\n')


def clean_conllu_file(input_file, output_file, tids=None):
    """Read and re-dump a conllu file to a normalized conllu file."""
    outputs = []
    for i, ex in enumerate(read_conllu(input_file)):
        if tids and i not in tids:
            continue
        words = [t['form'] for t in ex]
        pos = [t['upostag'].lower() for t in ex]
        heads = [t['head'] for t in ex]
        deprels = [t['deprel'].lower() for t in ex]
        idx = range(1, len(words) + 1)
        output = []
        for (j, w, p, h, dr) in zip(idx, words, pos, heads, deprels):
            w = w if w != '_' else 'x'  # Hack for eval.pl script.
            output.append('\t'.join(
                [str(j), w, '_', p, p, '_', str(h), dr, '_', '_']))
        output = '\n'.join(output)
        outputs.append(output)

    with open(output_file, 'w') as f:
        for ex in outputs:
            f.write(ex + '\n\n')


class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = '%s - %s - %s' % (
            record.levelname,
            time.strftime('%x %X'),
            datetime.timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return '%s - %s' % (prefix, message)


def init_logger(filepath=None, verbose=2):
    """Create a logger."""
    # Create log formatter
    log_formatter = LogFormatter()

    # Create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Create stream handler
    if verbose == 1:
        log_level = logging.INFO
    elif verbose == 2:
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # Create file handler if specified (set level to debug)
    if filepath:
        file_handler = logging.FileHandler(filepath, "w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger

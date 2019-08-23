"""Convert UD directory to expected layout."""

import argparse
import glob
import os
import shutil
import tqdm

from clparse.data.iso import convert_lang
from clparse.utils import init_dir

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, default=None)
parser.add_argument('--output-dir', type=str, default='out')


def main(args):
    for ud_lang in tqdm.tqdm(os.listdir(args.input_dir), ncols=130):
        lang = convert_lang(ud_lang)
        ud_dir = os.path.join(args.input_dir, ud_lang)
        output_dir = os.path.join(args.output_dir, lang)
        init_dir(os.path.join(output_dir))

        with open(os.path.join(output_dir, 'train.conllu'), 'w') as dest:
            for f in glob.glob(os.path.join(ud_dir, '*train*.conllu')):
                with open(f) as src:
                    shutil.copyfileobj(src, dest)
        with open(os.path.join(output_dir, 'valid.conllu'), 'w') as dest:
            for f in glob.glob(os.path.join(ud_dir, '*dev.conllu')):
                with open(f) as src:
                    shutil.copyfileobj(src, dest)
        with open(os.path.join(output_dir, 'test.conllu'), 'w') as dest:
            for f in glob.glob(os.path.join(ud_dir, '*test.conllu')):
                with open(f) as src:
                    shutil.copyfileobj(src, dest)


if __name__ == '__main__':
    main(parser.parse_args())

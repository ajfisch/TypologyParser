# Typology Augmented Cross-Lingual Parsing

Code for Working Hard or Hardly Working: Challenges of Integrating Typology into Neural Dependency Parsers.


## About

TODO

## Installation

This repository requires Linux/OSX and Python 3 (tested on 3.7). It also requires installing [PyTorch](www.pytorch.org) version 1.0. Its other dependencies are listed in `requirements.txt`. GPU/CUDA support is strongly recommended for speed, but is not necessary.

Run the following commands to clone the repository and install the package:

```
git clone https://github.com/ajfisch/TypologyParser.git
cd TypologyParser; pip install Cython; pip install -r requirements.txt; python setup.py develop
```

## Data Download

The experiments in the paper were conducted using the [Universal Dependencies v1.2](https://universaldependencies.org/) and the [Galactic Dependencies v1.0](https://github.com/gdtreebank/gdtreebank) datasets.

To download Universal Dependencies and convert it to the expected format, run:

```
./download_ud.sh
```

This will download the 37 UD treebanks and put them in `data/ud_v1.2`, organized in sub-directories by ISO code:

```
ud_v1.2/
├── ar
|  ├── test.conllu
|  ├── train.conllu
|  └── valid.conllu
├── ...
├── ta
    ├── test.conllu
    ├── train.conllu
    └── valid.conllu
```

To download Galactic Dependencies and convert it to the expected format, follow the instructions at https://github.com/gdtreebank/gdtreebank to download and extract the treebanks (only the substrates for [training languages](experiments/wang_eisner_udv1.py) are needed).

Then run:

```
python preprocess/convert_gd.py \
  --langs cs es fr hi de it la_itt no ar pt en nl da fi got grc et la_proiel grc_proiel bg \
  --input-dir <path/to/gd> \
  --output-dir data/gd_v1.0
```

This will create a directory named `data/gd_v1.0` with 8,820 GD languages:

```
gd_v1.0/
├── gd_ar
|  ├── train.conllu
|  └── train.hdf5
├── ...
├── gd_pt~pt@V
    ├── train.conllu
    └── train.hdf5
```
The `train.hdf5` files contain preprocessed versions (tensorized) of the treebanks, and allow for fairly quick disk-based random access during training.

The typology features used in the experiments are provided in the `typologies` directory. The original (discrete, human-readable) WALS features can be found in `typologies/wals_mapping_udv1.pkl.gz`.

## Experiments
### Training Parsers with Typology

The `experiments/parsing_experiment.py` [script](experiments/parsing_experiment.py) is the main script for launching experiments.

Run it as:

```
python experiments/parsing_experiment.py
```

With arguments:

```
--output-dir Parent directory to write job logs and models to.
--main Script to run (e.g. train_baseline.py)
--udv1-dir Path to UD version 1.2 directory (data/ud_v1.2).
--split {full,cv} Full=test set, CV=5-fold cross validation on training.
--seeds Number of runs in parallel to execute with different random seeds.
--args ... Extra arguments passed to the main script.
```

This will print out a command, or a list of commands, to execute. You can then execute it via `eval $(...)` or by piping into the `experiments/launch.py` [script](experiments/launch.py) which reads in the commands and schedules them across your available GPUs (specified with `--devices`).

#### Running with Galactic Dependencies

If testing with GD, please add the flag `--gd-patterns '"data/gd_v1.0/*"'` to add all of the GD languages into training. To use a different subset, specify a different list of glob patterns (e.g. `--gd-patterns data/gd_v1.0/gd_it~*~en@V data/gd_v1.0/gd_de~pt@N*`).

By default, the training routine will sample a batch from UD languages with probability `beta` and from GD languages with probability `1 - beta`. Set the flag `--beta` to change this.

Best practices:
- UD data is loaded into memory whereas GD data is read from disk. For best speed performance, set the number of asyncronous data loading processes to 10-20 via `--num-workers 20`.
- Increase the steps between evaluations by setting `--eval-interval` to 10-20K.
- Get better stability for the longer training times with Adam by enabling `amsgrad` and increasing `eps` by setting `--optimizer adam,amsgrad=true,eps=1e-3`.

### TODO:
- [ ] Typology as Quantization
- [ ] Corpus-Specific Typology
- [ ] Predicting Typology from Embeddings
- [ ] Typology vs. Parser Transferability

## Citation

Please cite the EMNLP-IJCNLP 2019 paper if you use this in your work:

```
@inproceedings{fisch2019typology,
  title={Working Hard or Hardly Working: Challenges of Integrating Typology into Neural Dependency Parsers},
  author={Fisch, Adam and Guo, Jiang and Barzilay, Regina},
  booktitle={Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2019}
}
```

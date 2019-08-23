"""Compute features from Liu 2010."""

from clparse.utils import read_conllu

# top-20 shared dependency relations across languages: for udv1
DEPRELS = ['cc',
           'conj',
           'case',
           'nsubj',
           'nmod',
           'dobj',
           'mark',
           'advcl',
           'amod',
           'advmod',
           'neg',
           'nummod',
           'xcomp',
           'ccomp',
           'cop',
           'acl',
           'aux',
           'punct',
           'det',
           'appos',
           'iobj',
           'dep',
           'csubj',
           'parataxis',
           'mwe',
           'name',
           'nsubjpass',
           'compound',
           'auxpass',
           'csubjpass',
           'vocative',
           'discourse']


def compute_features(filename):
    """ Compute Liu 2010 features from conllu file."""
    deprel_stats = {}
    for tree in read_conllu(filename):
        for tok in tree:
            deprel = tok['deprel']
            if deprel not in deprel_stats:
                deprel_stats[deprel] = {'r': 0, 'l': 0}
            if tok['head'] > tok['id']:
                deprel_stats[deprel]['r'] += 1
            else:
                deprel_stats[deprel]['l'] += 1

    # features = {} # for named vector
    features = []
    for deprel in DEPRELS:
        if deprel in deprel_stats.keys():
            fval = (float(deprel_stats[deprel]['r']) /
                    (deprel_stats[deprel]['r'] + deprel_stats[deprel]['l']))
            features.append(fval)
        else:
            features.append(.5)

    return features

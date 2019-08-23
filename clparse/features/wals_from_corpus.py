"""Compute corpus-consistent WALS features."""

import collections
from clparse.utils import read_conllu

WALS_CONDITIONS = {
    '82A': {'rel': ['nsubj', 'csubj'], 'pos': ['VERB/NOUN', 'VERB/PRON']},
    '83A': {'rel': ['dobj', 'iobj'], 'pos': ['VERB/NOUN', 'VERB/PRON']},
    '85A': {'pos': ['NOUN/ADP', 'PRON/ADP']},
    '86A': {'pos': ['NOUN/NOUN']},
    '87A': {'pos': ['NOUN/ADJ']},
    '88A': {'rel': ['det'], 'pos': ['*/DET']}
}

STATS_TO_WALS = {
    '82A': {'r': '2 VS', 'l': '1 SV', 'o': '3 No dominant order'},
    '83A': {'r': '2 VO', 'l': '1 OV', 'o': '3 No dominant order'},
    '85A': {'r': '1 Postpositions', 'l': '2 Prepositions'},
    '86A': {'r': '2 Noun-Genitive', 'l': '1 Genitive-Noun',
            'o': '3 No dominant order'},
    '87A': {'r': '2 Noun-Adjective', 'l': '1 Adjective-Noun',
            'o': '3 No dominant order'},
    '88A': {'r': '2 Noun-Demonstrative', 'l': '1 Demonstrative-Noun',
            'o': '3 No dominant order'}
}


def flatten(wals_feat):
    """Flatten the discrete WALS features to one-hot representation.

    Args:
      wals_feat: A dictionary of WALS features to dominant order.
    """
    onehot_per_feat = {}
    for w, feat in STATS_TO_WALS.items():
        onehot_per_feat[w] = {}
        for i, (d, fval) in enumerate(feat.items()):
            vec = [0.0] * len(feat)
            vec[i] = 1.0
            onehot_per_feat[w][fval] = vec

    # Generate one-hot representation.
    onehot_fsets = [onehot_per_feat[w][wals_feat[w]]
                    for w in sorted(STATS_TO_WALS.keys())]
    wals_feat_onehot = [fval for fset in onehot_fsets for fval in fset]
    return wals_feat_onehot


def unflatten(wals_feat_onehot):
    """Unflatten one-hot WALS representation to discrete features.

    Args:
      wals_feat_onehot: A binary vector representing the active WALS features.
    """
    wals_feats = {}
    start = 0
    for i, w in enumerate(sorted(STATS_TO_WALS.keys())):
        end = start + len(STATS_TO_WALS[w])
        onehot_subvec = wals_feat_onehot[start:end]
        index = onehot_subvec.index(1.0)
        wals_feats[w] = list(STATS_TO_WALS[w].items())[index][1]
        start = end
    return wals_feats


def compute_features(filename, threshold=3.0, one_hot=True):
    """Compute WALS features from conllu file.

    Args:
      filename: Path to conllu file.
      threshold: A scalar threshold for determining dominancy.
        - If a feature has a "No Dominant Order" option, the frequency f(d) of
          order d must be >= threshold x f(d') to be considered the dominant
          order. Otherwise, the feature is marked with "No Dominant Order."
        - If a feature does not have a "No Dominant Order" option, order d is
          considered dominant simply if f(d) > f(d').
      one_hot: Return WALS features as a one-hot representation.

    Returns:
      WALS feature vector.
    """
    # Collect WALS statistics (directionalities).
    wals_stats = collections.defaultdict(dict)
    for w, _ in WALS_CONDITIONS.items():
        wals_stats[w]['r'] = 0
        wals_stats[w]['l'] = 0

    # Iterate over parsed trees.
    for tree in read_conllu(filename):
        for tok in tree:
            for w, cond in WALS_CONDITIONS.items():
                # Check that the dependency type matches the condition's
                # relevant dependency type, if any. Skip if not relevant.
                deprel = tok['deprel']
                if 'rel' in cond:
                    if deprel not in cond['rel']:
                        continue

                # Skip tokens whose head is ROOT.
                if tok['head'] == 0:
                    continue

                # Check if the modifier and head tags match the pattern.
                pos_m = tok['upostag']
                pos_h = tree[tok['head'] - 1]['upostag']
                if "%s/%s" % (pos_h, pos_m) in cond['pos'] or \
                   (pos_m == "DET" and "*/DET" in cond['pos']):
                    # If the head is to the left, this is right-leaning: h-->m.
                    if tok['head'] < tok['id']:
                        wals_stats[w]['r'] += 1
                    # If the head is to the right, it is left-leaning: m<--h.
                    elif tok['head'] > tok['id']:
                        wals_stats[w]['l'] += 1
                    else:
                        raise ValueError('Should not have head as self!')

    # Determine WALS values based on these statistics.
    # Use Laplace smoothing.
    wals_features = collections.defaultdict(dict)
    for w, stats in wals_stats.items():
        if float(stats['r'] + 1) / (stats['l'] + 1) >= threshold:
            direction = 'r'
        elif float(stats['l'] + 1) / (stats['r'] + 1) >= threshold:
            direction = 'l'
        else:
            direction = 'o'

        # Pick the weakly dominant order if "No Dominant Order" is not usable.
        if direction == 'o' and ('o' not in STATS_TO_WALS[w]):
            direction = 'r' if stats['r'] > stats['l'] else 'l'

        wals_features[w] = STATS_TO_WALS[w][direction]

    # Convert to one-hot if specified.
    if one_hot:
        wals_features = flatten(wals_features)

    return wals_features

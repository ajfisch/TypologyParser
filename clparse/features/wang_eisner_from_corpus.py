"""Compute features from Wang and Eisner, 2018a."""

import collections
import statistics
from clparse.utils import read_conllu

CONTEXT_WINDOW = [1, 3, 8, 100, -1, -3, -8, -100]

POSSIBLE_TAGS = ['PUNCT', 'PROPN', 'ADJ', 'NOUN', 'VERB', 'DET', 'ADP',
                 'AUX', 'PRON', 'PART', 'SCONJ', 'NUM', 'ADV', 'CCONJ',
                 'X', 'INTJ', 'SYM']


def compute_features(filename):
    """Compute Wang and Eisner 2018a features from conllu file."""
    g_uni = {}
    g_bi = {}
    for w in CONTEXT_WINDOW:
        g_uni[w] = collections.defaultdict(list)  # {w: \pi_t^w}
        g_bi[w] = collections.defaultdict(list)   # {w: \pi_{t,s}^w}

    # Gather statistics.
    for ex in read_conllu(filename):
        # Get part-of-speech tags.
        ex_pos = [t['upostag'] for t in ex]

        # Only sentences of <= 40 tokens are considered.
        if len(ex_pos) > 40:
            continue

        for w in CONTEXT_WINDOW:
            for j, s in enumerate(ex_pos):
                # Right context.
                if w > 0:
                    context = ex_pos[j + 1:j + w + 1]

                # Left context.
                else:
                    context = ex_pos[max(j + w, 0):j]

                # Count the frequencies of the tags in the context.
                tag_counts = collections.Counter(context)

                # For each tag type, store its context frequency
                # per window type (unigram + bigram).
                for t in POSSIBLE_TAGS:
                    cnt = tag_counts[t] if t in tag_counts else 0
                    tag_frac = float(cnt) / abs(w)

                    # Map window --> token --> frequencies.
                    g_uni[w][t].append(tag_frac)

                    # Map window --> bigram --> frequencies.
                    g_bi[w][(s, t)].append(tag_frac)

    # Convert to final mean fractions and ratios.
    pi_uni = {}
    pi_bi = {}
    features_uni = []
    features_bi = []

    for w in CONTEXT_WINDOW:
        pi_uni[w] = {}
        pi_bi[w] = {}

        for t in POSSIBLE_TAGS:
            pi_uni[w][t] = statistics.mean(g_uni[w][t])

            # Unigram features are only taken for POSITIVE w.
            if w > 0:
                features_uni.append(pi_uni[w][t])

            # Bigram features are used for bigram:unigram ratios only.
            for s in POSSIBLE_TAGS:
                if (s, t) not in g_bi[w]:
                    pi_bi[w][(s, t)] = 0.0
                else:
                    pi_bi[w][(s, t)] = statistics.mean(g_bi[w][(s, t)])

                if pi_uni[w][t] == 0.0:
                    # If token doesn't exist, default to 1.
                    assert pi_bi[w][(s, t)] == 0.0, ('Unigram frequency is 0'
                                                     ' but bigram is not!')
                    features_bi.append(1.0)
                else:
                    # Bound ratio by 1.
                    features_bi.append(min(pi_bi[w][(s, t)] / pi_uni[w][t], 1))

    # Concat features.
    full_features = features_uni + features_bi

    return full_features

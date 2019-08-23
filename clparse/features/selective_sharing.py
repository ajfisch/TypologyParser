r"""Given a sequence of POS tags and a set of typological features,
compute the selectively shared feature vectors for each word pair.

WALS feature descriptions:
  - 81A: Order of Subject, Object and Verb
  - 82A: Order of Subject and Verb
  - 83A: Order of Object and Verb
  - 85A: Order of Adposition and Noun Phrase
  - 86A: Order of Genitive and Noun
  - 87A: Order of Adjective and Noun
  - 88A: Order of Demonstrative and Noun
  - 89A: Order of Numeral and Noun

Feature templates (Täckström et al., 2013, Zhang and Barzilay, 2015):
  - unlabeled attachment
     0. dir \otimes 81A \otimes \delta(h.p=VERB \and m.p=NOUN)
     1. dir \otimes 81A \otimes \delta(h.p=VERB \and m.p=PRON)
     2. dir \otimes 85A \otimes \delta(h.p=ADP \and m.p=NOUN)
     3. dir \otimes 85A \otimes \delta(h.p=ADP \and m.p=PRON)
     4. dir \otimes 86A \otimes \delta(h.p=NOUN \and m.p=NOUN)
     5. dir \otimes 87A \otimes \delta(h.p=NOUN \and m.p=ADJ)
  - labeled attachment (replace the 0th and 1th features above, not used yet)
     0. dir \otimes 82A \otimes \delta(h.p=VERB \and m.p=NOUN \and subj \in l)
     1. dir \otimes 82A \otimes \delta(h.p=VERB \and m.p=PRON \and subj \in l)
     2. dir \otimes 83A \otimes \delta(h.p=VERB \and m.p=NOUN \and obj \in l)
     3. dir \otimes 83A \otimes \delta(h.p=VERB \and m.p=PRON \and obj \in l)
"""

import torch


# conditions for firing each WALS type
WALS_COND = {
    '81A': ['VERB/NOUN', 'VERB/PRON'],
    '85A': ['NOUN/ADP', 'PRON/ADP'],
    '86A': ['NOUN/NOUN'],
    '87A': ['NOUN/ADJ']
}

# unlabeled: 81A, 85A, 86A, 87A
WALS_TYPES = ['81A', '85A', '86A', '87A']


def selective_sharing_features(lang, seq, lang2wals, feature_space):
    """Compute selective sharing features.
    Args:
      lang: language id
      seq: a sequence of POS tags
      lang2wals: a dict storing language-specific WALS features (pre-generated)
      feature_space: selectively-shared WALS feature space (pre-generated)

    Returns:
      ss_features: <float32> [seq_len, seq_len, num_features]
        - e.g. ss_features[i, j] is the feature vector for
          arc j->i (j: head, i: modifier)

    Example usage:

    # load feature space and lang2wals (udv1_wals.pkl, udv2_wals.pkl)
    sent = "ROOT NOUN NOUN VERB PRON ADV ADJ NOUN PUNCT".split()
    ss = selective_sharing_features("en", sent, lang2wals, feature_space)
    """
    ss_features = torch.zeros((len(seq), len(seq), len(feature_space)))

    for i, p_m in enumerate(seq):
        for j, p_h in enumerate(seq):
            if i == j:
                continue
            direction = 'R' if j < i else 'L'
            cond = p_h + '/' + p_m
            ss_features[i][j] = torch.tensor([.0] * len(feature_space))

            for w in WALS_TYPES:
                if cond in WALS_COND[w]:
                    feature_val = '#'.join(
                        [w, direction, lang2wals[lang][w], cond])
                    if feature_val in feature_space:
                        feature_index = feature_space[feature_val]
                        ss_features[i][j][feature_index] = 1.

    return ss_features.numpy()

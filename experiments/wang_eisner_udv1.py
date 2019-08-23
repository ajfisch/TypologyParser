"""Data splits for UD v1.2 from Wang and Eisner 2018a.

In total there are 37 treebanks. Two ("la" and "fi_ftb") are removed per
Wang and Eisner 2018a, as the same language appears in the training splits.

This leaves 35 total treebanks.

Note: the number of treebanks is not the same as the number of unique languages.
There are 18 unique training languages and 15 unique testing languages, giving
33 total unique languages.
"""

TEST = ['hr', 'ga', 'he', 'hu', 'fa',
        'ta', 'cu', 'el', 'ro', 'sl',
        'ja_ktc', 'sv', 'id', 'eu', 'pl']

TRAIN = ['cs', 'es', 'fr', 'hi', 'de',
         'it', 'la_itt', 'no', 'ar', 'pt',
         'en', 'nl', 'da', 'fi', 'got',
         'grc', 'et', 'la_proiel', 'grc_proiel', 'bg']

FOLDS = [['ar', 'da', 'no', 'grc_proiel'],
         ['cs', 'et', 'pt', 'grc'],
         ['de', 'got', 'it', 'la_proiel'],
         ['bg', 'fi', 'fr', 'la_itt'],
         ['nl', 'en', 'hi', 'es']]

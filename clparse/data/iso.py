"""Language ISO code mapping for UD v1.2."""

ISO_CODE_TO_NAME = {
    'ar': 'Arabic',
    'bg': 'Bulgarian',
    'cs': 'Czech',
    'cu': 'Old_Church_Slavonic',
    'da': 'Danish',
    'de': 'German',
    'el': 'Greek',
    'en': 'English',
    'es': 'Spanish',
    'et': 'Estonian',
    'eu': 'Basque',
    'fa': 'Persian',
    'fi': 'Finnish',
    'fi_ftb': 'Finnish-FTB',
    'fr': 'French',
    'ga': 'Irish',
    'got': 'Gothic',
    'grc': 'Ancient_Greek',
    'grc_proiel': 'Ancient_Greek-PROIEL',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hr': 'Croatian',
    'hu': 'Hungarian',
    'id': 'Indonesian',
    'it': 'Italian',
    'ja_ktc': 'Japanese-KTC',
    'la': 'Latin',
    'la_itt': 'Latin-ITT',
    'la_proiel': 'Latin-PROIEL',
    'nl': 'Dutch',
    'no': 'Norwegian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'sl': 'Slovenian',
    'sv': 'Swedish',
    'ta': 'Tamil',
}


def convert_lang(lang):
    """Convert language to ISO code."""
    if lang.startswith('UD_'):
        lang = lang[3:]
    if lang.startswith('GD_'):
        lang = lang[3:]
    return {v: k for k, v in ISO_CODE_TO_NAME.items()}.get(lang)


def convert_code(code):
    """Convert ISO code to language."""
    return ISO_CODE_TO_NAME.get(code)

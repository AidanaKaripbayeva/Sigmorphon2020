"""Constructs a hard-coded collection of languages with their hierarchies and alphabets."""
from . import LanguageCollection
from ..alphabets.standard import *


# Define language_families
language_collection = LanguageCollection()
language_collection.add_language_family('extras')
language_collection.add_language('empty', 'extras', empty)
language_collection.add_language('stop_start', 'extras', stop_start)
language_collection.add_language('symbols', 'extras', symbols)
language_collection.add_language('tones', 'extras', tones)
language_collection.add_language('other', 'extras', other)
language_collection.add_language_family('pie')
language_collection.add_language('roman', 'pie', roman)
language_collection.add_language('latin', 'pie', latin_diacritic)
language_collection.add_language_family('slavic')
language_collection.add_language('cyrillic', 'slavic', cyrillic)
language_collection.add_language_family('turkic')
language_collection.add_language('turkish', 'turkic', turkish_latin)
language_collection.add_language('cyrillic_kazak', 'turkic', cyrillic_kazak)
language_collection.add_language('common_turkic', 'turkic', common_turkic_alphabet)
language_collection.add_language('turkic_ipa', 'turkic', common_turkic_ipa)
language_collection.add_language('turkic_cyrillic', 'turkic', common_turkic_cyrillic)
language_collection.get_master_alphabet()

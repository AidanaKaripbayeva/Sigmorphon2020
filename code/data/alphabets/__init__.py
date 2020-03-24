
import functools
import operator

from .alphabet import Alphabet, AlphabetCounts
from . import standard

def get_master_alphabet(alphabets=None, reindex=False):
    
    if alphabets is None:
        #default to the prior behaviour to not break existing example code.
        alphabets = [standard.kitchen_sink_alphabet]
    
    unified_alphabet = functools.reduce(lambda a, b: a + b, alphabets)
    if reindex:
        unified_alphabet = unified_alphabet.sort()
    return unified_alphabet

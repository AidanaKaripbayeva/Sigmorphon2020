import sys
from ast import literal_eval
#sys.path.append('data/uniread')
#from data.uniread import read_unimorph_tsv
# import data.languages
from random import choices
#sys.path.append('src')

def frequency_choice(lang_alphabet):
    # lang_collection = data.languages.LanguageCollection.get_default(alphabet_choice="individual")
    # lang_alphabet = lang_collection[lang_family][lang].alphabet
    count = lang_alphabet.counts
    total = sum(count.values())
    probabilities = {key: count[key]/total for key in count.keys()}
    prob_key = list(probabilities.keys())
    prob_value = list(probabilities.values())
    #print(sum(probabilities.values()))
    if round(sum(probabilities.values()),5) != 1.0:
        print(round(sum(probabilities.values()),5))
        assert sum(probabilities.values()) == 1.0 # they should all add up to one
    
    letter = str(choices(prob_key, prob_value)[0])
    return letter 


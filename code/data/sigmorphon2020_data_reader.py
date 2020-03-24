from .alphabets import Alphabet
from .languages import LanguageCollection
from .uniread import read_unimorph_tsv
from itertools import chain
import os
import re

from collections import defaultdict


def compile_language_collection_from_sigmorphon2020(root_dir,process_alphabets = True):
    language_collection = LanguageCollection()
    for language_family_name in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, language_family_name)) \
                and re.fullmatch(r'[a-zA-Z-]*', language_family_name):
            language_collection.add_language_family(language_family_name)
            for language_file in os.listdir(os.path.join(root_dir, language_family_name)):
                if re.fullmatch(r'[a-zA-Z-]*\.trn', language_file):
                    language_name = language_file[:-4]
                    
                    if process_alphabets:
                        
                        train_data = read_unimorph_tsv(
                            os.path.join(root_dir, language_family_name, language_name + '.trn'))
                        test_data = read_unimorph_tsv(
                            os.path.join(root_dir, language_family_name, language_name + '.dev'))
                        
                        letters = defaultdict(int)
                        
                        for index, row in chain(train_data.iterrows(), test_data.iterrows()):
                            word = row['lemma'] + row['form']
                            for letter in word:
                                letters[letter] += 1
                        alphabet = Alphabet("".join(sorted(letters.keys())), letters)
                        
                    else:
                        alphabet = Alphabet()
                    language_collection.add_language(language_name, language_family_name, alphabet)
    _ = language_collection.get_master_alphabet()
    return language_collection
    

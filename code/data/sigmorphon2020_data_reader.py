from .alphabets import Alphabet
from .languages import LanguageCollection
from .uniread import read_unimorph_tsv
import consts
from itertools import chain
from .dataset import pandas_to_dataset, UnimorphDataLoader
import os
import re


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
                        letters = set()
                        for index, row in chain(train_data.iterrows(), test_data.iterrows()):
                            word = row['lemma'] + row['form']
                            for letter in word:
                                letters.add(letter)
                        alphabet = Alphabet("".join(sorted(letters)))
                    else:
                        alphabet = Alphabet()
                    language_collection.add_language(language_name, language_family_name, alphabet)
    _ = language_collection.get_master_alphabet()
    return language_collection



def create_data_loader_from_sigmorphon2020(config, is_train=True):
    root_dir = config[consts.SIGMORPHON2020_ROOT]
    panda_data_list = []
    for language_family_name in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, language_family_name)) \
                and re.fullmatch(r'[a-zA-Z-]*', language_family_name):
            if len(config[consts.LANGUAGE_FAMILIES]) == 0 or language_family_name in config[consts.LANGUAGE_FAMILIES]:
                for language_file in os.listdir(os.path.join(root_dir, language_family_name)):
                    if is_train and re.fullmatch(r'[a-zA-Z-]*\.trn', language_file) or \
                            not is_train and re.fullmatch(r'[a-zA-Z-]*\.dev', language_file):
                        language_name = language_file[:-4]
                        if len(config[consts.LANGUAGES]) == 0 or language_name in config[consts.LANGUAGES]:
                            panda_data = read_unimorph_tsv(os.path.join(root_dir, language_family_name, language_file))
                            panda_data_list.append(panda_data)
    # TODO Commandline arguments to choose tag_converter, alphabet_converter_in and alphabet_converter_out.
    dataset = pandas_to_dataset(panda_data_list)
    data_loader = UnimorphDataLoader(dataset=dataset, batch_size=config[consts.BATCH_SIZE])
    return data_loader

from .alphabets import Alphabet
from .dataset import pandas_to_dataset, UnimorphDataLoader, UnimorphTagOneHotConverter
from .languages import LanguageCollection
from .uniread import read_unimorph_tsv
import consts
from itertools import chain
import os
import re

from collections import defaultdict

def compile_language_collection_from_sigmorphon2020(root_dir,process_alphabets = True):
    """
    Compiles a collection of languages from the SIGMORPHON2020 dataset.

    :param root_dir: Root directory of the SIGMORPHON2020 dataset.
    :return: A collection of languages, their families and alphabets, as a LanguageCollection object.
    """
    language_collection = LanguageCollection()
    # For each language family in the data set directory ...
    for language_family_name in os.listdir(root_dir):
        # Check if it is a family name and not e.g. the `.git` folder.
        if os.path.isdir(os.path.join(root_dir, language_family_name)) \
                and re.fullmatch(r'[a-zA-Z-]*', language_family_name):
            # Instantiate a new language family.
            language_collection.add_language_family(language_family_name)

            # For each language in the directory of this family ...
            for language_file in os.listdir(os.path.join(root_dir, language_family_name)):
                # Check if it is a language name and not e.g. the `.git` folder.
                if re.fullmatch(r'[a-zA-Z-]*\.trn', language_file):
                    # Extract the name of the language
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

    return language_collection


def create_data_loader_from_sigmorphon2020(config, is_train=True, tag_converter=None):
    """
    A factory method for data loader objects built for SIGMORPHON2020.

    :param config: The configurations of the experiment as a dictionary object. This, in particular, holds the list
    of languages and families of interest in this experiment.

    :param is_train: Whether this data loader is for the training data or the test data. Defaults to training data.
    :param tag_converter: The object used to parse tags. Defaults to a new UnimorphTagOneHotConverter object.
    :return: A data loader object.
    """
    # Locate the root directory for SIGMORPHON2020.
    root_dir = config[consts.SIGMORPHON2020_ROOT]

    panda_data_list = []
    # For each language family ...
    for language_family_name in os.listdir(root_dir):
        # Check if it is a language family name and not e.g. the `.git` folder.
        if os.path.isdir(os.path.join(root_dir, language_family_name)) \
                and re.fullmatch(r'[a-zA-Z-]*', language_family_name):
            # If the user has chosen to include this family in the dataset.
            if len(config[consts.LANGUAGE_FAMILIES]) == 0 or language_family_name in config[consts.LANGUAGE_FAMILIES]:
                # For each language in this family ...
                for language_file in os.listdir(os.path.join(root_dir, language_family_name)):
                    # Check if it is a language name and not e.g. the `.git` folder.
                    if is_train and re.fullmatch(r'[a-zA-Z-]*\.trn', language_file) or \
                            not is_train and re.fullmatch(r'[a-zA-Z-]*\.dev', language_file):
                        # Extract the name of the language.
                        language_name = language_file[:-4]
                        # If the user has chosen to include this language in the dataset.
                        if len(config[consts.LANGUAGES]) == 0 or language_name in config[consts.LANGUAGES]:
                            # Extract the data as a Pandas `DataFrame`.
                            panda_data = read_unimorph_tsv(os.path.join(root_dir, language_family_name, language_file))
                            # Collect these `DataFrame`s in a list.
                            panda_data_list.append(panda_data)
    # TODO Commandline arguments to choose tag_converter, alphabet_converter_in and alphabet_converter_out.
    if tag_converter is None:
        tag_converter = UnimorphTagOneHotConverter()

    # Instantiate a `DataSet` object from the collected list of Pandas `DataFrame`s.
    dataset = pandas_to_dataset(panda_data_list, tag_converter=tag_converter)
    # Instantiate a `DataLoader` object from the `DataSet` object with the proper batch size.
    data_loader = UnimorphDataLoader(dataset=dataset, batch_size=config[consts.BATCH_SIZE])

    return data_loader, tag_converter.get_output_dimension()

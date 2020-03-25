from . import alphabets  # relative import, only way to get the module to work
from . import uniread
from collections import OrderedDict, namedtuple
import consts
import os
import pandas
import re
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data
import yaml

TRAIN_MODE = 'trn'
DEV_MODE = 'dev'
TEST_MODE = 'tst'


class UnimorphDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, input_alphabet, output_alphabet, family_tensor, language_tensor, tags_tensor, lemma_tensor_list, form_tensor_list, tags_str_list, lemmata_str_list, forms_str_list, language_collection=None):
        super(UnimorphDataset,self).__init__()

        self.language_collection = language_collection
        self.alphabet_input = input_alphabet
        self.alphabet_output = output_alphabet
        
        #TODO: an assertion that makes sure everything is the same length.
        self.families = family_tensor
        self.languages = language_tensor
        self.tags = tags_tensor
        self.lemmata = lemma_tensor_list
        self.forms = form_tensor_list
        self.tags_str = tags_str_list
        self.lemmata_str = lemmata_str_list
        self.forms_str = forms_str_list
    
    def get_dimensionality(self):#Sahand wanted a way to ask the dataset about its dimensions.
        return {consts.NUM_FAMILIES:
                None if self.language_collection is None else len(self.language_collection.language_families),
                consts.NUM_LANGUAGES:
                None if self.language_collection is None else len(self.language_collection),
                consts.INPUT_SYMBOLS: len(self.alphabet_input),
                consts.OUTPUT_SYMBOLS: len(self.alphabet_output),
                consts.TAGS: self.tags[0].shape
                }

    def __getitem__(self, index):
        return (self.families[index], self.languages[index], self.tags[index], self.lemmata[index], self.forms[index],
                self.tags_str[index], self.lemmata_str[index], self.forms_str[index])

    def __add__(self, other):
        
        if self.alphabet_input != other.alphabet_input:
            raise Exception("You can't currently combine UnimorphDataset objects with different input alphabets.")
            #TODO: Add the ability to combine UnimorphDataset objects with different alphabets.
        if self.alphabet_output != other.alphabet_output:
            raise Exception("You can't currently combine UnimorphDataset objects with different output alphabets.")
            #TODO: Add that ability.
        
        if self.language_collection is not None and other.language_collection is not None and self.language_collection != other.language_collection:
            raise Exception("You can only add together datasets with compatible language collections.")
        
        return UnimorphDataset(self.alphabet_input, self.alphabet_output,
                                torch.cat([self.families, other.families]),
                                torch.cat([self.languages, other.languages]),
                                torch.cat([self.self.tags,other.tags]),
                                self.lemmata + other.lemmata,
                                self.forms + other.forms,
                                self.tags_str + other.tags_str,
                                self.lemmata_str + other.lemmata_str,
                                self.forms_str + other.forms_str,
                                language_collection = self.language_collection
                                )
    def __len__(self):
        # TODO: assert that everything is the same length?
        return len(self.forms_str)


from .feature_converters import *
        
def pandas_to_dataset(dataset_or_sets,tag_converter=None,alphabet_converter_in=None,alphabet_converter_out=None):
    if isinstance(dataset_or_sets, pandas.DataFrame):
        dataset_or_sets = [dataset_or_sets]
    assert isinstance(dataset_or_sets, list)
    
    total_dataframe = pandas.concat(dataset_or_sets,ignore_index=True)
    
    if tag_converter is None or tag_converter == "one_hot" or tag_converter == "bit_vector":
        tag_converter = UnimorphTagBitVectorConverter()#default schema
    elif tag_converter == "masked_vectors":
        tag_converter = UnimorphTagMaskedVectorConverter(mask_value=-1)#Marc asked for -1
    else:
        raise NotImplementedError("That tagconverter is not yet available.")
    
    if alphabet_converter_in is None:
        fooalpha = alphabets.get_master_alphabet()
        alphabet_converter_in = lambda x: fooalpha(x.lower())

    if alphabet_converter_out is None:  # TODO: Make this use the in alphabet if it was given.
        baralpha = alphabets.get_master_alphabet()
        baralpha = baralpha + alphabets.Alphabet(str(baralpha).upper())

        alphabet_converter_out = lambda x: baralpha(x)

    family_tensor = torch.LongTensor(total_dataframe["family"].to_numpy())
    lang_tensor = torch.LongTensor(total_dataframe["language"].to_numpy())

    tags_tensor = torch.stack([tag_converter(i) for i in total_dataframe["tags"]])

    lemma_tensor_list = [ torch.LongTensor(alphabet_converter_in(i)) for i in total_dataframe["lemma"] ]
    form_tensor_list = [ torch.LongTensor(alphabet_converter_out(i)) for i in total_dataframe["form"] ]

    return UnimorphDataset(alphabet_converter_in, alphabet_converter_out,
                    family_tensor, lang_tensor,
                    tags_tensor, lemma_tensor_list, form_tensor_list,
                    total_dataframe["tags"].to_list(),
                    total_dataframe["lemma"].to_list(),
                    total_dataframe["form"].to_list()
                    )

from .dataloader import *

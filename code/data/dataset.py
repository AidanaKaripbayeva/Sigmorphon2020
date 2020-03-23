import os
import yaml
import torch
import pandas

import re

from collections import OrderedDict

TRAIN_MODE = 'trn'
DEV_MODE = 'dev'
TEST_MODE = 'tst'
PADDING_TOKEN = 0
START_TOKEN_INT = 1
END_TOKEN_INT = 0
START_TOKEN_CHAR = chr(2) #ASCII START OF TEXT
END_TOKEN_CHAR = chr(3) #ASCII END OF TEXT

from . import alphabets #relative import, only way to get the module to work

class UnimorphDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, family_tensor, language_tensor, tags_tensor, lemma_tensor_list, form_tensor_list, tags_str_list, lemmata_str_list, forms_str_list ):
        super(UnimorphDataset,self).__init__()
        
        #TODO: an assertion that makes sure everything is the same length.
        self.families = family_tensor
        self.languages = language_tensor
        self.tags = tags_tensor
        self.lemmata = lemma_tensor_list
        self.forms =  form_tensor_list
        self.tags_str = tags_str_list
        self.lemmata_str = lemmata_str_list
        self.forms_str = forms_str_list
        
    
    def __getitem__(self, index):
        return (self.families[index], self.languages[index], self.tags[index], self.lemmata[index], self.forms[index],
                                self.tags_str[index], self.lemmata_str[index], self.forms_str[index])
    
    def __add__(self, other):
        return UnimorphDataset(
                                torch.cat([self.families, other.families]),
                                torch.cat([self.languages, other.languages]),
                                torch.cat([self.self.tags,other.tags]),
                                self.lemmata + other.lemmata,
                                self.forms + other.forms,
                                self.tags_str + other.tags_str,
                                self.lemmata_str + other.lemmata_str,
                                self.forms_str + other.forms_str
                                )
    def __len__(self):
        #TODO: assert that everything is the same length?
        return len(self.forms_str)

from .feature_converters import *
        
def pandas_to_dataset(dataset_or_sets,tag_converter=None,alphabet_converter_in=None,alphabet_converter_out=None):
    if isinstance(dataset_or_sets, pandas.DataFrame):
        dataset_or_sets = [dataset_or_sets]
    assert isinstance(dataset_or_sets, list)
    
    total_dataframe = pandas.concat(dataset_or_sets,ignore_index=True)
    
    if tag_converter is None or tag_converter is "masked_vectors":
        tag_converter = UnimorphTagMaskedVectorConverter(mask_value=-1)#Marc asked for -1
    elif tag_converter is "one_hot":
        tag_converter = UnimorphTagOneHotConverter()#default schema
    
    if alphabet_converter_in is None:
        fooalpha = alphabets.get_master_alphabet()
        alphabet_converter_in = lambda x: fooalpha(x.lower())
    
    if alphabet_converter_out is None:#TODO: Make this use the in alphabet if it was given.
        baralpha = alphabets.get_master_alphabet()
        baralpha = baralpha + alphabets.Alphabet(str(baralpha).upper())
        
        alphabet_converter_out = lambda x: baralpha(x)
        
    family_tensor = torch.LongTensor(total_dataframe["family"].to_numpy())
    lang_tensor = torch.LongTensor(total_dataframe["language"].to_numpy())
    
    tags_tensor = torch.stack([tag_converter(i) for i in total_dataframe["tags"]])
    
    lemma_tensor_list = [ torch.LongTensor(alphabet_converter_in(i)) for i in total_dataframe["lemma"] ]
    form_tensor_list = [ torch.LongTensor(alphabet_converter_out(i)) for i in total_dataframe["form"] ]
    
    return UnimorphDataset(family_tensor, lang_tensor,
                    tags_tensor, lemma_tensor_list, form_tensor_list,
                    total_dataframe["tags"].to_list(), total_dataframe["lemma"].to_list(), total_dataframe["form"].to_list()
                    )

import torch.utils.data
import torch.nn.utils.rnn as rnn_utils
class UnimorphDataLoader(torch.utils.data.DataLoader):
    
    @classmethod
    def packed_collate(cls,in_data):
        fams, langs, tags_tens, lem_tens, form_tens, tags_strs, lem_strs, form_strs = list(zip(*in_data))
        fams = [i.repeat(len(j)) for i,j in zip(fams,lem_tens)]
        langs = [i.repeat(len(j)) for i,j in zip(langs,lem_tens)]
        return ( rnn_utils.pack_sequence(fams,enforce_sorted=False),
                rnn_utils.pack_sequence(langs,enforce_sorted=False),
                torch.stack(tags_tens),
                rnn_utils.pack_sequence(lem_tens,enforce_sorted=False),
                rnn_utils.pack_sequence(form_tens,enforce_sorted=False),
                list(tags_strs),
                list(lem_strs),
                list(form_strs) )
    
    @classmethod
    def padded_collate(cls,in_data):
        fams, langs, tags_tens, lem_tens, form_tens, tags_strs, lem_strs, form_strs = list(zip(*in_data))
        fams = [i.repeat(len(j)) for i,j in zip(fams,lem_tens)]
        langs = [i.repeat(len(j)) for i,j in zip(langs,lem_tens)]
        return (rnn_utils.pad_sequence(fams,enforce_sorted=False),
                rnn_utils.pad_sequence(langs,enforce_sorted=False),
                torch.stack(tags_tens),
                rnn_utils.pad_sequence(lem_tens),
                rnn_utils.pad_sequence(form_tens),
                list(tags_strs),
                list(lem_strs),
                list(form_strs) )
    
    
    def __init__(self,*args, **kwargs):
        
        if "collate_fn" in kwargs:
            #TODO: raise a non-fatal warning, is that really what was wanted?
            raise Exception("You should not provide a collate function to UnimorphDataLoader, it has its own.")
        
        #Allow the user to specify a collate function using the "collate_type" argument
        collator_selection = self.packed_collate
        if "collate_type" in kwargs:
            if kwargs["collate_type"] in ["pack","packed"]:
                collator_selection = self.packed_collate
            elif kwargs["collate_type"] in ["pad","padded"]:
                collator_selection = self.padded_collate
            else:
                raise NotImplementedError("The selected collation type ({}) is unavailable.".kwargs["collate_type"])
            del kwargs["collate_type"]
        
        kwargs["collate_fn"] = collator_selection
        super(UnimorphDataLoader,self).__init__(*args,**kwargs)

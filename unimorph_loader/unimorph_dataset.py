import os
import yaml
import torch
import pandas

import re

from collections import OrderedDict

TRAIN_MODE = 'trn'
DEV_MODE = 'dev'
TEST_MODE = 'tst'
PADDING_TOKEN = 99
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

from . import uniread
group_splitter = re.compile(";")
tag_splitter = re.compile("/|\+")
non_detector = re.compile("non{(.*)}")
class UnimorphTagOneHotConverter(object):
    
    @classmethod
    def from_schemafile(blah,filename):
        a_schema = uniread.schema.load_unimorph_schema_from_yaml(filename)
        return UnimorphTagOneHotConverter(a_schema)
    
    def __init__(self,schema):
        self.schema = schema
        self.tag_to_id = OrderedDict()
        self.tag_to_group = dict()

        for i,group in enumerate(self.schema):
            for j,tag in enumerate(self.schema[group]):
                self.tag_to_id[tag] = len(self.tag_to_id)
                self.tag_to_group[tag] = group
    
    def __call__(self, tagstring):
        all_tags = list()
        for taggroup in group_splitter.split(";"):
            for one_tag in tag_splitter.split(taggroup):
                non_match = non_detector.match(one_tag)
                if non_match is None:
                    all_tags.append(one_tag)
                else:
                    for t in schema[tag_to_group[non_match.group(1)]]:
                        all_tags.append(t)
        
        hot_vector = torch.LongTensor(len(self.tag_to_id))
        hot_vector.zero_()
        
        for t in all_tags:
            hot_vector[self.tag_to_id[t]] = 1
        
        return hot_vector
                
        
def pandas_to_dataset(dataset_or_sets,tag_converter=UnimorphTagOneHotConverter.from_schemafile("./tags.yaml"),alphabet_converter_in=None,alphabet_converter_out=None):
    if isinstance(dataset_or_sets, pandas.DataFrame):
        dataset_or_sets = [dataset_or_sets]
    assert isinstance(dataset_or_sets, list)
    
    total_dataframe = pandas.concat(dataset_or_sets,ignore_index=True)
    
    tag_converter = lambda x:torch.LongTensor([0,1,0]) #TODO:
    
    
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
    

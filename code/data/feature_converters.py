import re
from collections import OrderedDict

import torch

from . import uniread

class UnimorphTagBitVectorConverter(object):
    """
    Tool for converting a unimorph tag string into a 0/1 bit vector.
    """
    group_splitter = re.compile(";")
    tag_splitter = re.compile("/|\+")
    non_detector = re.compile("non{(.*)}")
    
    @classmethod
    def from_schemafile(blah,filename):
        a_schema = uniread.schema.load_unimorph_schema_from_yaml(filename)
        return UnimorphTagBitVectorConverter(a_schema)
    
    def __init__(self,schema=None):
        if schema is None:
            schema = uniread.schema.load_default_schema()
        self.schema = schema
        self.tag_to_index = OrderedDict()
        self.tag_to_group = dict()

        for i,group in enumerate(self.schema):
            for j,tag in enumerate(self.schema[group]):
                self.tag_to_index[tag] = len(self.tag_to_index)
                self.tag_to_group[tag] = group
    
    def __call__(self, tagstring):
        all_tags = list()
        for taggroup in self.group_splitter.split(tagstring):
            for one_tag in self.tag_splitter.split(taggroup):
                non_match = self.non_detector.match(one_tag)
                if non_match is None:
                    all_tags.append(one_tag)
                else:
                    for t in self.schema[self.tag_to_group[non_match.group(1)]]:
                        all_tags.append(t)
        
        hot_vector = torch.LongTensor(len(self.tag_to_index))
        hot_vector.zero_()
        
        for t in all_tags:
            hot_vector[self.tag_to_index[t]] = 1

        return hot_vector.detach()
    
    def __len__(self):
        return len(self.tag_to_index)

class UnimorphTagMaskedVectorConverter(object):
    """
    This tag converter outputs an NxM matrix where every ROW represents a
    category of tags from the Unimorph schema. Every row is masked with a user specified
    mask value at the elements outside the category that row represents.
    Inside the category, it is a 0/1 hot vector.
    
    EX: Say there are two categories, the first with 3 possible tags and the second
    with 2 possible tags. If we use a mask_value of -1, then a conversion might look like:
        [[0,1,0,-1,-1],
        [-1,-1,-1,1,1]]
    This means that the second tag in the first category and both in the second category were present.
        
    """
    @classmethod
    def from_schemafile(blah,filename):
        a_schema = uniread.schema.load_unimorph_schema_from_yaml(filename)
        return UnimorphTagMaskedVectorConverter(a_schema)
    
    def __init__(self,schema=None,mask_value=0):
        if schema is None:
            schema = uniread.schema.load_default_schema()
        self.schema = schema
        self.one_hot_converter = UnimorphTagBitVectorConverter(schema)
        self.mask_value = mask_value
        self.mask = torch.CharTensor(len(self.schema),len(self.one_hot_converter)).zero_()
        mask_offset = 0
        for row_id, schema_group in enumerate(self.schema.values()):
            self.mask[row_id, mask_offset:mask_offset+len(schema_group)] = 1
            mask_offset += len(schema_group)
        
    
    def __call__(self, tagstring):
        hot_vector = self.one_hot_converter(tagstring)
        #hot_vector = hot_vector.repeat(len(self.schema),1)
        
        hot_vector = (self.mask_value)*(1-self.mask) + hot_vector*self.mask
        
        #TODO: Finish
        return hot_vector.detach()

from . import alphabets  # relative import, only way to get the module to work
from . import uniread
from collections import OrderedDict, namedtuple

import torch
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data

class UnimorphDataLoader(torch.utils.data.DataLoader):
    
    inputs_type = namedtuple("inputs",("family","language","tags","lemma"))
    outputs_type = namedtuple("outputs",("form","tags_str","lemma_str","form_str"))
    
    #I implemented this so we could pin memory to speed up GPU tranfers,
    #but at least on my laptop, it still doesn't pin memory anyway.
    #Going with the named tuples.
    _ = """
    #We need custom batch objects to facilitate memory pinning. Using named-tuples before didn't work.
    class CustomBatchInput(object):
        def __init__(self, family, language, tags, lemma):
            self.family = family
            self.language = language
            self.tags = tags
            self.lemma = lemma
        def pin_memory(self):
            self.family = self.family.pin_memory()
            self.language = self.language.pin_memory()
            self.tags = self.tags.pin_memory()
            self.lemma = self.lemma.pin_memory()
            return self
        
    class CustomBatchOutput(object):
        def __init__(self, form, tags_str, lemma_str, form_str):
            self.form = form
            self.tags_str = tags_str
            self.lemma_str = lemma_str
            self.form_str = form_str
        def pin_memory(self):
            self.form = self.form.pin_memory()
            #lists of strings, for output to user. Should not even be sent to GPU.
            self.tags_str #= self.tags_str.pin_memory()
            self.lemma_str #= self.lemma_str.pin_memory()
            self.form_str #= self.form_str.pin_memory()
            return self
        
    class UNICustomBatch(object):
        def __init__(self, inputs, outputs):
            self.inputs = inputs
            self.outputs = outputs
        def __getitem__(self,index):
            if index == 0:
                return self.inputs
            elif index == 1:
                return self.outputs
            else:
                raise IndexError
        def pin_memory(self):
            self.inputs.pin_memory()
            self.outputs.pin_memory()
            return self
    """
            
    @classmethod
    def packed_collate(cls,in_data):
        fams, langs, tags_tens, lem_tens, form_tens, tags_strs, lem_strs, form_strs = list(zip(*in_data))
        fams = [i.repeat(len(j)) for i,j in zip(fams,lem_tens)]
        langs = [i.repeat(len(j)) for i,j in zip(langs,lem_tens)]
        return ( cls.inputs_type(
                    rnn_utils.pack_sequence(fams,enforce_sorted=False),
                    rnn_utils.pack_sequence(langs,enforce_sorted=False),
                    torch.stack(tags_tens),
                    rnn_utils.pack_sequence(lem_tens, enforce_sorted=False)
                ),
                cls.outputs_type(
                    rnn_utils.pack_sequence(form_tens, enforce_sorted=False),
                    list(tags_strs),
                    list(lem_strs),
                    list(form_strs)
                    )
                )

    @classmethod
    def padded_collate(cls,in_data):
        fams, langs, tags_tens, lem_tens, form_tens, tags_strs, lem_strs, form_strs = list(zip(*in_data))
        fams = [i.repeat(len(j)) for i,j in zip(fams,lem_tens)]
        langs = [i.repeat(len(j)) for i,j in zip(langs,lem_tens)]
        return ( cls.inputs_type(
                    rnn_utils.pad_sequence(fams,enforce_sorted=False),
                    rnn_utils.pad_sequence(langs,enforce_sorted=False),
                    torch.stack(tags_tens),
                    rnn_utils.pad_sequence(lem_tens)
                ),
                cls.outputs_type(
                    rnn_utils.pad_sequence(form_tens),
                    list(tags_strs),
                    list(lem_strs),
                    list(form_strs)
                    )
                )
    
    
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

import json

from collections import OrderedDict
from abc import ABC

class AlphabetCounts(OrderedDict):
    
    def __init__(self,initialization_param=None):
        if isinstance(initialization_param,str):
            initialization_param = json.loads(initialization_param,object_pairs_hook=OrderedDict)
        super(AlphabetCounts,self).__init__(initialization_param)
    
    def __str__(self):
        return json.dumps(self,separators=(',', ':'))#no whitespace
    
    def __add__(self, other):
        #Preserve the order of the LHS first, and then take any others,
        #preserving LHS order for previously unseen keys..
        to_return = AlphabetCounts(self)
        full_keyset = list(self.keys())
        for k in other.keys():
            if not k in self:
                full_keyset.append(k)
        for k in full_keyset:
            to_return[k] = self.get(k,0) + other.get(k,0)
        return to_return
    
    def encode(self):
        return str(self)
    
    @staticmethod
    def decode(in_str: str):
        return AlphabetCounts(in_str)
    


class AlphabetConverterMixin(ABC):
    def __call__(self, in_str, include_start=True,include_stop=True):
        raise NotImplementedError("")

class Alphabet(AlphabetConverterMixin, object):
    #TODO: find the correct decorators to make these constant.
    stop_token = chr(3)#ASCII End of text
    stop_integer = 0
    start_token = chr(2)#ASCII Start of text
    start_integer = 1
    unknown_token = chr(26)#ASCII Substitution
    unknown_integer = 2
    NUM_SPECIAL = 3
    
    def __init__(self,in_str="",in_counts=None):
        super().__init__()
        self.letters = OrderedDict([(Alphabet.stop_token, Alphabet.stop_integer),
                                    (Alphabet.start_token, Alphabet.start_integer),
                                    (Alphabet.unknown_token, Alphabet.unknown_integer)]
                                    )
        self.__quickstring = None #str
        self.counts = None #AlphabetCounts()
        
        #works for any iterable, thus also a copy constructor for an Alphabet
        for i in sorted(in_str):
            #TODO: an assertion that this is a character and not a string.
            if not i in self.letters: #someone might provide duplicates even here.
                self.letters[i] = len(self.letters)
        
        self.__quickstring = "".join(list(self.letters.keys()))
        if in_counts is not None:
            if isinstance(in_counts, dict):
                assert set(self.letters.keys()) == set([Alphabet.stop_token,Alphabet.start_token,Alphabet.unknown_token]).union(set(in_counts.keys()))
            elif isinstance(in_counts, list):
                raise NotImplementedError("holding counts in a list is not yet supported")
            else:
                #import pdb; pdb.set_trace()
                print(type(in_counts))
                raise NotImplementedError("Unsupported type for counts")
            #enforce order of this alphabet on counts.
            self.counts = AlphabetCounts(([(one_char, in_counts.get(one_char, 0)) for one_char in self.__quickstring]))
        
    
    def __len__(self):
        return len(self.__quickstring)
    
    def __iter__(self):
        return iter(self.letters.keys())
        
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.__quickstring[index]
        elif isinstance(index,str) and 1 == len(index):
            return self.letters[index]
        else:
            assert False
    
    def __str__(self):
        return str(self.__quickstring)
    
    def __repr__(self):
        return __name__ + ".Alphabet(" + str(self) + ")"
    
    def __add__(self, other):
        use_counts = None
        if self.counts is None and other.counts is None:
            use_counts = None
        elif self.counts is not None and other.counts is not None:
            use_counts = self.counts + other.counts
        else:
            raise Exception("Either bother or neither alphabet should include counts information")
        return Alphabet(sorted(str(self)+str(other)), use_counts)
    
    def __call__(self, in_str,include_start=True,include_stop=True):
        #TODO: implement unknown token.
        retlist = [self.letters[i] for i in in_str]
        if include_start:
            retlist = [self.letters[self.start_token]] + retlist
        if include_stop:
            retlist.append(self.letters[self.stop_token])
        
        return retlist
    
    def __eq__(self, other):
        return self.__quickstring == other.__quickstring
    
    def sort(self):
        return Alphabet(sorted(str(self.__quickstring)), self.counts)

    def copy(self):
        return Alphabet(str(self),self.counts)
    

from collections import OrderedDict
from abc import ABC

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
    
    def __init__(self,in_str=""):
        super().__init__()
        self.letters = OrderedDict([(Alphabet.stop_token, Alphabet.stop_integer),
                                    (Alphabet.start_token, Alphabet.start_integer),
                                    (Alphabet.unknown_token, Alphabet.unknown_integer)]
                                    )
        #works for any iterable, thus also a copy constructor for an Alphabet
        for i in in_str:
            #TODO: an assertion that this is a character and not a string.
            if not i in self.letters: #someone might provide duplicates even here.
                self.letters[i] = len(self.letters)
        
        self.__quickstring = "".join(list(self.letters.keys()))
    
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
        return Alphabet(sorted(str(self)+str(other)))
    
    def __call__(self, in_str,include_start=True,include_stop=True):
        retlist = [self.letters[i] for i in in_str]
        if include_start:
            retlist = [self.letters[self.start_token]] + retlist
        if include_stop:
            retlist.append(self.letters[self.stop_token])
        
        return retlist
    
    def __eq__(self, other):
        return self.__quickstring == other.__quickstring
    
    def sort(self):
        return Alphabet(sorted(str(self.__quickstring)))

    def copy(self):
        return Alphabet(str(self))

    def encode(self):
        return self.__quickstring

    def decode(code: str):
        return Alphabet(code)

from collections import OrderedDict
from data.unimorph_loader.uniread import read_unimorph_tsv
import functools
import operator
import data.unimorph_loader.languages

class Alphabet(object):
    #TODO: find the correct decorators to make these constant.
    stop_token = chr(3)
    stop_integer = 0
    start_token = chr(2)
    start_integer = 1
    
    def __init__(self,in_str=""):
        self.letters = OrderedDict([(Alphabet.stop_token, Alphabet.stop_integer), (Alphabet.start_token, Alphabet.start_integer)])
        self.reverse_map = {Alphabet.stop_integer: Alphabet.stop_token, Alphabet.start_integer: Alphabet.start_token}
        #works for any iterable, thus also a copy constructor for an Alphabet
        for i in in_str:
            #TODO: an assertion that this is a character and not a string.
            if not i in self.letters: #someone might provide duplicates even here.
                self.letters[i] = len(self.letters)
                self.reverse_map[self.letters[i]] = i
        
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

    def __setitem__(self, char: str, integral: int):
        old_integral = self.letters[char]
        self.reverse_map.pop(old_integral)
        self.letters[char] = integral
        self.reverse_map[integral] = char
    
    def __str__(self):
        return str(self.__quickstring)
    
    def __repr__(self):
        return __name__ + ".Alphabet(" + str(self) + ")"
    
    def __add__(self, other):
        return Alphabet(str(self)+str(other))
    
    def __call__(self, in_str,include_start=True,include_stop=True):
        retlist = [self.letters[i] for i in in_str]
        if include_start:
            retlist = [self.letters[self.start_token]] + retlist
        if include_stop:
            retlist.append(self.letters[self.stop_token])
        
        return retlist

    def copy(self):
        return Alphabet(str(self))

    def encode(self):
        return ("{}" + data.unimorph_loader.languages.separator
                + "{}" + data.unimorph_loader.languages.separator
                + "{}").format(self.__quickstring, self.letters, self.reverse_map)

    def decode(code: str):
        quick_string, letters, reverse_map = code.split(data.unimorph_loader.languages.separator)
        out = Alphabet()
        out.letters = eval(letters)
        out.reverse_map = reverse_map
        out.__quickstring = quick_string
        return out


def get_master_alphabet(alphabets, reindex=False):
    unified_alphabet = functools.reduce(lambda a, b: a + b, alphabets)
    if reindex:
        for alphabet in alphabets:
            for character in alphabet:
                alphabet[character] = unified_alphabet[character]
    return unified_alphabet

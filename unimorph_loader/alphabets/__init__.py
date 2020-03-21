from collections import OrderedDict

class Alphabet(object):
    #TODO: find the correct decorators to make these constant.
    stop_token = chr(3)
    stop_integer = 0
    start_token = chr(2)
    start_integer = 1
    
    def __init__(self,in_str=""):
        self.letters = OrderedDict([(Alphabet.stop_token, Alphabet.stop_integer), (Alphabet.start_token, Alphabet.start_integer)])
        #works for any iterable, thus also a copy constructor for an Alphabet
        for i in in_str:
            #TODO: an assertion that this is a character and not a string.
            if not i in self.letters: #someone might provide duplicates even here.
                self.letters[i] = len(self.letters)
        
        self.__quickstring = "".join(list(self.letters.keys()))
        
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
        return Alphabet(str(self)+str(other))
    
    def copy(self):
        return Alphabet(str(self))
        
empty = Alphabet()
stop_start = empty
symbols = Alphabet(' !"\'()*+,-./0123456789:;=?@^_~')
roman = Alphabet('abcdefghijklmnopqrstuvwxyz')
latin_diacritic = Alphabet('ßàáâäåæèéêëìíîïðñòóôõöøùúûüýþāăąċčďđēĕęġīĭįļņŋōŏœŗšŧūŭųźžƿǟǣǫǿțȭȯȱȳɂɔɛʉʔ')
turkish_latin = Alphabet("abcçdefgğhıijklmnoöprsştuüvyz")
cyrillic = Alphabet("абвгдежзийклмнопрстуфхцчшщъыьэюяёіѣҥ")
tones = Alphabet("⁰¹²³⁴⁵ᵈᵊᵖˀ")
other = Alphabet("|´ʼίӓӧӱӹᐟḑḗạậẹệọộụ–’")

#https://en.wikipedia.org/wiki/Kazakh_alphabets
cyrillic_kazak = Alphabet("аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя")

#https://en.wikipedia.org/wiki/Common_Turkic_Alphabet
common_turkic_alphabet = Alphabet("aäbcçdefgğhıijklmnñoöpqrsştuüvwxyzʼ")
common_turkic_ipa = Alphabet("ɑæbdʒtʃdefgɣhɯiʒcklmnŋoøpqrsʃtuyvwxjzʔ")
common_turkic_cyrillic = Alphabet('аәәебџчжддѕфгғҕһҳхыикқлљмнњңоөпрсҫшцттуүвўјзз́ҙ')

def get_master_alphabet(include_unseen_alphabets=True):
    #all the alphabets that I've
    alphabets_to_process = [stop_start, symbols, roman, latin_diacritic, turkish_latin, cyrillic, tones, other]
    
    if include_unseen_alphabets:
        alphabets_to_process.extend([cyrillic_kazak, common_turkic_alphabet, common_turkic_ipa, common_turkic_cyrillic ])
    
    master_alphabet = reduce(lambda a,b:a+b, alphabets_to_process)
    
    return master_alphabet

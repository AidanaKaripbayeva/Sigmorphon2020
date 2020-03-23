from collections import OrderedDict

from ..alphabets import Alphabet, get_master_alphabet


separator = "\t"


class Language:
    def __init__(self, language_id, name, family, alphabet):
        self.id = language_id
        self.name = name
        self.family_id = None
        self.family = family
        self.alphabet = alphabet
    
    def __repr__(self):
        return "Language({})".format(self.encode())

    def encode(self):
        return ("{}" + separator + "{}" + separator + "{}" + separator + "{}" + separator + "{}").format(self.id, self.family_id, self.name, self.family, self.alphabet)

    def decode(code: str, alphabet):
        original_id, name, family = code.split(separator)
        out = Language(name, family, None)
        out.id = original_id
        out.name = name
        out.family = family
        out.alphabet = alphabet
        return out


class LanguageFamily:
    def __init__(self, language_family_id, name):
        self.id = language_family_id
        self.name = name
        self.languages = OrderedDict()
        self._master_alphabet = None
    
    def __len__(self):
        return len(self.languages)
    
    def __iter__(self):
        return self.languages.__iter__()
    
    def __getitem__(self, index):
        return self.languages[index]
    
    def add_language(self, language):
        self.languages[language.name] = language
        language.family_id = self.id
        self._master_alphabet = None
    
    def get_master_alphabet(self):
        if self._master_alphabet is None:
            alphabets = [language.alphabet for _,language in self.languages.items()]
            self._master_alphabet = get_master_alphabet(alphabets, reindex=False)
        return self._master_alphabet

    def encode(self):
        return ("{}" + separator + "{}").format(self.id, self.name)

    def decode(code: str):
        original_id, name = code.split(separator)
        out = LanguageFamily(name)
        out.id = original_id
        return out


class LanguageCollection:
    
    @classmethod
    def save_tsv(cls, filelike, lang_collection):
        for lang in lang_collection.list():
            filelike.write("\t".join( [str(lang.id), str(lang.family_id), str(lang.name), str(lang.family), str(lang.alphabet)])  + "\n")
    
    @classmethod
    def from_tsv(cls, filelike):
        to_return = LanguageCollection()
        for row in filelike:#TODO: do this better with getline/readline or the inbuilt csv features in python
            id, fam_id, name, fam, alphstring = row.strip().split("\t")
            #This isn't ideal, but for the moment I am going to have to assume that the
            #TSV file was written in order.
            to_return.add_language(name, fam, Alphabet(alphstring))
        return to_return
            
            
    
    @classmethod
    def get_default(cls,alphabet_choice="individual"):
        import importlib.resources
        with importlib.resources.path(__package__,"{}_alphabets.tsv".format(alphabet_choice)) as presaved_filename:
            with open(presaved_filename) as presaved_filelike:
    		          return cls.from_tsv(presaved_filelike)
        pass
    
    def __init__(self):
        self._master_alphabet = None
        self.language_families = OrderedDict()
        self.language_count = 0

    def add_language_family(self, name):
        language_family = LanguageFamily(len(self.language_families), name)
        self.language_families[name] = language_family
        self._master_alphabet = None
        return language_family

    def add_language(self, name, family, alphabet):
        language = Language(self.language_count, name, family, alphabet)
        if not family in self:
            self.add_language_family(family)
        self.language_families[family].add_language(language)
        self.language_count += 1
        return language

    def get_master_alphabet(self):
        if self._master_alphabet is None:
            alphabets = [language.alphabet
                         for _, family in self.language_families.items()
                         for _, language in family.languages.items()
                         ]
            self._master_alphabet = get_master_alphabet(alphabets, reindex=True)
        return self._master_alphabet
    
    def __len__(self):
        return self.language_count
    
    def __iter__(self):
        return self.language_families.__iter__()
    
    def __getitem__(self,index):
        return self.language_families[index]
    
    def keys(self):
        return self.language_families.keys()
    
    def list(self):
        def lang_iterator(foo):
            for fam in foo.language_families:
                for lang in foo.language_families[fam]:
                    yield foo.language_families[fam][lang]
        return list(lang_iterator(self))
        
    def find_language(self, language_name):
        """
        Search the LanguageCollection for a specific language by name.
        You can use this to get the id of the language and its family.
        """
        for one_family in self.language_families.values():
            if language_name in one_family.languages:
                return one_family.languages[language_name]
        raise KeyError("{} not found".format(language_name))

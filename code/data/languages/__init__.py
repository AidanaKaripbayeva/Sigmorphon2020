from collections import OrderedDict

from ..alphabets import Alphabet, AlphabetCounts, get_master_alphabet

import pdb
separator = "\t"


class Language:
    """
    Data class holding the information about a language, including its name, its family name and its alphabet.
    """
    def __init__(self, language_id, name, family, alphabet):
        """Protected method. Only to be used by LanngugeCollection."""
        self.id = language_id
        self.name = name
        self.family_id = None
        self.family = family
        self.alphabet = alphabet
    
    def __repr__(self):
        return "Language({})".format(self.name)


class LanguageFamily:
    """
    Data class holding the information about a language family, including its name, the names of its known languages
    and its master alphabet. A master alphabet is a hypothetical alphabet that is created by taking the union of the
    alpabets of the languages in this family.
    """
    def __init__(self, language_family_id, name):
        """Protected method. Only to be used by LanngugeCollection."""
        self.id = language_family_id
        self.name = name
        self.languages = OrderedDict()
        self._id_to_name = OrderedDict() #dictionary instead of list to support stable removal
        self._master_alphabet = None

    def __len__(self):
        return len(self.languages)
    
    def __iter__(self):
        return self.languages.__iter__()
    
    def __getitem__(self, index):
        if isinstance(index,int):
            return self.languages[self._id_to_name[index]]
        return self.languages[index]
        return self.languages[index]
    
    def add_language(self, language):
        """Protected method. Only to be used by LanguageCollection class."""
        self.languages[language.name] = language
        self._id_to_name[language.id] = language.name
        language.family_id = self.id
        self._master_alphabet = None
    
    def get_master_alphabet(self):
        """
        Lazily constructs and returns the master alphabet of this family.

        :return: An Alphabet object.
        """
        if self._master_alphabet is None:
            alphabets = [language.alphabet for _,language in self.languages.items()]
            self._master_alphabet = get_master_alphabet(alphabets, reindex=True)
        return self._master_alphabet
    
    @property
    def alphabet(self):
        return self.get_master_alphabet()


class LanguageCollection:
    """
    A data class containing a list of language families and their master alphabet. A master alphabet is a
    hypothetical alphabet that is created by taking the union of the alpabets of of a language collection.
    """
    
    @classmethod
    def save_tsv(cls, filelike, lang_collection):
        for lang in lang_collection.list():
            filelike.write("\t".join( [str(lang.id), str(lang.family_id), str(lang.name), str(lang.family), str(lang.alphabet), str( lang.alphabet.counts ) ])  + "\n")
    
    @classmethod
    def from_tsv(cls, filelike):
        to_return = LanguageCollection()
        for row in filelike:#TODO: do this better with getline/readline or the inbuilt csv features in python
            id, fam_id, name, fam, alphstring, alphcounts = row.strip().split("\t")
            #This isn't ideal, but for the moment I am going to have to assume that the
            #TSV file was written in order.
            deserialized_counts = AlphabetCounts.decode(alphcounts)
            to_return.add_language(name, fam, Alphabet(alphstring, deserialized_counts) )
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
        self._id_to_name = OrderedDict() #dict to support stable removal later
        self.language_count = 0
        self.family_count = 0

    def add_language_family(self, name):
        """
        Instantiates and adds a new family of languages to the collection.

        :param name: The name of the family as a string.
        :return: The instantiated LanguageFamily object.
        """
        language_family = LanguageFamily(self.family_count, name) #TODO: will screw up if we remove a family and later ad annother.
        self.language_families[name] = language_family
        self._id_to_name[language_family.id] = name
        self.family_count += 1
        self._master_alphabet = None
        return language_family

    def add_language(self, name, family, alphabet):
        """
        Instantiates and adds a new language to the corresponding language family in the collection.

        :param name: The name of the language as a string.
        :param family: The name of the language family as a string.
        :param alphabet: The alphabet of the language as an Alphabet object.
        :return: The instantiated Language object.
        """
        language = Language(self.language_count, name, family, alphabet)
        if not family in self:
            self.add_language_family(family)
        self.language_families[family].add_language(language)
        self.language_count += 1
        self._master_alphabet = None
        return language

    def get_master_alphabet(self):
        """
        Lazily constructs and returns the master alphabet of the collection.

        :return: An Alphabet object.
        """
        if self._master_alphabet is None:
            alphabets = [family.get_master_alphabet() for family in self.language_families.values()
                         ]
            self._master_alphabet = get_master_alphabet(alphabets, reindex=True)
        return self._master_alphabet
    
    @property
    def alphabet(self):
        return self.get_master_alphabet()
    
    def __len__(self):
        return len(self.language_families)
    
    def __iter__(self):
        return self.language_families.__iter__()
    
    def __getitem__(self,index):
        if isinstance(index,int):
            return self.language_families[self._id_to_name[index]]
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
    
    def get_alphabet_for_subset(self, families, languages):
        """
        This is a mistake, and it's gross and hackish to do it this way.
        I no longer have the energy to do it the right way, and as long as this keeps
        a certain project partner happy without breaking the correct behaviour,
        maybe that's good enough.
        """
        if families is None:
            families = list()
        if languages is None:
            languages = list()
        
        #no languages or families requested. All.
        #Should have been handled in calling code.
        #if 0==len(families) and 0 == len(languages):
        #    families =
        
        sub_alphabets = list([Alphabet()])
        for fam in families:
            sub_alphabets.append(self[fam].get_master_alphabet())
        for lang in languages:
            sub_alphabets.append(self.find_language(lang).alphabet)
        return get_master_alphabet(sub_alphabets, reindex=True)

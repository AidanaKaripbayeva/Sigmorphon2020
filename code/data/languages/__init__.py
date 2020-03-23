from ..alphabets import Alphabet, get_master_alphabet


separator = "\t"


class Language:
    def __init__(self, language_id, name, family, alphabet):
        self.id = language_id
        self.name = name
        self.family = family
        self.alphabet = alphabet

    def encode(self):
        return ("{}" + separator + "{}" + separator + "{}").format(self.id, self.name, self.family)

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
        self.languages = {}
        self._master_alphabet = None

    def __iter__(self):
        return self.languages.__iter__()

    def add_language(self, language):
        self.languages[language.name] = language
        self._master_alphabet = None

    def get_master_alphabet(self):
        if self._master_alphabet is None:
            alphabets = [language.alphabet for _, language in self.languages]
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
    def __init__(self):
        self._master_alphabet = None
        self.language_families = {}
        self.language_count = 0

    def add_language_family(self, name):
        language_family = LanguageFamily(len(self.language_families), name)
        self.language_families[name] = language_family
        self._master_alphabet = None
        return language_family

    def add_language(self, name, family, alphabet):
        language = Language(self.language_count, name, family, alphabet)
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
        
    def find_language(self, language_name):
        """
        Search the LanguageCollection for a specific language by name.
        You can use this to get the id of the language and its family.
        """
        for one_family in self.language_families.values():
            if language_name in one_family.languages:
                return one_family.languages[language_name]
        raise KeyError("{} not found".format(language_name))

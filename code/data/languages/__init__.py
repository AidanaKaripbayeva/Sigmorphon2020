from ..alphabets import get_master_alphabet

separator = "\t"


class Language:
    """
    Data class holding the information about a language, including its name, its family name and its alphabet.
    """
    def __init__(self, language_id, name, family, alphabet):
        """Protected method. Only to be used by LanngugeCollection."""
        self.id = language_id
        self.name = name
        self.family = family
        self.alphabet = alphabet


class LanguageFamily:
    """
    Data class holding the information about a language family, including its name, the names of its known languages
    and its master alphabet. A master alphabet is a hypothetical alphabet that is created by taking the union of the
    alpabets of the langauges in this family.
    """
    def __init__(self, language_family_id, name):
        """Protected method. Only to be used by LanngugeCollection."""
        self.id = language_family_id
        self.name = name
        self.languages = {}
        self._master_alphabet = None

    def __iter__(self):
        return self.languages.__iter__()

    def add_language(self, language):
        """Protected method. Only to be used by LanguageCollection class."""
        self.languages[language.name] = language
        self._master_alphabet = None

    def get_master_alphabet(self):
        """
        Lazily constructs and returns the master alphabet of this family.

        :return: An Alphabet object.
        """
        if self._master_alphabet is None:
            alphabets = [language.alphabet for _, language in self.languages]
            self._master_alphabet = get_master_alphabet(alphabets, reindex=False)
        return self._master_alphabet


class LanguageCollection:
    """
    A data class containing a list of language families and their master alphabet. A master alphabet is a
    hypothetical alphabet that is created by taking the union of the alpabets of of a langauge collection.
    """
    def __init__(self):
        self._master_alphabet = None
        self.language_families = {}
        self.language_count = 0

    def add_language_family(self, name):
        """
        Instantiates and adds a new family of languages to the collection.

        :param name: The name of the family as a string.
        :return: The instantiated LanguageFamily object.
        """
        language_family = LanguageFamily(len(self.language_families), name)
        self.language_families[name] = language_family
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
        self.language_families[family].add_language(language)
        self.language_count += 1
        return language

    def get_master_alphabet(self):
        """
        Lazily constructs and returns the master alphabet of the collection.

        :return: An Alphabet object.
        """
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

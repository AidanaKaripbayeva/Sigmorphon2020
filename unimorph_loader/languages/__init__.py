from data.unimorph_loader.alphabets import Alphabet, get_master_alphabet
from data.unimorph_loader.uniread import read_unimorph_tsv
import os
import re
separator = "#"


class Language:
    _count = 0

    def __init__(self, name, family, alphabet):
        self.id = Language._count
        self.name = name
        self.family = family
        self.alphabet = alphabet
        Language._count += 1

    def __getstate__(self):
        raise Exception("Not serializable")

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
    _count = 0

    def __init__(self, name):
        self.id = LanguageFamily._count
        self.name = name
        self.languages = {}
        self._master_alphabet = None
        LanguageFamily._count += 1

    def __iter__(self):
        return self.languages.__iter__()

    def __getstate__(self):
        raise Exception("Not serializable")

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
        self.language_families = {}
        self._master_alphabet = None

    def add_language_family(self, language_family):
        self.language_families[language_family.name] = language_family
        self._master_alphabet = None

    def get_master_alphabet(self):
        if self._master_alphabet is None:
            alphabets = [language.alphabet
                         for _, family in self.language_families.items()
                         for _, language in family.languages.items()
                         ]
            self._master_alphabet = get_master_alphabet(alphabets, reindex=True)
        return self._master_alphabet

    def serialize(self, filename):
        with open(filename, 'w+') as file:
            print(str(len(self.language_families)), file=file)
            num_language = 0
            for _, language_family in self.language_families.items():
                print(language_family.encode(), file=file)
                num_language += len(language_family.languages)
            print(num_language, file=file)
            for _, language_family in self.language_families.items():
                for _, language in language_family.languages.items():
                    print(language.alphabet.encode(), file=file)
                    print(language.encode(), file=file)
            print(self._master_alphabet.encode(), file=file)

    def deserialize(self, filename):
        with open(filename, 'r') as file:
            num_language_families = int(file.readline()[:-1])
            for i in range(num_language_families):
                line = file.readline()[:-1]
                language_family = LanguageFamily.decode(line)
                self.language_families[language_family.name] = language_family
            num_languages = int(file.readline()[:-1])
            for i in range(num_languages):
                line = file.readline()[:-1]
                alphabet = Alphabet.decode(line)
                line = file.readline()[:-1]
                language = Language.decode(line, alphabet)
                self.language_families[language.family].add_language(language)
            line = file.readline()[:-1]
            self._master_alphabet = Alphabet.decode(line)


def read_language_collection_from_dataset(root_dir):
    language_collection = LanguageCollection()
    for language_family_name in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, language_family_name))\
                and re.match(r'[a-zA-Z-]*', language_family_name):
            language_family = LanguageFamily(language_family_name)
            for language_file in os.listdir(os.path.join(root_dir, language_family_name)):
                if re.match(r'[a-zA-Z-]*\.trn', language_file):
                    language_name = language_file[:-4]
                    train_data = read_unimorph_tsv(os.path.join(root_dir, language_family_name, language_name + '.trn'))
                    test_data = read_unimorph_tsv(os.path.join(root_dir, language_family_name, language_name + '.dev'))
                    letters = set()
                    for word in train_data['lemma'] + train_data['form'] + test_data['lemma'] + test_data['form']:
                        for letter in word:
                            letters.add(letter)
                    alphabet = Alphabet("".join(sorted(letters)))
                    language = Language(language_name, language_family_name, alphabet)
                    language_family.add_language(language)
        language_collection.add_language_family(language_family)
    _ = language_collection.get_master_alphabet()
    return language_collection

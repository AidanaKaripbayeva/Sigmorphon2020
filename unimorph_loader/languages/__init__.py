from data.unimorph_loader.alphabets import Alphabet
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
        LanguageFamily._count += 1

    def __iter__(self):
        return self.languages.__iter__()

    def __getstate__(self):
        raise Exception("Not serializable")

    def add_language(self, language):
        self.languages[language.id] = language

    def get_master_alphabet(self):
        master_alphabet = Alphabet()
        for language in self.languages:
            master_alphabet += language.alphabet
        return master_alphabet

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
        self._master_alphabet = Alphabet()

    def add_language_family(self, language_family):
        self.language_families[language_family.id] = language_family

    def get_master_alphabet(self):
        return self._master_alphabet

    def serialize(self, filename):
        with open(filename, 'w') as file:
            print(str(len(self.language_families)), file=file)
            num_language = 0
            for language_family in self.language_families:
                print(language_family.encode(), file=file)
                num_language += len(language_family.languages)
            print(num_language, file=file)
            for language_family in self.language_families:
                for language in language_family.languages:
                    print(language.alphabet.encode(), file=file)
                    print(language.encode(), file=file)
            print(self._master_alphabet.encode)

    def deserialize(self, filename):
        with open(filename, 'r') as file:
            num_language_families = int(file.readline())
            for i in range(num_language_families):
                line = file.readline()
                language_family = LanguageFamily.decode(line)
                self.language_families[language_family.id] = language_family
            num_languages = int(file.readline())
            for i in range(num_languages):
                line = file.readline()
                alphabet = Alphabet.decode(line)
                line = file.readline()
                language = Language.decode(line, alphabet)
                self.language_families[language.family].add_language(language)
            line = file.readline()
            self._master_alphabet = Alphabet.decode(line)

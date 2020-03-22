class Language:
    _count = 0

    def __init__(self, name, family, alphabet):
        self.id = Language._count
        self.name = name
        self.family = family
        self.alphabet = alphabet
        Language._count += 1

    def __getstate__(self):
        return self.__dict_, Language._count

    def __setstate__(self, state):
        self.__dict__ = state[0]
        Language._count = state[1]


class LanguageFamily:
    _count = 0

    def __init__(self, name):
        self.id = LanguageFamily._count
        self.name = name
        self.languages = []
        LanguageFamily._count += 1

    def __iter__(self):
        return self.languages.__iter__()

    def __getstate__(self):
        return self.__dict_, LanguageFamily._count

    def __setstate__(self, state):
        self.__dict__ = state[0]
        LanguageFamily._count = state[1]

    def add(self, language):
        self.languages.append(language)

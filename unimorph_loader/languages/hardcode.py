from data.unimorph_loader.alphabets import Alphabet
from . import LanguageCollection, LanguageFamily, Language

# Define alphabets
empty = Alphabet()
stop_start = empty
symbols = Alphabet(' !"\'()*+,-./0123456789:;=?@^_~')
roman = Alphabet('abcdefghijklmnopqrstuvwxyz')
latin_diacritic = Alphabet('ßàáâäåæèéêëìíîïðñòóôõöøùúûüýþāăąċčďđēĕęġīĭįļņŋōŏœŗšŧūŭųźžƿǟǣǫǿțȭȯȱȳɂɔɛʉʔ')
turkish_latin = Alphabet("abcçdefgğhıijklmnoöprsştuüvyz")
cyrillic = Alphabet("абвгдежзийклмнопрстуфхцчшщъыьэюяёіѣҥ")
tones = Alphabet("⁰¹²³⁴⁵ᵈᵊᵖˀ")
other = Alphabet("|´ʼίӓӧӱӹᐟḑḗạậẹệọộụ–’")

# https://en.wikipedia.org/wiki/Kazakh_alphabets
cyrillic_kazak = Alphabet("аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя")

# https://en.wikipedia.org/wiki/Common_Turkic_Alphabet
common_turkic_alphabet = Alphabet("aäbcçdefgğhıijklmnñoöpqrsştuüvwxyzʼ")
common_turkic_ipa = Alphabet("ɑæbdʒtʃdefgɣhɯiʒcklmnŋoøpqrsʃtuyvwxjzʔ")
common_turkic_cyrillic = Alphabet('аәебџчжддѕфгғҕһҳхыикқлљмнњңоөпрсҫшцттуүвўјзз́ҙ')


# Define language_families
language_collection = LanguageCollection()
extras_family = LanguageFamily('extras')
extras_family.add_language(Language('empty', 'extras', empty))
extras_family.add_language(Language('stop_start', 'extras', stop_start))
extras_family.add_language(Language('symbols', 'extras', symbols))
extras_family.add_language(Language('tones', 'extras', tones))
extras_family.add_language(Language('other', 'extras', other))
language_collection.add_language_family(extras_family)
pie_family = LanguageFamily('pie')
pie_family.add_language(Language('roman', 'pie', roman))
pie_family.add_language(Language('latin', 'pie', latin_diacritic))
language_collection.add_language_family(pie_family)
slavic_family = LanguageFamily('slavic')
slavic_family.add_language(Language('cyrillic', 'slavic', cyrillic))
language_collection.add_language_family(slavic_family)
turkic_family = LanguageFamily('turkic')
turkic_family.add_language(Language('turkish', 'turkic', turkish_latin))
turkic_family.add_language(Language('cyrillic_kazak', 'turkic', cyrillic_kazak))
turkic_family.add_language(Language('common_turkic', 'turkic', common_turkic_alphabet))
turkic_family.add_language(Language('turkic_ipa', 'turkic', common_turkic_ipa))
turkic_family.add_language(Language('turkic_cyrillic', 'turkic', common_turkic_cyrillic))
language_collection.add_language_family(turkic_family)

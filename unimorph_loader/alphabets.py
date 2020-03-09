symbols = ' !"\'()*+,-./0123456789:;=?@^_~'
roman = 'abcdefghijklmnopqrstuvwxyz'
latin_diacritic = 'ßàáâäåæèéêëìíîïðñòóôõöøùúûüýþāăąċčďđēĕęġīĭįļņŋōŏœŗšŧūŭųźžƿǟǣǫǿțȭȯȱȳɂɔɛʉʔ'
turkish_latin = "abcçdefgğhıijklmnoöprsştuüvyz"
cyrillic = "абвгдежзийклмнопрстуфхцчшщъыьэюяёіѣҥ"
tones = "⁰¹²³⁴⁵ᵈᵊᵖˀ"
other = "|´ʼίӓӧӱӹᐟḑḗạậẹệọộụ–’"

#https://en.wikipedia.org/wiki/Kazakh_alphabets
cyrillic_kazak = "аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя"

#https://en.wikipedia.org/wiki/Common_Turkic_Alphabet
common_turkic_alphabet = "aäbcçdefgğhıijklmnñoöpqrsştuüvwxyzʼ"
common_turkic_ipa = "ɑæbdʒtʃdefgɣhɯiʒcklmnŋoøpqrsʃtuyvwxjzʔ"
common_turkic_cyrillic = 'аәәебџчжддѕфгғҕһҳхыикқлљмнњңоөпрсҫшцттуүвўјзз́ҙ'

def get_master_alphabet(include_unseen_alphabets=True):
    from collections import OrderedDict
    master_alphabet = OrderedDict()
    #all the alphabets that I've
    alphabets_to_process = [symbols, roman, latin_diacritic, turkish_latin, cyrillic, tones, other]
    if include_unseen_alphabets:
        alphabets_to_process.extend([cyrillic_kazak, common_turkic_alphabet, common_turkic_ipa, common_turkic_cyrillic ])

    for one_alpha in alphabets_to_process:
        for i in one_alpha:
            master_alphabet[i] = True

    return "".join(list(master_alphabet.keys()))

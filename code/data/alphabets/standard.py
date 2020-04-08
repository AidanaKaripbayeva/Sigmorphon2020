#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from .alphabet import Alphabet

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

kitchen_sink_alphabet = symbols + roman + latin_diacritic + turkish_latin + cyrillic + tones + other + cyrillic_kazak + common_turkic_alphabet + common_turkic_ipa + common_turkic_cyrillic

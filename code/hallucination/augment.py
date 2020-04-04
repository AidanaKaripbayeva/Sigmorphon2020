'''
Borrowed from https://github.com/antonisa/inflection
'''
import sys
sys.path.insert(0, 'src')

import align
import argparse
import codecs
import os
from random import random, choice
import re
from hall_choice import frequency_choice

import data.languages


def read_data(filename):
    with codecs.open(filename, 'r', 'utf-8') as inp:
        lines = inp.readlines()
    inputs = []
    outputs = []
    tags = []
    for l in lines:
        l = l.strip().split('\t')
        if l:
            inputs.append(list(l[0].strip()))
            outputs.append(list(l[1].strip()))
            tags.append(l[2].strip().split(";"))
    return inputs, outputs, tags

def find_good_range(a,b):
    mask = [(a[i]==b[i] and a[i] != u" ") for i in range(len(a))]
    if sum(mask) == 0:
        return []
        # Some times the alignment is off-by-one
        b = ' ' + b
        mask = [(a[i]==b[i] and a[i] != u" ") for i in range(len(a))]
    ranges = []
    prev = False
    for i,k in enumerate(mask):
        if k and prev:
            prev = True
        elif k and not prev:
            start = i
            prev = True
        elif prev and not k:
            end = i
            ranges.append((start, end))
            prev = False
        elif not prev and not k:
            prev = False
    if prev:
        ranges.append((start,i+1))
    ranges = [c for c in ranges if c[1]-c[0]>2]
    return ranges


def augment(inputs, outputs, tags, characters, lang_family, L2):
    print(lang_family, L2)
    lang_alphabet = load_the_alphabet(lang_family, L2)
    
    temp = [(''.join(inputs[i]), ''.join(outputs[i])) for i in range(len(outputs))]
    aligned = align.Aligner(temp, align_symbol=' ').alignedpairs

    vocab = list(characters)
    try:
        vocab.remove(u" ")
    except:
        pass

    new_inputs = []
    new_outputs = []
    new_tags = []
    for k,item in enumerate(aligned):
        #print(''.join(inputs[k]) + '\t' + ''.join(outputs[k]))
        i,o = item[0],item[1]
        good_range = find_good_range(i,o)
        #print(good_range)
        if good_range:
            new_i, new_o = list(i), list(o)
            for r in good_range:
                s = r[0]
                e = r[1]
                if (e-s>5): #arbitrary value
                    s += 1
                    e -= 1
                for j in range(s,e):
                    # TODO: write a function instead of choice to do the frequency stuff things yay
                    # define the new function
                    if random() > 0.5: #arbitrary value
                        #nc = choice(vocab)
                        nc = frequency_choice(lang_alphabet)
                        #print(nc)
                        new_i[j] = nc
                        new_o[j] = nc
                       
            new_i1 = [c for l,c in enumerate(new_i) if (c.strip() or (new_o[l]==' ' and new_i[l] == ' '))]
            new_o1 = [c for l,c in enumerate(new_o) if (c.strip() or (new_i[l]==' ' and new_o[l] == ' '))]
            new_inputs.append(new_i1)
            new_outputs.append(new_o1)
            new_tags.append(tags[k])
        else:
            new_inputs.append([])
            new_outputs.append([])
            new_tags.append([])

    return new_inputs, new_outputs, new_tags

def load_the_alphabet(lang_family, lang):
    lang_collection = data.languages.LanguageCollection.get_default(alphabet_choice="individual")
    lang_alphabet = lang_collection[lang_family][lang].alphabet
    return lang_alphabet

def get_chars(l):
    flat_list = [char for word in l for char in word]
    return list(set(flat_list))

def main_augment_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath", help="path to data", type=str)
    parser.add_argument("language_family", help="language_family", type=str)
    parser.add_argument("language", help="language", type=str)
    
    parser.add_argument("--examples", help="number of hallucinated examples to create (def: 10000)", default=10000, type=int)
    parser.add_argument("--use_dev", help="whether to use the development set (def: False)", action="store_true")
    args = parser.parse_args()
    
    LANG_FAM = args.language_family
    DATA_PATH = args.datapath + "/" + LANG_FAM
    
    L2 = args.language
    LOW_PATH = os.path.join(DATA_PATH, L2+".trn")
    DEV_PATH = os.path.join(DATA_PATH, L2+".dev")
    
    N = args.examples
    usedev = args.use_dev

    lowi, lowo, lowt = read_data(LOW_PATH)
    devi, devo, devt = read_data(DEV_PATH)

    if usedev:
        vocab = get_chars(lowi+lowo+devi+devo)
    else:
        vocab = get_chars(lowi+lowo)

    i,o,t = [], [], []
    while len(i) < N:
        if usedev:
            # Do augmentation also using examples from dev
            ii,oo,tt = augment(devi+lowi, devo+lowo, devt+lowt, vocab, LANG_FAM, L2)
        else:
            # Just augment the training set
            ii,oo,tt = augment(lowi, lowo, lowt, vocab, LANG_FAM, L2)
        ii = [c for c in ii if c]
        oo = [c for c in oo if c]
        tt = [c for c in tt if c]
        i += ii
        o += oo
        t += tt
        if len(ii) == 0:
            break

    # Wait is this needed?
    i = [c for c in i if c]
    o = [c for c in o if c]
    t = [c for c in t if c]

    with codecs.open(os.path.join(DATA_PATH,L2+".hall"), 'w', 'utf-8') as outp:
        for k in range(min(N, len(i))):
            outp.write(''.join(i[k]) + '\t' + ''.join(o[k]) + '\t' + ';'.join(t[k]) + '\n')

#from https://towardsdatascience.com/pairwise-sequence-alignment-using-biopython-d1a9d0ba861f
#NW in paper https://www.aclweb.org/anthology/W16-2002.pdf

#%load_ext autoreload
#%autoreload 2
import sys
from collections import Counter
#sys.path.insert(1, '/home/aidana/TurkicSigmorphon2020/code')

#from data import uniread

from operator import itemgetter
from itertools import *
import pandas as pd
import numpy as np
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import time

def new_table():
    columns = ['N', 'PROPN', 'ADJ', 'PRO','PRE','CLF','ART','DET',
               'V','ADV','AUX','V.AGT','V.PTCP','V.PTCP.PST','V.MSDR','V.CVB','V.CVB.GEN','V.CVB.SIM',
               'ADP','COMP','CONJ','NUM','PART','INTJ', 'Total']
    rows = ['Suffix', 'Prefix', 'Infix','Circumfix','Transfix','Unchanged', 'Fail to find']

    #df[columns][rows][0] <- how lemma is different from stem
    #df[columns][rows][1] <- how form is different from stem

    zero_data = [[[0,0] for i in range(len(columns))] for j in range(len(rows))]
    df = pd.DataFrame(zero_data,index=rows, columns=columns)
    
    return df


def findOccurrences_no_equal(s, ch):
    return [i for i, letter in enumerate(s) if letter != ch]

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

def MaxLength(lst): 
    maxList = max(lst, key = len) 
    maxLength = max(map(len, lst)) 
      
    return maxList, maxLength 

def my_format_alignment(align1, align2, score, begin, end): 
    s = []
    match = 0
    gap=0
    mismatch=0
    for a, b in zip(align1[begin:end], align2[begin:end]): 
        if a == b: 
            s.append('|')
            match+=1  
        elif a == "-" or b == "-": 
            s.append(' ')
            gap+=1
        else: 
            s.append('.')
            mismatch+=1

    return match,mismatch,gap,s

def NW_algorithm(lemma,form):
    al = []
    alignments = pairwise2.align.globalms(lemma, form, 2, -1, -1, -0.1)
    #for a in alignments:
     #   print(format_alignment(*a)) 

    match, mismatch, gap, al = my_format_alignment(*alignments[0])
    #print('Matches = ', match, ', Mismatch = ', mismatch, ', Gap = ', gap)
    #print(al)
    
    return match, mismatch, gap, al, alignments

def finding_stem_list(al):
    pos = findOccurrences(al, '|')   
    groups = []
    for r, g in groupby(enumerate(pos), lambda x: x[0]-x[1]):
        groups.append(list(map(itemgetter(1), g)))

    stem_list = []
    for i in groups:
        if (len(i) >= k):
            stem_list.append(i)
    
    return stem_list


def update_table_with_inf(word, stems, item, df, tag):    
    first_stem_pos = word.find(stems[0])
    last_stem_pos = word.find(stems[-1])
    
    if(first_stem_pos > 0 or last_stem_pos+len(stems[-1]) < len(word)):
        df[tag]['Transfix'][item] = df[tag]['Transfix'][item] + 1
        #print("df[tag]['Transfix']")
    else:
        df[tag]['Infix'][item] = df[tag]['Infix'][item] + 1
        #print("df[tag]['Infix']")
    
    
def update_table_no_inf(word, stem, item, df, tag):
    stem_pos = word.find(stem) 
    if(stem_pos > 0 and stem_pos+len(stem) < len(word)):
        df[tag]['Circumfix'][item] = df[tag]['Circumfix'][item] + 1
        #print('df[tag][Circumfix]')
    elif (stem_pos == 0 and stem_pos+len(stem) < len(word)):
        df[tag]['Suffix'][item] = df[tag]['Suffix'][item] + 1
        #print("df[tag]['Suffix']")
    elif (stem_pos > 0 and stem_pos+len(stem) == len(word)):
        df[tag]['Prefix'][item] = df[tag]['Prefix'][item] + 1
        #print("df[tag]['Prefix']")
    else:
        df[tag]['Unchanged'][item] = df[tag]['Unchanged'][item] + 1
        #print("df[tag]['Unchanged']")


        
      
        
k = 3 
austronesian_names = ['cebtrn', 'hiltrn', 'maotrn', 'mlgtrn','tgltrn']
germanic_names = ['angtrn', 'dantrn', 'deutrn', 'engtrn', 'frrtrn', 'gmhtrn', 'isltrn', 'nldtrn', 'nobtrn', 'swetrn']
niger_congo_names = ['akatrn', 'gaatrn', 'kontrn', 'lintrn', 'lugtrn', 'nyatrn', 'sottrn', 'swatrn', 'zultrn']
oto_manguean_names = ['azgtrn', 'clytrn', 'cpatrn', 'ctptrn', 'czntrn', 'otetrn', 'otmtrn', 'peitrn', 'xtytrn', 'zpvtrn']
uralic_names = ['esttrn', 'fintrn','izhtrn','krltrn','livtrn','mdftrn','mhrtrn','myvtrn','smetrn','veptrn','vottrn']

families_string = ['austronesian', 'germanic', 'niger_congo', 'oto_manguean', 'uralic']
families_names = [austronesian_names, germanic_names, niger_congo_names, oto_manguean_names, uralic_names]

filename = 'affix_analysis_k3.csv'
beginning = ['This is an analysis of affixes for Sigmorphon2020']
pd.DataFrame(beginning).T.to_csv(filename, index=False, header=False)

for count_fam,fam in enumerate(families_names):
    family_language = ['Family language => ' + families_string[count_fam]]
    with open(filename, 'a') as f: 
            pd.DataFrame(family_language).to_csv(f, index=False, header=False)
    for count_lang, lang in enumerate(fam):    
        result_to_file = []
        result_to_file = ['Affix analysis table for ' + lang]
        dataset = pd.read_csv('languages/' + families_string[count_fam] + '/' + lang +'.csv')
        df = new_table()
        t1 = time.time()
        for i in range(dataset.shape[0]):

            lemma = str(dataset['lemma'][i]) #damdam
            form = str(dataset['form'][i]) #mandaramdam
            tag = str(dataset['tags'][i]).partition(";")[0]



            match, mismatch, gap, al, alignments = NW_algorithm(lemma, form)



            stem_list = finding_stem_list(al)
            if(match < k or len(stem_list)==0):
                df[tag]['Fail to find'][0] = df[tag]['Fail to find'][0] + 1
                df[tag]['Fail to find'][1] = df[tag]['Fail to find'][1] + 1
                #print('Fail to find')
                continue


            if len(stem_list) > 1:
                stems =[]
                for count,i in enumerate(stem_list):
                    stems.append(alignments[0][0][stem_list[count][0]:stem_list[count][-1]+1])

                update_table_with_inf(lemma, stems, 0, df, tag) 
                update_table_with_inf(form, stems, 1, df, tag)  
                #print(lemma, form, stems)

            else:
                stem = alignments[0][0][stem_list[0][0]:stem_list[0][-1]+1]
                update_table_no_inf(lemma, stem, 0, df, tag)
                update_table_no_inf(form, stem, 1, df, tag)
                #print(lemma, form, stem)


        t2 = time.time()
        hours, rem = divmod(t2-t1, 3600)
        minutes, seconds = divmod(rem, 60)
        print ("It took ", hours, "hours, ", minutes, "minutes, ", seconds, "seconds to finish this task")

        with open(filename, 'a') as f: 
            pd.DataFrame(result_to_file).to_csv(f, index=False, header=False)
            df.to_csv(f)
    
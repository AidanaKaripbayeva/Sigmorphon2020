import os
import sys
import logging
import argparse
import glob
import itertools
from collections import defaultdict

from data.languages import LanguageCollection
from data.alphabets import Alphabet, AlphabetCounts

import data.uniread as uniread

import consts

def process_dataset_and_recreate_alphabets(directory_path,extensions=["trn"]):
    if not os.path.exists(directory_path):
        raise Exception("specified directory does not exist")
    if not os.path.isdir(directory_path):
        raise Exception("Specified file is not a directory")
    
    full_language_collection = LanguageCollection.get_default("individual")
    all_matching_files = list()
    
    for file_extension in extensions:
        all_matching_files.extend(glob.glob(os.path.join(directory_path,"*","*.{}".format(file_extension))))
    
    
    #clear existing alphabets
    for fam in full_language_collection:
        for lang in full_language_collection[fam]:
            full_language_collection[fam][lang].alphabet = Alphabet()
    
    for one_file in all_matching_files:
        family_name = os.path.basename(os.path.dirname(one_file))
        lang_name = os.path.basename(one_file).split(".",1)[0]
        
        assert family_name is not os.path.basename(directory_path), "language file directly in base path."
        
        #TODO: This may not be stable, maybe it would be good to sort first.
        #Would not be a problem if we just hardcoded, since the langauges aren't going to change...
        if family_name not in full_language_collection:
            #encountered a previously unknown family.
            full_language_collection.add_language_family(family_name)
        
        if lang_name not in full_language_collection[family_name]:
            full_language_collection[family_name].add_language( lang_name, family_name, Alphabet() )
        
        a_language = full_language_collection.find_language( lang_name )
        #process the file(s) for alphabets
        file_contents = uniread.read_unimorph_tsv(one_file,
                                family      =   a_language.family_id,
                                language    =   a_language.id
                                )
                                
        all_seen_characters = defaultdict(int)
        for i in itertools.chain(file_contents["form"],file_contents["lemma"]):
            for j in i:
                all_seen_characters[j] += 1
        
        
        all_seen_characters_str = list(all_seen_characters.keys())
        all_seen_characters_str.sort()
        
        a_language.alphabet = a_language.alphabet + Alphabet("".join(all_seen_characters_str), all_seen_characters)
        a_language.alphabet = a_language.alphabet.sort()
    
    return full_language_collection

def str_list(in_str):
    return in_str.split(",")

def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument("--tsv-filename", default="processed.tsv", type=str, help="""Name of languages/alphabets TSV file to create,
            created relative to root path""")
    aparser.add_argument("--types",default=["trn"],type=str_list,
            help="""Which types of data (file extensions) to base the alphabets on.
            Use comma separation for multiple types, ex: 'trn,dev'.
            Default to 'trn'. """)
    aparser.add_argument('root_dir', type=str, help='Root directory for a (possibly modified) SIGMORPHON 2020 dataset')
    
    
    args = aparser.parse_args()
    
    lang_collection = process_dataset_and_recreate_alphabets(args.root_dir,args.types)
    
    with open(os.path.join(args.root_dir, args.tsv_filename), "w") as output_file:
        LanguageCollection.save_tsv(output_file, lang_collection)
    
    
if __name__ == "__main__":
    main()

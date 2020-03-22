from . import schema

import pandas as _pn


def read_unimorph_tsv(filename, family=-1, language=-1):
	the_dataset = _pn.read_csv(filename,sep="\t",names=["lemma","form","tags"],dtype=str,na_values=[],keep_default_na=False )
	
	the_dataset["family"] = family
	the_dataset["language"] = language
	
	return the_dataset

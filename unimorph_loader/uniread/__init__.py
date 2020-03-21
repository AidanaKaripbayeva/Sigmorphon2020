from provided import *

import pandas as _pn

def read_unimorph_tsv(filename):
	return _pn.read_csv(filename,sep="\t",names=["lemma","form","features"],dtype=str,na_values=[],keep_default_na=False )

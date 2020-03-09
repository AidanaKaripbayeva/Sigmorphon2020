
import pandas as _pn

def read_unimorph_tsv(filename):
	return _pn.read_csv(filename,sep="\t",names=["lemma","form","features"],dtype=str,na_values=[],keep_default_na=False )

from collections import OrderedDict
def load_unimorph_tags(filename):
	import yaml as _yml
	tag_list = None
	with open(filename,"r") as infile:
		tag_list = _yml.safe_load(infile)
	#
	assert set(tag_list["categories"].keys()) == set(tag_list["ordering"])
	#
	thetags = OrderedDict()
	for cat in tag_list["ordering"]:
		tags_for_cat = list() #makeshift ordered set and back to list
		for one_or_more_tags in tag_list["categories"][cat]:
			tags_for_cat.extend(one_or_more_tags.replace("/","+").split("+") )
		thetags[cat] = list(OrderedDict([(i,True) for i in tags_for_cat]).keys())
	#
	return thetags

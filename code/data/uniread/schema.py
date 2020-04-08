import yaml as _yml
from collections import OrderedDict


def load_unimorph_schema_from_yaml(filename):
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
        
        tags_for_cat = [i for i in tags_for_cat if not (i.startswith("non{") and i.endswith("}"))]
        #Using as an ordered set
        thetags[cat] = list(OrderedDict([(i,True) for i in tags_for_cat]).keys())
    #
    return thetags


def load_default_schema():
    import importlib_resources
    with importlib_resources.path(__package__,"default_tags.yaml") as default_tags_filename:
        return load_unimorph_schema_from_yaml(default_tags_filename)

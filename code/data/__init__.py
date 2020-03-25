import glob, os

from collections import namedtuple, OrderedDict

from .languages import LanguageCollection
from . import dataset, dataloader
from . import uniread

class SigmorphonData_Factory(object):
    available_files = namedtuple("available_files",("train","dev","test"))
    
    @classmethod
    def get_instance():
        pass

    def __init__(self, dir,lang_config_filename="lang_config.tsv"):
        self.dir = dir
        self.available_languages = list()
        self.known_language_collection = None # Handled later
        self.known_languages = list() #
        self.unknown_languages = list() #Not used
        self.language_files = OrderedDict() # OrderedDict
        
        #OrderedDict([(l.name, self.available_files(None,None,None)) for
        #                                    lname in self.known_language_collection.list()])
                                            
        #Default language collections
        self.known_language_collection = LanguageCollection.get_default("individual")
        #user specified language collections
        full_expected_config_file = os.path.join(dir, lang_config_filename)
        if os.path.exists(full_expected_config_file) and os.path.isfile(full_expected_config_file):
            with open(full_expected_config_file) as lc_infile:
                self.known_language_collection = LanguageCollection.from_tsv(lc_infile)
        
        #Just a list of language names.
        self.known_languages = [l.name for l in self.known_language_collection.list()]
        
        #empty.
        self.language_files = OrderedDict([(l.name, self.available_files(None,None,None)) for
                                            l in self.known_language_collection.list()])
        
        for one_known_language in self.known_languages:
            one_train_file = glob.glob(os.path.join(dir,"*","{}.trn".format(one_known_language)))
            one_train_file = [None] if 0 == len(one_train_file) else one_train_file
            
            one_dev_file = glob.glob(os.path.join(dir,"*","{}.dev".format(one_known_language)))
            one_dev_file = [None] if 0 == len(one_dev_file) else one_dev_file
            
            one_test_file = list() #handled once they tell us what the test files are.
            one_test_file = [None] if 0 == len(one_test_file) else one_test_file
            
            if len(one_train_file) > 1 or len(one_dev_file) > 1 or len(one_test_file) > 1:
                raise Exception("There are duplicate files for language {}".format(one_known_language))
            self.language_files[one_known_language] = self.available_files(one_train_file[0], one_dev_file[0],one_test_file[0])
        
    def get_dataset(self, types=["train"], families=None, languages=None, dataloader_kwargs={},**kwargs):
        """
        Types is what kind of datafile type to load, it must be a list, options are "train" and "dev".
        
        families is a list of fmaily names or None.
        languages is a list of language names or None.
        
        If both families and languages are None, all languages will be loaded.
        
        
        """
        
        assert types is not None, "You must specify which kinds of files you want as a listlike. Empty for nothing."
        types = set(types)
        
        
        if families is None and languages is None:
            languages = set(self.known_languages)
        
        if languages is None:
            languages = set()
            
        if families is not None:
            for f in families:
                for l in self.known_language_collection[f]:
                    languages.add(l)
        
        #TODO: More detailed code about alphabets.
        
        
        #TODO: It might be better to load each dataset completely before going to the next, instead of loading all pandas first.
        pandas_dataframes = list()
        for l_name in languages:
            lang_id = -1
            fam_id = -1
            
            the_language = self.known_language_collection.find_language(l_name)
            lang_id = the_language.id
            fam_id = the_language.family_id
            
            l_files = self.language_files[l_name]
            if "train" in types and l_files.train is not None:
                pandas_dataframes.append(uniread.read_unimorph_tsv(l_files.train, family=fam_id, language=lang_id))
            if "dev" in types and l_files.dev is not None:
                pandas_dataframes.append(uniread.read_unimorph_tsv(l_files.dev, family=fam_id, language=lang_id))
            if "test" in types and l_files.test is not None:
                pandas_dataframes.append(uniread.read_unimorph_tsv(l_files.test, family=fam_id, language=lang_id))
        
        #TODO: Optionally this should just get an alphabet of the languages/families requested
        alphabet_inout = self.known_language_collection.get_master_alphabet()
        
        ds = dataset.pandas_to_dataset(
                    pandas_dataframes,
                    tag_converter="bit_vector",
                    alphabet_converter_in=alphabet_inout,
                    alphabet_converter_out=alphabet_inout
        )
        ds.language_collection = self.known_language_collection
        
        dl = dataloader.UnimorphDataLoader(ds,**dataloader_kwargs)
        
        return dl
    

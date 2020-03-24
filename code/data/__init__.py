import glob, os


from .languages import LanguageCollection

class Sigmorphon2020Task0Data(object):
    
    @classmethod
    def get_instance():
        pass

    def __init__(self, dir):
        self.dir = dir
        self.known_languages = LanguageCollection.get_default("full")
        self.available_languages = list()
        
        
        #process the directory to find what languages are available.
        available_training = [os.path.basename(i) for i in glob.glob(os.path.join(dir, "*", "*.trn")) ]
        available_dev = glob.glob(os.path.join(dir, "*", "*.dev"))
        available_test = list() #glob.glob(os.path.join(dir, "*", "*.tst"))
        
        

import codecs
import consts
import re
import os
import torch
import yaml

TRAIN_MODE = 'trn'
DEV_MODE = 'dev'
TEST_MODE = 'tst'
PADDING_TOKEN = 99
START_TOKEN = 98
END_TOKEN = 97
int2category = None
category2int = None
tag2category = None
tag2int = None
char2int = {}
int2char = {}


def create_dataset(root_dir, mode):
    datasets = []
    for family in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, family)) and re.match(r'[a-zA-Z-]*', family):
            datasets.append(create_dataset_from_family(root_dir, family, mode))
    return torch.utils.data.dataset.ChainDataset(datasets)


def create_dataset_from_family(root_dir, family, mode):
    datasets = []
    for language in os.listdir(os.path.join(root_dir, family)):
        match = re.match(r'([a-zA-Z-]*)' + r'\.' + mode, language)
        if match:
            datasets.append(create_dataset_from_language(root_dir, family, match.group(1), mode=mode))
    return torch.utils.data.dataset.ChainDataset(datasets)


def create_dataset_from_language(root_dir, family, language, mode):
    global char2int, int2char
    file_name = os.path.join(root_dir, family, language + '.' + mode)
    is_train = (mode != TEST_MODE)
    dataset = UnimorphDataset(file_name, family=family, language=language, is_train=is_train)
    if language not in char2int:
        alphabet = set()
        if is_train:
            for stem, _, _, _, target in dataset:
                alphabet = alphabet.union(stem + target)
        else:
            for stem, _, _, _ in dataset:
                alphabet = alphabet.union(stem)
        int2char[language] = list(alphabet)
        char2int[language] = {c: i for i, c in enumerate(alphabet)}
    return dataset


def load_scheme(root_dir):
    global int2category, category2int, tag2category, tag2int
    with open(os.path.join(root_dir, 'tags.yaml'), 'r') as file:
        content = yaml.safe_load(file)
        int2category = list(content[consts.ORDERING])
        category2int = {c: i for i, c in enumerate(content[consts.ORDERING])}
        tag2category = {tag: category
                        for category in content[consts.ORDERING]
                        for tag in content[consts.CATEGORIES][category]}
        tag2int = {tag: i + 1
                   for category in content[consts.ORDERING]
                   for i, tag in enumerate(content[consts.CATEGORIES][category])}


def embedding(word, language):
    global char2int
    return torch.Tensor([START_TOKEN] + [char2int[language][c] for c in word] + [END_TOKEN])


def reverse_embedding(embedded, language):
    global int2char
    text = []
    for i, e in enumerate(embedded):
        if e == START_TOKEN and i == 0:
            continue
        if e == END_TOKEN:
            break
        if e >= len(int2char[language]):
            text.append('<{}>'.format(e))
        else:
            text.append(int2char[language][e])
    return ''.join(text)


def decode_tags(tags):
    global category2int, tag2int, tag2category
    decoded = torch.zeros(len(category2int))
    for tag in tags.split(';'):
        decoded[category2int[tag2category[tag]]] = tag2int[tag]
    return decoded


class UnimorphDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_name, family=None, language=None, is_train=True):
        super(torch.utils.data.IterableDataset).__init__()
        self.family = family
        self.language = language
        self.file_name = file_name
        self.is_train = is_train

    def __iter__(self):
        with codecs.open(self.file_name, 'r', 'utf-8') as file:
            if self.is_train:
                return iter([(line.strip().split('\t')[0].strip(),
                              line.strip().split('\t')[2].strip(),
                              self.family,
                              self.language,
                              line.strip().split('\t')[1].strip()) for line in file.readlines()])
            else:
                return iter([(line.strip().split('\t')[0].strip(),
                              line.strip().split('\t')[1].strip(),
                              self.family,
                              self.language) for line in file.readlines()])

    def __len__(self):
        with codecs.open(self.file_name, 'r', 'utf-8') as file:
            return len(file.readlines())

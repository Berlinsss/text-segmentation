from torch.utils.data import Dataset
from text_manipulation import word_model
from text_manipulation import extract_sentence_words
from pathlib2 import Path
import re
import wiki_utils
import os

import utils

logger = utils.setup_logger(__name__, 'train.log')

section_delimiter = "========"


def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files


def get_cache_path(wiki_folder):
    cache_file_path = wiki_folder / 'paths_cache'
    return cache_file_path


def cache_wiki_filenames(wiki_folder):
    cache_file_path = get_cache_path(wiki_folder)
    file_paths = []

    for filename in os.listdir(str(wiki_folder)):
        file_path = os.path.join(str(wiki_folder), filename)

        if os.path.isfile(file_path):
            file_paths.append(file_path)

    for path in file_paths:
        print(path)

    with cache_file_path.open('w+') as f:
        for file in file_paths:
            f.write(unicode(file) + u'\n')

#
# def clean_section(section):
#     cleaned_section = section.strip('\n')
#     return cleaned_section


# def get_scections_from_text(txt, high_granularity=True):
#     sections_to_keep_pattern = wiki_utils.get_seperator_foramt() if high_granularity else wiki_utils.get_seperator_foramt(
#         (1, 2))
#     if not high_granularity:
#         # if low granularity required we should flatten segments within segemnt level 2
#         pattern_to_ommit = wiki_utils.get_seperator_foramt((3, 999))
#         txt = re.sub(pattern_to_ommit, "", txt)
#
#         #delete empty lines after re.sub()
#         sentences = [s for s in txt.strip().split("\n") if len(s) > 0 and s != "\n"]
#         txt = '\n'.join(sentences).strip('\n')
#
#
#     all_sections = re.split(sections_to_keep_pattern, txt)
#     non_empty_sections = [s for s in all_sections if len(s) > 0]
#
#     return non_empty_sections


def get_text(path):
    file = open(str(path), "r")
    raw_content = file.read()
    file.close()

    clean_txt = raw_content.decode('utf-8').strip()

    return clean_txt


def read_wiki_file(path, word2vec, remove_special_tokens=False):
    data = []
    targets = []

    text = get_text(path)
    text_words = extract_sentence_words(text, remove_special_tokens=remove_special_tokens)
    if len(text_words) >= 1:
        data.append([word_model(word, word2vec) for word in text_words])
    else:
        # raise ValueError('Sentence in wikipedia file is empty')
        logger.info('text in wikipedia file is empty')

    return data, targets, path


class InputDataSet(Dataset):
    def __init__(self, root, word2vec):

        root_path = Path(root)
        cache_path = get_cache_path(root_path)
        if not cache_path.exists():
            cache_wiki_filenames(root_path)
        self.textfiles = cache_path.read_text().splitlines()

        if len(self.textfiles) == 0:
            raise RuntimeError('Found 0 images in subfolders of: {}'.format(root))

        self.root = root
        self.word2vec = word2vec

    def __getitem__(self, index):
        path = self.textfiles[index]

        return read_wiki_file(Path(path), self.word2vec, remove_special_tokens=True)

    def __len__(self):
        return len(self.textfiles)

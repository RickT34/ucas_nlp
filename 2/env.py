from tqdm import tqdm
import torch
from pathlib import Path
import itertools as it

DIR_DATA = Path('data')
PATH_CORPUS_RAW = DIR_DATA / "ChineseCorpus199801.txt"
PATH_DICT = DIR_DATA / "dictionary.bin"
PATH_CORPUS = DIR_DATA / "corpus.bin"
PATH_WORDS = DIR_DATA / "words.bin"
PATH_WORDS_COUNT = DIR_DATA / "wordscount.bin"

DICT_COUNT = 2048

DEVICE = 'cuda'
TOK_PAD = 0
TOK_UNK = 1
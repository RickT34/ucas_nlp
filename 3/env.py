from tqdm import tqdm
from pathlib import Path
import itertools as it

DIR_DATA = Path('data')
PATH_CORPUS_RAW = DIR_DATA / "ChineseCorpus199801.txt"
PATH_DATASET = DIR_DATA / "dataset.pkl"

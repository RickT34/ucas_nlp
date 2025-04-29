from curses.ascii import isalpha
from pathlib import Path
from tqdm import tqdm
import re

INDOR = Path('data/arxivpapers/')

pat = re.compile(r"[a-zA-Z']+")
res = []
for i, file in enumerate(tqdm(list(INDOR.glob('*.txt')))):
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
        text = text.replace('-\n', '')
        res.append(' '.join(pat.findall(text)))

PATH = Path('data/cleaned_eng.txt')
PATH.write_text('\n'.join(res))
PATH = Path('data/cleaned_eng_lower.txt')
PATH.write_text('\n'.join(res).lower())

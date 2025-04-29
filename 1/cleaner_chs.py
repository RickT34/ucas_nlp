from pathlib import Path
import pickle
from tqdm import tqdm
import jieba
import re

is_chinese = lambda x: "\u4e00" <= x <= "\u9fa5"

INDOR = Path('data/ppnews/')

res = []
for i, file in tqdm(enumerate(INDOR.glob('*.pkl'))):
    if i > 10:
        break
    l = pickle.load(open(file, 'rb'))
    seg_lists = (jieba.cut(t) for t in l)
    f = lambda x:' '.join(' '.join(w for w in seg_list if all(map(is_chinese, w))) for seg_list in seg_lists)
    res.append(' '.join(map(f, seg_lists)))
    

PATH = Path('data/cleaned_chs.txt')
PATH.write_text('\n'.join(res))

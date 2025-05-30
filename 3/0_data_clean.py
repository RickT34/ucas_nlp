from env import *
import pickle

texts = []
with open(PATH_CORPUS_RAW, "r", encoding="utf-8") as f:
    for line in f.readlines():
        s = line.split()
        texts.append(tuple(t.split("/")[0] for t in s[1:]))

pickle.dump(
    list(texts),
    open(PATH_DATASET, "wb")
)

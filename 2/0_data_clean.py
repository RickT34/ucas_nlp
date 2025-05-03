from env import *

texts = []
with open(PATH_CORPUS_RAW, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        s = line.split()
        texts.append(tuple(t.split('/')[0] for t in s[1:]))

words = list({w for w in it.chain(*texts)})
words2id = {w:i for i,w in enumerate(words)}

wordcount = torch.zeros(len(words), dtype=torch.long)

for c in texts:
    for i in c:
        wordcount[words2id[i]]+=1
        
wordsid_filtered = wordcount.argsort(descending=True)[:DICT_COUNT-2].tolist()

words_filtered = ['<PAD>', '<UNK>'] + sorted(words[i] for i in wordsid_filtered)
words2id_filtered = {w:i for i,w in enumerate(words_filtered)}
corpus = list(map(
    lambda l:torch.tensor(
        list(words2id_filtered.get(w, TOK_UNK) for w in l), dtype=torch.long
    ),
    texts
))

torch.save(words_filtered, PATH_WORDS)
torch.save(wordcount, PATH_WORDS_COUNT)
torch.save(words2id_filtered, PATH_DICT)
torch.save(corpus, PATH_CORPUS)
print("Finished")

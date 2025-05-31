from env import PATH_DATASET
import pickle
import random

dataset = pickle.load(PATH_DATASET.open("rb"))
random.seed(42)
random.shuffle(dataset)

n = 10

for sample in dataset[-n:]:
    print(f"输入: {''.join(sample)}")
    print(f"输出: {' '.join(sample)}")
    print()

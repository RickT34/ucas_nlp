from openai import OpenAI
import os
import pickle
from env import *
import random
from argpipe import pipewarp, Batch, debug

MODEL_NAME = "qwen-turbo-latest"


@pipewarp
def get_client():
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return "client", client


@pipewarp
def ask_llm(system_prompt: str, prompt: str, model_name: str, client: OpenAI):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return "answers", [completion.choices[0].message.content]


dataset = None


@pipewarp
def get_data_samples(samples_num: int):
    global dataset
    if dataset is None:
        dataset = pickle.load(PATH_DATASET.open("rb"))
        random.shuffle(dataset)
    assert samples_num <= len(dataset), "Not enough data samples"
    res = dataset[:samples_num]
    dataset = dataset[samples_num:]
    return "samples", res


@pipewarp
def perpare_prompt(sample: list[str]):
    return "prompt", "".join(sample)


def compute_acc(samples: list, answers: list, method):
    assert len(samples) == len(answers), "Number of samples and answers do not match"
    score = 0
    for s, a in zip(samples, answers):
        score += method(s, a)
    return "accuracy", score / len(samples), score, len(samples)


exp = (
    get_client
    | get_data_samples
    | Batch({"samples": "sample"}, perpare_prompt | ask_llm)
    | compute_acc
)


print(
    exp.exec(
        samples_num=3,
        model_name=MODEL_NAME,
        system_prompt="进行句子分词, 使用空格进行分隔. 请只输出分词结果.",
        method=lambda s, a: sum(1 for x, y in zip(s, a.split()) if x == y),
    )
)

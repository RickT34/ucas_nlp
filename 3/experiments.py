from openai import OpenAI
import os
import pickle
from env import *
import random
from argpipe import pipewarp, Batch, debug
import json
import yaml
import argparse


@pipewarp
def get_config(config_file: Path):
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["system_prompt"] = open(
        "prompts/" + config["system_prompt"], "r", encoding="utf-8"
    ).read()
    return config


@pipewarp
def get_client():
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return "client", client


def ask_llm(system_prompt: str, prompt: str, model_name: str, client: OpenAI):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return "llm_answer", completion.choices[0].message.content


def get_data_samples(samples_num: int):
    dataset = pickle.load(PATH_DATASET.open("rb"))
    random.shuffle(dataset)
    return "samples", dataset[:samples_num]


@pipewarp
def prepare_prompt(sample: list[str]):
    return "prompt", "".join(sample)


def answer_postprocess(llm_answer: str):
    return "answer", llm_answer.strip().split("\n")[-1].split(" ")


def compute_scores(sample: list[str], answer: list[str]):
    def _trans_to_class(s: list[str]):
        return "1".join("0" * (len(i) - 1) for i in s)

    matched = "".join(sample) == "".join(answer)
    acc = 0
    precision = 0
    recall = 0
    f1 = 0
    if matched:
        s = _trans_to_class(sample)
        a = _trans_to_class(answer)
        mch = lambda x, y: sum(1 for i, j in zip(s, a) if i == x and j == y)
        TP = mch("1", "1")
        TN = mch("0", "0")
        FP = mch("0", "1")
        FN = mch("1", "0")
        try:
            acc = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            print(f"Warning: division by zero: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
            print(f"Sample: {sample}, Answer: {answer}")
            matched = False
    return "result", {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": matched,
    }


def summery(results: list[dict]):
    accs = [r["accuracy"] for r in results if r["matched"]]
    precisions = [r["precision"] for r in results if r["matched"]]
    recalls = [r["recall"] for r in results if r["matched"]]
    f1s = [r["f1"] for r in results if r["matched"]]
    return "summery", {
        "accuracy": sum(accs) / len(accs),
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1s) / len(f1s),
        "matched": len(accs) / len(results),
    }


def save_result(
    model_name: str,
    system_prompt: str,
    samples: list,
    answers: list,
    answers_raw: list,
    results: list,
    summery: dict,
    output_dir: str,
    config_file: Path,
):
    d = {"summery": summery, "model_name": model_name, "system_prompt": system_prompt}
    d["raw"] = list({"sample": s, "answer": a, "result": r, "answer_raw": a_raw} for s, a, r, a_raw in zip(samples, answers, results, answers_raw))  # type: ignore
    file = output_dir + f"/result_{config_file.name}_{len(samples)}.json"
    json.dump(d, open(file, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    print(f"Result saved to {file}")


if __name__ == "__main__":

    workflow = (
        get_client  # 获取客户端
        | get_data_samples  # 获取数据集
        | Batch(  # 分别对每一种模型/提示词进行实验
            fork={"config_files": "config_file"},
            func=get_config  # 获取配置文件
            | Batch(  # 对数据集中每一个样本进行处理
                fork={"samples": "sample"},
                func=prepare_prompt  # 制作提示词
                | ask_llm  # 调用语言模型进行回答
                | answer_postprocess  # 处理回答
                | compute_scores,  # 计算得分
                gather={
                    "results": "result",
                    "answers": "answer",
                    "answers_raw": "llm_answer",
                },
            )
            | summery  # 总结得分
            | save_result,  # 保存结果
        )
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, required=True)
    parser.add_argument("--samples_num", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    config_files = list(Path(args.config_dir).glob("*.yaml"))

    os.makedirs(args.output_dir, exist_ok=True)

    workflow.exec(
        config_files=config_files,
        output_dir=args.output_dir,
        samples_num=args.samples_num,
    )

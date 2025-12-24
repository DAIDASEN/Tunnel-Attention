#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1 (LongBench 版): Qwen3 小模型长上下文痛点验证（基于 API）

核心变化：
  - 不再使用随机噪声 + needle，而是直接用官方 LongBench / LongBench-E 数据集。
  - 默认选 3 个典型长上下文任务：
      * passage_retrieval_en_e  （Synthetic Retrieval, Accuracy）
      * passage_count_e          （Synthetic Counting, Accuracy）
      * multifieldqa_en_e        （Single-doc QA, F1）

依赖：
  pip install openai python-dotenv tqdm pandas datasets

环境变量：
  DASHSCOPE_API_KEY: 你的 DashScope API key
  QWEN_BASE_URL:     可选，默认为 https://dashscope.aliyuncs.com/compatible-mode/v1
"""

import os
import time
import argparse
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import re
import string as pystring

# ============== 基础配置 ==============

load_dotenv(override=True)

DEFAULT_BASE_URL = os.getenv(
    "QWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DEFAULT_BASE_URL,
)

# 你也可以在命令行里覆盖
DEFAULT_MODELS = [
    "qwen3-4b",          # 小模型
    "qwen3-30b-a3b",     # 稍大模型，作对照
    "qwen3-235b-a22b",   # 更大模型，作对照
]

# 默认评估的 LongBench(-E) 子任务
DEFAULT_TASKS = [
    "passage_retrieval_en_e",
    "passage_count_e",
    "multifieldqa_en_e",
]


# ============== 通用工具函数 ==============

def call_qwen_chat(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 64,
    temperature: float = 0.0,
    extra_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    统一封装一次 chat.completions 调用，自动重试几次。
    """
    if extra_params is None:
        extra_params = {}

    extra_body = {
        "enable_thinking": False
    }

    def is_data_inspection_failed(err: Exception) -> bool:
        """
        DashScope / Qwen 兼容 OpenAI 接口会在输入触发安全审核时返回：
          code: data_inspection_failed
        这类错误属于“不可重试”，重试只会浪费时间/额度。
        """
        msg = f"{err}"
        return ("data_inspection_failed" in msg) or ("inappropriate content" in msg)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body,
                **extra_params,
            )
            content = resp.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            if is_data_inspection_failed(e):
                print(f"[WARN] 命中内容安全审核，跳过该样本（{model}）: {e}")
                raise
            print(f"[WARN] API 调用失败（{model}, 尝试 {attempt+1}/3）: {e}")
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"API 调用连续失败 3 次，模型={model}")


def normalize_answer(s: str) -> str:
    """
    文本归一化，用于 F1 / EM 度量：
      - 小写
      - 去掉标点
      - 去掉英文冠词
      - 压缩空白
    """
    if s is None:
        return ""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in pystring.punctuation)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score_single(prediction: str, ground_truth: str) -> float:
    """
    单个 gold answer 的 token-level F1（SQuAD 风格）。
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_f1_over_list(prediction: str, gold_list: List[str]) -> float:
    """
    对多个 gold，取最大 F1。
    """
    if not gold_list:
        return 0.0
    return max(f1_score_single(prediction, g) for g in gold_list)


def exact_match_score(prediction: str, gold_list: List[str]) -> float:
    """
    EM：normalized prediction 是否等于任一 normalized gold。
    """
    if prediction is None:
        return 0.0
    norm_pred = normalize_answer(prediction)
    for g in gold_list:
        if norm_pred == normalize_answer(str(g)):
            return 1.0
    return 0.0


def length_bucket(length_val: int) -> str:
    """
    LongBench 的 length 字段是字符数/词数，我们粗略按照 0-4k / 4k-8k / 8k+ 分桶。
    """
    if length_val < 4000:
        return "0-4k"
    elif length_val < 8000:
        return "4k-8k"
    else:
        return "8k+"


def infer_task_type(dataset_name: str) -> str:
    """
    根据 LongBench 的 dataset 名字粗略判断任务类型，用于决定 metric 和提示模板。

    这里 dataset_name 可以是 'passage_count_e'，我们只去掉末尾的 '_e'（LongBench-E 后缀）。
    """
    base = dataset_name[:-2] if dataset_name.endswith("_e") else dataset_name

    if base in ["passage_retrieval_en", "passage_retrieval_zh"]:
        return "synthetic_retrieval"
    if base == "passage_count":
        return "synthetic_count"
    if base in ["trec", "lsht"]:
        return "classification"
    if base in ["gov_report", "qmsum", "multi_news", "vcsum", "samsum"]:
        return "summarization"
    if base in ["lcc", "repobench-p"]:
        return "code"
    # 默认按 QA 处理
    return "qa"


def build_user_content(example: Dict[str, Any], task_type: str) -> str:
    """
    把 LongBench 的 {context, input} 拼成给 LLM 的 user prompt，
    并根据任务类型加一点说明。
    """
    ctx = example.get("context", "")
    inp = example.get("input", "")
    all_classes = example.get("all_classes", None)

    if task_type == "classification" and all_classes:
        label_list = ", ".join(map(str, all_classes))
        instructions = (
            "You must classify the input into one of the given labels.\n"
            f"Possible labels are: {label_list}.\n"
            "Answer with EXACTLY one label from the list.\n"
        )
    elif task_type == "synthetic_count":
        instructions = (
            "You must determine how many DISTINCT passages appear in the text.\n"
            "Answer with ONLY an integer, e.g., 3.\n"
        )
    elif task_type == "synthetic_retrieval":
        instructions = (
            "You must decide which passage the query refers to, "
            "based on the context above.\n"
            "Your answer should be exactly the name or identifier of that passage "
            "as it appears in the context.\n"
        )
    elif task_type == "summarization":
        instructions = (
            "Write a concise and accurate answer for the task below, "
            "based only on the given context.\n"
        )
    else:  # qa / code
        instructions = (
            "Answer the question based only on the context above. "
            "If the answer is not in the context, say you don't know.\n"
        )

    user_content = (
        "You are given a long context and a task.\n\n"
        "[CONTEXT START]\n"
        f"{ctx}\n"
        "[CONTEXT END]\n\n"
        f"Task: {inp}\n\n"
        f"{instructions}"
    )
    return user_content


def score_example(
    task_type: str,
    prediction: str,
    gold_answers: List[str],
) -> Tuple[float, str]:
    """
    返回 (score, metric_name)，
    - QA / Summarization：F1
    - Synthetic / 分类：Accuracy (EM)
    """
    if task_type in ["qa", "summarization"]:
        return max_f1_over_list(prediction, gold_answers), "f1"
    if task_type in ["synthetic_retrieval", "synthetic_count", "classification"]:
        return exact_match_score(prediction, gold_answers), "accuracy"
    # 其他类型默认 F1
    return max_f1_over_list(prediction, gold_answers), "f1"


# ============== 结果结构 ==============

@dataclass
class LongBenchResult:
    experiment: str              # 固定填 "longbench_phase1"
    model: str
    dataset: str                 # e.g. passage_retrieval_en_e
    task_type: str               # qa / synthetic_retrieval / synthetic_count / ...
    example_id: str
    context_length: int
    length_bucket: str           # 0-4k / 4k-8k / 8k+
    gold_answers: str            # 用 " ||| " 拼接
    prediction: str
    metric: str                  # "f1" or "accuracy"
    score: float                 # 单样本分数


# ============== LongBench 主实验 ==============

def run_longbench_phase1(
    models: List[str],
    tasks: List[str],
    max_examples_per_task: int,
    output_csv: str,
    trust_remote_code: bool = True,
) -> None:
    """
    在指定的 LongBench / LongBench-E 任务上，对多个 Qwen3 模型做评估。

    每个 (model, dataset)：
      - 从 test split 中随机抽 max_examples_per_task 条样本
      - 调用 API 推理
      - 用简单的 F1/EM 评价
      - 输出到 CSV
    """
    results: List[LongBenchResult] = []
    skipped_due_to_safety = 0

    # 为了不同任务/模型之间可复现，统一一个随机种子
    rng = random.Random(42)

    try:
        for dataset_name in tasks:
            print(f"\n=== 加载 LongBench 任务: {dataset_name} ===")
            # 按官方文档：load_dataset('THUDM/LongBench', dataset_name, split='test')
            # 注意：需要较新的 datasets 版本 (pip install -U datasets)
            try:
                lb_data = load_dataset(
                    "THUDM/LongBench",
                    dataset_name,
                    split="test",
                    trust_remote_code=trust_remote_code,
                )
            except ValueError as e:
                msg = str(e)
                if "trust_remote_code" in msg:
                    raise ValueError(
                        "加载 THUDM/LongBench 失败：该数据集包含自定义代码，需要允许执行。"
                        "请使用 --trust_remote_code（默认已开启），或检查你的 datasets 版本。"
                    ) from e
                raise

            n_total = len(lb_data)
            if max_examples_per_task is not None and max_examples_per_task > 0:
                k = min(max_examples_per_task, n_total)
                indices = list(range(n_total))
                rng.shuffle(indices)
                selected_indices = indices[:k]
            else:
                selected_indices = list(range(n_total))

            task_type = infer_task_type(dataset_name)
            default_metric = "accuracy" if task_type in (
                "synthetic_retrieval",
                "synthetic_count",
                "classification",
            ) else "f1"

            print(
                f"  任务类型: {task_type}, 预期 metric: {default_metric}, "
                f"样本数: {len(selected_indices)} / {n_total}"
            )

            for model in models:
                print(f"\n---- 评估模型 {model} on {dataset_name} ----")
                for idx in tqdm(selected_indices, desc=f"{dataset_name}-{model}"):
                    example = lb_data[int(idx)]

                    user_content = build_user_content(example, task_type)
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a careful and precise assistant. "
                                "Always base your answer strictly on the given context."
                            ),
                        },
                        {"role": "user", "content": user_content},
                    ]

                    try:
                        pred = call_qwen_chat(
                            model=model,
                            messages=messages,
                            max_tokens=64,
                            temperature=0.0,
                        )
                    except Exception as e:
                        msg = f"{e}"
                        if ("data_inspection_failed" in msg) or ("inappropriate content" in msg):
                            skipped_due_to_safety += 1
                        print(
                            f"[WARN] API error: model={model}, "
                            f"dataset={dataset_name}, idx={idx}: {e}"
                        )
                        pred = ""

                    # LongBench: answers 是 "all true answers" 的 list
                    ans_field = example.get("answers", [])
                    if isinstance(ans_field, list):
                        gold_answers = [str(a) for a in ans_field]
                    else:
                        gold_answers = [str(ans_field)]

                    score, metric_name = score_example(
                        task_type=task_type,
                        prediction=pred,
                        gold_answers=gold_answers,
                    )

                    length_val = int(example.get("length", 0))
                    bucket = length_bucket(length_val)
                    ex_id = example.get("_id", str(idx))

                    results.append(
                        LongBenchResult(
                            experiment="longbench_phase1",
                            model=model,
                            dataset=dataset_name,
                            task_type=task_type,
                            example_id=str(ex_id),
                            context_length=length_val,
                            length_bucket=bucket,
                            gold_answers=" ||| ".join(gold_answers),
                            prediction=pred,
                            metric=metric_name,
                            score=score,
                        )
                    )
    except KeyboardInterrupt:
        print("\n[INFO] 收到 Ctrl-C，准备保存已完成的结果并退出……")

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[LongBench Phase1] 结果已保存到 {output_csv}")
    if skipped_due_to_safety > 0:
        print(f"[INFO] 因内容安全审核跳过/置空的样本数（累计）: {skipped_due_to_safety}")

    # 简单打印下按 model / dataset 聚合的平均分，帮助快速 sanity check
    print("\n=== 简单汇总（按模型 & 数据集平均） ===")
    if len(df) == 0:
        print("(empty)")
    else:
        print(df.groupby(["model", "dataset"])["score"].mean())


# ============== CLI 入口 ==============

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 (LongBench): Qwen3 小模型长上下文痛点实验（API 版）"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help="要评估的模型名称列表（API 模型 ID）",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=DEFAULT_TASKS,
        help=(
            "LongBench/LongBench-E 子任务名字列表，"
            "例如：passage_retrieval_en_e passage_count_e multifieldqa_en_e"
        ),
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=100,
        help="每个 (模型, 任务) 最多评估的样本数（从 test 中随机采样）",
    )
    # 兼容你说的“epochs”：本脚本不训练，epochs 在这里等价于每个(模型,任务)评测样本数
    parser.add_argument(
        "--epochs",
        type=int,
        dest="max_examples",
        help="同 --max_examples（评测样本数）。脚本不训练，无真实 epoch 概念。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="phase1_longbench_results.csv",
        help="结果保存的 CSV 文件路径",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="允许 datasets 执行 HF 数据集仓库中的自定义加载代码（THUDM/LongBench 需要）。默认开启。",
    )
    parser.add_argument(
        "--no_trust_remote_code",
        action="store_false",
        dest="trust_remote_code",
        help="禁用 trust_remote_code（安全更严格，但 THUDM/LongBench 通常无法加载）。",
    )
    args = parser.parse_args()

    run_longbench_phase1(
        models=args.models,
        tasks=args.tasks,
        max_examples_per_task=args.max_examples,
        output_csv=args.output,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()

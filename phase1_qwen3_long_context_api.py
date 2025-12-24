#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1: Qwen3 小模型长上下文痛点验证（基于 API）

包含三个实验：
  1) 长度扫描：accuracy vs context length
  2) 位置扫描：Lost-in-the-middle
  3) 检索 vs 推理：能找不能推

依赖：
  pip install openai python-dotenv tqdm pandas

环境变量：
  DASHSCOPE_API_KEY: 你的 DashScope API key
  QWEN_BASE_URL:     可选，默认为 https://dashscope.aliyuncs.com/compatible-mode/v1     
"""
import os
import time
import random
import string
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============== 基础配置 ==============

# 加载 .env 文件
load_dotenv(override=True)

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 修复：移除末尾空格

# 从环境变量读取 API key，不再硬编码
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DEFAULT_BASE_URL,
)

# 你实际用的 Qwen3 小模型名字（请到控制台 / 文档里确认）
DEFAULT_MODELS = [
    "qwen3-4b",           # 小模型
    "qwen3-30b-a3b",        # 稍大模型，作对照
    "qwen3-235b-a22b",      # 更大模型，作对照
]


# ============== 工具函数：API 调用 & 文本生成 ==============

def call_qwen_chat(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 32,
    temperature: float = 0.0,
    extra_params: Optional[Dict[str, Any]] = None,
) -> str:
    """统一封装一次 chat.completions 调用，自动重试几次。"""
    if extra_params is None:
        extra_params = {}
    
    # Qwen3 模型需要设置 enable_thinking 参数，但需要通过 extra_body 传递
    extra_body = {
        "enable_thinking": False
    }

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body,  # 关键：通过 extra_body 传递
                **extra_params,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[WARN] API 调用失败（{model}, 尝试 {attempt+1}/3）: {e}")
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"API 调用连续失败 3 次，模型={model}")

# 简单随机文本生成，用英文单词凑长上下文
# 你后面可以换成从真实 corpus 采样

def random_word() -> str:
    length = random.randint(3, 10)
    return "".join(random.choices(string.ascii_lowercase, k=length))


def random_sentence(min_words=8, max_words=16) -> str:
    n = random.randint(min_words, max_words)
    words = [random_word() for _ in range(n)]
    s = " ".join(words)
    return s.capitalize() + "."


def random_paragraph(target_words: int) -> str:
    """生成大约 target_words 个词的随机段落。"""
    words = 0
    sentences = []
    while words < target_words:
        sent = random_sentence()
        w = len(sent.split())
        sentences.append(sent)
        words += w
    return " ".join(sentences)


def build_context_with_single_needle(
    total_words: int,
    needle_sentence: str,
    position_fraction: float,
) -> str:
    """
    构造一个长上下文，在大约 total_words 个词中插入 needle_sentence。

    position_fraction ∈ (0, 1): 针的位置在整体中的相对位置。
    """
    assert 0.0 < position_fraction < 1.0

    # 预留 needle 句子自己的长度
    needle_words = len(needle_sentence.split())
    remaining_words = max(total_words - needle_words, needle_words * 2)

    before_words = int(remaining_words * position_fraction)
    after_words = remaining_words - before_words

    before = random_paragraph(before_words)
    after = random_paragraph(after_words)

    # 在中间插入 needle
    context = f"{before} {needle_sentence} {after}"
    return context


def extract_first_integer(text: str) -> Optional[int]:
    """从模型输出里提取第一个整数，找不到就返回 None。"""
    import re
    m = re.search(r"-?\d+", text)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


# ============== 数据结构：记录每个样本结果 ==============

@dataclass
class ResultRow:
    experiment: str
    model: str
    context_len_tokens: int
    position_fraction: float
    example_id: int
    task_type: str  # "retrieval" / "reasoning"
    gold_answer: str
    model_answer_raw: str
    model_answer_parsed: str
    correct: int


# ============== 实验 1：长度扫描（Length Sweep） ==============

def run_length_sweep(
    models: List[str],
    target_token_lengths: List[int],
    num_examples_per_len: int,
    output_csv: str,
) -> None:
    """
    实验 1：在不同（近似）上下文长度下测 Needle-in-a-Haystack 检索准确率。

    用 approximate: tokens ≈ words / 0.7
    """
    results: List[ResultRow] = []

    # 非常粗略的 token ↔ words 换算
    approx_words_per_token = 0.7

    for model in models:
        print(f"\n=== [Length Sweep] 模型: {model} ===")
        for L_tokens in target_token_lengths:
            target_words = int(L_tokens * approx_words_per_token)
            print(f"\n  - 目标上下文长度 ≈ {L_tokens} tokens (~{target_words} words)")

            for ex_id in tqdm(range(num_examples_per_len), desc=f"{model} L={L_tokens}"):
                # 随机挑一个 6 位整数作为密码
                secret_code = random.randint(100000, 999999)
                needle_sentence = f"The secret code is {secret_code}."

                # 随机位置插入 needle（避免所有都固定在中间）
                pos = random.uniform(0.1, 0.9)
                context = build_context_with_single_needle(
                    total_words=target_words,
                    needle_sentence=needle_sentence,
                    position_fraction=pos,
                )

                user_content = (
                    "You will be given a long text. Read it carefully.\n\n"
                    f"TEXT START\n{context}\nTEXT END\n\n"
                    "Question: According to the text above, what is the secret code?\n"
                    "Answer with ONLY the 6-digit number."
                )

                messages = [
                    {"role": "system", "content": "You are an information extraction assistant."},
                    {"role": "user", "content": user_content},
                ]

                resp = call_qwen_chat(
                    model=model,
                    messages=messages,
                    max_tokens=16,
                    temperature=0.0,
                )

                parsed = extract_first_integer(resp)
                correct = int(parsed == secret_code)

                row = ResultRow(
                    experiment="length_sweep",
                    model=model,
                    context_len_tokens=L_tokens,
                    position_fraction=pos,
                    example_id=ex_id,
                    task_type="retrieval",
                    gold_answer=str(secret_code),
                    model_answer_raw=resp,
                    model_answer_parsed=str(parsed) if parsed is not None else "",
                    correct=correct,
                )
                results.append(row)

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[Length Sweep] 结果已保存到 {output_csv}")
    print(df.groupby(["model", "context_len_tokens"])["correct"].mean())


# ============== 实验 2：位置扫描（Lost in the Middle） ==============

def run_position_sweep(
    models: List[str],
    fixed_context_tokens: int,
    positions: List[float],
    num_examples_per_pos: int,
    output_csv: str,
) -> None:
    """
    实验 2：在固定长度的上下文下，改变答案位置，观察准确率 vs 位置。

    复现 Lost-in-the-Middle 现象。
    """
    results: List[ResultRow] = []
    approx_words_per_token = 0.7
    target_words = int(fixed_context_tokens * approx_words_per_token)

    for model in models:
        print(f"\n=== [Position Sweep] 模型: {model}, L≈{fixed_context_tokens} tokens ===")
        for pos in positions:
            assert 0.0 < pos < 1.0
            print(f"\n  - 答案位置相对比例: {pos:.2f}")

            for ex_id in tqdm(range(num_examples_per_pos), desc=f"{model} pos={pos:.2f}"):
                secret_code = random.randint(100000, 999999)
                needle_sentence = f"The secret code is {secret_code}."

                context = build_context_with_single_needle(
                    total_words=target_words,
                    needle_sentence=needle_sentence,
                    position_fraction=pos,
                )

                user_content = (
                    "You will be given a long text. Read it carefully.\n\n"
                    f"TEXT START\n{context}\nTEXT END\n\n"
                    "Question: According to the text above, what is the secret code?\n"
                    "Answer with ONLY the 6-digit number."
                )

                messages = [
                    {"role": "system", "content": "You are an information extraction assistant."},
                    {"role": "user", "content": user_content},
                ]

                resp = call_qwen_chat(
                    model=model,
                    messages=messages,
                    max_tokens=16,
                    temperature=0.0,
                )

                parsed = extract_first_integer(resp)
                correct = int(parsed == secret_code)

                row = ResultRow(
                    experiment="position_sweep",
                    model=model,
                    context_len_tokens=fixed_context_tokens,
                    position_fraction=pos,
                    example_id=ex_id,
                    task_type="retrieval",
                    gold_answer=str(secret_code),
                    model_answer_raw=resp,
                    model_answer_parsed=str(parsed) if parsed is not None else "",
                    correct=correct,
                )
                results.append(row)

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[Position Sweep] 结果已保存到 {output_csv}")
    print(df.groupby(["model", "position_fraction"])["correct"].mean())


# ============== 实验 3：检索 vs 推理 ==============

def build_context_with_three_numbers(
    total_words: int,
    numbers: List[int],
    position_fractions: List[float],
) -> str:
    """
    构造一个长上下文，里面有三条包含数字的事实句，其他都是噪声。
    numbers: [n1, n2, n3]
    position_fractions: 对应每条事实的大致位置
    """
    assert len(numbers) == 3
    assert len(position_fractions) == 3

    # 预估三个事实句长度
    fact_sentences = []
    for i, n in enumerate(numbers):
        fact_sentences.append(
            f"In report {i+1}, the number of widgets is {n}."
        )

    fact_words = sum(len(s.split()) for s in fact_sentences)
    remaining_words = max(total_words - fact_words, fact_words * 2)

    # 把 [0,1] 区间切成几段，分别填噪声+事实
    segments = []
    last_fraction = 0.0
    used_words = 0

    # 为简单起见，按比例分配噪声词数
    for idx, frac in enumerate(position_fractions):
        noise_fraction = max(frac - last_fraction, 0.05)
        noise_words = int(remaining_words * noise_fraction)
        segments.append(("noise", noise_words, None))
        last_fraction = frac

        # 插入一个事实句
        segments.append(("fact", 0, fact_sentences[idx]))

    # 尾部再加点噪声
    tail_noise_words = max(remaining_words - sum(w for t, w, _ in segments if t == "noise"), 0)
    segments.append(("noise", tail_noise_words, None))

    texts = []
    for t, w, s in segments:
        if t == "noise":
            if w > 0:
                texts.append(random_paragraph(w))
        else:
            texts.append(s)

    return " ".join(texts)


def run_retrieval_vs_reasoning(
    models: List[str],
    target_token_lengths: List[int],
    num_examples_per_len: int,
    output_csv: str,
) -> None:
    """
    实验 3：在相同上下文长度下，对比

      - 单一密码检索任务（retrieval）
      - 三个数字求和任务（reasoning）

    看小模型在“能找不能推”上的退化情况。
    """
    results: List[ResultRow] = []
    approx_words_per_token = 0.7

    for model in models:
        print(f"\n=== [Retrieval vs Reasoning] 模型: {model} ===")
        for L_tokens in target_token_lengths:
            target_words = int(L_tokens * approx_words_per_token)
            print(f"\n  - 目标上下文长度 ≈ {L_tokens} tokens (~{target_words} words)")

            # ------- (a) 检索任务：和实验 1 一样 -------
            for ex_id in tqdm(range(num_examples_per_len), desc=f"{model} L={L_tokens} [retrieval]"):
                secret_code = random.randint(100000, 999999)
                needle_sentence = f"The secret code is {secret_code}."

                pos = random.uniform(0.1, 0.9)
                context = build_context_with_single_needle(
                    total_words=target_words,
                    needle_sentence=needle_sentence,
                    position_fraction=pos,
                )

                user_content = (
                    "You will be given a long text. Read it carefully.\n\n"
                    f"TEXT START\n{context}\nTEXT END\n\n"
                    "Question: According to the text above, what is the secret code?\n"
                    "Answer with ONLY the 6-digit number."
                )

                messages = [
                    {"role": "system", "content": "You are an information extraction assistant."},
                    {"role": "user", "content": user_content},
                ]

                resp = call_qwen_chat(
                    model=model,
                    messages=messages,
                    max_tokens=16,
                    temperature=0.0,
                )

                parsed = extract_first_integer(resp)
                correct = int(parsed == secret_code)

                row = ResultRow(
                    experiment="retrieval_vs_reasoning_retrieval",
                    model=model,
                    context_len_tokens=L_tokens,
                    position_fraction=pos,
                    example_id=ex_id,
                    task_type="retrieval",
                    gold_answer=str(secret_code),
                    model_answer_raw=resp,
                    model_answer_parsed=str(parsed) if parsed is not None else "",
                    correct=correct,
                )
                results.append(row)

            # ------- (b) 推理任务：三个数字求和 -------
            for ex_id in tqdm(range(num_examples_per_len), desc=f"{model} L={L_tokens} [reasoning]"):
                nums = [random.randint(10, 99) for _ in range(3)]
                pos_fracs = sorted([
                    random.uniform(0.1, 0.3),
                    random.uniform(0.35, 0.65),
                    random.uniform(0.7, 0.9),
                ])
                context = build_context_with_three_numbers(
                    total_words=target_words,
                    numbers=nums,
                    position_fractions=pos_fracs,
                )

                gold_sum = sum(nums)

                user_content = (
                    "You will be given a long text that contains several reports.\n"
                    "Each report describes the number of widgets.\n\n"
                    f"TEXT START\n{context}\nTEXT END\n\n"
                    "Question: According to the reports in the text, there are exactly three reports "
                    "about widgets: report 1, report 2, and report 3.\n"
                    "What is the sum of the numbers of widgets in these three reports?\n"
                    "Answer with ONLY the integer."
                )

                messages = [
                    {"role": "system", "content": "You are a careful mathematical reasoner."},
                    {"role": "user", "content": user_content},
                ]

                resp = call_qwen_chat(
                    model=model,
                    messages=messages,
                    max_tokens=32,
                    temperature=0.0,
                )

                parsed = extract_first_integer(resp)
                correct = int(parsed == gold_sum)

                row = ResultRow(
                    experiment="retrieval_vs_reasoning_reasoning",
                    model=model,
                    context_len_tokens=L_tokens,
                    position_fraction=-1.0,  # 不适用
                    example_id=ex_id,
                    task_type="reasoning",
                    gold_answer=str(gold_sum),
                    model_answer_raw=resp,
                    model_answer_parsed=str(parsed) if parsed is not None else "",
                    correct=correct,
                )
                results.append(row)

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[Retrieval vs Reasoning] 结果已保存到 {output_csv}")
    print(df.groupby(["model", "context_len_tokens", "task_type"])["correct"].mean())


# ============== CLI 入口 ==============

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Qwen3 小模型长上下文痛点实验（API 版）")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["len_sweep", "pos_sweep", "retrieval_vs_reasoning"],
        help="选择要运行的实验：len_sweep / pos_sweep / retrieval_vs_reasoning",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help="要评估的模型名称列表（API 模型 ID）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="phase1_results.csv",
        help="结果保存的 CSV 文件路径",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=50,
        help="每个长度 / 每个位置 / 每个任务的样本数（API 成本会随之线性增加）",
    )
    args = parser.parse_args()

    models = args.models

    if args.experiment == "len_sweep":
        # 这里给几个典型的目标长度（单位：近似 tokens）
        target_lengths = [4096, 8192, 16384, 32768]
        run_length_sweep(
            models=models,
            target_token_lengths=target_lengths,
            num_examples_per_len=args.num_examples,
            output_csv=args.output,
        )

    elif args.experiment == "pos_sweep":
        fixed_L = 16384  # 你可以改成 32768 看更极端的 Lost-in-the-middle
        positions = [0.05, 0.25, 0.5, 0.75, 0.95]
        run_position_sweep(
            models=models,
            fixed_context_tokens=fixed_L,
            positions=positions,
            num_examples_per_pos=args.num_examples,
            output_csv=args.output,
        )

    elif args.experiment == "retrieval_vs_reasoning":
        target_lengths = [8192, 16384, 32768]
        run_retrieval_vs_reasoning(
            models=models,
            target_token_lengths=target_lengths,
            num_examples_per_len=args.num_examples,
            output_csv=args.output,
        )

    else:
        raise ValueError(f"未知实验类型：{args.experiment}")


if __name__ == "__main__":
    main()
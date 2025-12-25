#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 0: 在本地模型上查看 self-attention 模式（单例 probe 版）

功能：
  1. 构造一段长文本 + needle 句子（The secret code is XXXXXX）。
  2. 用本地 Transformer 模型跑一次前向，把每层每个 head 的 attention 拿出来。
  3. 导出：
      - 指定层/头的 full attention 热力图（PNG）
      - 各 head 对“最后一个 token”的距离分布统计
      - 答案 token 上的 attention mass（测模型是不是真的在看答案位置）

依赖：
  pip install torch transformers matplotlib
"""

import os
import math
import random
import string
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt


# ============== 文本构造：沿用你 Phase1 的 needle 逻辑 ==============

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

    needle_words = len(needle_sentence.split())
    remaining_words = max(total_words - needle_words, needle_words * 2)

    before_words = int(remaining_words * position_fraction)
    after_words = remaining_words - before_words

    before = random_paragraph(before_words)
    after = random_paragraph(after_words)

    context = f"{before} {needle_sentence} {after}"
    return context


# ============== 模型与 attention 采集 ==============

def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
):
    print(f"[INFO] 加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 一般 Qwen / Llama 类模型用这个
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def run_model_with_attentions(
    tokenizer,
    model,
    prompt: str,
    device: str = "cuda",
):
    """
    对给定 prompt 跑一遍模型前向，返回：
      - attentions: 每层的 attention tensor 列表
      - input_ids: 输入 token ids
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        # 很关键：output_attentions=True, use_cache=False
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )

    # outputs.attentions 是长度为 num_layers 的 tuple
    # 每个元素 shape: (batch, num_heads, seq_len, seq_len)
    attentions = outputs.attentions
    return attentions, input_ids


# ============== 工具：定位答案 token 的位置 ==============

def find_subsequence_indices(
    seq: torch.Tensor,
    subseq: torch.Tensor,
) -> Optional[Tuple[int, int]]:
    """
    在 seq 中找 subseq 的起止位置，返回 (start, end_exclusive)。
    找不到返回 None。
    """
    seq_list = seq.tolist()
    sub_list = subseq.tolist()
    n, m = len(seq_list), len(sub_list)
    for i in range(n - m + 1):
        if seq_list[i : i + m] == sub_list:
            return i, i + m
    return None


# ============== 可视化：画 attention 热力图 + 距离分布 ==============

def plot_full_attention_heatmap(
    attn: torch.Tensor,
    out_path: str,
    title: str = "",
    vmax: Optional[float] = None,
):
    """
    attn: (seq_len, seq_len) 单头 or 某层某头
      约定：行 = query，列 = key
    """
    attn_np = attn.cpu().float().numpy()
    if vmax is None:
        vmax = attn_np.max()

    plt.figure(figsize=(6, 5))
    plt.imshow(attn_np, aspect="auto", interpolation="nearest", vmin=0.0, vmax=vmax)
    plt.colorbar(label="attention weight")
    plt.xlabel("Key position (被看的 token)")
    plt.ylabel("Query position (在看别人)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] attention 热力图已保存: {out_path}")


def analyze_last_token_attention_distance(
    attn_layer: torch.Tensor,
    save_hist_path: Optional[str] = None,
    layer_idx: int = -1,
):
    """
    对某一层的 attention (batch=1) 分析：
      - 每个 head 中，最后一个 query token 对所有 key 的注意力分布
      - 统计距离分布：越远的 token 占了多少权重

    attn_layer: (1, num_heads, seq_len, seq_len)
    """
    attn = attn_layer[0]  # (num_heads, seq_len, seq_len)
    num_heads, seq_len, _ = attn.shape

    # 最后一个 token 的 query index
    q_idx = seq_len - 1
    positions = torch.arange(seq_len)

    head_mean_distances = []
    head_local_mass = []

    # 用于画距离直方图（合并所有 head）
    all_dists = []
    all_weights = []

    for h in range(num_heads):
        row = attn[h, q_idx]  # (seq_len,)
        row = row / (row.sum() + 1e-9)

        d = q_idx - positions  # 距离（越大越靠前）
        d = d.float()

        # 只看过去 (d >= 0)
        mask = d >= 0
        d = d[mask]
        w = row[mask]

        mean_dist = (d * w).sum().item()
        local_mass = row[q_idx-16 : q_idx+1].sum().item()  # 最近 16 个 token 的权重和

        head_mean_distances.append(mean_dist)
        head_local_mass.append(local_mass)

        all_dists.extend(d.cpu().tolist())
        all_weights.extend(w.cpu().tolist())

    print(f"[INFO] Layer {layer_idx}:")
    print(f"  每个 head 对最后 token 的平均距离: {[round(x,1) for x in head_mean_distances]}")
    print(f"  每个 head 最近 16 token 的注意力质量: {[round(x,3) for x in head_local_mass]}")
    print(f"  平均最近 16 token mass = {sum(head_local_mass)/len(head_local_mass):.3f}")

    if save_hist_path is not None:
        # 简单画一个距离 vs 权重的加权直方图
        import numpy as np

        d_arr = np.array(all_dists)
        w_arr = np.array(all_weights)
        # 把距离按区间分桶，比如 [0,8),[8,32),[32,128),[128,+inf)
        bins = [0, 8, 32, 128, 512, 1e9]
        labels = ["0-8", "8-32", "32-128", "128-512", "512+"]

        mass_per_bin = []
        for i in range(len(bins) - 1):
            mask = (d_arr >= bins[i]) & (d_arr < bins[i+1])
            mass_per_bin.append(w_arr[mask].sum())

        mass_sum = sum(mass_per_bin) + 1e-9
        mass_per_bin = [m / mass_sum for m in mass_per_bin]

        plt.figure(figsize=(5, 3))
        plt.bar(labels, mass_per_bin)
        plt.ylabel("Normalized attention mass")
        plt.xlabel("Distance to last token (in tokens)")
        plt.title(f"Layer {layer_idx} last-token attention distance")
        plt.tight_layout()
        plt.savefig(save_hist_path, dpi=200)
        plt.close()
        print(f"[INFO] 距离分布直方图已保存: {save_hist_path}")


def compute_answer_attention_mass(
    attn_layer: torch.Tensor,
    answer_span: Tuple[int, int],
    layer_idx: int = -1,
):
    """
    统计在某层里，“最后一个 token”对答案 span 的总 attention mass。
    attn_layer: (1, num_heads, seq_len, seq_len)
    answer_span: (start_idx, end_idx_excl)
    """
    attn = attn_layer[0]  # (num_heads, seq_len, seq_len)
    num_heads, seq_len, _ = attn.shape
    q_idx = seq_len - 1
    start, end = answer_span

    masses = []
    for h in range(num_heads):
        row = attn[h, q_idx]  # (seq_len,)
        row = row / (row.sum() + 1e-9)
        mass = row[start:end].sum().item()
        masses.append(mass)

    print(f"[INFO] Layer {layer_idx}: 每个 head 对答案 span 的注意力质量: {[round(x,3) for x in masses]}")
    print(f"       平均答案 attention mass = {sum(masses)/len(masses):.3f}")


# ============== 主流程：整合在一起 ==============

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 0: Attention Probe for Long Context")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace 模型名，比如 Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda / cpu",
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=2048,
        help="目标上下文 token 数（近似），用于控制生成的随机文本长度",
    )
    parser.add_argument(
        "--needle_pos",
        type=float,
        default=0.5,
        help="needle 在上下文中的相对位置 (0,1) 之间，比如 0.1,0.5,0.9",
    )
    args = parser.parse_args()

    # 1. 构造带 needle 的长上下文
    approx_words_per_token = 0.7
    total_words = int(args.target_tokens * approx_words_per_token)

    secret_code = random.randint(100000, 999999)
    needle_sentence = f"The secret code is {secret_code}."
    context = build_context_with_single_needle(
        total_words=total_words,
        needle_sentence=needle_sentence,
        position_fraction=args.needle_pos,
    )

    prompt = (
        "You will be given a long text. Read it carefully.\n\n"
        f"TEXT START\n{context}\nTEXT END\n\n"
        "Question: According to the text above, what is the secret code?\n"
        "Answer with ONLY the 6-digit number."
    )

    print(f"[INFO] 构造完成，secret_code = {secret_code}")

    # 2. 加载模型 & 跑前向，拿 attention
    tokenizer, model = load_model_and_tokenizer(args.model_name, device=args.device)
    attentions, input_ids = run_model_with_attentions(tokenizer, model, prompt, device=args.device)

    seq_len = input_ids.shape[1]
    print(f"[INFO] 实际 token 长度 = {seq_len}")

    # 3. 找到答案句子的 token 范围
    answer_ids = tokenizer(
        needle_sentence,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].to(input_ids.device)

    span = find_subsequence_indices(input_ids[0], answer_ids)
    if span is None:
        print("[WARN] 在 token 序列中找不到 needle 子序列，说明 tokenization 有点不对齐。后续统计会忽略答案 span。")
    else:
        print(f"[INFO] 答案句子 token 范围: {span}")

    os.makedirs("phase0_outputs", exist_ok=True)

    # 4. 选一个层/头画 full attention 热力图（用于直观观察）
    #   通常最后一两层更有“决策含义”，我们用最后一层 layer = -1
    last_layer_idx = len(attentions) - 1
    last_layer_attn = attentions[last_layer_idx]  # (1, num_heads, seq_len, seq_len)
    num_heads = last_layer_attn.shape[1]

    # 先画 head 0 的 full attention
    head0_attn = last_layer_attn[0, 0]  # (seq_len, seq_len)
    heatmap_path = os.path.join(
        "phase0_outputs",
        f"attn_layer{last_layer_idx}_head0_full.png",
    )
    plot_full_attention_heatmap(
        head0_attn,
        heatmap_path,
        title=f"Layer {last_layer_idx} Head 0 Attention",
    )

    # 5. 分析“最后一个 token”的距离分布
    dist_hist_path = os.path.join(
        "phase0_outputs",
        f"attn_layer{last_layer_idx}_last_token_distance.png",
    )
    analyze_last_token_attention_distance(
        last_layer_attn,
        save_hist_path=dist_hist_path,
        layer_idx=last_layer_idx,
    )

    # 6. 统计最后一个 token 对答案 span 的 attention mass
    if span is not None:
        compute_answer_attention_mass(
            last_layer_attn,
            answer_span=span,
            layer_idx=last_layer_idx,
        )

    print("[INFO] Phase 0 probe 完成。请查看 phase0_outputs/ 下的 PNG 图和命令行统计。")


if __name__ == "__main__":
    main()

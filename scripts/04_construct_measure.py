# -*- coding: utf-8 -*-
"""
步骤④ 指标构建 Measure Construction
=====================================
计算公式（论文式3-1）：
    ShortTerm_it = MD&A中命中短期视域词典的词次总数 / MD&A总词数 × 100

同步计算备选口径（稳健性检验用）：
    ShortTermSent_it = 含词典词的句子数 / MD&A总句数 × 100

输出：results/measure_panel.csv —— 企业-年度面板（可直接导入 Stata / Python）
"""
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_PATH = os.path.join(BASE, "data", "processed", "tokens.csv")
SENT_PATH = os.path.join(BASE, "data", "processed", "sentences.csv")
DICT_PATH = os.path.join(BASE, "dictionary", "shortterm_dict.txt")
META_PATH = os.path.join(BASE, "data", "raw", "metadata.csv")
OUT_PATH = os.path.join(BASE, "results", "measure_panel.csv")


def load_dictionary() -> set:
    words = set()
    with open(DICT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                words.add(line)
    return words


def main():
    dict_words = load_dictionary()
    tokens = pd.read_csv(TOK_PATH, dtype={"code": str, "year": str})
    sents = pd.read_csv(SENT_PATH, dtype={"code": str, "year": str})
    meta = pd.read_csv(META_PATH, dtype={"code": str, "year": str})

    # L1 原始计数：在清洗后的句子文本上按词典词做子串计数（多字词不会被分词边界
    # 切断，保证"短期收益""权宜之计"等词条也能命中） -> L2 标准化比率
    sent_text = sents.groupby(["code", "year"])["sentence"].apply("".join).reset_index()
    sent_text.columns = ["code", "year", "text"]
    sent_text["dict_hits"] = sent_text["text"].apply(
        lambda t: sum(t.count(w) for w in dict_words))
    tokens = tokens.merge(sent_text[["code", "year", "dict_hits"]], on=["code", "year"])
    tokens["ShortTerm"] = (tokens["dict_hits"] / tokens["n_words"] * 100).round(4)

    # 备选口径：含词典词的句子数 / 总句数 × 100
    sent_hits = sents.assign(hit=sents["sentence"].astype(str).apply(
        lambda s: int(any(w in s for w in dict_words))))
    sent_agg = sent_hits.groupby(["code", "year"]).agg(
        n_sentences=("hit", "size"), hit_sentences=("hit", "sum")).reset_index()
    sent_agg["ShortTermSent"] = (sent_agg["hit_sentences"] / sent_agg["n_sentences"] * 100).round(4)

    panel = tokens[["code", "year", "n_words", "dict_hits", "ShortTerm"]].merge(
        sent_agg[["code", "year", "ShortTermSent"]], on=["code", "year"])
    panel = panel.merge(meta.drop(columns=["source"]), on=["code", "year"], how="left")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    panel.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[measure] 企业-年度面板 {panel.shape[0]} 行 -> results/measure_panel.csv")
    print(f"[measure] ShortTerm 均值 {panel['ShortTerm'].mean():.3f}，"
          f"标准差 {panel['ShortTerm'].std():.3f}，"
          f"最小 {panel['ShortTerm'].min():.3f}，最大 {panel['ShortTerm'].max():.3f}")


if __name__ == "__main__":
    main()

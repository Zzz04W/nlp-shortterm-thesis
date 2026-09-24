# -*- coding: utf-8 -*-
"""
步骤③ 特征提取（词典法路线） Dictionary Construction
======================================================
借鉴胡楠等（2021,《管理世界》）基于短期视域（short-term horizon）理论构建的
管理者短视词典（43个种子词），并结合本文语料进行扩展与人工校验：

  1) 理论种子词：直接取自文献的短期视域词典；
  2) 共现扩展：在本文语料上以 PMI（点互信息）筛选与种子词强共现的候选词，
     生成"待人工校验"候选表（本科论文建议：最终词典仍以人工确认的种子词为准，
     保证透明与可复现——对应课件"路线B：主题/模型只用来定词"的原则）；
  3) 输出最终词典 dictionary/shortterm_dict.txt 与候选词表
     dictionary/expansion_candidates.csv。

使用方法：python scripts/03_build_dictionary.py [--expand]
"""
import os
import sys
import math
from collections import Counter

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_PATH = os.path.join(BASE, "data", "processed", "tokens.csv")
DICT_PATH = os.path.join(BASE, "dictionary", "shortterm_dict.txt")
CAND_PATH = os.path.join(BASE, "dictionary", "expansion_candidates.csv")

# 理论种子词：短期视域词典（整理自胡楠等, 2021；含四字成语与常用短视表述）
SEED_WORDS = [
    "尽快", "立刻", "马上", "立即", "赶快", "早日", "加快", "抓紧", "急需",
    "争分夺秒", "只争朝夕", "迫在眉睫", "时不我待", "分秒必争",
    "眼下", "眼前", "当下", "目前", "现今", "如今",
    "短期", "短期内", "近期", "年内", "今年",
    "短期收益", "短期回报", "短期利益", "快速见效", "立竿见影", "一步到位",
    "权宜之计", "应急", "投机", "速成", "速胜",
    "急功近利", "急于求成", "饮鸩止渴", "杀鸡取卵", "竭泽而渔",
]

# 扩展时排除的宽泛高频词（防止污染词典）
BLACKLIST = set([
    "公司", "发展", "经营", "市场", "业务", "产品", "客户", "行业", "管理",
    "建设", "推进", "提升", "工作", "目标", "战略", "项目", "收入", "利润",
])


def write_final_dictionary():
    with open(DICT_PATH, "w", encoding="utf-8") as f:
        f.write("# 管理者短视（短期视域）词典 v1.0\n")
        f.write("# 来源：整理自胡楠等（2021）《管理世界》短期视域词典（43词），"
                "结合本文语料校验调整\n")
        f.write("# 每行一个词，# 开头为注释\n")
        for w in SEED_WORDS:
            f.write(w + "\n")
    print(f"[dictionary] 最终词典 {len(SEED_WORDS)} 词 -> dictionary/shortterm_dict.txt")


def pmi_expansion(top_k: int = 20):
    """基于种子词的共现 PMI 扩展：输出候选词供人工校验。"""
    df = pd.read_csv(TOK_PATH)
    docs = [d.split() for d in df["tokens"].astype(str)]

    n_docs = len(docs)
    doc_freq = Counter()
    seed_tf = Counter()
    word_tf = Counter()
    seed_df = Counter()
    for doc in docs:
        word_tf.update(doc)
        doc_freq.update(set(doc))
        seeds_in = [w for w in doc if w in SEED_WORDS]
        if seeds_in:
            seed_df.update(set(seeds_in))
            seed_tf.update(seeds_in)

    # 与"含种子词文档"总体相比，PMI 显著偏高的词
    seed_doc_ids = [i for i, doc in enumerate(docs) if any(w in SEED_WORDS for w in doc)]
    cand_tf, cand_df = Counter(), Counter()
    for i in seed_doc_ids:
        cand_tf.update(docs[i])
        cand_df.update(set(docs[i]))

    rows = []
    for w, f_cand in cand_df.items():
        if w in SEED_WORDS or w in BLACKLIST or len(w) < 2:
            continue
        if f_cand < 3:  # 出现文档数过低，不可靠
            continue
        p_w_cand = f_cand / max(len(seed_doc_ids), 1)
        p_w_all = doc_freq[w] / n_docs
        if p_w_all == 0:
            continue
        pmi = math.log(p_w_cand / p_w_all)
        rows.append({"word": w, "pmi": round(pmi, 3),
                     "df_in_seed_docs": int(f_cand),
                     "df_all": int(doc_freq[w])})
    cand = pd.DataFrame(rows)
    if cand.empty:
        print("[dictionary] 语料中共现候选词不足，跳过扩展（不影响最终词典）")
        return
    cand = cand.sort_values("pmi", ascending=False).head(top_k)
    cand.to_csv(CAND_PATH, index=False, encoding="utf-8-sig")
    print(f"[dictionary] PMI 扩展候选词 {len(cand)} 个 -> dictionary/expansion_candidates.csv")
    print(cand.to_string(index=False))


if __name__ == "__main__":
    write_final_dictionary()
    if "--expand" in sys.argv:
        pmi_expansion()

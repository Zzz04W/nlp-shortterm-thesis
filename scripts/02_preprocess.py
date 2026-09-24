# -*- coding: utf-8 -*-
"""
步骤② 文本预处理 Preprocessing
================================
五步流水线：文档解析 -> 文本清洗 -> 中文分词 -> 去停用词 -> 词形归一

输入：data/raw/{code}_{year}.txt
输出：data/processed/tokens.csv（code, year, n_words, n_sentences, tokens）
      data/processed/sentences.csv（句子级中间表，供句子口径稳健性检验使用）
"""
import os
import re
import glob

import jieba
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "data", "raw")
OUT_DIR = os.path.join(BASE, "data", "processed")
STOPWORDS_PATH = os.path.join(BASE, "dictionary", "cn_stopwords.txt")

# ------------------------------------------------------------------ 文本清洗
def clean_text(text: str) -> str:
    """清洗：去页眉页脚/页码/乱码/多余空白（对应课件"最常见的坑"第1条）。"""
    text = re.sub(r"（本页共\s*\d+\s*页）", "", text)
    text = re.sub(r"第\s*\d+\s*页", "", text)
    text = re.sub(r"股份有限公司董事会", "", text)  # 去落款（避免贪婪匹配误删正文）
    text = re.sub(r"[a-zA-Z0-9_]+", " ", text)      # 去英文与数字串
    text = re.sub(r"[^\u4e00-\u9fa5。；，]", " ", text)  # 仅保留中文与常用标点
    text = re.sub(r"\s+", "", text)
    return text

# ------------------------------------------------------------------ 停用词
def load_stopwords() -> set:
    """通用停用词 + 财经领域自定义停用词（对应课件"最常见的坑"第2条）。"""
    sw = set()
    if os.path.exists(STOPWORDS_PATH):
        with open(STOPWORDS_PATH, encoding="utf-8") as f:
            sw |= {line.strip() for line in f if line.strip()}
    # 财经领域补充停用词（通用表未覆盖）
    finance_sw = [
        "公司", "本公司", "我公司", "报告期", "报告期内", "年度", "同比", "较上年",
        "期末", "期初", "万元", "亿元", "上市公司", "证券", "股份", "有限", "集团",
        "董事会", "管理层", "股东", "股东大会", "董事", "监事", "高级管理人员",
        "披露", "公告", "审计", "会计", "财务", "经营情况", "主营业务",
    ]
    sw |= set(finance_sw)
    return sw

# ------------------------------------------------------------------ 分句
def split_sentences(text: str) -> list:
    sents = [s.strip() for s in re.split(r"[。；]", text) if s.strip()]
    return sents

# ------------------------------------------------------------------ 主流程
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stopwords = load_stopwords()
    jieba.setLogLevel(60)

    tok_rows, sent_rows = [], []
    for fp in sorted(glob.glob(os.path.join(RAW_DIR, "*_20*.txt"))):
        fname = os.path.basename(fp)
        code, year = fname.replace(".txt", "").split("_")

        with open(fp, encoding="utf-8") as f:
            raw = f.read()
        cleaned = clean_text(raw)

        sents = split_sentences(cleaned)
        # jieba 精确模式（对应课件：搜索引擎模式会切碎专有名词）
        all_tokens = []
        for s in sents:
            toks = [t for t in jieba.lcut(s) if t.strip() and t not in stopwords
                    and not re.match(r"^[。；，\s]+$", t)]
            all_tokens.append(toks)
            sent_rows.append({
                "code": code, "year": year, "sentence": s,
                "n_words_sent": len(toks), "tokens_sent": " ".join(toks),
            })

        tokens = [t for ts in all_tokens for t in ts]
        tok_rows.append({
            "code": code, "year": year,
            "n_sentences": len(sents),
            "n_words": len(tokens),
            "tokens": " ".join(tokens),
        })

    pd.DataFrame(tok_rows).to_csv(
        os.path.join(OUT_DIR, "tokens.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(sent_rows).to_csv(
        os.path.join(OUT_DIR, "sentences.csv"), index=False, encoding="utf-8-sig")
    print(f"[preprocess] 完成 {len(tok_rows)} 个企业-年度观测的分词与停用词过滤")
    print(f"[preprocess] 输出 -> data/processed/tokens.csv, sentences.csv")

if __name__ == "__main__":
    main()

# 管理者短视主义与企业研发投入：基于A股制造业上市公司年报MD&A的文本测度

> 🌐 公开仓库地址：https://github.com/Zzz04W/nlp-shortterm-thesis
>
> 毕业论文方法预备课（管理研究中的NLP测度构建方法）· 预演毕业论文配套材料
> 按「NLP 测度构建五步流水线」完整实现：语料获取 → 文本预处理 → 特征提取 → 指标构建 → 信效度检验

## 仓库结构

```
├── paper/                     预演论文全文（Word）与内容稿
├── scripts/                   五步流水线 Python 脚本（可一键复现）
│   ├── 01_build_corpus.py         ① 语料获取（含巨潮真实数据接口预留 + 演示语料生成）
│   ├── 02_preprocess.py           ② 文本预处理（清洗/分词/去停用词）
│   ├── 03_build_dictionary.py     ③ 特征提取·词典法（种子词典 + PMI 扩展候选）
│   ├── 04_construct_measure.py    ④ 指标构建（企业-年度面板）
│   └── 05_validity_analysis.py    ⑤ 信效度检验（聚合/区分/预测效度 + 回归 + 稳健性）
├── dictionary/
│   ├── shortterm_dict.txt         管理者短视（短期视域）词典 v1.0（41词）
│   ├── cn_stopwords.txt           通用停用词表
│   └── expansion_candidates.csv   PMI 扩展候选词（待人工校验）
├── data/
│   ├── raw/                       MD&A 文本语料（600 份）+ metadata.csv 元数据面板
│   └── processed/                 tokens.csv / sentences.csv 中间结果
├── results/
│   ├── measure_panel.csv          ★ 企业-年度测量结果面板（600 观测）
│   ├── desc_stats.csv             描述性统计
│   ├── corr_matrix.csv            效度检验相关矩阵
│   ├── group_test.csv             条件均值检验（高低短视组）
│   ├── ols_results.txt            OLS 回归（3个模型）
│   └── robustness.csv             稳健性检验汇总
└── docs/
    └── 数据来源说明.md             每个文件的来源与处理步骤
```

## 一键复现

```bash
pip install -r requirements.txt
python scripts/01_build_corpus.py        # 语料获取
python scripts/02_preprocess.py          # 预处理（jieba 精确模式）
python scripts/03_build_dictionary.py    # 词典构建
python scripts/04_construct_measure.py   # 测度计算 -> results/measure_panel.csv
python scripts/05_validity_analysis.py   # 效度检验与回归
```

## 主要结果速览

| 检验 | 结果 | 结论 |
|---|---|---|
| 聚合效度 | ShortTerm × 句子口径 r = 0.794, p < 0.01 | 通过 |
| 区分效度 | 与 size（r=-0.023, p=0.572）、lev、age 均不显著相关 | 通过 |
| 条件均值检验 | 高短视组研发强度 3.994 vs 低短视组 4.576，t = -5.711 | 通过 |
| 预测效度（OLS） | 研发强度 ~ 短视测度 β = -0.343（p < 0.01，控制企业特征与年份FE后 β = -0.344） | 通过 |
| 稳健性 | 更换分母口径 / 取消缩尾，结论不变 | 通过 |

## 数据说明

本仓库语料为**程序化合成的演示语料**（模拟年报MD&A行文风格，600 份企业-年度文本），
用于保证作业全流程在无数据库权限时也可复现；接入真实年报（巨潮资讯网）的接口已在
`01_build_corpus.py` 中预留。详见 `docs/数据来源说明.md`。

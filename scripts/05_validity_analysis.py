# -*- coding: utf-8 -*-
"""
步骤⑤ 信效度检验 Reliability & Validity
========================================
四类检验（对应课件第19页）：
  1) 描述性统计
  2) 聚合效度：本文词典口径 ShortTerm 与备选句子口径 ShortTermSent 的相关
  3) 区分效度：与公司规模 size、资产负债率 lev、上市年限 age 的低相关
  4) 预测效度：以 ShortTerm 解释研发投入强度 rd_intensity 的 OLS 回归
     （模型1 无控制变量；模型2 加控制变量；模型3 加年份固定效应）
稳健性：
  a) 更换分母口径（句子比率）重新回归；
  b) 连续变量 1%/99% 缩尾后重新回归；
  c) 条件均值检验：按 ShortTerm 中位数分组比较 rd_intensity（t检验）。

输出：results/desc_stats.csv, corr_matrix.csv, group_test.csv,
      ols_results.txt, robustness.csv
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_PATH = os.path.join(BASE, "results", "measure_panel.csv")
OUT = os.path.join(BASE, "results")


def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def main():
    df = pd.read_csv(PANEL_PATH, dtype={"code": str})
    for c in ["size", "lev", "age", "roa", "rd_intensity", "ShortTerm", "ShortTermSent"]:
        df[c + "_w"] = winsorize(df[c]) if c in ["rd_intensity", "ShortTerm", "ShortTermSent"] else df[c]

    # ---------------------------------------------------------- 描述性统计
    desc = df[["ShortTerm", "ShortTermSent", "rd_intensity", "size", "lev", "age", "roa"]] \
        .describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(3)
    desc.to_csv(os.path.join(OUT, "desc_stats.csv"), encoding="utf-8-sig")
    print("==== 描述性统计 ====")
    print(desc)

    # ---------------------------------------------------------- 效度检验
    # 聚合效度
    r_conv, p_conv = stats.pearsonr(df["ShortTerm"], df["ShortTermSent"])
    # 区分效度（与"不该相关"的变量低相关）
    r_size, p_size = stats.pearsonr(df["ShortTerm"], df["size"])
    r_lev, p_lev = stats.pearsonr(df["ShortTerm"], df["lev"])
    r_age, p_age = stats.pearsonr(df["ShortTerm"], df["age"])
    corr = pd.DataFrame({
        "pair": ["聚合效度: ShortTerm × ShortTermSent",
                 "区分效度: ShortTerm × size",
                 "区分效度: ShortTerm × lev",
                 "区分效度: ShortTerm × age",
                 "预测效度: ShortTerm × rd_intensity"],
        "pearson_r": [round(r_conv, 3), round(r_size, 3), round(r_lev, 3),
                      round(r_age, 3), None],
        "p_value": [f"{p_conv:.2e}", f"{p_size:.3f}", f"{p_lev:.3f}",
                    f"{p_age:.3f}", None],
    })
    r_pred, p_pred = stats.pearsonr(df["ShortTerm"], df["rd_intensity"])
    corr.loc[4, "pearson_r"] = round(r_pred, 3)
    corr.loc[4, "p_value"] = f"{p_pred:.2e}"
    corr.to_csv(os.path.join(OUT, "corr_matrix.csv"), index=False, encoding="utf-8-sig")
    print("\n==== 效度检验 ====")
    print(corr.to_string(index=False))

    # ---------------------------------------------------------- 条件均值检验
    med = df["ShortTerm"].median()
    hi = df[df["ShortTerm"] >= med]["rd_intensity"]
    lo = df[df["ShortTerm"] < med]["rd_intensity"]
    t, p_t = stats.ttest_ind(hi, lo, equal_var=False)
    grp = pd.DataFrame({
        "group": ["高短视组(>=中位数)", "低短视组(<中位数)", "组间差异"],
        "n": [len(hi), len(lo), None],
        "rd_intensity_mean": [round(hi.mean(), 3), round(lo.mean(), 3), round(hi.mean() - lo.mean(), 3)],
        "t_stat": [None, None, round(t, 3)],
        "p_value": [None, None, f"{p_t:.2e}"],
    })
    grp.to_csv(os.path.join(OUT, "group_test.csv"), index=False, encoding="utf-8-sig")
    print("\n==== 条件均值检验 ====")
    print(grp.to_string(index=False))

    # ---------------------------------------------------------- 回归（预测效度）
    lines = []
    m1 = smf.ols("rd_intensity_w ~ ShortTerm_w", data=df).fit(cov_type="HC1")
    m2 = smf.ols("rd_intensity_w ~ ShortTerm_w + size + lev + age + roa + soe",
                 data=df).fit(cov_type="HC1")
    df["year_f"] = df["year"].astype(str)
    m3 = smf.ols("rd_intensity_w ~ ShortTerm_w + size + lev + age + roa + soe + C(year_f)",
                 data=df).fit(cov_type="HC1")

    for name, m in [("模型1：无控制变量", m1),
                    ("模型2：加企业特征控制变量", m2),
                    ("模型3：模型2 + 年份固定效应", m3)]:
        lines.append(f"\n---- {name} ----")
        lines.append(m.summary().as_text())
        lines.append(f"R-squared = {m.rsquared:.4f}, N = {int(m.nobs)}")
    with open(os.path.join(OUT, "ols_results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n==== 回归（预测效度）====")
    for name, m in [("模型1", m1), ("模型2", m2), ("模型3", m3)]:
        b, pv = m.params["ShortTerm_w"], m.pvalues["ShortTerm_w"]
        print(f"{name}: ShortTerm 系数 = {b:.4f} (p = {pv:.2e}), R2 = {m.rsquared:.4f}")

    # ---------------------------------------------------------- 稳健性
    m_r1 = smf.ols("rd_intensity_w ~ ShortTermSent_w + size + lev + age + roa + soe",
                   data=df).fit(cov_type="HC1")
    m_r2 = smf.ols("rd_intensity ~ ShortTerm + size + lev + age + roa + soe",
                   data=df).fit(cov_type="HC1")
    rob = pd.DataFrame({
        "设定": ["基准（模型2）", "更换分母口径（句子比率）", "不缩尾"],
        "解释变量系数": [round(m2.params["ShortTerm_w"], 3),
                    round(m_r1.params["ShortTermSent_w"], 3),
                    round(m_r2.params["ShortTerm"], 3)],
        "p值": [f"{m2.pvalues['ShortTerm_w']:.2e}",
             f"{m_r1.pvalues['ShortTermSent_w']:.2e}",
             f"{m_r2.pvalues['ShortTerm']:.2e}"],
        "R2": [round(m2.rsquared, 3), round(m_r1.rsquared, 3), round(m_r2.rsquared, 3)],
    })
    rob.to_csv(os.path.join(OUT, "robustness.csv"), index=False, encoding="utf-8-sig")
    print("\n==== 稳健性 ====")
    print(rob.to_string(index=False))
    print("\n[validity] 全部结果已输出至 results/ 目录")


if __name__ == "__main__":
    main()

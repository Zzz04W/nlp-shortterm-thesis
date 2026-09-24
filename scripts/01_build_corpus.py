# -*- coding: utf-8 -*-
"""
步骤① 语料获取 Corpus Acquisition
==================================
研究：管理者短视主义与企业研发投入——基于A股制造业上市公司年报MD&A的文本测度

本脚本提供两种模式：
  mode = "real"  ：按巨潮资讯网（cninfo）年报披露页规则批量下载真实年报PDF/文本，
                   接口函数 get_cninfo_annual_reports() 已给出URL构造与请求骨架，
                   需要使用者填入自己的查询参数与频率控制后运行。
  mode = "demo"  ：生成与年报MD&A行文风格一致的演示语料（程序化合成，
                   保证作业全流程在无数据库权限时也可一键复现），
                   同时生成企业-年度元数据（行业、规模、杠杆、产权性质、
                   研发强度等"财务数据库字段"的模拟值）。

输出：
  data/raw/{code}_{year}.txt   每个企业-年度一份 MD&A 文本
  data/raw/metadata.csv        企业-年度元数据面板（模拟"财务数据库"字段）

注意：真实研究中，MD&A文本来自巨潮资讯网年报PDF（PyMuPDF解析），
     财务变量来自CSMAR/Wind数据库，时间对齐规则见论文第3章。
"""
import os
import re
import random
import math

import pandas as pd

random.seed(20260924)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "data", "raw")
MODE = "demo"  # 可改为 "real"

N_FIRMS = 120
YEARS = [2019, 2020, 2021, 2022, 2023]

# ---------------------------------------------------------------- 真实数据接口
def get_cninfo_annual_reports(code: str, year: int) -> str:
    """真实模式示例：构造巨潮资讯网年报检索请求（骨架，未填鉴权参数）。

    实际使用时：1) 在 http://www.cninfo.com.cn/new/hisAnnouncement/query
    以 POST 方式按 股票代码+公告类别(年度报告)+时间区间 检索；
    2) 控制请求频率（>=2秒/次），遵守robots协议；3) 下载PDF后用PyMuPDF
    抽取MD&A章节文本（定位"管理层讨论与分析"至"公司未来发展的展望"）。
    """
    import requests  # noqa: F401
    url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    payload = {
        "stock": code,
        "tabName": "fulltext",
        "category": "category_ndbg_szsh",  # 年度报告
        "seDate": f"{year + 1}-01-01~{year + 1}-06-30",  # t年年报于t+1年4月底前披露
    }
    raise NotImplementedError("请按注释填入请求头、分页与下载逻辑后使用真实数据")

# ---------------------------------------------------------------- 演示语料
# 句子模板池：不带短期导向色彩的常规MD&A语句
ROUTINE_SENTS = [
    "报告期内公司围绕年度经营计划扎实推进各项重点工作总体经营情况保持稳定",
    "公司持续优化产品结构深化渠道建设主营业务收入实现稳健增长",
    "面对复杂的外部经营环境公司管理层积极应对努力克服不利因素影响",
    "公司进一步完善治理结构规范运作信息披露真实准确完整",
    "公司加强成本管控提升运营效率毛利率水平保持基本平稳",
    "公司积极拓展国内外市场客户结构与订单质量进一步改善",
    "公司高度重视安全生产与质量管理体系运行情况良好",
    "公司稳步推进产能建设生产运营秩序正常",
]

# 含"短期视域"色彩的语句模板（占位符{w}在生成时替换为具体词典词）
SHORTTERM_SENTS = [
    "公司将{w}推进重点项目的落地实施确保年内见到成效",
    "管理层表示要{w}扭转当前被动局面实现业绩快速回升",
    "公司要求各事业部{w}消化库存回笼资金缓解业绩压力",
    "为完成全年目标公司将{w}调整产品定价策略抢占市场份额",
    "公司{w}布局热门赛道力求{w}形成新的利润增长点",
    "管理层认为当前窗口稍纵即逝必须{w}扩大产能投放",
    "公司将采取权宜之计压缩各项开支以确保{w}兑现业绩承诺",
    "公司{w}回笼资金以应对眼前的流动性压力",
]

RND_SENTS = [
    "公司坚持创新驱动发展战略持续加大研发投入研发费用同比增长较快",
    "公司围绕核心技术开展攻关新申请专利数量保持稳定增长",
    "公司与高校科研院所共建联合实验室推进产学研协同创新",
    "公司加大研发人才引进力度研发人员占比稳步提升",
    "公司在新材料新工艺方向的前瞻性研发布局取得阶段性成果",
]

RISK_SENTS = [
    "公司可能面临原材料价格波动市场需求不及预期等风险",
    "宏观经济波动与行业竞争加剧可能对公司业绩产生不利影响",
    "汇率波动及贸易环境变化对公司海外业务构成一定不确定性",
]

CITY_PREFIX = ["华", "中", "东", "南", "北", "金", "恒", "瑞", "泰", "宏", "天", "利", "康", "威", "晟"]
CORE_WORD = ["精工", "智造", "电子", "新材", "装备", "光电", "动力", "医化", "环科", "智控"]
SUFFIX = ["科技", "股份", "制造", "实业"]


def make_text(shortterm_level: float, rnd_emphasis: float, rng: random.Random) -> str:
    """按短视水平 shortterm_level 与研发强调度 rnd_emphasis 生成一段MD&A文本。"""
    n_routine = rng.randint(6, 9)
    n_short = max(0, int(round(1.2 + 4.5 * shortterm_level + rng.gauss(0, 0.8))))
    n_rnd = max(1, int(round(1.5 + 2.5 * rnd_emphasis + rng.gauss(0, 0.6))))
    n_risk = rng.randint(1, 3)

    sents = [rng.choice(ROUTINE_SENTS) + "。" for _ in range(n_routine)]
    for _ in range(n_short):
        tpl = rng.choice(SHORTTERM_SENTS)
        sents.append(tpl.format(w=rng.choice(SHORT_WORDS)) + "。")
    sents += [rng.choice(RND_SENTS) + "。" for _ in range(n_rnd)]
    sents += [rng.choice(RISK_SENTS) + "。" for _ in range(n_risk)]
    rng.shuffle(sents)

    header = "{name}{year}年年度报告管理层讨论与分析（模拟文本）".format(
        name="XX股份有限公司", year="")
    body = "".join(sents)
    tail = "（本页共12页）XX股份有限公司董事会二零二四年四月"
    return header + body + tail


# 短期视域词典（与 dictionary/shortterm_dict.txt 保持一致，由 03 脚本落盘）
SHORT_WORDS = [
    "尽快", "立刻", "马上", "立即", "赶快", "早日", "加快", "抓紧", "急需",
    "争分夺秒", "只争朝夕", "迫在眉睫", "时不我待", "分秒必争",
    "眼下", "眼前", "当下", "目前", "现今", "如今",
    "短期", "短期内", "近期", "年内", "今年",
    "短期收益", "短期回报", "短期利益", "快速见效", "立竿见影", "一步到位",
    "权宜之计", "应急", "投机", "速成", "速胜",
    "急功近利", "急于求成", "饮鸩止渴", "杀鸡取卵", "竭泽而渔",
]


def build_demo():
    os.makedirs(RAW_DIR, exist_ok=True)
    rows = []
    firm_latent = {f"f{i}": random.gauss(0, 1) for i in range(N_FIRMS)}

    for i in range(N_FIRMS):
        # 交易所前缀轮换 + 顺序编号，保证代码唯一且形似真实A股代码
        code = f"{['00', '30', '60'][i % 3]}{i + 1:04d}"
        name = random.choice(CITY_PREFIX) + random.choice(CORE_WORD) + random.choice(SUFFIX)
        ind = random.choice(["专用设备制造", "电子元器件制造", "医药制造", "化学原料制造", "汽车零部件制造"])
        soe = 1 if random.random() < 0.25 else 0
        size0 = random.gauss(22.0, 1.1)
        age0 = random.randint(6, 25)

        prev_s = firm_latent[f"f{i}"]
        for y in YEARS:
            s = 0.7 * prev_s + 0.3 * random.gauss(0, 1)  # 短视具有企业内持续性
            prev_s = s
            size = size0 + 0.05 * (y - 2019) + random.gauss(0, 0.15)
            lev = min(max(random.gauss(0.42, 0.14), 0.05), 0.9)
            age = age0 + (y - 2019)
            roa = random.gauss(0.04, 0.05)

            # 真实生成过程：短视抑制研发（用于检验流水线能否还原该关系）
            rd = 4.2 - 1.35 * s + 0.45 * (size - 22.0) + random.gauss(0, 0.9)
            rd = min(max(rd, 0.2), 15.0)

            text = make_text(
                shortterm_level=1 / (1 + math.exp(-s)),
                rnd_emphasis=min(max(rd / 8.0, 0), 1),
                rng=random,
            )
            with open(os.path.join(RAW_DIR, f"{code}_{y}.txt"), "w", encoding="utf-8") as f:
                f.write(text)
            rows.append({
                "code": code, "name": name, "year": y, "industry": ind,
                "soe": soe, "size": round(size, 3), "lev": round(lev, 3),
                "age": age, "roa": round(roa, 3), "rd_intensity": round(rd, 3),
                "latent_shortterm": round(s, 4),
                "source": "演示语料（程序化合成，接口见01脚本real模式）",
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RAW_DIR, "metadata.csv"), index=False, encoding="utf-8-sig")
    print(f"[demo] 已生成 {N_FIRMS} 家公司 × {len(YEARS)} 年 = {len(df)} 份MD&A演示文本")
    print(f"[demo] 元数据面板 -> data/raw/metadata.csv")


if __name__ == "__main__":
    if MODE == "demo":
        build_demo()
    else:
        for c in ["000001", "300750", "600519"]:
            get_cninfo_annual_reports(c, 2023)

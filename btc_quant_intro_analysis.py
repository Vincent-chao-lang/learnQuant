#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
btc_quant_intro_analysis.py

流水线：
1) 从 Stooq 获取 BTCUSD 日线数据 -> 标准化为 date, price_usd -> 合并更新本地 CSV
2) 量化分析（结构/分布/风险，不做方向预测）
3) 输出中文 Markdown 报告 + 直方图（图片按日期命名，不覆盖历史）
4) 邮件发送（Gmail SMTP）：附件包含 MD + HTML；若 pandoc 可用则附加 PDF

依赖：
pip install numpy pandas scikit-learn matplotlib requests python-dotenv
（可选）安装 pandoc 以生成 PDF：brew install pandoc 或 apt install pandoc

.env（放在脚本同目录，且加入 .gitignore）：
SMTP_USER=你的gmail@gmail.com
SMTP_PASS=你的Gmail App Password（去掉空格）
TO_EMAIL=收件人邮箱（可填自己）
"""

import argparse
import os
import ssl
import smtplib
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# 读取 .env（强烈建议）
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # 没装 python-dotenv 也能跑，只是需要你用 export 设置环境变量
    pass


# ----------------------------
# 0) 小工具：Markdown 表格（不依赖 tabulate）
# ----------------------------
def df_to_md_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    df = df.head(max_rows).copy()
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(map(str, cols)) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for v in row.values:
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------
# 1) 数据更新：Stooq -> 标准化 -> 合并 CSV
# ----------------------------
def fetch_btcusd_daily_from_stooq(timeout: int = 30) -> pd.DataFrame:
    """
    Stooq 日线 CSV：
    https://stooq.com/q/d/l/?s=btcusd&i=d

    常见列：Date, Open, High, Low, Close, Volume
    我们只用 Close 作为 price_usd，并标准化输出：date, price_usd
    """
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    from io import StringIO
    raw = pd.read_csv(StringIO(resp.text))

    # 统一列名为小写，兼容 Date/Close 等大小写
    raw.columns = [c.strip().lower() for c in raw.columns]
    if "date" not in raw.columns or "close" not in raw.columns:
        raise RuntimeError(f"Stooq CSV 缺少 date/close 列，实际列：{list(raw.columns)}")

    df = raw[["date", "close"]].copy()
    df.rename(columns={"close": "price_usd"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df

def get_local_last_date(csv_path: Path) -> pd.Timestamp | None:
    if not csv_path.exists():
        return None
    local = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
    if local.empty:
        return None
    return local["date"].iloc[-1]

def fetch_btcusd_daily_increment_from_coingecko(
    last_date: pd.Timestamp | None,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    CoinGecko 免费备源（不带 key）：
    - 只拉最近 2 天 market_chart
    - 按 UTC 日期聚合成日序列（每一天取最后一个价格点）
    - 如果 last_date 给了，就只返回 > last_date 的增量
    输出标准：date, price_usd
    """
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": 2}  # 只取最近 2 天
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()

    data = resp.json()
    prices = data.get("prices", [])
    if not prices:
        return pd.DataFrame(columns=["date", "price_usd"])

    df = pd.DataFrame(prices, columns=["ts_ms", "price_usd"])
    # 用 UTC 来划分“哪一天”，避免本地时区导致跨日错位
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.date
    df["date"] = pd.to_datetime(df["date"])
    df = (
        df.sort_values("ts_ms")
          .groupby("date", as_index=False)["price_usd"]
          .last()
          .sort_values("date")
          .reset_index(drop=True)
    )

    if last_date is not None:
        df = df[df["date"] > last_date].reset_index(drop=True)

    return df[["date", "price_usd"]]

# def safe_fetch_btcusd_increment_from_coingecko(last_date, timeout=60) -> pd.DataFrame:
#     try:
#         return fetch_btcusd_daily_increment_from_coingecko(last_date=last_date, timeout=timeout)
#     except Exception as e:
#         # 不让备源失败影响主流程
#         print(f"⚠️ CoinGecko 备源失败（将跳过增量补齐）：{type(e).__name__}: {e}")
#         return pd.DataFrame(columns=["date", "price_usd"])

import time
import random

def safe_fetch_btcusd_increment_from_coingecko(last_date, timeout=60) -> pd.DataFrame:
    for attempt in range(3):
        try:
            return fetch_btcusd_daily_increment_from_coingecko(last_date=last_date, timeout=timeout)
        except Exception as e:
            wait = (2 ** attempt) + random.random()
            print(f"⚠️ CoinGecko 备源失败（第{attempt+1}/3次）：{type(e).__name__}: {e}；{wait:.1f}s后重试")
            time.sleep(wait)
    print("⚠️ CoinGecko 备源连续失败，跳过增量补齐，继续使用本地数据")
    return pd.DataFrame(columns=["date", "price_usd"])


def fetch_btcusd_daily_primary_with_backup(csv_path: Path, timeout: int = 30) -> pd.DataFrame:
    """
    主源：Stooq（全量，标准化后返回）
    备源：CoinGecko 免费接口（仅增量，返回最近新增日期）
    输出统一：date, price_usd
    """
    try:
        df = fetch_btcusd_daily_from_stooq(timeout=timeout)
        print("✅ 数据源：Stooq（主源）")
        return df
    except Exception as e:
        print(f"⚠️ Stooq 获取失败，切换 CoinGecko（免费备源，仅增量）。原因：{e}")
        last_date = get_local_last_date(csv_path)
        inc = fetch_btcusd_daily_increment_from_coingecko(last_date=last_date, timeout=max(60, timeout))
        print(f"✅ 数据源：CoinGecko（免费备源）增量行数：{len(inc)}")
        return inc


# def update_price_csv(csv_path: Path) -> Tuple[pd.DataFrame, bool]:
#     """
#     更新本地 CSV（标准 schema：date, price_usd）
#     - CSV 不存在：下载全量并保存
#     - CSV 存在：下载全量后 merge 去重（小数据量更稳）
#     返回：(最新 df, 是否发生写入更新)
#     """
#     remote = fetch_btcusd_daily_primary_with_backup(csv_path)

#     if not csv_path.exists():
#         ensure_dir(csv_path.parent)
#         remote.to_csv(csv_path, index=False)
#         return remote, True

#     local = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
#     if "price_usd" not in local.columns:
#         raise RuntimeError(f"本地 CSV 不是标准格式（缺少 price_usd）：{csv_path}")

#     merged = (
#         pd.concat([local, remote], ignore_index=True)
#         .drop_duplicates(subset=["date"], keep="last")
#         .sort_values("date")
#         .reset_index(drop=True)
#     )

#     updated = (len(merged) != len(local)) or (merged["date"].iloc[-1] != local["date"].iloc[-1])
#     if updated:
#         merged.to_csv(csv_path, index=False)

#     return merged, updated


def update_price_csv(csv_path: Path) -> tuple[pd.DataFrame, bool, str]:
    """
    返回：(最新df, 是否更新, 本次数据源说明)
    规则：
      - 默认用 Stooq 全量 merge（稳）
      - 如果 Stooq 请求失败：用 CoinGecko 免费增量补
      - 如果 Stooq 成功但“没新增日期”（周末常见）：也尝试用 CoinGecko 免费增量补
        （这样周末也能补到新日线点）
    """
    source_note = "stooq"

    last_date = get_local_last_date(csv_path)

    # 1) 先尝试 Stooq
    stooq_df = None
    try:
        stooq_df = fetch_btcusd_daily_from_stooq(timeout=30)
    except Exception as e:
        stooq_df = None
        stooq_err = e

    # 2) 如果本地不存在，优先用可用的全量（Stooq），否则用 CoinGecko（会较少但至少能起）
    if not csv_path.exists():
        ensure_dir(csv_path.parent)
        if stooq_df is not None and not stooq_df.empty:
            stooq_df.to_csv(csv_path, index=False)
            return stooq_df, True, "stooq(主源)"
        inc = fetch_btcusd_daily_increment_from_coingecko(last_date=None, timeout=60)
        inc.to_csv(csv_path, index=False)
        return inc, True, "coingecko(备源-增量)"

    # 3) 读取本地
    local = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if "price_usd" not in local.columns:
        raise RuntimeError(f"本地 CSV 缺少 price_usd：{csv_path}")

    local_last = local["date"].iloc[-1] if not local.empty else None

    # 4) Stooq 成功：merge 全量
    if stooq_df is not None and not stooq_df.empty:
        merged = (
            pd.concat([local, stooq_df], ignore_index=True)
              .drop_duplicates(subset=["date"], keep="last")
              .sort_values("date")
              .reset_index(drop=True)
        )
        updated = (len(merged) != len(local)) or (merged["date"].iloc[-1] != local_last)

        # 4a) 如果 Stooq 没新增（周末常见），尝试用 CoinGecko 增量补
        #     目的：周末也尽量补到“新的一天”
        if not updated:
            #inc = fetch_btcusd_daily_increment_from_coingecko(last_date=local_last, timeout=60)
            inc = safe_fetch_btcusd_increment_from_coingecko(last_date=local_last, timeout=60)
            if not inc.empty:
                merged2 = (
                    pd.concat([merged, inc], ignore_index=True)
                      .drop_duplicates(subset=["date"], keep="last")
                      .sort_values("date")
                      .reset_index(drop=True)
                )
                updated2 = (len(merged2) != len(merged)) or (merged2["date"].iloc[-1] != merged["date"].iloc[-1])
                if updated2:
                    merged2.to_csv(csv_path, index=False)
                    return merged2, True, "coingecko(周末备源-增量)"
            else:
                return merged, False, "stooq(主源；周末可能无更新)"
        # 正常写回（Stooq 有变化）
        if updated:
            merged.to_csv(csv_path, index=False)
        return merged, updated, "stooq(主源)"

    # 5) Stooq 失败：用 CoinGecko 增量补
    inc = fetch_btcusd_daily_increment_from_coingecko(last_date=local_last, timeout=60)
    if inc.empty:
        # 兜底：Stooq 失败 + CoinGecko 也没取到新增，就返回原数据但标记原因
        return local, False, f"coingecko(备源-无新增；stooq失败:{type(stooq_err).__name__})"
    merged = (
        pd.concat([local, inc], ignore_index=True)
          .drop_duplicates(subset=["date"], keep="last")
          .sort_values("date")
          .reset_index(drop=True)
    )
    merged.to_csv(csv_path, index=False)
    return merged, True, "coingecko(备源-增量；stooq失败)"


# ----------------------------
# 2) 分析：特征、相似行情、聚类状态、风险识别
# ----------------------------
def load_and_features(csv_path: Path) -> pd.DataFrame:
    """
    输入 CSV 必须包含列：date, price_usd
    输出包含 ret/vol/trend 等特征，dropna 后用于分析
    """
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if "price_usd" not in df.columns:
        raise ValueError("CSV 必须包含列：price_usd（本脚本会在更新阶段保证）")

    df["price_usd"] = df["price_usd"].astype(float)
    df["log_price"] = np.log(df["price_usd"])
    df["ret"] = df["log_price"].diff()  # 日对数收益

    df["vol_7d"] = df["ret"].rolling(7).std()
    df["vol_30d"] = df["ret"].rolling(30).std()

    df["ma_30"] = df["price_usd"].rolling(30).mean()
    df["trend_30d"] = df["price_usd"] / df["ma_30"] - 1

    df = df.dropna().reset_index(drop=True)
    if len(df) < 80:
        raise ValueError(f"有效样本太短（{len(df)} 行），建议至少 80 天以上")
    return df


def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["ret"].dropna()
    vol30 = df["vol_30d"].dropna()
    out = {
        "开始日期": df["date"].iloc[0].date(),
        "结束日期": df["date"].iloc[-1].date(),
        "样本天数": len(df),
        "日对数收益均值": float(ret.mean()),
        "日对数收益中位数": float(ret.median()),
        "日收益为正比例(胜率)": float((ret > 0).mean()),
        "30日年化波动均值(≈)": float(vol30.mean() * np.sqrt(365)),
    }
    return pd.DataFrame([out])


def build_window_matrix(df: pd.DataFrame, window: int) -> Tuple[np.ndarray, np.ndarray]:
    feats = ["ret", "vol_7d", "trend_30d"]
    X, start_idx = [], []
    for i in range(len(df) - window):
        w = df.iloc[i:i + window][feats].values
        X.append(w.flatten())
        start_idx.append(i)
    return np.asarray(X), np.asarray(start_idx)


@dataclass
class SimilarPack:
    table: pd.DataFrame
    dist: pd.DataFrame
    hist_path: Path


def similar_windows_future_dist(
    df: pd.DataFrame,
    out_dir: Path,
    report_date: str,
    window: int = 20,
    topk: int = 12,
    horizon: int = 10,
) -> SimilarPack:
    X, start_idx = build_window_matrix(df, window=window)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    q = Xs[-1].reshape(1, -1)
    sims = cosine_similarity(q, Xs)[0]

    idx_sorted = np.argsort(sims)[::-1]
    idx_sorted = idx_sorted[idx_sorted != (len(Xs) - 1)]
    top = idx_sorted[:topk]

    rows = []
    for j in top:
        i = int(start_idx[j])
        w_start = df.iloc[i]["date"].date()
        w_end = df.iloc[i + window - 1]["date"].date()

        future_slice = df.iloc[i + window:i + window + horizon]
        if len(future_slice) < horizon:
            continue
        future_ret = float(future_slice["ret"].sum())

        rows.append({
            "相似窗口开始": w_start,
            "相似窗口结束": w_end,
            "相似度": float(sims[j]),
            f"未来{horizon}天对数收益": future_ret
        })

    table = pd.DataFrame(rows).sort_values("相似度", ascending=False).reset_index(drop=True)
    if table.empty:
        raise RuntimeError("相似窗口结果为空：可能数据太短或 horizon/window 参数过大")

    col = f"未来{horizon}天对数收益"
    vals = table[col].values

    dist = pd.DataFrame([{
        "样本数": int(len(vals)),
        "均值": float(np.mean(vals)),
        "标准差": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
        "最小值(min)": float(np.min(vals)),
        "10%分位(p10)": float(np.quantile(vals, 0.10)),
        "中位数(p50)": float(np.quantile(vals, 0.50)),
        "90%分位(p90)": float(np.quantile(vals, 0.90)),
        "最大值(max)": float(np.max(vals)),
        "胜率(>0)": float(np.mean(vals > 0)),
    }])

    ensure_dir(out_dir)
    hist_path = out_dir / f"future_hist_{report_date}.png"

    # 解决中文字体 warning：尽量选系统可用字体（不会因缺字体而失败）
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure()
    plt.hist(vals, bins=min(12, max(5, len(vals))))
    plt.axvline(np.mean(vals))
    plt.title(f"相似行情下未来{horizon}天收益分布（Top{topk}, 窗口{window}天）")
    plt.xlabel("未来对数收益")
    plt.ylabel("次数")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()

    return SimilarPack(table=table, dist=dist, hist_path=hist_path)


@dataclass
class ClusterPack:
    df_state: pd.DataFrame
    state_means: pd.DataFrame


def cluster_market_states(df: pd.DataFrame, k: int = 6) -> ClusterPack:
    feats = ["ret", "vol_7d", "vol_30d", "trend_30d"]
    X = df[feats].values
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(Xs)

    df_state = df.copy()
    df_state["状态"] = labels

    means = df_state.groupby("状态")[feats].mean().reset_index()
    means = means.rename(columns={
        "ret": "日收益均值",
        "vol_7d": "7日波动均值",
        "vol_30d": "30日波动均值",
        "trend_30d": "30日趋势均值",
    })
    return ClusterPack(df_state=df_state, state_means=means)


def risk_by_state(df_state: pd.DataFrame, horizon: int = 10) -> Tuple[pd.DataFrame, int]:
    df = df_state.copy()
    df["未来收益"] = df["ret"].rolling(horizon).sum().shift(-horizon)
    df = df.dropna(subset=["未来收益"]).reset_index(drop=True)

    g = df.groupby("状态")["未来收益"]
    risk = pd.DataFrame({
        "状态": g.size().index,
        "样本数": g.size().values,
        "均值": g.mean().values,
        "胜率(>0)": g.apply(lambda x: float((x > 0).mean())).values,
        "p10(更看风险)": g.quantile(0.10).values,
        "p25": g.quantile(0.25).values,
        "min(最差)": g.min().values,
        "max(最好)": g.max().values,
    })

    risk = risk.sort_values(["均值", "p10(更看风险)", "胜率(>0)"], ascending=[True, True, True]).reset_index(drop=True)
    worst_state = int(risk.iloc[0]["状态"])
    return risk, worst_state


# ----------------------------
# 3) 报告：术语附录
# ----------------------------
def glossary_section() -> str:
    lines = []
    lines.append("\n\n---\n")
    lines.append("## 📘 附录：报告术语白话解释（给不懂量化的读者）\n")

    lines.append("### 量化分析\n")
    lines.append("用数据和统计方法约束判断，不靠感觉；这份报告不做“方向预测”。\n\n")

    lines.append("### 分布\n")
    lines.append("不是一个结果，而是一组可能结果的范围。左边通常代表亏损，右边代表盈利。\n\n")

    lines.append("### 市场状态 / 状态编号（k=6）\n")
    lines.append("把市场按行为特征分成 6 类环境；编号只是标签，是否危险要看风险统计。\n\n")

    lines.append("### 波动（7日/30日）\n")
    lines.append("价格上下晃动的剧烈程度。波动越大，短期越不稳定。\n\n")

    lines.append("### 趋势（30日趋势）\n")
    lines.append("最近 30 天整体偏向上涨还是下跌。\n\n")

    lines.append("### 未来窗口（horizon，如未来10天）\n")
    lines.append("用来统计“类似情况下后面通常发生什么”，不是预测。\n\n")

    lines.append("### p10（10%分位数）/ min（最差）\n")
    lines.append("p10 表示最差的 10% 情况通常会亏多少；min 是历史最极端一次亏损，用来理解左尾风险。\n\n")

    lines.append("### 最容易亏钱的状态\n")
    lines.append("综合均值、p10、胜率后历史上更不友好的环境，不代表必亏，只表示更容易出现不利分布。\n\n")

    lines.append("### 如何用这份报告\n")
    lines.append("它不是告诉你怎么赢，而是帮助你避免在历史上不友好的环境里做激进决策。\n")

    return "".join(lines)


# ----------------------------
# 4) 报告生成（Markdown + 图片引用）
# ----------------------------
def generate_report(
    csv_path: Path,
    out_dir: Path,
    k: int,
    window: int,
    topk: int,
    horizon: int,
) -> Tuple[Path, str]:
    df = load_and_features(csv_path)
    latest_date = df["date"].iloc[-1].date()
    report_date = str(latest_date)

    base = basic_summary(df)

    cluster = cluster_market_states(df, k=k)
    current_state = int(cluster.df_state.iloc[-1]["状态"])

    risk_table, worst_state = risk_by_state(cluster.df_state, horizon=horizon)

    sim = similar_windows_future_dist(
        df=df,
        out_dir=out_dir,
        report_date=report_date,
        window=window,
        topk=topk,
        horizon=horizon,
    )

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: List[str] = []
    lines.append("# BTC 量化分析入门版报告（结构 / 分布 / 风险）\n")
    lines.append(f"> 生成时间：{now_utc}  ｜  数据截止：{report_date}\n")

    lines.append("## 阅读指南（给不懂量化的人）\n")
    lines.append("- 这份报告**不预测价格**，只回答：在当前环境下，未来结果的**分布**大致如何。\n")
    lines.append("- 重点关注：**p10 / min（左尾风险）**、胜率、以及当前是否接近“最容易亏钱状态”。\n")
    lines.append("- 如果当前状态=最容易亏钱状态：优先考虑**降低风险暴露**，而不是加大操作频率。\n")

    lines.append("\n## 0）市场基本画像（描述世界，不做判断）\n")
    lines.append(df_to_md_table(base, max_rows=1))

    lines.append("\n\n## 1）市场状态（市场不是涨跌，而是状态切换）\n")
    lines.append(f"- 当前状态编号：**{current_state}**（k={k}）\n")
    lines.append("### 各状态的平均特征（用于解释状态大概是什么环境）\n")
    lines.append(df_to_md_table(cluster.state_means, max_rows=50))

    lines.append("\n\n## 2）风险识别（看左尾，而不是只看均值）\n")
    lines.append(f"- 历史上最容易亏钱的状态编号：**{worst_state}**（按均值+p10+胜率综合排序）\n")
    lines.append("### 各状态未来收益分布摘要（未来窗口为 horizon 天）\n")
    lines.append(df_to_md_table(risk_table, max_rows=50))

    lines.append("\n\n## 3）相似行情 → 未来分布（分布，不是预测）\n")
    lines.append("### 未来收益分布摘要（在“最相似的历史片段”条件下）\n")
    lines.append(df_to_md_table(sim.dist, max_rows=1))

    lines.append("\n\n### 收益分布直方图\n")
    # 使用相对路径引用图片，确保 Markdown 可显示
    lines.append(f"![相似行情下未来{horizon}天收益分布]({sim.hist_path.name})\n")

    lines.append("\n\n### 历史最相似的行情片段（Top 相似度）\n")
    lines.append(df_to_md_table(sim.table, max_rows=topk))

    lines.append("\n\n## 4）结论提醒（不做预测，只做风险约束）\n")
    if current_state == worst_state:
        lines.append("- **当前状态 == 最容易亏钱状态**：历史上在这种环境下，未来收益分布更不友好，建议优先控制风险。\n")
    else:
        lines.append("- 当前状态与“最容易亏钱状态”不同：不代表一定有利，但历史上风险结构相对没那么差。\n")
    lines.append("- 任何单次结果都可能偏离统计分布；报告价值在于帮助你避免“在不利环境下高频决策”。\n")

    # 附录：术语解释
    lines.append(glossary_section())

    ensure_dir(out_dir)
    stable_path = out_dir / "report.md"
    dated_path = out_dir / f"report_{report_date}.md"

    stable_path.write_text("\n".join(lines), encoding="utf-8")
    dated_path.write_text("\n".join(lines), encoding="utf-8")

    return stable_path, report_date


# ----------------------------
# 5) 邮件：MD + HTML（可选 PDF）
# ----------------------------
def md_to_html_simple(md_text: str) -> str:
    """
    Markdown -> HTML：
    - 若安装了 markdown 库，则支持更好的表格渲染
    - 否则用 <pre> 保底
    """
    try:
        import markdown  # pip install markdown
        body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>BTC Report</title></head>
<body>{body}</body></html>"""
    except Exception:
        safe = md_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>BTC Report</title></head>
<body><pre style="white-space:pre-wrap">{safe}</pre></body></html>"""


def try_make_pdf_with_pandoc(md_path: Path, pdf_path: Path) -> bool:
    """
    尝试用 pandoc 把 md 转 pdf（可选）。
    成功返回 True，否则 False（不会让程序失败）。
    """
    try:
        subprocess.run(
            ["pandoc", str(md_path), "-o", str(pdf_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except Exception:
        return False


# def send_email_gmail(subject: str, body_text: str, attachments: List[Path]) -> None:
#     """
#     Gmail SMTP SSL 465 发送邮件。
#     从环境变量取：
#       SMTP_USER / SMTP_PASS / TO_EMAIL
#     """
#     smtp_user = (os.getenv("SMTP_USER") or "").strip()
#     smtp_pass = (os.getenv("SMTP_PASS") or "").strip()
#     to_email = (os.getenv("TO_EMAIL") or "").strip()

#     if not smtp_user or not smtp_pass or not to_email:
#         raise RuntimeError("缺少环境变量：SMTP_USER / SMTP_PASS / TO_EMAIL（建议用 .env 配置）")

#     msg = smtplib.email.message.EmailMessage()  # type: ignore[attr-defined]
#     # 兼容旧版本：如果上面报错，就走 MIMEMultipart
#     try:
#         from email.message import EmailMessage
#         msg = EmailMessage()
#         msg["From"] = smtp_user
#         msg["To"] = to_email
#         msg["Subject"] = subject
#         msg.set_content(body_text)

#         for p in attachments:
#             if not p.exists():
#                 continue
#             data = p.read_bytes()
#             # 简单按扩展名设置类型
#             ext = p.suffix.lower()
#             if ext == ".md":
#                 maintype, subtype = "text", "markdown"
#             elif ext == ".html":
#                 maintype, subtype = "text", "html"
#             elif ext == ".pdf":
#                 maintype, subtype = "application", "pdf"
#             elif ext == ".png":
#                 maintype, subtype = "image", "png"
#             else:
#                 maintype, subtype = "application", "octet-stream"
#             msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=p.name)

#         context = ssl.create_default_context()
#         with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
#             server.login(smtp_user, smtp_pass)
#             server.send_message(msg)
#         return

#     except Exception:
#         # 回退到更传统的 MIME 方式
#         from email.mime.text import MIMEText
#         from email.mime.multipart import MIMEMultipart
#         from email.mime.application import MIMEApplication

#         m = MIMEMultipart()
#         m["From"] = smtp_user
#         m["To"] = to_email
#         m["Subject"] = subject
#         m.attach(MIMEText(body_text, "plain", "utf-8"))

#         for p in attachments:
#             if not p.exists():
#                 continue
#             part = MIMEApplication(p.read_bytes(), Name=p.name)
#             part["Content-Disposition"] = f'attachment; filename="{p.name}"'
#             m.attach(part)

#         context = ssl.create_default_context()
#         with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
#             server.login(smtp_user, smtp_pass)
#             server.sendmail(smtp_user, [to_email], m.as_string())


# def email_report_after_generated(out_dir: Path, report_date: str) -> None:
#     """
#     发送附件：
#     - 必发：report_YYYY-MM-DD.md（或 report.md）
#     - 附加：report_YYYY-MM-DD.html（自动生成）
#     - 可选：report_YYYY-MM-DD.pdf（若 pandoc 可用）
#     """
#     md_path = out_dir / f"report_{report_date}.md"
#     if not md_path.exists():
#         md_path = out_dir / "report.md"
#     if not md_path.exists():
#         raise FileNotFoundError(f"找不到报告文件：{md_path}")

#     md_text = md_path.read_text(encoding="utf-8", errors="ignore")

#     attachments: List[Path] = [md_path]

#     # HTML（自动生成）
#     html_path = out_dir / f"report_{report_date}.html"
#     html_path.write_text(md_to_html_simple(md_text), encoding="utf-8")
#     attachments.append(html_path)

#     # PDF（可选）
#     pdf_path = out_dir / f"report_{report_date}.pdf"
#     if try_make_pdf_with_pandoc(md_path, pdf_path):
#         attachments.append(pdf_path)

#     subject = f"BTC 分析报告 {report_date}"
#     body = (
#         f"你好，\n\n"
#         f"已生成 BTC 分析报告（日期：{report_date}）。\n"
#         f"- 附件：Markdown + HTML（若系统支持则附 PDF）。\n\n"
#         f"—— 自动化报告系统"
#     )

#     send_email_gmail(subject=subject, body_text=body, attachments=attachments)


## inline

from email.message import EmailMessage

def md_to_html_email(md_text: str, inline_hist_cid: str | None = None) -> str:
    """
    Markdown -> HTML（用于邮件正文）
    - 优先使用 markdown 库渲染表格
    - 并可选：把报告里的直方图替换成内嵌图片（cid）
    """
    # 先把 markdown 中对 png 的引用替换成 cid
    # 例如：![xxx](future_hist_2026-01-30.png) -> <img src="cid:hist" ...>
    # 这里我们用一个简单替换：只要提供 cid，就把任何 future_hist_*.png 替换为 cid
    if inline_hist_cid:
        import re
        md_text = re.sub(
            r"!\[([^\]]*)\]\((future_hist_[^)]+\.png)\)",
            r'<p><img alt="\1" src="cid:' + inline_hist_cid + r'" style="max-width:100%;height:auto;"></p>',
            md_text
        )

    try:
        import markdown  # pip install markdown
        body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    except Exception:
        safe = md_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body = f"<pre style='white-space:pre-wrap'>{safe}</pre>"

    # 加一点最基础的 HTML 样式，让表格好读
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>BTC Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; line-height: 1.5; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 8px; }}
  th {{ background: #f6f6f6; }}
  code, pre {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
  pre {{ padding: 10px; overflow-x: auto; }}
</style>
</head>
<body>
{body}
</body>
</html>"""
    return html


def send_email_gmail_inline_report(
    subject: str,
    text_fallback: str,
    html_body: str,
    inline_images: list[tuple[Path, str]] | None = None,  # [(path, cid)]
    attachments: list[Path] | None = None,                # 可选备份附件
) -> None:
    """
    Gmail SMTP SSL 465 发送邮件：
    - 邮件正文：HTML（可读） + 纯文本fallback
    - 可内嵌图片（cid）
    - 可选附件（比如 md / html / pdf 备份）
    """
    smtp_user = (os.getenv("SMTP_USER") or "").strip()
    smtp_pass = (os.getenv("SMTP_PASS") or "").strip()
    to_email = (os.getenv("TO_EMAIL") or "").strip()
    if not smtp_user or not smtp_pass or not to_email:
        raise RuntimeError("缺少环境变量：SMTP_USER / SMTP_PASS / TO_EMAIL（建议用 .env 配置）")

    msg = EmailMessage()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = subject

    # 纯文本版本（当邮箱不支持 HTML 时显示）
    msg.set_content(text_fallback)

    # HTML 正文
    msg.add_alternative(html_body, subtype="html")

    # 内嵌图片：必须加到 HTML 那个部分里（即 payload[-1]）
    if inline_images:
        html_part = msg.get_payload()[-1]
        for img_path, cid in inline_images:
            if not img_path.exists():
                continue
            data = img_path.read_bytes()
            maintype, subtype = "image", img_path.suffix.lower().lstrip(".") or "png"
            html_part.add_related(data, maintype=maintype, subtype=subtype, cid=f"<{cid}>")

    # 可选附件：作为备份（你想要可保留，不想要可不加）
    for p in (attachments or []):
        if not p.exists():
            continue
        data = p.read_bytes()
        ext = p.suffix.lower()
        if ext == ".md":
            maintype, subtype = "text", "markdown"
        elif ext == ".html":
            maintype, subtype = "text", "html"
        elif ext == ".pdf":
            maintype, subtype = "application", "pdf"
        elif ext == ".png":
            maintype, subtype = "image", "png"
        else:
            maintype, subtype = "application", "octet-stream"
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=p.name)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)


def email_report_as_body(out_dir: Path, report_date: str, attach_backup: bool = False) -> None:
    """
    把报告内容直接作为邮件正文发送（HTML + 文本备份）。
    - 默认不发附件（更像日报）
    - 如 attach_backup=True，可附带 md/html/pdf 作为备份
    """
    md_path = out_dir / f"report_{report_date}.md"
    if not md_path.exists():
        md_path = out_dir / "report.md"
    if not md_path.exists():
        raise FileNotFoundError(f"找不到报告文件：{md_path}")

    md_text = md_path.read_text(encoding="utf-8", errors="ignore")

    # 内嵌直方图
    hist_path = out_dir / f"future_hist_{report_date}.png"
    cid = "hist"
    html = md_to_html_email(md_text, inline_hist_cid=cid if hist_path.exists() else None)

    subject = f"BTC 分析报告 {report_date}"
    text_fallback = (
        f"BTC 分析报告 {report_date}\n\n"
        f"你的邮箱客户端可能不支持 HTML 显示。\n"
        f"请查看邮件 HTML 正文，或启用附件备份。\n\n"
        f"---\n\n{md_text}"
    )

    inline_images = []
    if hist_path.exists():
        inline_images.append((hist_path, cid))

    attachments = []
    if attach_backup:
        attachments.append(md_path)
        html_path = out_dir / f"report_{report_date}.html"
        html_path.write_text(html, encoding="utf-8")
        attachments.append(html_path)

        pdf_path = out_dir / f"report_{report_date}.pdf"
        if try_make_pdf_with_pandoc(md_path, pdf_path):
            attachments.append(pdf_path)

    send_email_gmail_inline_report(
        subject=subject,
        text_fallback=text_fallback,
        html_body=html,
        inline_images=inline_images,
        attachments=attachments,
    )



# ----------------------------
# 6) CLI + 主流程：更新 -> 分析 -> 报告 -> 邮件
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BTC 量化分析入门版：更新数据 -> 分析 -> 报告 -> 邮件")
    p.add_argument("--csv", default="./bitcoin_price_history_usd.csv", help="本地 CSV 路径（标准格式 date,price_usd）")
    p.add_argument("--out-dir", default="./reports", help="输出目录（report.md + 图片）")

    p.add_argument("--k", type=int, default=6, help="状态聚类数量（建议 5-7）")
    p.add_argument("--window", type=int, default=20, help="相似行情窗口长度（天）")
    p.add_argument("--topk", type=int, default=12, help="最相似窗口数量")
    p.add_argument("--horizon", type=int, default=10, help="未来分布统计窗口（天）")

    p.add_argument("--no-email", action="store_true", help="只生成报告，不发邮件")
    p.add_argument("--no-update", action="store_true", help="不更新数据，直接用本地 CSV 做分析（调试用）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if args.k < 2 or args.k > 20:
        raise ValueError("--k 建议在 2~20")
    if args.window < 5 or args.window > 120:
        raise ValueError("--window 建议在 5~120")
    if args.topk < 3:
        raise ValueError("--topk 至少 3")
    if args.horizon < 1 or args.horizon > 60:
        raise ValueError("--horizon 建议在 1~60")

    # 1) 更新数据（Stooq -> 标准化 -> 合并）
    if not args.no_update:
        #df_price, updated = update_price_csv(csv_path)
        df_price, updated, source_note = update_price_csv(csv_path)
        print(f"✅ 数据已就绪：{csv_path}（{'有更新' if updated else '无新增'}，共 {len(df_price)} 行；数据源：{source_note}）")
        #print(f"✅ 数据已就绪：{csv_path}（{'有更新' if updated else '无新增'}，共 {len(df_price)} 行）")
    else:
        if not csv_path.exists():
            raise FileNotFoundError(f"--no-update 模式下找不到 CSV：{csv_path}")
        print(f"ℹ️ 跳过更新，直接使用本地 CSV：{csv_path}")

    # 2) 分析并生成报告
    report_path, report_date = generate_report(
        csv_path=csv_path,
        out_dir=out_dir,
        k=args.k,
        window=args.window,
        topk=args.topk,
        horizon=args.horizon,
    )
    print(f"✅ 报告已生成：{report_path}（日期：{report_date}）")

    # 3) 发送邮件
    if not args.no_email:
        #email_report_after_generated(out_dir=out_dir, report_date=report_date)
        email_report_as_body(out_dir=out_dir, report_date=report_date, attach_backup=True)
        print("📩 邮件已发送")
    else:
        print("ℹ️ --no-email 已启用：不发送邮件")


if __name__ == "__main__":
    main()

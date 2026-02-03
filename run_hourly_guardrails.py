#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os, json, time, uuid, math, sqlite3, hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import yaml
import urllib.request
import urllib.parse


EPS = 1e-12


# -----------------------------
# Time helpers
# -----------------------------
def now_tz(tz: str) -> datetime:
    return datetime.now(ZoneInfo(tz))

def floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)

def iso(dt: datetime) -> str:
    return dt.isoformat()


# -----------------------------
# DB helpers
# -----------------------------
def db_connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def db_get_state(conn: sqlite3.Connection, key: str, default: Dict[str, Any]) -> Dict[str, Any]:
    row = conn.execute("SELECT value FROM radar_state WHERE key=?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except Exception:
        return default

def db_set_state(conn: sqlite3.Connection, key: str, value: Dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO radar_state(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value, ensure_ascii=False))
    )

def db_upsert_hour_ohlcv(conn: sqlite3.Connection, ts_hour: str, ohlcv: Dict[str, Any], quality_flags: Dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO radar_hourly_ohlcv(ts_hour, open, high, low, close, volume, quality_flags) "
        "VALUES(?,?,?,?,?,?,?) "
        "ON CONFLICT(ts_hour) DO UPDATE SET open=excluded.open, high=excluded.high, low=excluded.low, "
        "close=excluded.close, volume=excluded.volume, quality_flags=excluded.quality_flags",
        (
            ts_hour,
            ohlcv.get("open"), ohlcv.get("high"), ohlcv.get("low"),
            ohlcv["close"], ohlcv.get("volume"),
            json.dumps(quality_flags, ensure_ascii=False),
        )
    )

def db_insert_run(conn: sqlite3.Connection, run_row: Dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO radar_runs(run_id, ts_hour, started_at, finished_at, status, error_code, error_msg, confidence, sources_used) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (
            run_row["run_id"], run_row["ts_hour"], run_row["started_at"], run_row["finished_at"],
            run_row["status"], run_row.get("error_code"), run_row.get("error_msg"),
            run_row["confidence"], json.dumps(run_row["sources_used"], ensure_ascii=False)
        )
    )

def db_insert_events(conn: sqlite3.Connection, ts_hour: str, events: List[Dict[str, Any]]) -> None:
    for e in events:
        conn.execute(
            "INSERT INTO radar_events(event_id, ts_hour, event_type, severity, title, details) "
            "VALUES(?,?,?,?,?,?)",
            (
                str(uuid.uuid4()),
                ts_hour,
                e["code"],
                e.get("severity", "info"),
                e.get("title", e["code"]),
                json.dumps(e.get("detail", {}), ensure_ascii=False),
            )
        )


# -----------------------------
# Basic stats
# -----------------------------
def stdev(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(v)

def median(xs: List[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return float("nan")
    mid = n // 2
    return ys[mid] if n % 2 == 1 else 0.5 * (ys[mid - 1] + ys[mid])

def mad(xs: List[float], med: float) -> float:
    return median([abs(x - med) for x in xs]) if xs else float("nan")

def pctl(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    idx = int(p * (len(ys) - 1))
    return ys[max(0, min(len(ys)-1, idx))]

#数据源
def _http_get_json(url: str, timeout_sec: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "btc-guardrails/0.1",
            "Accept": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)

def _fetch_binance_1h_kline_close(symbol: str, timeout_sec: float) -> Dict[str, Any]:
    """
    Try multiple Binance domains in order.
    First success wins; failures are silent and fall through.
    """
    domains = [
        "https://api.binance.com",
        "https://api1.binance.com",
        "https://api2.binance.com",
    ]

    last_error = None

    for base in domains:
        try:
            params = urllib.parse.urlencode({
                "symbol": symbol,
                "interval": "1h",
                "limit": 1,
            })
            url = f"{base}/api/v3/klines?{params}"

            data = _http_get_json(url, timeout_sec=timeout_sec)

            if (
                not isinstance(data, list)
                or len(data) < 1
                or not isinstance(data[0], list)
                or len(data[0]) < 6
            ):
                raise ValueError("unexpected kline shape")

            k = data[0]
            o = float(k[1])
            h = float(k[2])
            l = float(k[3])
            c = float(k[4])
            v = float(k[5])

            # ✅ 成功立刻返回
            return {
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }

        except Exception as e:
            last_error = e
            continue

    # 所有域名都失败
    raise RuntimeError(f"binance all domains failed: {last_error}")


def _fetch_coingecko_spot_price(timeout_sec: float) -> Dict[str, Any]:
    # CoinGecko /simple/price (public; returns {"bitcoin":{"usd":...}})
    # Note: If your environment is rate-limited, this is only used when primary fails.
    params = urllib.parse.urlencode({"ids": "bitcoin", "vs_currencies": "usd"})
    url = f"https://api.coingecko.com/api/v3/simple/price?{params}"
    data = _http_get_json(url, timeout_sec=timeout_sec)

    usd = data.get("bitcoin", {}).get("usd")
    if usd is None:
        raise ValueError("coingecko price missing")
    c = float(usd)
    # Only close available; OHLC unknown
    return {"open": None, "high": None, "low": None, "close": c, "volume": None}

def ingest_hour_ohlcv(cfg: Dict[str, Any], ts_hour: str) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    主源成功 => 不调用 CoinGecko（0 影响）
    主源失败 => CoinGecko 单次短超时兜底；仍失败 => 返回 close=None（但脚本继续跑完）
    """
    tz = cfg["app"]["timezone"]
    now_dt = now_tz(tz)
    ts_dt = datetime.fromisoformat(ts_hour)
    staleness_sec = max(0, int((now_dt - ts_dt).total_seconds()))

    timeout_sec = float(cfg["sources"].get("timeout_sec", 2))
    total_budget = float(cfg["sources"].get("total_budget_sec", 4))

    symbol_binance = cfg.get("market", {}).get("symbol_binance", "BTCUSDT")

    providers = sorted(cfg["sources"]["providers"], key=lambda x: x.get("priority", 999))
    sources_used: List[Dict[str, Any]] = []
    raw_inputs: List[Dict[str, Any]] = []

    close = None
    open_ = high = low = volume = None
    source_fail_count = 0

    start_time = time.time()

    def within_budget() -> bool:
        return (time.time() - start_time) <= total_budget

    for p in providers:
        if not p.get("enabled", True):
            continue
        if not within_budget():
            break

        name = p["name"]
        t0 = time.time()
        ok = False
        payload: Dict[str, Any] = {}

        try:
            # ---- primary: Binance klines ----
            if name == "binance_spot_klines":
                payload = _fetch_binance_1h_kline_close(symbol_binance, timeout_sec=timeout_sec)
                ok = payload.get("close") is not None

            # ---- fallback: CoinGecko spot ----
            elif name == "coingecko_simple_price":
                # 只有主源没成功才会走到这一步（因为我们会 break）
                payload = _fetch_coingecko_spot_price(timeout_sec=min(timeout_sec, 2.0))
                ok = payload.get("close") is not None

            else:
                raise ValueError(f"unknown provider: {name}")

        except Exception as e:
            payload = {"error": str(e)}
            ok = False

        latency_ms = int((time.time() - t0) * 1000)
        sources_used.append({"source": name, "latency_ms": latency_ms, "ok": ok})
        raw_inputs.append({
            "source": name,
            "fetched_at": iso(now_dt),
            "payload_json": payload,
            "payload_sha256": sha256_json(payload),
            "ok": ok,
            "latency_ms": latency_ms,
        })

        if ok:
            close = float(payload["close"])
            open_ = float(payload["open"]) if payload.get("open") is not None else None
            high = float(payload["high"]) if payload.get("high") is not None else None
            low = float(payload["low"]) if payload.get("low") is not None else None
            volume = float(payload["volume"]) if payload.get("volume") is not None else None
            # ✅ 成功即停止：保证 CoinGecko 不会影响正常跑完逻辑
            break
        else:
            source_fail_count += 1
            # 失败也不要重试太多：我们靠 failover + 降级，不靠死磕
            continue

    missing_ratio = 0.0 if close is not None else 1.0
    ohlcv = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}

    quality = {
        "missing_ratio": float(missing_ratio),
        "staleness_sec": int(staleness_sec),
        "source_fail_count": int(source_fail_count),
        "sources_used": sources_used,
    }
    return ohlcv, quality, raw_inputs






# -----------------------------
# History fetch (from radar db only)
# -----------------------------
def fetch_closes(conn: sqlite3.Connection, end_ts_hour: str, hours_back: int) -> List[Tuple[str, float]]:
    end_dt = datetime.fromisoformat(end_ts_hour)
    start_dt = end_dt - timedelta(hours=hours_back)
    rows = conn.execute(
        "SELECT ts_hour, close FROM radar_hourly_ohlcv WHERE ts_hour >= ? AND ts_hour <= ? ORDER BY ts_hour ASC",
        (start_dt.isoformat(), end_dt.isoformat())
    ).fetchall()
    out = []
    for r in rows:
        c = r["close"]
        if c is None:
            continue
        try:
            out.append((r["ts_hour"], float(c)))
        except Exception:
            continue
    return out

def returns_from_closes(closes: List[float]) -> List[float]:
    rets = []
    for i in range(1, len(closes)):
        rets.append(math.log(max(closes[i], EPS) / max(closes[i - 1], EPS)))
    return rets


# -----------------------------
# Confidence / quality
# -----------------------------
def compute_confidence(missing_ratio: float, staleness_sec: int, source_fail_count: int, cfg: Dict[str, Any]) -> float:
    conf = 1.0
    conf -= 0.6 * min(max(missing_ratio, 0.0), 1.0)
    if staleness_sec > cfg["quality"]["stale_warn_sec"]:
        conf -= 0.2
    if source_fail_count >= 1:
        conf -= 0.2
    if source_fail_count >= 2:
        conf -= 0.3
    return max(0.0, min(1.0, conf))


# -----------------------------
# Ingest (YOU plug your real sources)
# -----------------------------
def sha256_json(obj: Any) -> str:
    b = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

def ingest_hour_ohlcv(cfg: Dict[str, Any], ts_hour: str) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Return:
      ohlcv: {"open":..,"high":..,"low":..,"close":..,"volume":..}
      quality: {"missing_ratio":..,"staleness_sec":..,"source_fail_count":..,"sources_used":[...]}
      raw_inputs: [{source,fetched_at,payload_json,ok,latency_ms,payload_sha256}]
    """
    tz = cfg["app"]["timezone"]
    now_dt = now_tz(tz)
    ts_dt = datetime.fromisoformat(ts_hour)
    staleness_sec = max(0, int((now_dt - ts_dt).total_seconds()))

    providers = sorted(cfg["sources"]["providers"], key=lambda x: x.get("priority", 999))
    sources_used: List[Dict[str, Any]] = []
    raw_inputs: List[Dict[str, Any]] = []

    close = None
    open_ = high = low = volume = None
    source_fail_count = 0

    for p in providers:
        if not p.get("enabled", True):
            continue

        name = p["name"]
        t0 = time.time()

        ok = False
        payload: Dict[str, Any] = {}
        try:
            # --------- TODO: replace this block with real fetch ----------
            # payload should contain at least {"close": 123.4}
            payload = {"close": None}
            ok = payload.get("close") is not None
            # -----------------------------------------------------------
        except Exception as e:
            payload = {"error": str(e)}
            ok = False

        latency_ms = int((time.time() - t0) * 1000)
        sources_used.append({"source": name, "latency_ms": latency_ms, "ok": ok})
        raw_inputs.append({
            "source": name,
            "fetched_at": iso(now_dt),
            "payload_json": payload,
            "payload_sha256": sha256_json(payload),
            "ok": ok,
            "latency_ms": latency_ms,
        })

        if ok:
            close = float(payload["close"])
            open_ = float(payload.get("open")) if payload.get("open") is not None else None
            high = float(payload.get("high")) if payload.get("high") is not None else None
            low = float(payload.get("low")) if payload.get("low") is not None else None
            volume = float(payload.get("volume")) if payload.get("volume") is not None else None
            break
        else:
            source_fail_count += 1

        # total budget guard (optional)
        if (time.time() - t0) > cfg["sources"]["timeout_sec"]:
            pass

    missing_ratio = 0.0 if close is not None else 1.0
    ohlcv = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}

    quality = {
        "missing_ratio": float(missing_ratio),
        "staleness_sec": int(staleness_sec),
        "source_fail_count": int(source_fail_count),
        "sources_used": sources_used,
    }
    return ohlcv, quality, raw_inputs


# -----------------------------
# Guardrails decision
# -----------------------------
def decide_guardrails(
    conn: sqlite3.Connection,
    ts_hour: str,
    ohlcv: Dict[str, Any],
    quality: Dict[str, Any],
    cfg: Dict[str, Any],
    prev_state: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns:
      guardrails: {trade_allowed, risk_mode, position_cap}
      reasons: list[{code,severity,detail}]
      new_state: stored in radar_state
    """
    reasons: List[Dict[str, Any]] = []

    missing_ratio = quality["missing_ratio"]
    staleness_sec = quality["staleness_sec"]
    source_fail_count = quality["source_fail_count"]

    confidence = compute_confidence(missing_ratio, staleness_sec, source_fail_count, cfg)

    # --- Data outage / degraded ---
    # Track consecutive bad confidence (stateful)
    bad_count = int(prev_state.get("bad_conf_hours", 0))
    if confidence < cfg["confidence"]["fail_below"]:
        bad_count += 1
    else:
        bad_count = 0

    # Build returns/vol regime using our own radar DB (no v1 dependency)
    closes = fetch_closes(conn, ts_hour, hours_back=cfg["history"]["hours_back_for_vol"])
    close_vals = [c for _, c in closes if c is not None]

    ret_1h = None
    rv_24h = None
    z_vol = None
    shock_p99 = None

    if len(close_vals) >= 2 and ohlcv["close"] is not None and not math.isnan(ohlcv["close"]):
        rets = returns_from_closes(close_vals)
        ret_1h = rets[-1] if rets else None
        if len(rets) >= 24:
            rv_24h = stdev(rets[-24:])

    # Shock threshold: p99(abs(ret_1h)) over ~30d if available; else hard threshold
    absret_hist = []
    if len(close_vals) >= cfg["history"]["min_hours_for_shock"] + 1:
        rets_hist = returns_from_closes(close_vals)
        absret_hist = [abs(r) for r in rets_hist if r is not None]
        shock_p99 = pctl(absret_hist, 0.99)

    hard_absret = cfg["shock"]["hard_absret1h"]

    shock = False
    if ret_1h is not None:
        if (shock_p99 is not None and abs(ret_1h) >= shock_p99) or abs(ret_1h) >= hard_absret:
            shock = True
            reasons.append({"code": "SHOCK_MOVE", "severity": "critical",
                            "detail": {"abs_ret_1h": abs(ret_1h), "p99": shock_p99, "hard": hard_absret}})

    # Vol regime z-score using MAD on rv_24h history (derived from closes)
    # Build rv_24h history by sliding window on historical returns (cheap enough for MVP)
    rv_hist = []
    if len(close_vals) >= (24 + cfg["history"]["min_hours_for_zstats"]) + 1:
        rets_hist = returns_from_closes(close_vals)
        # use last ~30d rets; compute rolling 24h stdev
        for i in range(24, len(rets_hist)):
            window = rets_hist[i-24:i]
            v = stdev(window)
            if v is not None:
                rv_hist.append(v)

    if rv_24h is not None and len(rv_hist) >= cfg["history"]["min_hours_for_zstats"]:
        med = median(rv_hist)
        m = mad(rv_hist, med) + EPS
        z_vol = (rv_24h - med) / m

    # Trend break proxy: 3h return
    ret_3h = None
    if len(close_vals) >= 4:
        ret_3h = math.log(max(close_vals[-1], EPS) / max(close_vals[-4], EPS))  # 3 hours span
    dir_bad = (-ret_3h) if ret_3h is not None else None

    # --- decide mode ---
    mode = "normal"

    # Freeze conditions
    if bad_count >= cfg["quality"]["outage_confirm_hours"]:
        mode = "freeze"
        reasons.append({"code": "DATA_OUTAGE", "severity": "critical",
                        "detail": {"confidence": confidence, "bad_conf_hours": bad_count}})
    elif shock:
        mode = "freeze"
    else:
        # Cautious conditions
        if confidence < cfg["confidence"]["degraded_below"]:
            mode = "cautious"
            reasons.append({"code": "DATA_DEGRADED", "severity": "warn",
                            "detail": {"confidence": confidence, "staleness_sec": staleness_sec, "missing_ratio": missing_ratio}})
        if z_vol is not None and z_vol >= cfg["regime"]["z_vol_cautious"]:
            mode = "cautious"
            reasons.append({"code": "VOL_REGIME", "severity": "warn", "detail": {"z_vol": z_vol, "rv_24h": rv_24h}})
        # simple trend break: 3h drop threshold (hard, no zstats)
        if dir_bad is not None and dir_bad >= cfg["regime"]["dir_3h_drop_cautious"]:
            mode = "cautious"
            reasons.append({"code": "TREND_BREAK", "severity": "warn", "detail": {"ret_3h": ret_3h}})

    # Hysteresis: freeze解除慢一点 / cautious解除慢一点（stateful）
    # Track stable good hours
    good_count = int(prev_state.get("good_hours", 0))
    if mode == "normal" and confidence >= cfg["confidence"]["normal_requires_conf"]:
        good_count += 1
    else:
        good_count = 0

    prev_mode = prev_state.get("mode", "normal")

    if prev_mode == "freeze" and mode != "freeze":
        # require several good hours before leaving freeze
        if good_count < cfg["hysteresis"]["freeze_release_good_hours"]:
            mode = "freeze"
            reasons.append({"code": "HOLD_FREEZE", "severity": "info",
                            "detail": {"good_hours": good_count, "need": cfg["hysteresis"]["freeze_release_good_hours"]}})

    if prev_mode == "cautious" and mode == "normal":
        if good_count < cfg["hysteresis"]["cautious_release_good_hours"]:
            mode = "cautious"
            reasons.append({"code": "HOLD_CAUTIOUS", "severity": "info",
                            "detail": {"good_hours": good_count, "need": cfg["hysteresis"]["cautious_release_good_hours"]}})

    # Map to switches
    trade_allowed = (mode != "freeze")
    position_cap = 0.0 if mode == "freeze" else (cfg["guardrails"]["cautious_position_cap"] if mode == "cautious" else 1.0)

    guardrails = {
        "trade_allowed": bool(trade_allowed),
        "risk_mode": mode,
        "position_cap": float(position_cap),
    }

    # events for audit/alert (only on mode change + major causes)
    events: List[Dict[str, Any]] = []
    if mode != prev_mode:
        sev = "critical" if mode == "freeze" else ("warn" if mode == "cautious" else "info")
        events.append({
            "code": "MODE_CHANGE",
            "severity": sev,
            "title": f"Guardrails mode -> {mode}",
            "detail": {"from": prev_mode, "to": mode}
        })
    # include primary reason codes as events (optional)
    for r in reasons:
        if r["code"] in cfg["alerts"]["send_on"]:
            events.append({"code": r["code"], "severity": r["severity"], "title": r["code"], "detail": r["detail"]})

    new_state = {
        "mode": mode,
        "bad_conf_hours": bad_count,
        "good_hours": good_count,
        "last_ts_hour": ts_hour,
        "last_confidence": confidence,
    }

    debug = {
        "confidence": confidence,
        "ret_1h": ret_1h,
        "rv_24h": rv_24h,
        "z_vol": z_vol,
        "shock_p99": shock_p99,
    }

    return guardrails, reasons, new_state, events, debug


# -----------------------------
# Output writers
# -----------------------------
def write_json(path: str, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# def write_latest_and_archive(out_dir: str, ts_hour: str, payload: Dict[str, Any]) -> Tuple[str, str]:
#     Path(out_dir).mkdir(parents=True, exist_ok=True)
#     dt = datetime.fromisoformat(ts_hour)
#     archive = str(Path(out_dir) / f"guardrails_{dt.strftime('%Y-%m-%d_%H')}.json")
#     latest = str(Path(out_dir) / "guardrails_latest.json")
#     write_json(archive, payload)
#     write_json(latest, payload)
#     return latest, archive


# 强制No freeze
# def write_latest_and_archive(out_dir: str, ts_hour: str, payload: Dict[str, Any]) -> Tuple[str, str]:
#     """
#     Write guardrails latest & archive.
#     Policy patch:
#       - If data outage / confidence <= 0
#       - AND config.guardrails.no_freeze_on_outage == true
#       - THEN force output to cautious (never freeze)
#     """

#     Path(out_dir).mkdir(parents=True, exist_ok=True)
#     dt = datetime.fromisoformat(ts_hour)

#     # ---------- no-freeze-on-outage patch ----------
#     try:
#         cfg_guard = payload.get("config", {}).get("guardrails", {})
#         no_freeze = bool(cfg_guard.get("no_freeze_on_outage", False))

#         status = payload.get("status")
#         confidence = float(payload.get("confidence", 0.0))

#         is_outage = (status == "fail") or (confidence <= 0.0)

#         if no_freeze and is_outage:
#             cap = cfg_guard.get(
#                 "outage_position_cap",
#                 cfg_guard.get("cautious_position_cap", 0.5),
#             )
#             try:
#                 cap = float(cap)
#             except Exception:
#                 cap = 0.5

#             cap = max(0.0, min(1.0, cap))
#             trade_allowed = bool(cfg_guard.get("outage_trade_allowed", True))

#             # 🔒 强制覆盖为 cautious（永不 freeze）
#             payload["guardrails"] = {
#                 "trade_allowed": trade_allowed,
#                 "risk_mode": "cautious",
#                 "position_cap": cap,
#             }

#             # 标注原因，方便 Phase 1 / 人类理解
#             payload.setdefault("reasons", []).append({
#                 "code": "OUTAGE_NO_FREEZE",
#                 "severity": "warning",
#                 "detail": {
#                     "note": "data outage; forced cautious by policy",
#                     "confidence": confidence,
#                 },
#             })
#     except Exception:
#         # 写文件函数里绝不抛异常，避免影响主流程
#         pass
#     # ---------- end patch ----------

#     archive = str(Path(out_dir) / f"guardrails_{dt.strftime('%Y-%m-%d_%H')}.json")
#     latest = str(Path(out_dir) / "guardrails_latest.json")

#     write_json(archive, payload)
#     write_json(latest, payload)

#     return latest, archive


def write_latest_and_archive(out_dir: str, ts_hour: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dt = datetime.fromisoformat(ts_hour)
    archive = str(Path(out_dir) / f"guardrails_{dt.strftime('%Y-%m-%d_%H')}.json")
    latest = str(Path(out_dir) / "guardrails_latest.json")

    # --- NEVER FREEZE ON OUTAGE (hard policy) ---
    try:
        status = payload.get("status")
        confidence = float(payload.get("confidence", 0.0))
        is_outage = (status == "fail") or (confidence <= 0.0)

        if is_outage:
            payload["guardrails"] = {
                "trade_allowed": True,
                "risk_mode": "cautious",
                "position_cap": 0.5,
            }
            payload.setdefault("reasons", []).append({
                "code": "OUTAGE_NO_FREEZE",
                "severity": "warning",
                "detail": {"note": "data outage; forced cautious (never freeze)"},
            })
    except Exception:
        pass
    # --- END POLICY ---

    write_json(archive, payload)
    write_json(latest, payload)
    return latest, archive

# -----------------------------
# Main
# -----------------------------
def main() -> int:
    # ------------------------------------
    # cfg_path = os.environ.get("RADAR_CONFIG", "config.guardrails.yaml")
    # with open(cfg_path, "r", encoding="utf-8") as f:
    #     cfg = yaml.safe_load(f)
    # -----------------------------
    # Load config (no env required)
    # -----------------------------
    script_dir = Path(__file__).resolve().parent
    cfg_path = script_dir / "config.guardrails.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tz = cfg["app"]["timezone"]
    ts_hour_dt = floor_to_hour(now_tz(tz))
    ts_hour = iso(ts_hour_dt)

    run_id = f"{ts_hour_dt.strftime('%Y%m%d%H')}_{uuid.uuid4().hex[:8]}"
    started_at = iso(now_tz(tz))

    conn = db_connect(cfg["db"]["path"])

    try:
        conn.execute("BEGIN;")

        # state
        prev_state = db_get_state(conn, "guardrails", {"mode": "normal", "bad_conf_hours": 0, "good_hours": 0})

        # ingest
        ohlcv, quality, raw_inputs = ingest_hour_ohlcv(cfg, ts_hour)

        # persist input snapshots (optional): if you kept radar_inputs table from full schema, you can store here.
        # for minimal schema, we skip it to stay tiny.

        # quality flags + confidence
        confidence = compute_confidence(
            quality["missing_ratio"], quality["staleness_sec"], quality["source_fail_count"], cfg
        )
        status = "ok"
        if confidence < cfg["confidence"]["fail_below"]:
            status = "fail"
        elif confidence < cfg["confidence"]["degraded_below"]:
            status = "degraded"

        # store ohlcv if close exists; if missing, still store NaN close? better: skip close insert when None
        # Here: if close is None, we do not upsert ohlcv row (avoid polluting history)
        if ohlcv.get("close") is not None:
            db_upsert_hour_ohlcv(conn, ts_hour, ohlcv, {
                "missing_ratio": quality["missing_ratio"],
                "staleness_sec": quality["staleness_sec"],
                "source_fail_count": quality["source_fail_count"]
            })

        guardrails, reasons, new_state, events, debug = decide_guardrails(
            conn, ts_hour, ohlcv, quality, cfg, prev_state
        )

        # Persist state + events
        db_set_state(conn, "guardrails", new_state)
        if events:
            db_insert_events(conn, ts_hour, events)

        finished_at = iso(now_tz(tz))
        db_insert_run(conn, {
            "run_id": run_id,
            "ts_hour": ts_hour,
            "started_at": started_at,
            "finished_at": finished_at,
            "status": status,
            "error_code": None,
            "error_msg": None,
            "confidence": confidence,
            "sources_used": quality["sources_used"],
        })

        # write outputs
        payload = {
            "ts_hour": ts_hour,
            "status": status,
            "confidence": confidence,
            "guardrails": guardrails,
            "reasons": reasons,
            "ttl_minutes": cfg["guardrails"]["ttl_minutes"],
            "debug": debug if cfg["outputs"].get("include_debug", False) else None,
            "version": {"schema": cfg["app"]["schema_version"], "rules": cfg["app"]["rules_version"]},
        }
        # remove debug if None for cleanliness
        if payload["debug"] is None:
            payload.pop("debug", None)

        latest, archive = write_latest_and_archive(cfg["outputs"]["json_dir"], ts_hour, payload)

        conn.execute("COMMIT;")
        # print(f"[OK] {ts_hour} mode={guardrails['risk_mode']} cap={guardrails['position_cap']} conf={confidence:.2f}")
        # print(f"[OK] latest: {latest}")
        # print(f"[OK] archive: {archive}")
        # ✅ 用最终写出的 guardrails 来打印
        final_gr = payload.get("guardrails", {})
        print(f"[OK] {ts_hour} mode={final_gr.get('risk_mode')} cap={final_gr.get('position_cap')} conf={payload.get('confidence'):.2f}")
        print(f"[OK] latest: {latest}")
        print(f"[OK] archive: {archive}") 
        return 0

    except Exception as e:
        conn.execute("ROLLBACK;")
        finished_at = iso(now_tz(cfg["app"]["timezone"]))
        try:
            conn.execute("BEGIN;")
            db_insert_run(conn, {
                "run_id": run_id,
                "ts_hour": ts_hour,
                "started_at": started_at,
                "finished_at": finished_at,
                "status": "fail",
                "error_code": type(e).__name__,
                "error_msg": str(e)[:300],
                "confidence": 0.0,
                "sources_used": [],
            })
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")

        print(f"[FAIL] {ts_hour} {type(e).__name__}: {e}")
        return 2
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
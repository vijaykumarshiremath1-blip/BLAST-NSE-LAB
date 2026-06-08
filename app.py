import io
import os
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, dcc, html, dash_table, no_update
from flask_caching import Cache
from plotly.subplots import make_subplots

NSE_EQUITY_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
DEFAULT_SYMBOL = "RELIANCE"
TOP_SCAN_LIMIT = 40
LOOKBACK_DAYS = 260
NSE_HOME = "https://www.nseindia.com"
NSE_REPORTS_URL = "https://www.nseindia.com/all-reports"
HIST_CACHE_DIR = Path("nse_hist_cache")
HIST_CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_HOME + "/",
}
CSV_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/csv,application/zip,application/octet-stream,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_HOME + "/",
}
THEME = {
    "bg": "#0b1220",
    "bg2": "#111a2b",
    "bg3": "#162033",
    "panel": "#121c2d",
    "panel2": "#172235",
    "border": "#26354f",
    "text": "#e6edf7",
    "muted": "#9fb0c8",
    "faint": "#6d7f99",
    "accent": "#3dd9b4",
    "accent2": "#5aa8ff",
    "warn": "#ffcc66",
    "danger": "#ff6b8a",
    "good": "#34d399",
    "bad": "#fb7185",
}
SIGNAL_COLORS = {
    "STRONG BUY": THEME["good"],
    "BUY": "#6ee7b7",
    "HOLD": THEME["warn"],
    "SELL": "#fda4af",
    "STRONG SELL": THEME["danger"],
    "NIL": THEME["faint"],
}
DEFAULT_INDICATORS = ["SMA20", "SMA50", "SMA200", "Bollinger", "Supertrend", "MACD", "RSI"]

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Blast NSE Lab Pro"
cache = Cache(app.server, config={
    "CACHE_TYPE": "FileSystemCache",
    "CACHE_DIR": "cache-directory",
    "CACHE_THRESHOLD": 4000,
    "CACHE_DEFAULT_TIMEOUT": 3600,
})

session = requests.Session()
session.headers.update(HEADERS)
try:
    session.get(NSE_HOME, timeout=15)
except Exception:
    pass


def fmt_num(value, decimals=2, default="-"):
    try:
        if value is None or pd.isna(value):
            return default
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return default


def fmt_int(value, default="-"):
    try:
        if value is None or pd.isna(value):
            return default
        return f"{int(value):,}"
    except Exception:
        return default


def fmt_pct(value, decimals=2, default="-"):
    try:
        if value is None or pd.isna(value):
            return default
        v = float(value)
        if abs(v) <= 1:
            v *= 100
        return f"{v:.{decimals}f}%"
    except Exception:
        return default


def fmt_currency_inr(value, default="-"):
    try:
        if value is None or pd.isna(value):
            return default
        return f"₹ {float(value):,.2f}"
    except Exception:
        return default


def style_card(children, height=None):
    style = {
        "background": f"linear-gradient(180deg, {THEME['panel2']}, {THEME['panel']})",
        "border": f"1px solid {THEME['border']}",
        "borderRadius": "18px",
        "padding": "18px",
        "boxShadow": "0 12px 28px rgba(0,0,0,0.28)",
    }
    if height:
        style["minHeight"] = height
    return html.Div(children, style=style)


def stat_card(title, value, sub=None, color=None):
    return style_card([
        html.Div(title, style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}),
        html.Div(value, style={"color": color or THEME["text"], "fontSize": "26px", "fontWeight": "800", "marginTop": "10px"}),
        html.Div(sub or "", style={"color": THEME["faint"], "fontSize": "13px", "marginTop": "8px"}),
    ], height="118px")


def metric_line(label, value, color=None):
    return html.Div([
        html.Span(label, style={"color": THEME["muted"]}),
        html.Strong(value, style={"color": color or THEME["text"]}),
    ], style={"display": "flex", "justifyContent": "space-between", "padding": "10px 0", "borderBottom": f"1px solid {THEME['border']}"})


def history_path(symbol):
    return HIST_CACHE_DIR / f"{symbol}.csv"


def bhavcopy_url(dt: datetime) -> str:
    return f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{dt.strftime('%Y%m%d')}_F_0000.csv.zip"


def normalize_bhavcopy(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().upper() for c in df.columns]
    mapping = {
        "DATE1": "DATE",
        "OPEN_PRICE": "OPEN",
        "HIGH_PRICE": "HIGH",
        "LOW_PRICE": "LOW",
        "CLOSE_PRICE": "CLOSE",
        "TOTAL_TRADED_QUANTITY": "VOLUME",
        "TTL_TRD_QNTY": "VOLUME",
    }
    df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
    if "SERIES" in df.columns:
        df = df[df["SERIES"].astype(str).str.strip().eq("EQ")]
    required = ["SYMBOL", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
    out = df[required].copy()
    out["DATE"] = pd.to_datetime(out["DATE"], dayfirst=True, errors="coerce")
    for c in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["DATE", "OPEN", "HIGH", "LOW", "CLOSE"])


def download_bhavcopy_day(dt: datetime) -> pd.DataFrame | None:
    try:
        r = session.get(bhavcopy_url(dt), headers=CSV_HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            with zf.open(csv_name) as f:
                raw = pd.read_csv(f)
        return normalize_bhavcopy(raw)
    except Exception:
        return None


def load_local_history(symbol: str) -> pd.DataFrame:
    p = history_path(symbol)
    if not p.exists():
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df = pd.read_csv(p)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)


def save_local_history(symbol: str, df: pd.DataFrame):
    df.to_csv(history_path(symbol), index=False)


def fetch_history(symbol: str, force_refresh=False) -> pd.DataFrame:
    existing = load_local_history(symbol)
    if not force_refresh and len(existing) >= 180:
        return existing
    rows = []
    end = datetime.today()
    for i in range(LOOKBACK_DAYS):
        dt = end - timedelta(days=i)
        if dt.weekday() >= 5:
            continue
        day = download_bhavcopy_day(dt)
        if day is None or day.empty:
            continue
        match = day[day["SYMBOL"].astype(str).str.strip().eq(symbol)]
        if not match.empty:
            rows.append(match.iloc[0])
    if rows:
        new_df = pd.DataFrame(rows).sort_values("DATE").drop_duplicates(subset=["DATE"])
        new_df = new_df.rename(columns={"DATE": "Date", "OPEN": "Open", "HIGH": "High", "LOW": "Low", "CLOSE": "Close", "VOLUME": "Volume"})
        merged = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        merged = merged.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        save_local_history(symbol, merged)
        return merged
    if not existing.empty:
        return existing
    raise ValueError(f"No NSE data available for {symbol} right now.")


def load_nse_stocks() -> pd.DataFrame:
    try:
        r = session.get(NSE_EQUITY_URL, headers=CSV_HEADERS, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [str(c).strip() for c in df.columns]
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()
        df = df[df["SERIES"].eq("EQ")].copy()
        return df[["SYMBOL", "NAME OF COMPANY"]].drop_duplicates(subset=["SYMBOL"]).sort_values("SYMBOL").reset_index(drop=True)
    except Exception:
        return pd.DataFrame([{"SYMBOL": DEFAULT_SYMBOL, "NAME OF COMPANY": "Reliance Industries Limited"}])


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def adx(df, period=14):
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).sum() / atr_val
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).sum() / atr_val
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()


def supertrend(df, period=10, multiplier=3):
    hl2 = (df["High"] + df["Low"]) / 2
    atr_val = atr(df, period)
   

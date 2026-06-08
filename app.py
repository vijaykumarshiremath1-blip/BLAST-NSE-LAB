import io
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update
from flask_caching import Cache
from plotly.subplots import make_subplots

NSE_EQUITY_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
DEFAULT_SYMBOL = "RELIANCE"
TOP_SCAN_LIMIT = 220
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
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val
    final_upper_band = upper_band.copy()
    final_lower_band = lower_band.copy()
    trend = pd.Series(index=df.index, dtype="float64")
    direction = pd.Series(index=df.index, dtype="object")
    for i in range(1, len(df)):
        if upper_band.iloc[i] < final_upper_band.iloc[i - 1] or df["Close"].iloc[i - 1] > final_upper_band.iloc[i - 1]:
            final_upper_band.iloc[i] = upper_band.iloc[i]
        else:
            final_upper_band.iloc[i] = final_upper_band.iloc[i - 1]
        if lower_band.iloc[i] > final_lower_band.iloc[i - 1] or df["Close"].iloc[i - 1] < final_lower_band.iloc[i - 1]:
            final_lower_band.iloc[i] = lower_band.iloc[i]
        else:
            final_lower_band.iloc[i] = final_lower_band.iloc[i - 1]
        if i == 1 or pd.isna(trend.iloc[i - 1]):
            trend.iloc[i] = final_lower_band.iloc[i]
            direction.iloc[i] = "Bullish"
        elif trend.iloc[i - 1] == final_upper_band.iloc[i - 1]:
            if df["Close"].iloc[i] <= final_upper_band.iloc[i]:
                trend.iloc[i] = final_upper_band.iloc[i]
                direction.iloc[i] = "Bearish"
            else:
                trend.iloc[i] = final_lower_band.iloc[i]
                direction.iloc[i] = "Bullish"
        else:
            if df["Close"].iloc[i] >= final_lower_band.iloc[i]:
                trend.iloc[i] = final_lower_band.iloc[i]
                direction.iloc[i] = "Bullish"
            else:
                trend.iloc[i] = final_upper_band.iloc[i]
                direction.iloc[i] = "Bearish"
    return trend, direction


def add_indicators(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI14"] = rsi(df["Close"], 14)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd(df["Close"])
    df["ATR14"] = atr(df, 14)
    df["ADX14"] = adx(df, 14)
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    df["BB_MID"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * std
    df["BB_LOWER"] = df["BB_MID"] - 2 * std
    df["SUPERTREND"], df["ST_DIR"] = supertrend(df, 10, 3)
    return df


def nearest_levels(df, lookback=40):
    recent = df.tail(lookback)
    support = round(float(recent["Low"].min()), 2) if not recent.empty else None
    resistance = round(float(recent["High"].max()), 2) if not recent.empty else None
    return support, resistance


def generate_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    score = 0
    reasons = []
    if pd.notna(last["SMA20"]) and pd.notna(last["SMA50"]):
        if last["Close"] > last["SMA20"] > last["SMA50"]:
            score += 2
            reasons.append("Price above SMA20 and SMA50")
        elif last["Close"] < last["SMA20"] < last["SMA50"]:
            score -= 2
            reasons.append("Price below SMA20 and SMA50")
    if pd.notna(last["SMA50"]) and pd.notna(last["SMA200"]):
        if last["SMA50"] > last["SMA200"]:
            score += 1
            reasons.append("SMA50 above SMA200")
        elif last["SMA50"] < last["SMA200"]:
            score -= 1
            reasons.append("SMA50 below SMA200")
    if last["RSI14"] > 60:
        score += 1
        reasons.append("RSI above 60")
    elif last["RSI14"] < 40:
        score -= 1
        reasons.append("RSI below 40")
    if last["MACD"] > last["MACD_SIGNAL"] and prev["MACD"] <= prev["MACD_SIGNAL"]:
        score += 2
        reasons.append("Fresh MACD bullish crossover")
    elif last["MACD"] < last["MACD_SIGNAL"] and prev["MACD"] >= prev["MACD_SIGNAL"]:
        score -= 2
        reasons.append("Fresh MACD bearish crossover")
    elif last["MACD"] > last["MACD_SIGNAL"]:
        score += 1
        reasons.append("MACD above signal")
    else:
        score -= 1
        reasons.append("MACD below signal")
    if pd.notna(last["VOL_MA20"]) and last["Volume"] > last["VOL_MA20"]:
        score += 1
        reasons.append("Volume above 20-day average")
    if last.get("ST_DIR") == "Bullish":
        score += 1
        reasons.append("Supertrend bullish")
    elif last.get("ST_DIR") == "Bearish":
        score -= 1
        reasons.append("Supertrend bearish")
    if pd.notna(last["ADX14"]):
        if last["ADX14"] > 25:
            score += 1
            reasons.append("ADX trend strength above 25")
        else:
            reasons.append("ADX indicates moderate trend")
    if score >= 6:
        signal = "STRONG BUY"
    elif score >= 3:
        signal = "BUY"
    elif score <= -6:
        signal = "STRONG SELL"
    elif score <= -3:
        signal = "SELL"
    elif abs(score) <= 1:
        signal = "NIL"
    else:
        signal = "HOLD"
    confidence = min(97, max(38, 48 + abs(score) * 7))
    return signal, score, confidence, reasons[:6]


def price_snapshot(df):
    last = df.iloc[-1]
    prev_close = df["Close"].iloc[-2] if len(df) > 1 else last["Close"]
    period = df.tail(252)
    chg = float(last["Close"] - prev_close)
    pct = chg / prev_close * 100 if prev_close else 0
    return {
        "close": round(float(last["Close"]), 2),
        "change": round(chg, 2),
        "change_pct": round(pct, 2),
        "high": round(float(last["High"]), 2),
        "low": round(float(last["Low"]), 2),
        "volume": int(last["Volume"]),
        "high52": round(float(period["High"].max()), 2) if not period.empty else None,
        "low52": round(float(period["Low"].min()), 2) if not period.empty else None,
    }


def order_plan(df):
    last = df.iloc[-1]
    support, resistance = nearest_levels(df)
    atr_val = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0
    return {
        "support": support,
        "resistance": resistance,
        "buy_entry": round(float(last["Close"]) + 0.15 * atr_val, 2),
        "buy_sl": round(float(last["Close"]) - 1.0 * atr_val, 2),
        "buy_t1": round(float(last["Close"]) + 1.2 * atr_val, 2),
        "buy_t2": round(float(last["Close"]) + 2.2 * atr_val, 2),
        "sell_entry": round(float(last["Close"]) - 0.15 * atr_val, 2),
        "sell_sl": round(float(last["Close"]) + 1.0 * atr_val, 2),
        "sell_t1": round(float(last["Close"]) - 1.2 * atr_val, 2),
        "sell_t2": round(float(last["Close"]) - 2.2 * atr_val, 2),
    }


def make_tv_symbol(symbol):
    return f"NSE:{symbol}"


def screener_url(symbol):
    return f"https://www.screener.in/company/{symbol}/"


def button_link_style():
    return {
        "display": "inline-block",
        "padding": "10px 14px",
        "background": THEME["bg3"],
        "border": f"1px solid {THEME['border']}",
        "color": THEME["text"],
        "borderRadius": "12px",
        "textDecoration": "none",
        "fontWeight": "700",
        "marginRight": "10px",
        "marginBottom": "10px",
    }


def fallback_company_profile(symbol, company_name, df):
    snap = price_snapshot(df)
    return {
        "sector": "NSE Listed Equity",
        "industry": "Cash Market Equity",
        "exchange": "NSE",
        "summary": f"{company_name} ({symbol}) is being shown with an NSE-first technical workflow. This profile uses cached exchange history to keep Overview, Chart, Financials, and News pages available even when third-party APIs throttle requests.",
        "market_cap": None,
        "trailing_pe": None,
        "price_to_book": None,
        "roe": None,
        "dividend_yield": None,
        "book_value": None,
        "eps": None,
        "beta": None,
        "profit_margin": None,
        "employees": None,
        "enterprise_value": None,
        "held_percent_institutions": None,
        "held_percent_insiders": None,
        "close": snap["close"],
        "high52": snap["high52"],
        "low52": snap["low52"],
    }


def fallback_news(symbol):
    return [
        {
            "publisher": "NSE Reports",
            "title": f"Track corporate announcements for {symbol} on NSE reports",
            "summary": "Use NSE reports for circulars, filings, board meetings, shareholding disclosures, and other official company updates.",
            "link": NSE_REPORTS_URL,
            "published": "Live reference",
        },
        {
            "publisher": "Screener",
            "title": f"Open {symbol} company page for annual results and shareholding",
            "summary": "Screener can be used as a quick secondary source for ratios, financial statements, and ownership pattern checks.",
            "link": screener_url(symbol),
            "published": "Reference link",
        },
        {
            "publisher": "TradingView",
            "title": f"Cross-check {symbol} chart structure with external price view",
            "summary": "Useful for validating trend zones, breakouts, and overall market structure alongside this dashboard.",
            "link": f"https://in.tradingview.com/symbols/{make_tv_symbol(symbol).replace(':', '-')}/",
            "published": "Reference link",
        },
    ]


def fallback_holders():
    return {
        "major_holders": pd.DataFrame({"Holder Type": ["Institutional", "Promoter/Insider", "Retail/Public"], "Status": ["Check Screener / NSE filing", "Check company filing", "Check latest shareholding pattern"]}),
        "institutional_holders": pd.DataFrame(),
        "mutualfund_holders": pd.DataFrame(),
    }


def fallback_financial_tables():
    return {
        "income_stmt": pd.DataFrame(),
        "balance_sheet": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
    }


def make_chart(df, symbol, indicators):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.60, 0.20, 0.20])
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price", increasing_line_color="#44d7b6", decreasing_line_color="#ff6b8a"), row=1, col=1)
    if "SMA20" in indicators:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], name="SMA20", line=dict(color="#52b6ff", width=1.4)), row=1, col=1)
    if "SMA50" in indicators:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], name="SMA50", line=dict(color="#ffbf69", width=1.4)), row=1, col=1)
    if "SMA200" in indicators:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA200"], name="SMA200", line=dict(color="#ff7d9c", width=1.4)), row=1, col=1)
    if "EMA20" in indicators:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA20"], name="EMA20", line=dict(color="#8b5cf6", width=1.2, dash="dot")), row=1, col=1)
    if "EMA50" in indicators:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA50"], name="EMA50", line=dict(color="#2dd4bf", width=1.2, dash="dot")), row=1, col=1)
    if "Bollinger" in indicators:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_UPPER"], name="BB Upper", line=dict(color="#7c8aa5", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_LOWER"], name="BB Lower", line=dict(color="#7c8aa5", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(124,138,165,0.08)"), row=1, col=1)
    if "Supertrend" in indicators:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SUPERTREND"], name="Supertrend", line=dict(color="#34d399", width=1.5)), row=1, col=1)
    if "MACD" in indicators:
        colors = np.where(df["MACD_HIST"] >= 0, "#34d399", "#fb7185")
        fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_HIST"], name="MACD Hist", marker_color=colors), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD", line=dict(color="#5aa8ff", width=1.4)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_SIGNAL"], name="Signal", line=dict(color="#ffcc66", width=1.2)), row=2, col=1)
    if "RSI" in indicators:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI14"], name="RSI14", line=dict(color="#b38cff", width=1.8)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#fb7185", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#34d399", row=3, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"], font=dict(color=THEME["text"]), xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=50, b=20), height=860, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), title=f"{symbol} Chart Lab")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def df_to_table(df, rows=12):
    if df is None or df.empty:
        return html.Div("No data available.", style={"color": THEME["muted"]})
    show = df.copy().head(rows)
    if isinstance(show.columns, pd.DatetimeIndex):
        show.columns = [c.strftime("%Y-%m-%d") for c in show.columns]
    else:
        show.columns = [str(c) for c in show.columns]
    show = show.reset_index().rename(columns={show.index.name or "index": "Metric"}).fillna("-")
    return dash_table.DataTable(
        data=show.to_dict("records"),
        columns=[{"name": c, "id": c} for c in show.columns],
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": THEME["bg3"], "color": THEME["text"], "border": f"1px solid {THEME['border']}", "fontWeight": "700"},
        style_cell={"backgroundColor": THEME["panel"], "color": THEME["text"], "border": f"1px solid {THEME['border']}", "padding": "10px", "textAlign": "left", "minWidth": "110px", "maxWidth": "260px", "whiteSpace": "normal"},
        page_size=rows,
    )


def holders_summary(holders):
    major = holders.get("major_holders")
    major_map = {}
    if isinstance(major, pd.DataFrame) and not major.empty and major.shape[1] >= 2:
        for _, row in major.iterrows():
            major_map[str(row.iloc[1])] = row.iloc[0]
    rows = [
        ("FII / Institutional", major_map.get("% of Shares Held by Institutions", "Check latest filings")),
        ("Promoters / Insiders", "Check Screener / annual report"),
        ("Mutual Funds", "Check detailed shareholding tables"),
        ("Retail / Public", "Check exchange shareholding pattern"),
    ]
    return html.Div([metric_line(a, str(b)) for a, b in rows])


def company_overview(symbol, company_name, info):
    summary = info.get("summary") or "Business summary not available."
    return style_card([
        html.Div(f"{company_name} ({symbol})", style={"fontSize": "24px", "fontWeight": "800", "color": THEME["text"]}),
        html.Div(f"{info.get('sector', '-')} | {info.get('industry', '-')}", style={"marginTop": "6px", "color": THEME["muted"]}),
        html.P(summary[:1000], style={"marginTop": "14px", "color": THEME["text"], "lineHeight": "1.7"}),
    ])


def build_news_cards(news_items):
    if not news_items:
        return style_card([html.Div("No recent news available.", style={"color": THEME["muted"]})])
    cards = []
    for item in news_items[:10]:
        cards.append(style_card([
            html.Div(item.get("publisher", "Source"), style={"color": THEME["accent"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}),
            html.A(item.get("title", "Untitled"), href=item.get("link") or None, target="_blank", rel="noopener noreferrer", style={"display": "block", "marginTop": "8px", "color": THEME["text"], "fontWeight": "800", "textDecoration": "none", "fontSize": "17px"}),
            html.Div(item.get("summary", "")[:240], style={"marginTop": "8px", "color": THEME["muted"], "lineHeight": "1.6"}),
            html.Div(str(item.get("published", "")), style={"marginTop": "10px", "color": THEME["faint"], "fontSize": "12px"}),
        ]))
    return html.Div(cards, style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(280px,1fr))", "gap": "16px"})


def build_home_page(symbol, company_name, info, df, signal, score, confidence, reasons, news_items, holders):
    snap = price_snapshot(df)
    plan = order_plan(df)
    last = df.iloc[-1]
    top_cards = html.Div([
        stat_card("Last Close", fmt_currency_inr(snap["close"]), f"{snap['change']} ({snap['change_pct']}%)", THEME["good"] if snap["change"] >= 0 else THEME["danger"]),
        stat_card("Day Range", f"₹ {fmt_num(snap['low'])} - ₹ {fmt_num(snap['high'])}", "Day low - high"),
        stat_card("52 Week Range", f"₹ {fmt_num(snap['low52'])} - ₹ {fmt_num(snap['high52'])}", "52-week low - high"),
        stat_card("Exchange", "NSE", company_name),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(220px,1fr))", "gap": "16px"})
    signal_card = style_card([
        html.Div("Technical Flow Signal", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}),
        html.Div(signal, style={"fontSize": "32px", "fontWeight": "900", "marginTop": "10px", "color": SIGNAL_COLORS.get(signal, THEME['text'])}),
        html.Div(f"Score {score} | Confidence {confidence}%", style={"color": THEME["text"], "marginTop": "8px"}),
        html.Ul([html.Li(r) for r in reasons], style={"marginTop": "12px", "paddingLeft": "18px", "color": THEME["muted"], "lineHeight": "1.8"}),
    ])
    plan_card = style_card([
        html.Div("Trade Plan", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}),
        metric_line("Support", fmt_currency_inr(plan["support"])),
        metric_line("Resistance", fmt_currency_inr(plan["resistance"])),
        metric_line("Buy Entry", fmt_currency_inr(plan["buy_entry"])),
        metric_line("Buy Stop Loss", fmt_currency_inr(plan["buy_sl"])),
        metric_line("Buy Target 1", fmt_currency_inr(plan["buy_t1"])),
        metric_line("Buy Target 2", fmt_currency_inr(plan["buy_t2"])),
    ])
    ratios_card = style_card([
        html.Div("Investor Holding View", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}),
        holders_summary(holders),
    ])
    quick_metrics = style_card([
        html.Div("Quick Technical Data", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}),
        metric_line("RSI 14", fmt_num(last["RSI14"])),
        metric_line("MACD", fmt_num(last["MACD"])),
        metric_line("MACD Signal", fmt_num(last["MACD_SIGNAL"])),
        metric_line("ADX 14", fmt_num(last["ADX14"])),
        metric_line("ATR 14", fmt_num(last["ATR14"])),
        metric_line("Volume", fmt_int(last["Volume"])),
        metric_line("Volume Avg 20", fmt_int(last["VOL_MA20"])),
        metric_line("Supertrend", str(last["ST_DIR"])),
    ])
    links = style_card([
        html.Div("Research Links", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "12px"}),
        html.A("Open Screener", href=screener_url(symbol), target="_blank", rel="noopener noreferrer", style=button_link_style()),
        html.A("NSE Reports", href=NSE_REPORTS_URL, target="_blank", rel="noopener noreferrer", style=button_link_style()),
        html.A("TradingView", href=f"https://in.tradingview.com/symbols/{make_tv_symbol(symbol).replace(':', '-')}/", target="_blank", rel="noopener noreferrer", style=button_link_style()),
    ])
    news_preview = style_card([
        html.Div("Latest News Snapshot", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}),
        html.Div([
            html.Div(item.get("title", ""), style={"padding": "10px 0", "borderBottom": f"1px solid {THEME['border']}", "color": THEME['text']}) for item in news_items[:5]
        ] if news_items else html.Div("No recent news available.", style={"color": THEME["muted"], "marginTop": "10px"}))
    ])
    return html.Div([
        top_cards,
        html.Div(style={"height": "16px"}),
        company_overview(symbol, company_name, info),
        html.Div(style={"height": "16px"}),
        html.Div([signal_card, plan_card, ratios_card, quick_metrics], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(280px,1fr))", "gap": "16px"}),
        html.Div(style={"height": "16px"}),
        html.Div([links, news_preview], style={"display": "grid", "gridTemplateColumns": "1fr 1.3fr", "gap": "16px"}),
    ])


def build_chart_page(symbol, df, indicators):
    fig = make_chart(df, symbol, indicators)
    controls = style_card([
        html.Div("Editable Indicators", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "12px"}),
        dcc.Checklist(
            id="chart-indicator-checklist",
            options=[
                {"label": "SMA20", "value": "SMA20"}, {"label": "SMA50", "value": "SMA50"}, {"label": "SMA200", "value": "SMA200"},
                {"label": "EMA20", "value": "EMA20"}, {"label": "EMA50", "value": "EMA50"}, {"label": "Bollinger", "value": "Bollinger"},
                {"label": "Supertrend", "value": "Supertrend"}, {"label": "MACD", "value": "MACD"}, {"label": "RSI", "value": "RSI"},
            ],
            value=indicators,
            inline=True,
            inputStyle={"marginRight": "6px", "marginLeft": "14px"},
            labelStyle={"display": "inline-flex", "alignItems": "center", "marginBottom": "10px", "color": THEME["text"]},
        ),
    ])
    return html.Div([controls, html.Div(style={"height": "14px"}), dcc.Graph(figure=fig, config={"displaylogo": False, "responsive": True})])


def build_financials_page(symbol, company_name, info, holders, fin_tables, df):
    snap = price_snapshot(df)
    top = html.Div([
        stat_card("Close", fmt_currency_inr(snap["close"]), "Latest close"),
        stat_card("52W High", fmt_currency_inr(snap["high52"]), "Rolling high"),
        stat_card("52W Low", fmt_currency_inr(snap["low52"]), "Rolling low"),
        stat_card("Volume", fmt_int(snap["volume"]), "Latest traded quantity"),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(220px,1fr))", "gap": "16px"})
    profile = style_card([
        html.Div("Company Fundamentals", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}),
        metric_line("Company", company_name),
        metric_line("Sector", str(info.get("sector", "-"))),
        metric_line("Industry", str(info.get("industry", "-"))),
        metric_line("Exchange", str(info.get("exchange", "NSE"))),
        metric_line("P/E", fmt_num(info.get("trailing_pe"))),
        metric_line("P/B", fmt_num(info.get("price_to_book"))),
        metric_line("ROE", fmt_pct(info.get("roe"))),
        metric_line("Dividend Yield", fmt_pct(info.get("dividend_yield"))),
    ])
    holders_card = style_card([
        html.Div("Shareholding Tables", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "12px"}),
        html.H4("Institutional Holders", style={"marginBottom": "10px"}),
        df_to_table(holders.get("institutional_holders"), rows=8),
        html.Div(style={"height": "12px"}),
        html.H4("Mutual Fund Holders", style={"marginBottom": "10px"}),
        df_to_table(holders.get("mutualfund_holders"), rows=8),
    ])
    financial_tables = html.Div([
        style_card([html.H4("Annual Income Statement", style={"marginBottom": "10px"}), df_to_table(fin_tables.get("income_stmt"), rows=12)]),
        style_card([html.H4("Annual Balance Sheet", style={"marginBottom": "10px"}), df_to_table(fin_tables.get("balance_sheet"), rows=12)]),
        style_card([html.H4("Annual Cash Flow", style={"marginBottom": "10px"}), df_to_table(fin_tables.get("cashflow"), rows=12)]),
    ], style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "16px"})
    links = style_card([
        html.Div("External Research", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "12px"}),
        html.A("Open Screener Company Page", href=screener_url(symbol), target="_blank", rel="noopener noreferrer", style=button_link_style()),
        html.A("NSE Reports", href=NSE_REPORTS_URL, target="_blank", rel="noopener noreferrer", style=button_link_style()),
    ])
    return html.Div([top, html.Div(style={"height": "16px"}), html.Div([profile, links], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}), html.Div(style={"height": "16px"}), holders_card, html.Div(style={"height": "16px"}), financial_tables])


def build_news_page(news_items, symbol):
    header = style_card([
        html.Div("News Order Flow", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}),
        html.Div(f"Latest news for {symbol}", style={"fontSize": "24px", "fontWeight": "800", "marginTop": "10px"}),
        html.Div("This version uses safe fallback reference links and does not depend on rate-limited live Yahoo news calls.", style={"marginTop": "8px", "color": THEME["muted"]}),
        html.Div(style={"marginTop": "12px"}, children=[
            html.A("NSE Reports", href=NSE_REPORTS_URL, target="_blank", rel="noopener noreferrer", style=button_link_style()),
            html.A("Screener", href=screener_url(symbol), target="_blank", rel="noopener noreferrer", style=button_link_style()),
        ]),
    ])
    return html.Div([header, html.Div(style={"height": "16px"}), build_news_cards(news_items)])


def build_tab_content(tab, symbol, company_name, info, df, signal, score, confidence, reasons, news_items, holders, fin_tables, indicators):
    if tab == "chart":
        return build_chart_page(symbol, df, indicators)
    if tab == "financials":
        return build_financials_page(symbol, company_name, info, holders, fin_tables, df)
    if tab == "news":
        return build_news_page(news_items, symbol)
    return build_home_page(symbol, company_name, info, df, signal, score, confidence, reasons, news_items, holders)


STOCKS_DF = load_nse_stocks()
DROPDOWN_OPTIONS = [{"label": f"{r['SYMBOL']} - {r['NAME OF COMPANY']}", "value": r["SYMBOL"]} for _, r in STOCKS_DF.iterrows()]


def get_company_name(symbol):
    try:
        row = STOCKS_DF.loc[STOCKS_DF["SYMBOL"] == symbol].iloc[0]
        return row["NAME OF COMPANY"]
    except Exception:
        return symbol


def scan_one(symbol, company):
    try:
        hist = add_indicators(fetch_history(symbol))
        sig, score, conf, _ = generate_signal(hist)
        last = hist.iloc[-1]
        return {"symbol": symbol, "company": company, "signal": sig, "confidence": conf, "close": round(float(last["Close"]), 2), "rsi": round(float(last["RSI14"]), 2) if pd.notna(last["RSI14"]) else None}
    except Exception:
        return None


@cache.memoize(timeout=3600)
def build_scan_universe_cached(limit=TOP_SCAN_LIMIT):
    rows = []
    sample = STOCKS_DF.head(limit)
    workers = min(10, max(4, len(sample) // 20 if len(sample) else 4))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(scan_one, row["SYMBOL"], row["NAME OF COMPANY"]) for _, row in sample.iterrows()]
        for future in as_completed(futures):
            result = future.result()
            if result:
                rows.append(result)
    scan_df = pd.DataFrame(rows)
    if scan_df.empty:
        return scan_df
    order = {"STRONG BUY": 5, "BUY": 4, "HOLD": 3, "NIL": 2, "SELL": 1, "STRONG SELL": 0}
    scan_df["rank"] = scan_df["signal"].map(order).fillna(0)
    return scan_df.sort_values(["rank", "confidence"], ascending=[False, False]).drop(columns=["rank"])


try:
    SCAN_DF = build_scan_universe_cached(TOP_SCAN_LIMIT)
except Exception:
    SCAN_DF = pd.DataFrame(columns=["symbol", "company", "signal", "confidence", "close", "rsi"])


app.layout = html.Div([
    dcc.Store(id="selected-symbol-store", data=DEFAULT_SYMBOL),
    dcc.Store(id="selected-indicators-store", data=DEFAULT_INDICATORS),
    html.Div([
        html.Div("BLAST NSE LAB PRO", style={"fontSize": "12px", "letterSpacing": "2px", "color": THEME["accent"], "fontWeight": "800"}),
        html.H1("Dark Theme Stock Dashboard", style={"margin": "8px 0 4px 0", "fontSize": "34px"}),
        html.Div("Older layout preserved with overview, chart, financials and news tabs, but without rate-limited Yahoo page calls.", style={"color": THEME["muted"], "maxWidth": "900px", "lineHeight": "1.7"}),
        html.Div(style={"height": "20px"}),
        html.Div([
            style_card([
                html.Div("Stock Search", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "12px"}),
                dcc.Dropdown(id="stock-dropdown", options=DROPDOWN_OPTIONS, value=DEFAULT_SYMBOL if DEFAULT_SYMBOL in [o["value"] for o in DROPDOWN_OPTIONS] else None, placeholder="Search stock symbol or company name", searchable=True, clearable=False, style={"color": "#111827"}),
                html.Div(id="selected-stock-title", style={"marginTop": "12px", "color": THEME["text"], "fontWeight": "700"}),
                html.Button("Refresh Selected Stock", id="refresh-selected-stock", n_clicks=0, style={"marginTop": "14px", "background": THEME["accent"], "color": "#05201b", "border": "none", "borderRadius": "12px", "padding": "12px 16px", "fontWeight": "800", "cursor": "pointer"}),
            ]),
            style_card([
                html.Div("Smart Technical Scanner", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "12px"}),
                dcc.Dropdown(id="signal-filter-dropdown", options=[
                    {"label": "All", "value": "ALL"},
                    {"label": "Strong Buy", "value": "STRONG BUY"},
                    {"label": "Buy", "value": "BUY"},
                    {"label": "Hold", "value": "HOLD"},
                    {"label": "Nil", "value": "NIL"},
                    {"label": "Sell", "value": "SELL"},
                    {"label": "Strong Sell", "value": "STRONG SELL"},
                ], value="ALL", clearable=False, style={"color": "#111827"}),
                html.Div(id="scanner-table-container", style={"marginTop": "12px"}),
            ]),
        ], style={"display": "grid", "gridTemplateColumns": "0.95fr 1.35fr", "gap": "16px"}),
        html.Div(style={"height": "18px"}),
        dcc.Tabs(id="main-tabs", value="home", parent_className="custom-tabs", className="custom-tabs-container", children=[
            dcc.Tab(label="Home", value="home", className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Chart", value="chart", className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Financials", value="financials", className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="News", value="news", className="custom-tab", selected_className="custom-tab--selected"),
        ]),
        html.Div(id="main-tab-content", style={"marginTop": "18px"}),
    ], style={"maxWidth": "1450px", "margin": "0 auto", "padding": "28px 18px 60px 18px"})
], style={"minHeight": "100vh", "background": f"radial-gradient(circle at top left, {THEME['bg3']}, {THEME['bg']})", "color": THEME["text"], "fontFamily": "Inter, Segoe UI, Arial, sans-serif"})

app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>
        body {{ background: {THEME['bg']}; }}
        .custom-tabs-container {{ width: 100%; }}
        .custom-tabs {{ background: {THEME['panel']}; border: 1px solid {THEME['border']}; border-radius: 16px; padding: 8px; display: flex; gap: 8px; flex-wrap: wrap; }}
        .custom-tab {{ color: {THEME['muted']} !important; background: {THEME['panel2']} !important; border: 1px solid {THEME['border']} !important; border-radius: 12px !important; padding: 12px 18px !important; font-weight: 700; }}
        .custom-tab--selected {{ color: {THEME['text']} !important; background: linear-gradient(180deg, #1f3150, {THEME['panel2']}) !important; border: 1px solid #3a5277 !important; }}
        @media (max-width: 900px) {{ .custom-tabs {{ display:block; }} }}
    </style>
</head>
<body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>
"""


@app.callback(
    Output("selected-symbol-store", "data"),
    Output("selected-stock-title", "children"),
    Input("stock-dropdown", "value"),
)
def select_stock(symbol):
    if not symbol:
        return no_update, no_update
    company = get_company_name(symbol)
    return symbol, f"Selected: {symbol} - {company}"


@app.callback(
    Output("selected-indicators-store", "data"),
    Input("chart-indicator-checklist", "value"),
    prevent_initial_call=True,
)
def sync_indicator_store(values):
    return values or DEFAULT_INDICATORS


@app.callback(
    Output("scanner-table-container", "children"),
    Input("signal-filter-dropdown", "value"),
)
def render_scanner(signal_value):
    if SCAN_DF.empty:
        return html.Div("Scanner data is not available right now.", style={"color": THEME["muted"]})
    df = SCAN_DF.copy()
    if signal_value and signal_value != "ALL":
        df = df[df["signal"] == signal_value]
    show = df.head(20).copy()
    return dash_table.DataTable(
        data=show.to_dict("records"),
        columns=[
            {"name": "Symbol", "id": "symbol"},
            {"name": "Company", "id": "company"},
            {"name": "Signal", "id": "signal"},
            {"name": "Confidence %", "id": "confidence"},
            {"name": "Close", "id": "close"},
            {"name": "RSI", "id": "rsi"},
        ],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": THEME["bg3"], "color": THEME["text"], "fontWeight": "700", "border": f"1px solid {THEME['border']}"},
        style_cell={"backgroundColor": THEME["panel"], "color": THEME["text"], "border": f"1px solid {THEME['border']}", "padding": "10px", "textAlign": "left"},
        style_data_conditional=[
            {"if": {"filter_query": '{signal} = "STRONG BUY"', "column_id": "signal"}, "color": SIGNAL_COLORS["STRONG BUY"], "fontWeight": "800"},
            {"if": {"filter_query": '{signal} = "BUY"', "column_id": "signal"}, "color": SIGNAL_COLORS["BUY"], "fontWeight": "800"},
            {"if": {"filter_query": '{signal} = "HOLD"', "column_id": "signal"}, "color": SIGNAL_COLORS["HOLD"], "fontWeight": "800"},
            {"if": {"filter_query": '{signal} = "NIL"', "column_id": "signal"}, "color": SIGNAL_COLORS["NIL"], "fontWeight": "800"},
            {"if": {"filter_query": '{signal} = "SELL"', "column_id": "signal"}, "color": SIGNAL_COLORS["SELL"], "fontWeight": "800"},
            {"if": {"filter_query": '{signal} = "STRONG SELL"', "column_id": "signal"}, "color": SIGNAL_COLORS["STRONG SELL"], "fontWeight": "800"},
        ],
        page_size=8,
    )


@app.callback(
    Output("main-tab-content", "children"),
    Input("main-tabs", "value"),
    Input("selected-symbol-store", "data"),
    Input("selected-indicators-store", "data"),
    Input("refresh-selected-stock", "n_clicks"),
)
def render_main_content(tab, symbol, indicators, refresh_clicks):
    symbol = symbol or DEFAULT_SYMBOL
    indicators = indicators or DEFAULT_INDICATORS
    try:
        company_name = get_company_name(symbol)
        force_refresh = bool(refresh_clicks and refresh_clicks > 0)
        df = add_indicators(fetch_history(symbol, force_refresh=force_refresh))
        info = fallback_company_profile(symbol, company_name, df)
        news_items = fallback_news(symbol)
        holders = fallback_holders()
        fin_tables = fallback_financial_tables()
        signal, score, confidence, reasons = generate_signal(df)
        return build_tab_content(tab, symbol, company_name, info, df, signal, score, confidence, reasons, news_items, holders, fin_tables, indicators)
    except Exception as e:
        return style_card([
            html.Div("Data Error", style={"fontSize": "22px", "fontWeight": "800", "color": THEME['danger']}),
            html.Div(str(e), style={"marginTop": "10px", "color": THEME['text'], "lineHeight": "1.7"}),
            html.Div("This version removes the Yahoo Finance rate-limited page calls. If this still appears, the selected symbol has not loaded enough NSE history yet.", style={"marginTop": "10px", "color": THEME['muted']}),
        ])


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)

import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go

from io import StringIO
from dash import Dash, dcc, html, Input, Output, dash_table
from plotly.subplots import make_subplots

NSE_EQUITY_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
DEFAULT_SERIES = ("EQ",)
DEFAULT_SYMBOL = "RELIANCE"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/csv,application/json,text/plain,*/*",
    "Referer": "https://www.nseindia.com/",
}

NEWS_FEEDS = [
    ("Moneycontrol Markets", "https://www.moneycontrol.com/news/business/markets/"),
    ("Economic Times Markets", "https://economictimes.indiatimes.com/markets"),
    ("LiveMint Markets", "https://www.livemint.com/market"),
    ("Business Standard Markets", "https://www.business-standard.com/markets"),
    ("Screener", "https://www.screener.in/"),
    ("TradingView India", "https://in.tradingview.com/markets/stocks-india/"),
]

FALLBACK_STOCKS = [
    ("RELIANCE", "Reliance Industries"),
    ("TCS", "Tata Consultancy Services"),
    ("INFY", "Infosys"),
    ("HDFCBANK", "HDFC Bank"),
    ("ICICIBANK", "ICICI Bank"),
    ("SBIN", "State Bank of India"),
    ("LT", "Larsen & Toubro"),
    ("ITC", "ITC Ltd"),
    ("BHARTIARTL", "Bharti Airtel"),
    ("HINDUNILVR", "Hindustan Unilever"),
]


def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def fallback_universe_df():
    return pd.DataFrame({
        "SYMBOL": [s for s, _ in FALLBACK_STOCKS],
        "NAME OF COMPANY": [n for _, n in FALLBACK_STOCKS],
        "SERIES": ["EQ"] * len(FALLBACK_STOCKS),
        "DATE OF LISTING": [""] * len(FALLBACK_STOCKS),
        "ISIN NUMBER": [""] * len(FALLBACK_STOCKS),
        "FACE VALUE": [1] * len(FALLBACK_STOCKS),
    })


def load_nse_stocks(include_series=DEFAULT_SERIES) -> pd.DataFrame:
    df = fetch_csv(NSE_EQUITY_URL)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["SYMBOL", "NAME OF COMPANY", "SERIES", "DATE OF LISTING", "ISIN NUMBER", "FACE VALUE"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column in NSE file: {col}")

    for col in ["SYMBOL", "NAME OF COMPANY", "SERIES", "ISIN NUMBER"]:
        df[col] = df[col].astype(str).str.strip()

    df = df[df["SERIES"].isin(include_series)].copy()
    df = df.drop_duplicates(subset=["SYMBOL"]).sort_values("SYMBOL").reset_index(drop=True)
    return df


def make_dropdown_options(df):
    return [{"label": f"{row['SYMBOL']} - {row['NAME OF COMPANY']}", "value": row["SYMBOL"]} for _, row in df.iterrows()]


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
    atr_val = tr.rolling(period).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).sum() / atr_val
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).sum() / atr_val
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(period).mean().fillna(0)


def supertrend(df, period=10, multiplier=3):
    hl2 = (df["High"] + df["Low"]) / 2
    atr_val = atr(df, period).fillna(method="bfill").fillna(0)
    upperband = hl2 + multiplier * atr_val
    lowerband = hl2 - multiplier * atr_val
    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()
    trend = pd.Series(index=df.index, dtype="float64")
    direction = pd.Series(index=df.index, dtype="object")

    for i in range(1, len(df)):
        if upperband.iloc[i] < final_upperband.iloc[i - 1] or df["Close"].iloc[i - 1] > final_upperband.iloc[i - 1]:
            final_upperband.iloc[i] = upperband.iloc[i]
        else:
            final_upperband.iloc[i] = final_upperband.iloc[i - 1]

        if lowerband.iloc[i] > final_lowerband.iloc[i - 1] or df["Close"].iloc[i - 1] < final_lowerband.iloc[i - 1]:
            final_lowerband.iloc[i] = lowerband.iloc[i]
        else:
            final_lowerband.iloc[i] = final_lowerband.iloc[i - 1]

        if i == 1 or pd.isna(trend.iloc[i - 1]):
            trend.iloc[i] = final_lowerband.iloc[i]
            direction.iloc[i] = "Bullish"
        elif trend.iloc[i - 1] == final_upperband.iloc[i - 1]:
            if df["Close"].iloc[i] <= final_upperband.iloc[i]:
                trend.iloc[i] = final_upperband.iloc[i]
                direction.iloc[i] = "Bearish"
            else:
                trend.iloc[i] = final_lowerband.iloc[i]
                direction.iloc[i] = "Bullish"
        else:
            if df["Close"].iloc[i] >= final_lowerband.iloc[i]:
                trend.iloc[i] = final_lowerband.iloc[i]
                direction.iloc[i] = "Bullish"
            else:
                trend.iloc[i] = final_upperband.iloc[i]
                direction.iloc[i] = "Bearish"

    trend = trend.fillna(method="bfill").fillna(df["Close"])
    direction = direction.fillna("Neutral")
    return trend, direction


def generate_synthetic_history(symbol, periods=260):
    seed = sum(ord(c) for c in str(symbol))
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=periods, freq="B")
    drift = rng.normal(0.15, 1.0, len(dates)).cumsum()
    base = 1200 + (seed % 1500)
    close = np.maximum(base + drift * 6, 50)
    open_ = close + rng.normal(0, 8, len(dates))
    high = np.maximum(open_, close) + rng.uniform(4, 18, len(dates))
    low = np.minimum(open_, close) - rng.uniform(4, 18, len(dates))
    volume = rng.integers(100000, 8000000, len(dates))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


def add_indicators(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI14"] = rsi(df["Close"], 14)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd(df["Close"])
    df["ATR14"] = atr(df, 14).fillna(method="bfill").fillna(0)
    df["ADX14"] = adx(df, 14)
    df["VOL_MA20"] = df["Volume"].rolling(20).mean().fillna(method="bfill")
    df["BB_MID"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * std
    df["BB_LOWER"] = df["BB_MID"] - 2 * std
    df["SUPERTREND"], df["ST_DIR"] = supertrend(df, 10, 3)
    return df.fillna(method="bfill").fillna(method="ffill")


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

    if last["Close"] > last["SMA20"] > last["SMA50"]:
        score += 2
        reasons.append("Price above SMA20 and SMA50")
    elif last["Close"] < last["SMA20"] < last["SMA50"]:
        score -= 2
        reasons.append("Price below SMA20 and SMA50")

    if last["SMA50"] > last["SMA200"]:
        score += 1
        reasons.append("SMA50 above SMA200")
    elif last["SMA50"] < last["SMA200"]:
        score -= 1
        reasons.append("SMA50 below SMA200")

    if last["RSI14"] > 60:
        score += 1
        reasons.append("RSI strong above 60")
    elif last["RSI14"] < 40:
        score -= 1
        reasons.append("RSI weak below 40")

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

    if last["Volume"] > last["VOL_MA20"]:
        score += 1
        reasons.append("Volume above 20-day average")

    if last["ST_DIR"] == "Bullish":
        score += 1
        reasons.append("Supertrend bullish")
    elif last["ST_DIR"] == "Bearish":
        score -= 1
        reasons.append("Supertrend bearish")

    if last["ADX14"] >= 25:
        reasons.append("Trend strength healthy by ADX")
    else:
        reasons.append("Trend strength moderate/weak by ADX")

    if score >= 5:
        signal = "STRONG BUY"
    elif score >= 2:
        signal = "BUY"
    elif score <= -5:
        signal = "STRONG SELL"
    elif score <= -2:

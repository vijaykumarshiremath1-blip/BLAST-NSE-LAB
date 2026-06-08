import time
from io import StringIO
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from dash import Dash, Input, Output, dcc, html, dash_table, no_update
from flask_caching import Cache
from plotly.subplots import make_subplots

NSE_EQUITY_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
DEFAULT_SERIES = ("EQ",)
DEFAULT_SYMBOL = "RELIANCE"
TOP_SCAN_LIMIT = 250

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/csv,application/json,text/plain,*/*",
    "Referer": "https://www.nseindia.com/",
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

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Blast NSE Lab Pro"

cache = Cache(app.server, config={
    "CACHE_TYPE": "FileSystemCache",
    "CACHE_DIR": "cache-directory",
    "CACHE_THRESHOLD": 4000,
    "CACHE_DEFAULT_TIMEOUT": 900,
})


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


def fmt_market_cap(value):
    try:
        if value is None or pd.isna(value):
            return "-"
        value = float(value)
        crore = value / 1e7
        if crore >= 100000:
            return f"₹ {crore/100000:.2f} Lakh Cr"
        return f"₹ {crore:,.0f} Cr"
    except Exception:
        return "-"


def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def load_nse_stocks(include_series=DEFAULT_SERIES) -> pd.DataFrame:
    df = fetch_csv(NSE_EQUITY_URL)
    df.columns = [str(c).strip() for c in df.columns]
    required = ["SYMBOL", "NAME OF COMPANY", "SERIES"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column in NSE file: {col}")
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    df = df[df["SERIES"].isin(include_series)].copy()
    df = df.drop_duplicates(subset=["SYMBOL"]).sort_values("SYMBOL").reset_index(drop=True)
    return df


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
    upperband = hl2 + multiplier * atr_val
    lowerband = hl2 - multiplier * atr_val
    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()
    trend = pd.Series(index=df.index, dtype="float64")
    direction = pd.Series(index=df.index, dtype="object")
    for i in range(1, len(df)):
        if pd.isna(final_upperband.iloc[i - 1]):
            final_upperband.iloc[i - 1] = upperband.iloc[i - 1]
        if pd.isna(final_lowerband.iloc[i - 1]):
            final_lowerband.iloc[i - 1] = lowerband.iloc[i - 1]
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


def generate_signal(df, info=None, news_count=0, deals_bias=0):
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
        if last["ADX14"] >= 25:
            score += 1
            reasons.append("ADX trend strength above 25")
        else:
            reasons.append("ADX indicates moderate trend")
    if info:
        rec = str(info.get("recommendationKey", "")).lower()
        if rec in {"buy", "strong_buy", "outperform"}:
            score += 1
            reasons.append("Analyst recommendation supportive")
        elif rec in {"sell", "underperform"}:
            score -= 1
            reasons.append("Analyst recommendation weak")
        if info.get("heldPercentInstitutions") not in (None, ""):
            if float(info.get("heldPercentInstitutions", 0) or 0) > 0.15:
                score += 1
                reasons.append("Institutional holding present")
    if news_count >= 5:
        score += 1
        reasons.append("Healthy recent news flow")
    score += deals_bias
    if deals_bias > 0:
        reasons.append("Deal flow supportive")
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


def get_company_name(symbol, stocks_df):
    try:
        row = stocks_df.loc[stocks_df["SYMBOL"] == symbol].iloc[0]
        return row["NAME OF COMPANY"]
    except Exception:
        return symbol


@cache.memoize(timeout=1800)
def fetch_stock_history(symbol):
    last_error = None
    for attempt in range(3):
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            df = ticker.history(period="1y", interval="1d", auto_adjust=False)
            if df.empty:
                raise ValueError(f"No market data found for {symbol}")
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
            return df
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            if "too many requests" in msg or "rate limited" in msg:
                time.sleep(3 * (attempt + 1))
            else:
                break
    raise ValueError(f"Yahoo Finance temporarily unavailable for {symbol}: {last_error}")


@cache.memoize(timeout=1800)
def fetch_stock_info(symbol):
    ticker = yf.Ticker(f"{symbol}.NS")
    info = ticker.info or {}
    try:
        fast = ticker.fast_info or {}
        for k, v in fast.items():
            info.setdefault(k, v)
    except Exception:
        pass
    return info


@cache.memoize(timeout=1800)
def fetch_stock_news(symbol):
    ticker = yf.Ticker(f"{symbol}.NS")
    news = getattr(ticker, "news", []) or []
    cleaned = []
    for item in news[:15]:
        content = item.get("content", {}) if isinstance(item, dict) else {}
        cleaned.append({
            "title": content.get("title") or item.get("title") or "Untitled",
            "summary": content.get("summary") or item.get("summary") or "",
            "publisher": content.get("provider", {}).get("displayName") or item.get("publisher") or "Source",
            "link": content.get("canonicalUrl", {}).get("url") or item.get("link") or "",
            "published": content.get("pubDate") or item.get("providerPublishTime") or "",
        })
    return cleaned


@cache.memoize(timeout=1800)
def fetch_holders_tables(symbol):
    ticker = yf.Ticker(f"{symbol}.NS")
    out = {}
    for name, attr in [("major_holders", "major_holders"), ("institutional_holders", "institutional_holders"), ("mutualfund_holders", "mutualfund_holders")]:
        try:
            df = getattr(ticker, attr)
            if isinstance(df, pd.DataFrame) and not df.empty:
                out[name] = df.reset_index(drop=True)
        except Exception:
            pass
    return out


@cache.memoize(timeout=1800)
def fetch_financial_tables(symbol):
    ticker = yf.Ticker(f"{symbol}.NS")
    tables = {}
    for key, attr in {
        "income_stmt": "income_stmt",
        "quarterly_income_stmt": "quarterly_income_stmt",
        "balance_sheet": "balance_sheet",
        "quarterly_balance_sheet": "quarterly_balance_sheet",
        "cashflow": "cashflow",
        "quarterly_cashflow": "quarterly_cashflow",
    }.items():
        try:
            df = getattr(ticker, attr)
            if isinstance(df, pd.DataFrame) and not df.empty:
                tables[key] = df.copy()
        except Exception:
            pass
    return tables


def price_snapshot(df, info=None):
    last = df.iloc[-1]
    prev_close = df["Close"].iloc[-2] if len(df) > 1 else last["Close"]
    chg = float(last["Close"] - prev_close)
    pct = (chg / prev_close * 100) if prev_close else 0
    return {
        "close": round(float(last["Close"]), 2),
        "change": round(chg, 2),
        "change_pct": round(pct, 2),
        "high": round(float(last["High"]), 2),
        "low": round(float(last["Low"]), 2),
        "volume": int(last["Volume"]),
        "high_52": info.get("fiftyTwoWeekHigh") if info else None,
        "low_52": info.get("fiftyTwoWeekLow") if info else None,
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
    return f"https://www.screener.in/company/{quote_plus(symbol)}/"


def moneycontrol_bulk_url(symbol):
    return "https://www.moneycontrol.com/markets/stock-deals/bulk-deals/"


def nse_bulk_url():
    return "https://www.nseindia.com/market-data/large-deals"


def style_card(children, height=None):
    style = {"background": f"linear-gradient(180deg, {THEME['panel2']}, {THEME['panel']})", "border": f"1px solid {THEME['border']}", "borderRadius": "18px", "padding": "18px", "boxShadow": "0 12px 28px rgba(0,0,0,0.28)"}
    if height:
        style["minHeight"] = height
    return html.Div(children, style=style)


def stat_card(title, value, sub=None, color=None):
    return style_card([html.Div(title, style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}), html.Div(value, style={"color": color or THEME["text"], "fontSize": "26px", "fontWeight": "800", "marginTop": "10px"}), html.Div(sub or "", style={"color": THEME["faint"], "fontSize": "13px", "marginTop": "8px"})], height="118px")


def metric_line(label, value):
    return html.Div([html.Span(label, style={"color": THEME["muted"]}), html.Strong(value, style={"color": THEME["text"]})], style={"display": "flex", "justifyContent": "space-between", "padding": "10px 0", "borderBottom": f"1px solid {THEME['border']}"})


def make_chart(df, symbol, indicators):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.60, 0.20, 0.20])
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price", increasing_line_color="#44d7b6", decreasing_line_color="#ff6b8a"), row=1, col=1)
    if "SMA20" in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", line=dict(color="#52b6ff", width=1.4)), row=1, col=1)
    if "SMA50" in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(color="#ffbf69", width=1.4)), row=1, col=1)
    if "SMA200" in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200", line=dict(color="#ff7d9c", width=1.4)), row=1, col=1)
    if "EMA20" in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(color="#8b5cf6", width=1.2, dash="dot")), row=1, col=1)
    if "EMA50" in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(color="#2dd4bf", width=1.2, dash="dot")), row=1, col=1)
    if "Bollinger" in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_UPPER"], name="BB Upper", line=dict(color="#7c8aa5", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOWER"], name="BB Lower", line=dict(color="#7c8aa5", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(124,138,165,0.08)"), row=1, col=1)
    if "Supertrend" in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df["SUPERTREND"], name="Supertrend", line=dict(color="#34d399", width=1.5)), row=1, col=1)
    if "MACD" in indicators:
        colors = np.where(df["MACD_HIST"] >= 0, "#34d399", "#fb7185")
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="MACD Hist", marker_color=colors), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#5aa8ff", width=1.4)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", line=dict(color="#ffcc66", width=1.2)), row=2, col=1)
    if "RSI" in indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14", line=dict(color="#b38cff", width=1.8)), row=3, col=1)
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
    show = show.reset_index().rename(columns={show.index.name or "index": "Metric"}).fillna("")
    return dash_table.DataTable(data=show.to_dict("records"), columns=[{"name": c, "id": c} for c in show.columns], style_as_list_view=True, style_table={"overflowX": "auto"}, style_header={"backgroundColor": THEME["bg3"], "color": THEME["text"], "border": f"1px solid {THEME['border']}", "fontWeight": "700"}, style_cell={"backgroundColor": THEME["panel"], "color": THEME["text"], "border": f"1px solid {THEME['border']}", "padding": "10px", "textAlign": "left", "minWidth": "110px", "maxWidth": "260px", "whiteSpace": "normal"}, page_size=rows)


def holders_summary(info, holders):
    major = holders.get("major_holders")
    major_map = {}
    if isinstance(major, pd.DataFrame) and major.shape[1] >= 2:
        for _, row in major.iterrows():
            major_map[str(row.iloc[1])] = row.iloc[0]
    rows = [("FII / Institutional", fmt_pct(info.get("heldPercentInstitutions")) if info else "-"), ("Promoters / Insiders", fmt_pct(info.get("heldPercentInsiders")) if info else "-"), ("Mutual Funds", major_map.get("% of Shares Held by Institutions", "-")), ("Retail / Public", "Check detailed shareholding in Screener / annual filing")]
    return html.Div([metric_line(a, b) for a, b in rows])


def company_overview(symbol, company_name, info):
    summary = info.get("longBusinessSummary") or info.get("description") or "Business summary not available."
    return style_card([html.Div(f"{company_name} ({symbol})", style={"fontSize": "24px", "fontWeight": "800", "color": THEME["text"]}), html.Div(f"{info.get('sector', '-')} • {info.get('industry', '-')}", style={"marginTop": "6px", "color": THEME["muted"]}), html.P(summary[:900], style={"marginTop": "14px", "color": THEME["text"], "lineHeight": "1.7"})])


def build_news_cards(news_items):
    if not news_items:
        return style_card(html.Div("No recent news available from Yahoo feed.", style={"color": THEME["muted"]}))
    cards = []
    for item in news_items[:10]:
        cards.append(style_card([html.Div(item.get("publisher", "Source"), style={"color": THEME["accent"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}), html.A(item.get("title", "Untitled"), href=item.get("link") or None, target="_blank", rel="noopener noreferrer", style={"display": "block", "marginTop": "8px", "color": THEME["text"], "fontWeight": "800", "textDecoration": "none", "fontSize": "17px"}), html.Div(item.get("summary", "")[:240], style={"marginTop": "8px", "color": THEME["muted"], "lineHeight": "1.6"}), html.Div(str(item.get("published", "")), style={"marginTop": "10px", "color": THEME["faint"], "fontSize": "12px"})]))
    return html.Div(cards, style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(280px,1fr))", "gap": "16px"})


def button_link_style():
    return {"display": "inline-block", "padding": "10px 14px", "background": THEME["bg3"], "border": f"1px solid {THEME['border']}", "color": THEME["text"], "borderRadius": "12px", "textDecoration": "none", "fontWeight": "700", "marginRight": "10px", "marginBottom": "10px"}


def build_home_page(symbol, company_name, info, df, signal, score, confidence, reasons, news_items, holders):
    snap = price_snapshot(df, info)
    plan = order_plan(df)
    last = df.iloc[-1]
    top_cards = html.Div([stat_card("Last Close", fmt_currency_inr(snap["close"]), f"{snap['change']} ({snap['change_pct']}%)", THEME["good"] if snap["change"] >= 0 else THEME["danger"]), stat_card("Day Range", f"₹ {fmt_num(snap['low'])} - ₹ {fmt_num(snap['high'])}", "Day low / high"), stat_card("52 Week Range", f"₹ {fmt_num(snap['low_52'])} - ₹ {fmt_num(snap['high_52'])}", "52-week low / high"), stat_card("Market Cap", fmt_market_cap(info.get("marketCap")), info.get("exchange", "NSE"))], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(220px,1fr))", "gap": "16px"})
    signal_card = style_card([html.Div("Technical + Flow Signal", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}), html.Div(signal, style={"fontSize": "32px", "fontWeight": "900", "marginTop": "10px", "color": SIGNAL_COLORS.get(signal, THEME['text'])}), html.Div(f"Score: {score}   |   Confidence: {confidence}%", style={"color": THEME["text"], "marginTop": "8px"}), html.Ul([html.Li(r) for r in reasons], style={"marginTop": "12px", "paddingLeft": "18px", "color": THEME["muted"], "lineHeight": "1.8"})])
    plan_card = style_card([html.Div("Trade Plan", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}), metric_line("Support", fmt_currency_inr(plan["support"])), metric_line("Resistance", fmt_currency_inr(plan["resistance"])), metric_line("Buy Entry", fmt_currency_inr(plan["buy_entry"])), metric_line("Buy Stop Loss", fmt_currency_inr(plan["buy_sl"])), metric_line("Buy Target 1", fmt_currency_inr(plan["buy_t1"])), metric_line("Buy Target 2", fmt_currency_inr(plan["buy_t2"])),])
    ratios_card = style_card([html.Div("Investor Holding View", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}), holders_summary(info, holders)])
    quick_metrics = style_card([html.Div("Quick Technical Data", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}), metric_line("RSI 14", fmt_num(last["RSI14"])), metric_line("MACD", fmt_num(last["MACD"])), metric_line("MACD Signal", fmt_num(last["MACD_SIGNAL"])), metric_line("ADX 14", fmt_num(last["ADX14"])), metric_line("ATR 14", fmt_num(last["ATR14"])), metric_line("Volume", fmt_int(last["Volume"])), metric_line("Volume Avg 20", fmt_int(last["VOL_MA20"])), metric_line("Supertrend", fmt_num(last["SUPERTREND"])),])
    links = style_card([html.Div("Research Links", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "12px"}), html.A("Open Screener", href=screener_url(symbol), target="_blank", rel="noopener noreferrer", style=button_link_style()), html.A("NSE Large Deals", href=nse_bulk_url(), target="_blank", rel="noopener noreferrer", style=button_link_style()), html.A("Moneycontrol Bulk Deals", href=moneycontrol_bulk_url(symbol), target="_blank", rel="noopener noreferrer", style=button_link_style()), html.A("TradingView", href=f"https://in.tradingview.com/symbols/{make_tv_symbol(symbol).replace(':','-')}/", target="_blank", rel="noopener noreferrer", style=button_link_style()),])
    news_preview = style_card([html.Div("Latest News Snapshot", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}), html.Div([html.Div(item.get("title", ""), style={"padding": "10px 0", "borderBottom": f"1px solid {THEME['border']}", "color": THEME['text']}) for item in news_items[:5]]) if news_items else html.Div("No recent news available.", style={"color": THEME["muted"], "marginTop": "10px"}),])
    return html.Div([top_cards, html.Div(style={"height": "16px"}), company_overview(symbol, company_name, info), html.Div(style={"height": "16px"}), html.Div([signal_card, plan_card, ratios_card, quick_metrics], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(280px,1fr))", "gap": "16px"}), html.Div(style={"height": "16px"}), html.Div([links, news_preview], style={"display": "grid", "gridTemplateColumns": "1fr 1.3fr", "gap": "16px"})])


def build_chart_page(symbol, df, indicators):
    fig = make_chart(df, symbol, indicators)
    controls = style_card([html.Div("Editable Indicators", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "12px"}), dcc.Checklist(id="chart-indicator-checklist", options=[{"label": "SMA20", "value": "SMA20"}, {"label": "SMA50", "value": "SMA50"}, {"label": "SMA200", "value": "SMA200"}, {"label": "EMA20", "value": "EMA20"}, {"label": "EMA50", "value": "EMA50"}, {"label": "Bollinger", "value": "Bollinger"}, {"label": "Supertrend", "value": "Supertrend"}, {"label": "MACD", "value": "MACD"}, {"label": "RSI", "value": "RSI"}], value=indicators, inline=True, inputStyle={"marginRight": "6px", "marginLeft": "14px"}, labelStyle={"display": "inline-flex", "alignItems": "center", "marginBottom": "10px", "color": THEME["text"]})])
    return html.Div([controls, html.Div(style={"height": "14px"}), dcc.Graph(figure=fig, config={"displaylogo": False, "responsive": True})])


def build_financials_page(symbol, info, holders, fin_tables):
    top = html.Div([stat_card("PE Ratio", fmt_num(info.get("trailingPE")), "Trailing PE"), stat_card("PB Ratio", fmt_num(info.get("priceToBook")), "Price to book"), stat_card("ROE", fmt_pct(info.get("returnOnEquity")), "Return on equity"), stat_card("Dividend Yield", fmt_pct(info.get("dividendYield")), "Dividend yield")], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(220px,1fr))", "gap": "16px"})
    profile = style_card([html.Div("Company Fundamentals", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}), metric_line("Sector", str(info.get("sector", "-"))), metric_line("Industry", str(info.get("industry", "-"))), metric_line("Employees", fmt_int(info.get("fullTimeEmployees"))), metric_line("Enterprise Value", fmt_market_cap(info.get("enterpriseValue"))), metric_line("Book Value", fmt_num(info.get("bookValue"))), metric_line("EPS", fmt_num(info.get("trailingEps"))), metric_line("Beta", fmt_num(info.get("beta"))), metric_line("Profit Margin", fmt_pct(info.get("profitMargins"))),])
    holders_card = style_card([html.Div("Shareholding Tables", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "12px"}), html.H4("Institutional Holders", style={"marginBottom": "10px"}), df_to_table(holders.get("institutional_holders"), rows=8), html.Div(style={"height": "12px"}), html.H4("Mutual Fund Holders", style={"marginBottom": "10px"}), df_to_table(holders.get("mutualfund_holders"), rows=8),])
    financial_tables = html.Div([style_card([html.H4("Annual Income Statement", style={"marginBottom": "10px"}), df_to_table(fin_tables.get("income_stmt"), rows=12)]), style_card([html.H4("Annual Balance Sheet", style={"marginBottom": "10px"}), df_to_table(fin_tables.get("balance_sheet"), rows=12)]), style_card([html.H4("Annual Cash Flow", style={"marginBottom": "10px"}), df_to_table(fin_tables.get("cashflow"), rows=12)])], style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "16px"})
    links = style_card([html.Div("External Research", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "12px"}), html.A("Open Screener Company Page", href=screener_url(symbol), target="_blank", rel="noopener noreferrer", style=button_link_style()), html.A("NSE Large Deals", href=nse_bulk_url(), target="_blank", rel="noopener noreferrer", style=button_link_style()), html.A("Moneycontrol Bulk Deals", href=moneycontrol_bulk_url(symbol), target="_blank", rel="noopener noreferrer", style=button_link_style())])
    return html.Div([top, html.Div(style={"height": "16px"}), html.Div([profile, links], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}), html.Div(style={"height": "16px"}), holders_card, html.Div(style={"height": "16px"}), financial_tables])


def build_news_page(news_items, symbol):
    header = style_card([html.Div("News & Order Flow", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px"}), html.Div(f"Latest news for {symbol}", style={"fontSize": "24px", "fontWeight": "800", "marginTop": "10px"}), html.Div("Bulk or block deal discovery is provided through NSE and Moneycontrol links, while recent feed items appear below.", style={"marginTop": "8px", "color": THEME["muted"]}), html.Div(style={"marginTop": "12px"}, children=[html.A("NSE Large Deals", href=nse_bulk_url(), target="_blank", rel="noopener noreferrer", style=button_link_style()), html.A("Moneycontrol Bulk Deals", href=moneycontrol_bulk_url(symbol), target="_blank", rel="noopener noreferrer", style=button_link_style())])])
    return html.Div([header, html.Div(style={"height": "16px"}), build_news_cards(news_items)])


def build_tab_content(tab, symbol, company_name, info, df, signal, score, confidence, reasons, news_items, holders, fin_tables, indicators):
    if tab == "chart":
        return build_chart_page(symbol, df, indicators)
    if tab == "financials":
        return build_financials_page(symbol, info, holders, fin_tables)
    if tab == "news":
        return build_news_page(news_items, symbol)
    return build_home_page(symbol, company_name, info, df, signal, score, confidence, reasons, news_items, holders)


try:
    STOCKS_DF = load_nse_stocks()
    DROPDOWN_OPTIONS = [{"label": f"{r['SYMBOL']} - {r['NAME OF COMPANY']}", "value": r["SYMBOL"]} for _, r in STOCKS_DF.iterrows()]
except Exception:
    STOCKS_DF = pd.DataFrame(columns=["SYMBOL", "NAME OF COMPANY"])
    DROPDOWN_OPTIONS = []


def _scan_one(symbol, company):
    try:
        hist = fetch_stock_history(symbol)
        hist = add_indicators(hist)
        sig, score, conf, _ = generate_signal(hist)
        last = hist.iloc[-1]
        return {"symbol": symbol, "company": company, "signal": sig, "confidence": conf, "close": round(float(last["Close"]), 2), "rsi": round(float(last["RSI14"]), 2) if pd.notna(last["RSI14"]) else None}
    except Exception:
        return None


@cache.memoize(timeout=1800)
def build_scan_universe_cached(limit=TOP_SCAN_LIMIT):
    rows = []
    sample = STOCKS_DF.head(limit)
    workers = min(12, max(4, len(sample) // 20 if len(sample) else 4))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_scan_one, row["SYMBOL"], row["NAME OF COMPANY"]) for _, row in sample.iterrows()]
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

DEFAULT_INDICATORS = ["SMA20", "SMA50", "SMA200", "Bollinger", "Supertrend", "MACD", "RSI"]

app.layout = html.Div([
    dcc.Store(id="selected-symbol-store", data=DEFAULT_SYMBOL),
    dcc.Store(id="selected-indicators-store", data=DEFAULT_INDICATORS),
    html.Div([
        html.Div([html.Div("BLAST NSE LAB PRO", style={"fontSize": "12px", "letterSpacing": "2px", "color": THEME["accent"], "fontWeight": "800"}), html.H1("Dark Theme Stock Dashboard", style={"margin": "8px 0 4px 0", "fontSize": "34px"}), html.Div("Cleaner watchlist, stronger visibility, quick stock intelligence, chart lab, financials and live news in one screen.", style={"color": THEME["muted"], "maxWidth": "900px", "lineHeight": "1.7"})]),
        html.Div(style={"height": "20px"}),
        html.Div([
            style_card([html.Div("Stock Search", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "12px"}), dcc.Dropdown(id="stock-dropdown", options=DROPDOWN_OPTIONS, value=DEFAULT_SYMBOL if DEFAULT_SYMBOL in [o["value"] for o in DROPDOWN_OPTIONS] else None, placeholder="Search stock symbol or company name", searchable=True, clearable=False, style={"color": "#111827"}), html.Div(id="selected-stock-title", style={"marginTop": "12px", "color": THEME["text"], "fontWeight": "700"})]),
            style_card([html.Div("Smart Technical Scanner", style={"color": THEME["muted"], "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "12px"}), dcc.Dropdown(id="signal-filter-dropdown", options=[{"label": "All", "value": "ALL"}, {"label": "Strong Buy", "value": "STRONG BUY"}, {"label": "Buy", "value": "BUY"}, {"label": "Hold", "value": "HOLD"}, {"label": "Nil", "value": "NIL"}, {"label": "Sell", "value": "SELL"}, {"label": "Strong Sell", "value": "STRONG SELL"}], value="ALL", clearable=False, style={"color": "#111827"}), html.Div(id="scanner-table-container", style={"marginTop": "12px"})]),
        ], style={"display": "grid", "gridTemplateColumns": "0.95fr 1.35fr", "gap": "16px"}),
        html.Div(style={"height": "18px"}),
        dcc.Tabs(id="main-tabs", value="home", parent_className="custom-tabs", className="custom-tabs-container", children=[dcc.Tab(label="Home", value="home", className="custom-tab", selected_className="custom-tab--selected"), dcc.Tab(label="Chart", value="chart", className="custom-tab", selected_className="custom-tab--selected"), dcc.Tab(label="Financials", value="financials", className="custom-tab", selected_className="custom-tab--selected"), dcc.Tab(label="News", value="news", className="custom-tab", selected_className="custom-tab--selected")]),
        html.Div(id="main-tab-content", style={"marginTop": "18px"}),
    ], style={"maxWidth": "1450px", "margin": "0 auto", "padding": "28px 18px 60px 18px"})
], style={"minHeight": "100vh", "background": f"radial-gradient(circle at top left, {THEME['bg3']}, {THEME['bg']})", "color": THEME["text"], "fontFamily": "Inter, Segoe UI, Arial, sans-serif"})

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { background: #0b1220; }
            .custom-tabs-container { width: 100%; }
            .custom-tabs { background: #121c2d; border: 1px solid #26354f; border-radius: 16px; padding: 8px; display: flex; gap: 8px; flex-wrap: wrap; }
            .custom-tab { color: #9fb0c8 !important; background: #172235 !important; border: 1px solid #26354f !important; border-radius: 12px !important; padding: 12px 18px !important; font-weight: 700; }
            .custom-tab--selected { color: #e6edf7 !important; background: linear-gradient(180deg, #1f3150, #172235) !important; border: 1px solid #3a5277 !important; }
            @media (max-width: 900px) { .custom-tabs { display:block; } }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


@app.callback(Output("selected-symbol-store", "data"), Output("selected-stock-title", "children"), Input("stock-dropdown", "value"))
def select_stock(symbol):
    if not symbol:
        return no_update, ""
    company = get_company_name(symbol, STOCKS_DF)
    return symbol, f"Selected: {symbol} - {company}"


@app.callback(Output("selected-indicators-store", "data"), Input("chart-indicator-checklist", "value"), prevent_initial_call=True)
def sync_indicator_store(values):
    return values or DEFAULT_INDICATORS


@app.callback(Output("scanner-table-container", "children"), Input("signal-filter-dropdown", "value"))
def render_scanner(signal_value):
    if SCAN_DF.empty:
        return html.Div("Scanner data is not available right now.", style={"color": THEME["muted"]})
    df = SCAN_DF.copy()
    if signal_value and signal_value != "ALL":
        df = df[df["signal"] == signal_value]
    show = df.head(20).copy()
    return dash_table.DataTable(
        data=show.to_dict("records"),
        columns=[{"name": "Symbol", "id": "symbol"}, {"name": "Company", "id": "company"}, {"name": "Signal", "id": "signal"}, {"name": "Confidence %", "id": "confidence"}, {"name": "Close", "id": "close"}, {"name": "RSI", "id": "rsi"}],
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


@app.callback(Output("main-tab-content", "children"), Input("main-tabs", "value"), Input("selected-symbol-store", "data"), Input("selected-indicators-store", "data"))
def render_main_content(tab, symbol, indicators):
    symbol = symbol or DEFAULT_SYMBOL
    indicators = indicators or DEFAULT_INDICATORS
    try:
        company_name = get_company_name(symbol, STOCKS_DF)
        df = add_indicators(fetch_stock_history(symbol))
        info = fetch_stock_info(symbol)
        news_items = fetch_stock_news(symbol)
        holders = fetch_holders_tables(symbol)
        fin_tables = fetch_financial_tables(symbol)
        signal, score, confidence, reasons = generate_signal(df, info=info, news_count=len(news_items), deals_bias=0)
        return build_tab_content(tab, symbol, company_name, info, df, signal, score, confidence, reasons, news_items, holders, fin_tables, indicators)
    except Exception as e:
        return style_card([html.Div("Data Error", style={"fontSize": "22px", "fontWeight": "800", "color": THEME["danger"]}), html.Div(str(e), style={"marginTop": "10px", "color": THEME["text"], "lineHeight": "1.7"})])


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)import pandas as pd
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

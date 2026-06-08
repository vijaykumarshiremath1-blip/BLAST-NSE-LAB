import io
import zipfile
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, dcc, html, dash_table
from flask_caching import Cache
from plotly.subplots import make_subplots

APP_NAME = "Screener 2.0"
DEFAULT_SYMBOL = "RELIANCE"
LOOKBACK_DAYS = 220
NSE_HOME = "https://www.nseindia.com"
NSE_REPORTS_URL = "https://www.nseindia.com/all-reports"
EQ_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
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
    "bg": "#08111f",
    "bg2": "#0d1728",
    "panel": "#111c2d",
    "panel2": "#162338",
    "border": "#26354f",
    "text": "#e7eef7",
    "muted": "#a2b2c9",
    "faint": "#72839d",
    "accent": "#28d3a5",
    "blue": "#61a8ff",
    "good": "#34d399",
    "warn": "#ffcc66",
    "danger": "#fb7185",
    "purple": "#b78cff",
}
SIGNAL_COLORS = {
    "STRONG BUY": "#22c55e",
    "BUY": "#86efac",
    "HOLD": "#ffcc66",
    "SELL": "#fda4af",
    "STRONG SELL": "#fb7185",
    "NIL": "#94a3b8",
}
NEWS_URL = {
    "Moneycontrol": "https://www.moneycontrol.com/stocks/marketstats/nsemovers/index.php",
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets",
    "NSE Reports": NSE_REPORTS_URL,
}

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = APP_NAME
cache = Cache(app.server, config={
    "CACHE_TYPE": "FileSystemCache",
    "CACHE_DIR": "cache-directory",
    "CACHE_THRESHOLD": 2000,
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


def fmt_currency(value, default="-"):
    try:
        if value is None or pd.isna(value):
            return default
        return f"₹ {float(value):,.2f}"
    except Exception:
        return default


def style_card(children, extra=None):
    style = {
        "background": f"linear-gradient(180deg, {THEME['panel2']}, {THEME['panel']})",
        "border": f"1px solid {THEME['border']}",
        "borderRadius": "18px",
        "padding": "18px",
        "boxShadow": "0 12px 26px rgba(0,0,0,0.22)",
    }
    if extra:
        style.update(extra)
    return html.Div(children, style=style)


def stat_card(title, value, subtitle="", color=None):
    return style_card([
        html.Div(title, style={"fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "color": THEME["muted"]}),
        html.Div(value, style={"fontSize": "30px", "fontWeight": "800", "marginTop": "8px", "color": color or THEME["text"]}),
        html.Div(subtitle, style={"fontSize": "13px", "marginTop": "8px", "color": THEME["faint"]}),
    ], {"minHeight": "122px"})


def kv_row(label, value, color=None):
    return html.Div([
        html.Div(label, style={"color": THEME["muted"]}),
        html.Div(value, style={"color": color or THEME["text"], "fontWeight": "700"}),
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
    required = ["SYMBOL", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
    if not all(c in df.columns for c in required):
        raise ValueError("Unexpected bhavcopy format")
    if "SERIES" in df.columns:
        df = df[df["SERIES"].astype(str).str.strip().eq("EQ")]
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


def build_symbol_history(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    existing = load_local_history(symbol)
    end = datetime.today()
    rows = []
    for i in range(days):
        dt = end - timedelta(days=i)
        if dt.weekday() >= 5:
            continue
        day = download_bhavcopy_day(dt)
        if day is None or day.empty:
            continue
        match = day[day["SYMBOL"].astype(str).str.strip().eq(symbol)]
        if not match.empty:
            rows.append(match.iloc[0])
    if not rows and existing.empty:
        raise ValueError(f"No NSE bhavcopy history available for {symbol}.")
    if rows:
        new_df = pd.DataFrame(rows).sort_values("DATE").drop_duplicates(subset=["DATE"])
        new_df = new_df.rename(columns={"DATE": "Date", "OPEN": "Open", "HIGH": "High", "LOW": "Low", "CLOSE": "Close", "VOLUME": "Volume"})
        merged = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        merged = merged.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        save_local_history(symbol, merged)
        return merged
    return existing


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def macd(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    line = fast_ema - slow_ema
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


def atr(df, period=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def adx(df, period=14):
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_val = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).sum() / atr_val
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).sum() / atr_val
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()


def supertrend(df, period=10, multiplier=3):
    hl2 = (df["High"] + df["Low"]) / 2
    a = atr(df, period)
    upper = hl2 + multiplier * a
    lower = hl2 - multiplier * a
    final_upper = upper.copy()
    final_lower = lower.copy()
    trend = pd.Series(index=df.index, dtype="float64")
    direction = pd.Series(index=df.index, dtype="object")
    for i in range(1, len(df)):
        final_upper.iloc[i] = upper.iloc[i] if upper.iloc[i] < final_upper.iloc[i-1] or df["Close"].iloc[i-1] > final_upper.iloc[i-1] else final_upper.iloc[i-1]
        final_lower.iloc[i] = lower.iloc[i] if lower.iloc[i] > final_lower.iloc[i-1] or df["Close"].iloc[i-1] < final_lower.iloc[i-1] else final_lower.iloc[i-1]
        prev_trend = trend.iloc[i-1] if i > 0 else np.nan
        if pd.isna(prev_trend):
            trend.iloc[i] = final_lower.iloc[i]
            direction.iloc[i] = "Bullish"
        elif prev_trend == final_upper.iloc[i-1]:
            if df["Close"].iloc[i] <= final_upper.iloc[i]:
                trend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = "Bearish"
            else:
                trend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = "Bullish"
        else:
            if df["Close"].iloc[i] >= final_lower.iloc[i]:
                trend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = "Bullish"
            else:
                trend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = "Bearish"
    return trend, direction


def add_indicators(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI14"] = rsi(df["Close"])
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd(df["Close"])
    df["ATR14"] = atr(df)
    df["ADX14"] = adx(df)
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_MID"] = mid
    df["BB_UPPER"] = mid + 2 * std
    df["BB_LOWER"] = mid - 2 * std
    df["SUPERTREND"], df["ST_DIR"] = supertrend(df)
    return df


def signal_from_df(df):
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
        else:
            score -= 1
            reasons.append("SMA50 below SMA200")
    if pd.notna(last["RSI14"]):
        if last["RSI14"] > 60:
            score += 1
            reasons.append("RSI above 60")
        elif last["RSI14"] < 40:
            score -= 1
            reasons.append("RSI below 40")
    if pd.notna(last["MACD"]) and pd.notna(last["MACD_SIGNAL"]):
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
    if pd.notna(last["ADX14"]) and last["ADX14"] >= 25:
        score += 1
        reasons.append("ADX above 25")
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
    confidence = min(97, max(40, 50 + abs(score) * 6))
    return signal, score, confidence, reasons[:6]


def support_resistance(df, lookback=40):
    recent = df.tail(lookback)
    if recent.empty:
        return None, None
    return round(float(recent["Low"].min()), 2), round(float(recent["High"].max()), 2)


def company_profile(symbol):
    row = STOCKS_DF.loc[STOCKS_DF["SYMBOL"] == symbol]
    company = row.iloc[0]["NAME OF COMPANY"] if not row.empty else symbol
    return {
        "name": company,
        "sector": "NSE Listed Equity",
        "exchange": "NSE",
        "series": "EQ",
        "website": NSE_REPORTS_URL,
        "about": f"{company} is shown here through an NSE-focused technical dashboard using cached bhavcopy history for quicker reloads.",
    }


def build_news_items(symbol, company):
    return [
        {"headline": f"Track {symbol} company updates on NSE reports", "source": "NSE", "url": NEWS_URL["NSE Reports"], "summary": f"Use the NSE reports portal for circulars, corporate announcements, and filings related to {company}."},
        {"headline": f"Watch broader market sentiment before trading {symbol}", "source": "Economic Times Markets", "url": NEWS_URL["Economic Times Markets"], "summary": "Market-wide risk, rates, and sector momentum can affect technical follow-through."},
        {"headline": f"Check active movers and sentiment for {symbol}", "source": "Moneycontrol", "url": NEWS_URL["Moneycontrol"], "summary": "Cross-check price action with active movers, volume spikes, and market mood before acting."},
    ]


def make_chart(df, symbol):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.60, 0.20, 0.20])
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#34d399", decreasing_line_color="#fb7185"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], name="SMA20", line=dict(color="#60a5fa", width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], name="SMA50", line=dict(color="#fbbf24", width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA200"], name="SMA200", line=dict(color="#f472b6", width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_UPPER"], name="BB Upper", line=dict(color="#64748b", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_LOWER"], name="BB Lower", line=dict(color="#64748b", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(100,116,139,0.08)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SUPERTREND"], name="Supertrend", line=dict(color="#22c55e", width=1.4)), row=1, col=1)
    colors = np.where(df["MACD_HIST"] >= 0, "#34d399", "#fb7185")
    fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_HIST"], name="MACD Hist", marker_color=colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD", line=dict(color="#60a5fa", width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_SIGNAL"], name="Signal", line=dict(color="#fbbf24", width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI14"], name="RSI14", line=dict(color="#b78cff", width=1.8)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#fb7185", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#34d399", row=3, col=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"]),
        height=850,
        margin=dict(l=20, r=20, t=48, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        title=f"{symbol} Technical Chart"
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


@cache.memoize(timeout=86400)
def load_stock_master_records():
    r = session.get(EQ_LIST_URL, headers=CSV_HEADERS, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [str(c).strip() for c in df.columns]
    df = df[df["SERIES"].astype(str).str.strip().eq("EQ")].copy()
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    df = df.drop_duplicates(subset=["SYMBOL"]).sort_values("SYMBOL")
    return df[["SYMBOL", "NAME OF COMPANY"]].to_dict("records")


def get_stock_master_df():
    try:
        return pd.DataFrame(load_stock_master_records())
    except Exception:
        return pd.DataFrame([{"SYMBOL": DEFAULT_SYMBOL, "NAME OF COMPANY": "Reliance Industries Limited"}])


STOCKS_DF = get_stock_master_df()
DROPDOWN_OPTIONS = [{"label": f"{r['SYMBOL']} - {r['NAME OF COMPANY']}", "value": r["SYMBOL"]} for _, r in STOCKS_DF.iterrows()]


app.layout = html.Div([
    dcc.Store(id="loaded-symbol", data=DEFAULT_SYMBOL),
    html.Div([
        html.Div("SCREENER 2.0", style={"fontSize": "12px", "letterSpacing": "2px", "fontWeight": "800", "color": THEME["accent"]}),
        html.H1("Screener 2.0 - Full Dashboard", style={"margin": "8px 0 4px", "fontSize": "34px"}),
        html.Div("Multi-section stock view with overview, chart, technicals, news, and financial summary cards. Data loads on demand for the selected stock.", style={"color": THEME["muted"], "maxWidth": "940px", "lineHeight": "1.7"}),
        html.Div(style={"height": "18px"}),
        html.Div([
            style_card([
                html.Div("Stock Search", style={"fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "color": THEME["muted"], "marginBottom": "12px"}),
                dcc.Dropdown(id="stock-dropdown", options=DROPDOWN_OPTIONS, value=DEFAULT_SYMBOL, clearable=False, searchable=True, style={"color": "#111827"}),
                html.Div(id="selected-title", style={"marginTop": "12px", "fontWeight": "700", "color": THEME["text"]}),
            ]),
            style_card([
                html.Div("Actions", style={"fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "color": THEME["muted"], "marginBottom": "12px"}),
                html.Button("Load / Refresh", id="load-button", n_clicks=0, style={"background": THEME["accent"], "color": "#06261f", "fontWeight": "800", "padding": "12px 16px", "border": "none", "borderRadius": "12px", "cursor": "pointer", "marginRight": "10px"}),
                html.A("NSE Reports", href=NSE_REPORTS_URL, target="_blank", rel="noopener noreferrer", style={"display": "inline-block", "padding": "12px 16px", "background": THEME["bg2"], "color": THEME["text"], "border": f"1px solid {THEME['border']}", "borderRadius": "12px", "textDecoration": "none", "fontWeight": "700"}),
                html.Div("The first load may take time while NSE bhavcopy history is downloaded and cached locally.", style={"marginTop": "12px", "color": THEME["faint"], "fontSize": "13px"}),
            ]),
        ], style={"display": "grid", "gridTemplateColumns": "1.15fr 1fr", "gap": "16px"}),
        html.Div(style={"height": "18px"}),
        dcc.Loading(type="circle", color=THEME["accent"], children=html.Div(id="dashboard-content"))
    ], style={"maxWidth": "1480px", "margin": "0 auto", "padding": "28px 18px 60px"})
], style={"minHeight": "100vh", "background": f"radial-gradient(circle at top left, {THEME['bg2']}, {THEME['bg']})", "color": THEME["text"], "fontFamily": "Inter, Segoe UI, Arial, sans-serif"})

app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>body { background:#08111f; }</style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>
"""


@app.callback(Output("selected-title", "children"), Input("stock-dropdown", "value"))
def update_selected_title(symbol):
    row = STOCKS_DF.loc[STOCKS_DF["SYMBOL"] == symbol]
    company = row.iloc[0]["NAME OF COMPANY"] if not row.empty else symbol
    return f"Selected: {symbol} - {company}"


@app.callback(
    Output("dashboard-content", "children"),
    Output("loaded-symbol", "data"),
    Input("load-button", "n_clicks"),
    State("stock-dropdown", "value"),
    prevent_initial_call=False,
)
def render_dashboard(n_clicks, symbol):
    symbol = symbol or DEFAULT_SYMBOL
    try:
        hist = load_local_history(symbol)
        if hist.empty or len(hist) < 120 or n_clicks:
            hist = build_symbol_history(symbol)
        if hist.empty:
            return style_card([html.Div("No data available for this symbol right now.", style={"color": THEME["muted"]})]), symbol
        df = add_indicators(hist)
        company = company_profile(symbol)
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        price_change = float(last["Close"] - prev["Close"])
        pct_change = (price_change / prev["Close"] * 100) if prev["Close"] else 0
        signal, score, confidence, reasons = signal_from_df(df)
        period = df.tail(252)
        support, resistance = support_resistance(df)
        atr_val = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0
        chart = make_chart(df, symbol)
        news_items = build_news_items(symbol, company["name"])

        top_cards = html.Div([
            stat_card("Last Close", fmt_currency(last["Close"]), f"{price_change:.2f} ({pct_change:.2f}%)", THEME["good"] if price_change >= 0 else THEME["danger"]),
            stat_card("Signal", signal, f"Score {score} | Confidence {confidence}%", SIGNAL_COLORS.get(signal, THEME["text"])),
            stat_card("52 Week Range", f"₹ {fmt_num(period['Low'].min())} - ₹ {fmt_num(period['High'].max())}", "Rolling 52-week low / high"),
            stat_card("Volume", fmt_int(last["Volume"]), "Latest traded quantity"),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(220px,1fr))", "gap": "16px"})

        overview_tab = style_card([
            html.Div("Overview", style={"fontSize": "24px", "fontWeight": "800", "marginBottom": "12px"}),
            html.Div(company["about"], style={"color": THEME["muted"], "lineHeight": "1.8", "marginBottom": "14px"}),
            kv_row("Company", company["name"]),
            kv_row("Sector", company["sector"]),
            kv_row("Exchange", company["exchange"]),
            kv_row("Series", company["series"]),
            kv_row("Current Signal", signal, SIGNAL_COLORS.get(signal, THEME["text"])),
            kv_row("Confidence", f"{confidence}%", THEME["accent"]),
        ])

        financial_tab = style_card([
            html.Div("Financial Report", style={"fontSize": "24px", "fontWeight": "800", "marginBottom": "12px"}),
            kv_row("Latest Close", fmt_currency(last["Close"])),
            kv_row("Session Open", fmt_currency(last["Open"])),
            kv_row("Session High", fmt_currency(last["High"])),
            kv_row("Session Low", fmt_currency(last["Low"])),
            kv_row("Average True Range", fmt_num(last["ATR14"])),
            kv_row("Volume vs 20DMA", f"{(last['Volume'] / last['VOL_MA20']):.2f}x" if pd.notna(last['VOL_MA20']) and last['VOL_MA20'] else "-"),
            html.Div("This section gives a trading-focused financial snapshot from price and volume history when live fundamentals are not available.", style={"marginTop": "12px", "color": THEME["faint"], "lineHeight": "1.7"}),
        ])

        technical_tab = style_card([
            html.Div("Technicals", style={"fontSize": "24px", "fontWeight": "800", "marginBottom": "12px"}),
            kv_row("RSI 14", fmt_num(last["RSI14"]), THEME["purple"]),
            kv_row("MACD", fmt_num(last["MACD"]), THEME["blue"]),
            kv_row("MACD Signal", fmt_num(last["MACD_SIGNAL"]), THEME["warn"]),
            kv_row("ADX 14", fmt_num(last["ADX14"]), THEME["accent"]),
            kv_row("SMA 20", fmt_num(last["SMA20"])),
            kv_row("SMA 50", fmt_num(last["SMA50"])),
            kv_row("SMA 200", fmt_num(last["SMA200"])),
            kv_row("Supertrend", str(last["ST_DIR"])),
            html.Div([html.Div("Applicability", style={"fontWeight": "800", "marginTop": "14px", "marginBottom": "6px"}), html.Ul([html.Li(r) for r in reasons], style={"paddingLeft": "18px", "color": THEME["muted"], "lineHeight": "1.8"})]),
        ])

        plan_tab = style_card([
            html.Div("Trade Levels", style={"fontSize": "24px", "fontWeight": "800", "marginBottom": "12px"}),
            kv_row("Support", fmt_currency(support)),
            kv_row("Resistance", fmt_currency(resistance)),
            kv_row("Buy Entry", fmt_currency(float(last["Close"]) + 0.15 * atr_val), THEME["good"]),
            kv_row("Buy Stop Loss", fmt_currency(float(last["Close"]) - 1.0 * atr_val), THEME["danger"]),
            kv_row("Buy Target 1", fmt_currency(float(last["Close"]) + 1.2 * atr_val), THEME["blue"]),
            kv_row("Sell Entry", fmt_currency(float(last["Close"]) - 0.15 * atr_val), THEME["warn"]),
        ])

        news_cards = html.Div([
            style_card([
                html.Div(item["headline"], style={"fontSize": "18px", "fontWeight": "800", "marginBottom": "8px"}),
                html.Div(item["summary"], style={"color": THEME["muted"], "lineHeight": "1.7", "marginBottom": "10px"}),
                html.Div(item["source"], style={"fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "color": THEME["faint"], "marginBottom": "10px"}),
                html.A("Open source page", href=item["url"], target="_blank", rel="noopener noreferrer", style={"color": THEME["accent"], "textDecoration": "none", "fontWeight": "700"}),
            ]) for item in news_items
        ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(260px,1fr))", "gap": "16px"})

        data_view = df.tail(60)[["Date", "Open", "High", "Low", "Close", "Volume", "RSI14", "MACD", "ADX14"]].copy()
        data_view["Date"] = data_view["Date"].dt.strftime("%Y-%m-%d")
        for c in ["Open", "High", "Low", "Close", "RSI14", "MACD", "ADX14"]:
            data_view[c] = data_view[c].round(2)

        tabs = dcc.Tabs(id="section-tabs", value="overview", colors={"border": THEME["border"], "primary": THEME["accent"], "background": THEME["panel"]}, children=[
            dcc.Tab(label="Overview", value="overview", children=[html.Div(style={"height": "14px"}), overview_tab]),
            dcc.Tab(label="Chart", value="chart", children=[html.Div(style={"height": "14px"}), style_card([dcc.Graph(figure=chart, config={"displaylogo": False, "responsive": True})], {"padding": "8px 12px 12px"})]),
            dcc.Tab(label="Technicals", value="technicals", children=[html.Div(style={"height": "14px"}), html.Div([technical_tab, plan_tab], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(320px,1fr))", "gap": "16px"})]),
            dcc.Tab(label="Financial Report", value="financials", children=[html.Div(style={"height": "14px"}), financial_tab]),
            dcc.Tab(label="News", value="news", children=[html.Div(style={"height": "14px"}), news_cards]),
            dcc.Tab(label="Data", value="data", children=[html.Div(style={"height": "14px"}), style_card([
                html.Div("Recent Data", style={"fontSize": "22px", "fontWeight": "800", "marginBottom": "12px"}),
                dash_table.DataTable(
                    data=data_view.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in data_view.columns],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": THEME["bg2"], "color": THEME["text"], "fontWeight": "700", "border": f"1px solid {THEME['border']}"},
                    style_cell={"backgroundColor": THEME["panel"], "color": THEME["text"], "border": f"1px solid {THEME['border']}", "padding": "10px", "textAlign": "left"},
                    page_size=12,
                )
            ])]),
        ])

        return html.Div([
            top_cards,
            html.Div(style={"height": "16px"}),
            tabs,
        ]), symbol
    except Exception as e:
        return style_card([
            html.Div("Unable to load stock data", style={"fontSize": "24px", "fontWeight": "800", "color": THEME["danger"]}),
            html.Div(str(e), style={"marginTop": "10px", "color": THEME["text"], "lineHeight": "1.7"}),
            html.Div("Try another stock, wait a few seconds, and click Load / Refresh again.", style={"marginTop": "10px", "color": THEME["muted"]}),
        ]), symbol


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)

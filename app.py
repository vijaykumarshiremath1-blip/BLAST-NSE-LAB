import io
import zipfile
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update
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
    "bg": "#0b1220",
    "bg2": "#10192a",
    "panel": "#121c2d",
    "panel2": "#172235",
    "border": "#26354f",
    "text": "#e6edf7",
    "muted": "#9fb0c8",
    "faint": "#70819b",
    "accent": "#29d3a4",
    "blue": "#5aa8ff",
    "good": "#34d399",
    "warn": "#ffcc66",
    "danger": "#fb7185",
}

SIGNAL_COLORS = {
    "STRONG BUY": "#22c55e",
    "BUY": "#6ee7b7",
    "HOLD": "#ffcc66",
    "SELL": "#fda4af",
    "STRONG SELL": "#fb7185",
    "NIL": "#94a3b8",
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


def fmt_currency_inr(value, default="-"):
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
        "boxShadow": "0 10px 24px rgba(0,0,0,0.25)",
    }
    if extra:
        style.update(extra)
    return html.Div(children, style=style)


def stat_card(title, value, subtitle="", color=None):
    return style_card([
        html.Div(title, style={"fontSize": "12px", "color": THEME["muted"], "textTransform": "uppercase", "letterSpacing": "1px"}),
        html.Div(value, style={"fontSize": "28px", "fontWeight": "800", "color": color or THEME["text"], "marginTop": "8px"}),
        html.Div(subtitle, style={"fontSize": "13px", "color": THEME["faint"], "marginTop": "8px"}),
    ], {"minHeight": "120px"})


def kv_row(label, value):
    return html.Div([
        html.Div(label, style={"color": THEME["muted"]}),
        html.Div(value, style={"color": THEME["text"], "fontWeight": "700"}),
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
    need = ["SYMBOL", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
    if not all(c in df.columns for c in need):
        raise ValueError("Unexpected bhavcopy format")
    if "SERIES" in df.columns:
        df = df[df["SERIES"].astype(str).str.strip().eq("EQ")]
    out = df[need].copy()
    out["DATE"] = pd.to_datetime(out["DATE"], dayfirst=True, errors="coerce")
    for c in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["DATE", "OPEN", "HIGH", "LOW", "CLOSE"])
    return out


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
        if not existing.empty:
            merged = pd.concat([existing, new_df], ignore_index=True)
            merged = merged.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        else:
            merged = new_df.reset_index(drop=True)
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
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
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


def signal_from_df(df: pd.DataFrame):
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


def support_resistance(df: pd.DataFrame, lookback=40):
    recent = df.tail(lookback)
    if recent.empty:
        return None, None
    return round(float(recent["Low"].min()), 2), round(float(recent["High"].max()), 2)


def make_chart(df: pd.DataFrame, symbol: str, indicators: list[str]):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.60, 0.20, 0.20])
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#34d399", decreasing_line_color="#fb7185"
    ), row=1, col=1)
    overlay_map = [
        ("SMA20", "#60a5fa", None),
        ("SMA50", "#fbbf24", None),
        ("SMA200", "#f472b6", None),
        ("EMA20", "#8b5cf6", "dot"),
        ("EMA50", "#22d3ee", "dot"),
    ]
    for col, color, dash_style in overlay_map:
        if col in indicators:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[col], name=col, line=dict(color=color, width=1.4, dash=dash_style)), row=1, col=1)
    if "Bollinger" in indicators:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_UPPER"], name="BB Upper", line=dict(color="#64748b", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_LOWER"], name="BB Lower", line=dict(color="#64748b", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(100,116,139,0.08)"), row=1, col=1)
    if "Supertrend" in indicators:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SUPERTREND"], name="Supertrend", line=dict(color="#22c55e", width=1.5)), row=1, col=1)
    if "MACD" in indicators:
        colors = np.where(df["MACD_HIST"] >= 0, "#34d399", "#fb7185")
        fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_HIST"], name="MACD Hist", marker_color=colors), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD", line=dict(color="#60a5fa", width=1.4)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_SIGNAL"], name="Signal", line=dict(color="#fbbf24", width=1.2)), row=2, col=1)
    if "RSI" in indicators:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI14"], name="RSI14", line=dict(color="#c084fc", width=1.8)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#fb7185", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#34d399", row=3, col=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"]),
        height=840,
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
DEFAULT_COMPANY = STOCKS_DF.loc[STOCKS_DF["SYMBOL"] == DEFAULT_SYMBOL, "NAME OF COMPANY"]
DEFAULT_COMPANY = DEFAULT_COMPANY.iloc[0] if not DEFAULT_COMPANY.empty else DEFAULT_SYMBOL
DEFAULT_INDICATORS = ["SMA20", "SMA50", "SMA200", "Bollinger", "Supertrend", "MACD", "RSI"]


def get_company_name(symbol: str):
    row = STOCKS_DF.loc[STOCKS_DF["SYMBOL"] == symbol]
    if row.empty:
        return symbol
    return row.iloc[0]["NAME OF COMPANY"]


def build_default_table_message(msg: str):
    return style_card([html.Div(msg, style={"color": THEME["muted"], "fontSize": "15px"})])


app.layout = html.Div([
    dcc.Store(id="symbol-store", data=DEFAULT_SYMBOL),
    dcc.Store(id="indicator-store", data=DEFAULT_INDICATORS),
    html.Div([
        html.Div("SCREENER 2.0", style={"fontSize": "12px", "letterSpacing": "2px", "fontWeight": "800", "color": THEME["accent"]}),
        html.H1("Screener 2.0 - Better NSE Dashboard", style={"margin": "8px 0 4px", "fontSize": "34px"}),
        html.Div("Fast startup, Yahoo-free NSE bhavcopy history, cleaner charts, and on-demand loading for a smoother Windows experience.", style={"color": THEME["muted"], "maxWidth": "900px", "lineHeight": "1.7"}),
        html.Div(style={"height": "18px"}),
        html.Div([
            style_card([
                html.Div("Stock Search", style={"fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "color": THEME["muted"], "marginBottom": "12px"}),
                dcc.Dropdown(id="stock-dropdown", options=DROPDOWN_OPTIONS, value=DEFAULT_SYMBOL, clearable=False, searchable=True, style={"color": "#111827"}),
                html.Div(id="selected-title", style={"marginTop": "12px", "fontWeight": "700", "color": THEME["text"]}),
            ]),
            style_card([
                html.Div("Actions", style={"fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "color": THEME["muted"], "marginBottom": "12px"}),
                html.Button("Load / Refresh Selected Stock", id="load-button", n_clicks=0, style={"background": THEME["accent"], "color": "#06261f", "fontWeight": "800", "padding": "12px 16px", "border": "none", "borderRadius": "12px", "cursor": "pointer", "marginRight": "10px"}),
                html.A("NSE All Reports", href=NSE_REPORTS_URL, target="_blank", rel="noopener noreferrer", style={"display": "inline-block", "padding": "12px 16px", "background": THEME["bg2"], "color": THEME["text"], "border": f"1px solid {THEME['border']}", "borderRadius": "12px", "textDecoration": "none", "fontWeight": "700"}),
                html.Div("Tip: first load of a symbol may take some time because data is downloaded from NSE and cached locally.", style={"marginTop": "12px", "color": THEME["faint"], "fontSize": "13px"}),
            ]),
        ], style={"display": "grid", "gridTemplateColumns": "1.1fr 1fr", "gap": "16px"}),
        html.Div(style={"height": "18px"}),
        dcc.Loading(
            id="main-loading",
            type="circle",
            color=THEME["accent"],
            children=html.Div(id="main-content")
        ),
    ], style={"maxWidth": "1450px", "margin": "0 auto", "padding": "28px 18px 60px"})
], style={"minHeight": "100vh", "background": f"radial-gradient(circle at top left, {THEME['bg2']}, {THEME['bg']})", "color": THEME["text"], "fontFamily": "Inter, Segoe UI, Arial, sans-serif"})

app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      body { background:#0b1220; }
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


@app.callback(Output("selected-title", "children"), Input("stock-dropdown", "value"))
def update_selected_title(symbol):
    if not symbol:
        return ""
    return f"Selected: {symbol} - {get_company_name(symbol)}"


@app.callback(Output("indicator-store", "data"), Input("chart-indicator-checklist", "value"), prevent_initial_call=True)
def sync_indicator_store(values):
    return values or DEFAULT_INDICATORS


@app.callback(
    Output("main-content", "children"),
    Output("symbol-store", "data"),
    Input("load-button", "n_clicks"),
    State("stock-dropdown", "value"),
    State("indicator-store", "data"),
    prevent_initial_call=False,
)
def render_dashboard(n_clicks, symbol, indicators):
    symbol = symbol or DEFAULT_SYMBOL
    indicators = indicators or DEFAULT_INDICATORS
    try:
        hist = load_local_history(symbol)
        if hist.empty or len(hist) < 120 or n_clicks:
            hist = build_symbol_history(symbol)
        if hist.empty:
            return build_default_table_message("No data available for this symbol right now."), symbol
        df = add_indicators(hist)
        signal, score, confidence, reasons = signal_from_df(df)
        company = get_company_name(symbol)
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        price_change = float(last["Close"] - prev["Close"])
        pct_change = (price_change / prev["Close"] * 100) if prev["Close"] else 0
        period = df.tail(252)
        support, resistance = support_resistance(df)
        atr_val = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0
        chart = make_chart(df, symbol, indicators)

        top_cards = html.Div([
            stat_card("Last Close", fmt_currency_inr(last["Close"]), f"{price_change:.2f} ({pct_change:.2f}%)", THEME["good"] if price_change >= 0 else THEME["danger"]),
            stat_card("Day Range", f"₹ {fmt_num(last['Low'])} - ₹ {fmt_num(last['High'])}", "Latest session low / high"),
            stat_card("52 Week Range", f"₹ {fmt_num(period['Low'].min())} - ₹ {fmt_num(period['High'].max())}", "Rolling 52-week low / high"),
            stat_card("Volume", fmt_int(last["Volume"]), "Latest traded quantity"),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(220px,1fr))", "gap": "16px"})

        signal_card = style_card([
            html.Div("Signal Engine", style={"fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "color": THEME["muted"]}),
            html.Div(signal, style={"fontSize": "32px", "fontWeight": "900", "marginTop": "10px", "color": SIGNAL_COLORS.get(signal, THEME['text'])}),
            html.Div(f"Score: {score} | Confidence: {confidence}%", style={"marginTop": "8px", "color": THEME["text"]}),
            html.Ul([html.Li(r) for r in reasons], style={"marginTop": "12px", "paddingLeft": "18px", "color": THEME["muted"], "lineHeight": "1.8"}),
        ])

        plan_card = style_card([
            html.Div("Trade Reference", style={"fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "color": THEME["muted"]}),
            kv_row("Support", fmt_currency_inr(support)),
            kv_row("Resistance", fmt_currency_inr(resistance)),
            kv_row("Buy Entry", fmt_currency_inr(float(last["Close"]) + 0.15 * atr_val)),
            kv_row("Buy Stop Loss", fmt_currency_inr(float(last["Close"]) - 1.0 * atr_val)),
            kv_row("Buy Target 1", fmt_currency_inr(float(last["Close"]) + 1.2 * atr_val)),
            kv_row("Sell Entry", fmt_currency_inr(float(last["Close"]) - 0.15 * atr_val)),
        ])

        tech_card = style_card([
            html.Div("Technical Snapshot", style={"fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "color": THEME["muted"]}),
            kv_row("RSI 14", fmt_num(last["RSI14"])),
            kv_row("MACD", fmt_num(last["MACD"])),
            kv_row("MACD Signal", fmt_num(last["MACD_SIGNAL"])),
            kv_row("ADX 14", fmt_num(last["ADX14"])),
            kv_row("ATR 14", fmt_num(last["ATR14"])),
            kv_row("SMA 20", fmt_num(last["SMA20"])),
            kv_row("SMA 50", fmt_num(last["SMA50"])),
            kv_row("SMA 200", fmt_num(last["SMA200"])),
        ])

        data_view = df.tail(60)[["Date", "Open", "High", "Low", "Close", "Volume", "RSI14", "MACD", "ADX14"]].copy()
        data_view["Date"] = data_view["Date"].dt.strftime("%Y-%m-%d")
        for c in ["Open", "High", "Low", "Close", "RSI14", "MACD", "ADX14"]:
            data_view[c] = data_view[c].round(2)

        controls = style_card([
            html.Div("Chart Controls", style={"fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "color": THEME["muted"], "marginBottom": "12px"}),
            dcc.Checklist(
                id="chart-indicator-checklist",
                options=[
                    {"label": "SMA20", "value": "SMA20"},
                    {"label": "SMA50", "value": "SMA50"},
                    {"label": "SMA200", "value": "SMA200"},
                    {"label": "EMA20", "value": "EMA20"},
                    {"label": "EMA50", "value": "EMA50"},
                    {"label": "Bollinger", "value": "Bollinger"},
                    {"label": "Supertrend", "value": "Supertrend"},
                    {"label": "MACD", "value": "MACD"},
                    {"label": "RSI", "value": "RSI"},
                ],
                value=indicators,
                inline=True,
                inputStyle={"marginRight": "6px", "marginLeft": "14px"},
                labelStyle={"display": "inline-flex", "alignItems": "center", "marginBottom": "10px", "color": THEME["text"]},
            )
        ])

        return html.Div([
            top_cards,
            html.Div(style={"height": "16px"}),
            style_card([
                html.Div(f"{company} ({symbol})", style={"fontSize": "24px", "fontWeight": "800"}),
                html.Div("Yahoo-free build using NSE bhavcopy files. Loads faster because data is fetched only for the selected stock.", style={"marginTop": "8px", "color": THEME["muted"], "lineHeight": "1.7"}),
            ]),
            html.Div(style={"height": "16px"}),
            html.Div([signal_card, plan_card, tech_card], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(280px,1fr))", "gap": "16px"}),
            html.Div(style={"height": "16px"}),
            controls,
            html.Div(style={"height": "14px"}),
            dcc.Graph(figure=chart, config={"displaylogo": False, "responsive": True}),
            html.Div(style={"height": "16px"}),
            style_card([
                html.Div("Recent Data", style={"fontSize": "22px", "fontWeight": "800", "marginBottom": "12px"}),
                dash_table.DataTable(
                    data=data_view.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in data_view.columns],
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": THEME["bg2"], "color": THEME["text"], "fontWeight": "700", "border": f"1px solid {THEME['border']}"},
                    style_cell={"backgroundColor": THEME["panel"], "color": THEME["text"], "border": f"1px solid {THEME['border']}", "padding": "10px", "textAlign": "left"},
                    page_size=12,
                )
            ]),
        ]), symbol
    except Exception as e:
        return style_card([
            html.Div("Unable to load stock data", style={"fontSize": "24px", "fontWeight": "800", "color": THEME["danger"]}),
            html.Div(str(e), style={"marginTop": "10px", "color": THEME["text"], "lineHeight": "1.7"}),
            html.Div("Try another stock or click Load / Refresh again after a few seconds.", style={"marginTop": "10px", "color": THEME["muted"]}),
        ]), symbol


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)

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

def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

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
    return [
        {"label": f"{row['SYMBOL']} - {row['NAME OF COMPANY']}", "value": row["SYMBOL"]}
        for _, row in df.iterrows()
    ]

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

    if pd.notna(last["VOL_MA20"]) and last["Volume"] > last["VOL_MA20"]:
        score += 1
        reasons.append("Volume above 20-day average")

    if last["ST_DIR"] == "Bullish":
        score += 1
        reasons.append("Supertrend bullish")
    elif last["ST_DIR"] == "Bearish":
        score -= 1
        reasons.append("Supertrend bearish")

    if pd.notna(last["ADX14"]):
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
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = min(95, max(35, 50 + abs(score) * 7))
    return signal, score, confidence, reasons

def order_plan(df):
    last = df.iloc[-1]
    support, resistance = nearest_levels(df)
    atr_val = float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0

    entry_buy = round(float(last["Close"]) + 0.15 * atr_val, 2)
    sl_buy = round(float(last["Close"]) - 1.0 * atr_val, 2)
    t1_buy = round(float(last["Close"]) + 1.2 * atr_val, 2)
    t2_buy = round(float(last["Close"]) + 2.2 * atr_val, 2)

    entry_sell = round(float(last["Close"]) - 0.15 * atr_val, 2)
    sl_sell = round(float(last["Close"]) + 1.0 * atr_val, 2)
    t1_sell = round(float(last["Close"]) - 1.2 * atr_val, 2)
    t2_sell = round(float(last["Close"]) - 2.2 * atr_val, 2)

    return {
        "support": support,
        "resistance": resistance,
        "buy_entry": entry_buy,
        "buy_sl": sl_buy,
        "buy_t1": t1_buy,
        "buy_t2": t2_buy,
        "sell_entry": entry_sell,
        "sell_sl": sl_sell,
        "sell_t1": t1_sell,
        "sell_t2": t2_sell,
    }

def build_chart(df, symbol):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.60, 0.20, 0.20]
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
        ),
        row=1, col=1
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", line=dict(color="#00E5FF", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(color="#F9A826", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200", line=dict(color="#FF4D6D", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_UPPER"], name="BB Upper", line=dict(color="#888", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_LOWER"],
            name="BB Lower",
            line=dict(color="#888", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(120,120,120,0.08)"
        ),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["SUPERTREND"], name="Supertrend", line=dict(color="#52D273", width=1.3)), row=1, col=1)

    colors = np.where(df["MACD_HIST"] >= 0, "#52D273", "#FF5C7A")
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="MACD Hist", marker_color=colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#00E5FF", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", line=dict(color="#F9A826", width=1.2)), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14", line=dict(color="#B388FF", width=1.7)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#FF5C7A", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#52D273", row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#081018",
        plot_bgcolor="#081018",
        font=dict(color="#E8F1F8"),
        xaxis_rangeslider_visible=False,
        height=850,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        title=f"{symbol} Technical Dashboard"
    )
    return fig

def fetch_stock_history(symbol):
    ticker = yf.Ticker(f"{symbol}.NS")
    df = ticker.history(period="1y", interval="1d", auto_adjust=False)
    if df.empty:
        raise ValueError(f"No market data found for {symbol}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    return df

def make_tv_symbol(symbol):
    return f"NSE:{symbol}"

def screener_url(symbol):
    return f"https://www.screener.in/company/{symbol}/"

def price_snapshot(df):
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
    }

def card_style():
    return {
        "background": "linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03))",
        "border": "1px solid rgba(255,255,255,0.08)",
        "borderRadius": "16px",
        "padding": "16px",
        "minHeight": "105px",
        "boxShadow": "0 10px 24px rgba(0,0,0,0.18)",
    }

def small_title(text):
    return html.Div(text, style={"color": "#8EA7BA", "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "0.8px"})

def big_value(text, color="#EAF2F8", size="24px"):
    return html.Div(text, style={"fontSize": size, "fontWeight": "800", "color": color, "marginTop": "10px"})

def tab_style():
    return {
        "backgroundColor": "#0d1620",
        "color": "#A9BDCC",
        "border": "1px solid #203040",
        "padding": "12px",
        "fontWeight": "700"
    }

def tab_selected_style():
    return {
        "backgroundColor": "#122232",
        "color": "#EAF2F8",
        "border": "1px solid #14B8A6",
        "padding": "12px",
        "fontWeight": "800"
    }

def info_panel(title, children):
    return html.Div(
        style={
            "background": "rgba(255,255,255,0.04)",
            "border": "1px solid rgba(255,255,255,0.08)",
            "borderRadius": "18px",
            "padding": "16px",
        },
        children=[
            html.H4(title, style={"marginTop": "0", "marginBottom": "12px"}),
            children
        ]
    )

def metric_row(label, value):
    val = "-" if pd.isna(value) else round(float(value), 2)
    return html.Div(
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "padding": "8px 0",
            "borderBottom": "1px solid rgba(255,255,255,0.06)"
        },
        children=[
            html.Span(label, style={"color": "#9EB3C4"}),
            html.Strong(str(val), style={"color": "#EAF2F8"})
        ]
    )

def trade_box(label, value, color):
    return html.Div(
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "padding": "10px 12px",
            "marginBottom": "8px",
            "background": "rgba(255,255,255,0.03)",
            "border": "1px solid rgba(255,255,255,0.06)",
            "borderRadius": "12px"
        },
        children=[
            html.Span(label, style={"color": "#A8BBC8"}),
            html.Strong(f"₹ {value}", style={"color": color, "fontSize": "16px"})
        ]
    )

def section_subtitle(text):
    return html.Div(text, style={"marginBottom": "10px", "fontWeight": "800", "fontSize": "16px", "color": "#D8E6EF"})

def link_btn():
    return {
        "padding": "10px 14px",
        "borderRadius": "12px",
        "textDecoration": "none",
        "background": "#102130",
        "border": "1px solid #1D3B47",
        "color": "#DFF7FF",
        "fontWeight": "700",
        "display": "inline-block"
    }

try:
    stocks_df = load_nse_stocks()
    dropdown_options = make_dropdown_options(stocks_df)
    universe_msg = f"NSE universe loaded: {len(stocks_df)} stocks"
except Exception as e:
    stocks_df = pd.DataFrame(columns=["SYMBOL", "NAME OF COMPANY", "SERIES", "DATE OF LISTING", "ISIN NUMBER", "FACE VALUE"])
    dropdown_options = []
    universe_msg = f"NSE load failed: {e}"

app = Dash(__name__)
server = app.server
app.title = "Blast NSE Lab"

app.layout = html.Div(
    style={
        "background": "linear-gradient(135deg, #061018 0%, #0b1723 40%, #101826 100%)",
        "minHeight": "100vh",
        "color": "#EAF2F8",
        "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
        "padding": "18px",
    },
    children=[
        dcc.Store(id="stocks-store", data=stocks_df.to_dict("records")),
        dcc.Store(id="selected-symbol-store", data=DEFAULT_SYMBOL),
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "gap": "16px",
                "marginBottom": "18px",
                "background": "rgba(255,255,255,0.04)",
                "border": "1px solid rgba(255,255,255,0.08)",
                "borderRadius": "18px",
                "padding": "18px 20px",
                "backdropFilter": "blur(8px)",
            },
            children=[
                html.Div([
                    html.Div("BLAST NSE LAB", style={"fontSize": "26px", "fontWeight": "800", "letterSpacing": "1px"}),
                    html.Div("Compact technical scanner • chart room • trade map • news deck", style={"color": "#9FB4C7", "fontSize": "13px", "marginTop": "4px"}),
                ]),
                html.Div(universe_msg, style={"color": "#8BE9A8", "fontSize": "13px"})
            ]
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "2.2fr 1fr", "gap": "16px", "marginBottom": "16px"},
            children=[
                html.Div(
                    style={
                        "background": "rgba(255,255,255,0.04)",
                        "border": "1px solid rgba(255,255,255,0.08)",
                        "borderRadius": "18px",
                        "padding": "16px",
                    },
                    children=[
                        html.Label("NSE Stock Search", style={"fontWeight": "700", "marginBottom": "8px", "display": "block"}),
                        dcc.Dropdown(
                            id="stock-dropdown",
                            options=dropdown_options,
                            value=DEFAULT_SYMBOL,
                            multi=False,
                            placeholder="Select NSE stock",
                            style={"color": "#111"}
                        ),
                    ]
                ),
                html.Div(
                    style={
                        "background": "rgba(255,255,255,0.04)",
                        "border": "1px solid rgba(255,255,255,0.08)",
                        "borderRadius": "18px",
                        "padding": "16px",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "space-between",
                        "gap": "10px",
                    },
                    children=[
                        html.Button(
                            "Refresh Universe",
                            id="refresh-button",
                            n_clicks=0,
                            style={
                                "background": "#14B8A6",
                                "color": "#041016",
                                "border": "none",
                                "padding": "12px 16px",
                                "borderRadius": "12px",
                                "fontWeight": "700",
                                "cursor": "pointer",
                                "width": "100%"
                            }
                        )
                    ]
                ),
            ]
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(6, 1fr)", "gap": "12px", "marginBottom": "16px"},
            children=[
                html.Div(id="card-price", style=card_style()),
                html.Div(id="card-change", style=card_style()),
                html.Div(id="card-signal", style=card_style()),
                html.Div(id="card-rsi", style=card_style()),
                html.Div(id="card-trend", style=card_style()),
                html.Div(id="card-levels", style=card_style()),
            ]
        ),
        dcc.Tabs(
            id="main-tabs",
            value="tab-overview",
            colors={"border": "#203040", "primary": "#14B8A6", "background": "#0d1620"},
            children=[
                dcc.Tab(label="Overview", value="tab-overview", style=tab_style(), selected_style=tab_selected_style()),
                dcc.Tab(label="TradingView", value="tab-tv", style=tab_style(), selected_style=tab_selected_style()),
                dcc.Tab(label="Orders & Signal", value="tab-orders", style=tab_style(), selected_style=tab_selected_style()),
                dcc.Tab(label="News & Links", value="tab-news", style=tab_style(), selected_style=tab_selected_style()),
                dcc.Tab(label="Universe", value="tab-universe", style=tab_style(), selected_style=tab_selected_style()),
            ]
        ),
        html.Div(id="tab-content", style={"marginTop": "16px"})
    ]
)

@app.callback(
    Output("stocks-store", "data"),
    Output("stock-dropdown", "options"),
    Input("refresh-button", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_universe(n):
    df = load_nse_stocks()
    return df.to_dict("records"), make_dropdown_options(df)

@app.callback(
    Output("selected-symbol-store", "data"),
    Input("stock-dropdown", "value"),
)
def sync_selected_symbol(symbol):
    return symbol or DEFAULT_SYMBOL

@app.callback(
    Output("card-price", "children"),
    Output("card-change", "children"),
    Output("card-signal", "children"),
    Output("card-rsi", "children"),
    Output("card-trend", "children"),
    Output("card-levels", "children"),
    Input("selected-symbol-store", "data"),
)
def update_top_cards(symbol):
    try:
        df = add_indicators(fetch_stock_history(symbol))
        snap = price_snapshot(df)
        signal, score, confidence, reasons = generate_signal(df)
        plan = order_plan(df)
        st_dir = df["ST_DIR"].iloc[-1]
        rsi_val = round(float(df["RSI14"].iloc[-1]), 2)

        change_color = "#52D273" if snap["change"] >= 0 else "#FF5C7A"
        signal_color = "#52D273" if "BUY" in signal else "#FF5C7A" if "SELL" in signal else "#F9A826"
        trend_color = "#52D273" if st_dir == "Bullish" else "#FF5C7A"

        return (
            [small_title("Last Price"), big_value(f"₹ {snap['close']}"), html.Div(f"H {snap['high']} • L {snap['low']}", style={"color": "#94A9B8", "marginTop": "6px"})],
            [small_title("Day Change"), big_value(f"{snap['change']} ({snap['change_pct']}%)", change_color), html.Div(f"Vol {snap['volume']:,}", style={"color": "#94A9B8", "marginTop": "6px"})],
            [small_title("AI Signal"), big_value(signal, signal_color), html.Div(f"Score {score} • Confidence {confidence}%", style={"color": "#94A9B8", "marginTop": "6px"})],
            [small_title("RSI 14"), big_value(str(rsi_val), "#B388FF"), html.Div("30 oversold • 70 overbought", style={"color": "#94A9B8", "marginTop": "6px"})],
            [small_title("Trend Engine"), big_value(st_dir, trend_color), html.Div(f"ADX {round(float(df['ADX14'].iloc[-1]), 2)}", style={"color": "#94A9B8", "marginTop": "6px"})],
            [small_title("S/R Levels"), big_value(f"{plan['support']} / {plan['resistance']}", "#00E5FF", size="20px"), html.Div("Support / Resistance", style={"color": "#94A9B8", "marginTop": "6px"})],
        )
    except Exception as e:
        err = [small_title("Status"), big_value("Data Error", "#FF5C7A"), html.Div(str(e), style={"color": "#94A9B8", "marginTop": "6px", "fontSize": "12px"})]
        return err, err, err, err, err, err

@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    Input("selected-symbol-store", "data"),
    Input("stocks-store", "data"),
)
def render_tab(tab, symbol, stocks_data):
    try:
        df = add_indicators(fetch_stock_history(symbol))
        signal, score, confidence, reasons = generate_signal(df)
        plan = order_plan(df)
        chart = build_chart(df.tail(220), symbol)
        tv_symbol = make_tv_symbol(symbol)
        last = df.iloc[-1]

        if tab == "tab-overview":
            return html.Div(
                style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px"},
                children=[
                    info_panel("Technical Chart", dcc.Graph(figure=chart, config={"displayModeBar": True})),
                    html.Div(
                        style={"display": "grid", "gap": "16px"},
                        children=[
                            info_panel(
                                "Indicator Snapshot",
                                html.Div([
                                    metric_row("SMA20", last["SMA20"]),
                                    metric_row("SMA50", last["SMA50"]),
                                    metric_row("SMA200", last["SMA200"]),
                                    metric_row("RSI14", last["RSI14"]),
                                    metric_row("MACD", last["MACD"]),
                                    metric_row("MACD Signal", last["MACD_SIGNAL"]),
                                    metric_row("ATR14", last["ATR14"]),
                                    metric_row("ADX14", last["ADX14"]),
                                    metric_row("Supertrend", last["SUPERTREND"]),
                                ])
                            ),
                            info_panel(
                                "Model Readout",
                                html.Div([
                                    html.Div(
                                        f"Signal: {signal}",
                                        style={
                                            "fontSize": "22px",
                                            "fontWeight": "800",
                                            "color": "#14E0A1" if "BUY" in signal else "#FF5C7A" if "SELL" in signal else "#F9A826"
                                        }
                                    ),
                                    html.Div(f"Confidence: {confidence}%", style={"marginTop": "6px", "color": "#A6BDCB"}),
                                    html.Ul([html.Li(r) for r in reasons], style={"marginTop": "10px", "color": "#D7E4ED", "paddingLeft": "18px"}),
                                    html.Div("Model is rule-based technical scoring, not guaranteed financial advice.", style={"marginTop": "12px", "fontSize": "12px", "color": "#7E97AA"})
                                ])
                            ),
                        ]
                    )
                ]
            )

        if tab == "tab-tv":
            tv_html = f"""
            <html>
            <head>
            <meta charset="utf-8"/>
            <style>
            body{{margin:0;background:#081018;font-family:Arial;color:#fff;}}
            .wrap{{padding:8px;}}
            .box{{border-radius:16px;overflow:hidden;border:1px solid rgba(255,255,255,.08);}}
            </style>
            </head>
            <body>
                <div class="wrap">
                    <div class="box">
                        <iframe
                            src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_chart&symbol={tv_symbol}&interval=D&hidesidetoolbar=0&symboledit=1&saveimage=1&toolbarbg=0b1723&theme=dark&style=1&timezone=Asia%2FKolkata"
                            style="width:100%;height:760px;border:0;"
                            allowtransparency="true"
                            scrolling="no">
                        </iframe>
                    </div>
                </div>
            </body>
            </html>
            """
            return html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "16px"},
                children=[
                    info_panel(
                        "TradingView Advanced View",
                        html.Iframe(srcDoc=tv_html, style={"width": "100%", "height": "790px", "border": "0", "borderRadius": "16px"})
                    ),
                    info_panel(
                        "Platform Links",
                        html.Div([
                            html.A("Open in Screener", href=screener_url(symbol), target="_blank", rel="noopener noreferrer", style=link_btn()),
                            html.A("Open in TradingView India", href=f"https://in.tradingview.com/symbols/{tv_symbol.replace(':', '-')}/", target="_blank", rel="noopener noreferrer", style=link_btn()),
                        ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"})
                    )
                ]
            )

        if tab == "tab-orders":
            return html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                children=[
                    info_panel(
                        "Auto Trade Plan",
                        html.Div([
                            section_subtitle("Long Setup"),
                            trade_box("Buy Entry", plan["buy_entry"], "#14E0A1"),
                            trade_box("Stop Loss", plan["buy_sl"], "#FF5C7A"),
                            trade_box("Target 1", plan["buy_t1"], "#00E5FF"),
                            trade_box("Target 2", plan["buy_t2"], "#00E5FF"),
                            html.Hr(style={"borderColor": "rgba(255,255,255,0.08)", "margin": "16px 0"}),
                            section_subtitle("Short Setup"),
                            trade_box("Sell Entry", plan["sell_entry"], "#FFB020"),
                            trade_box("Stop Loss", plan["sell_sl"], "#FF5C7A"),
                            trade_box("Target 1", plan["sell_t1"], "#00E5FF"),
                            trade_box("Target 2", plan["sell_t2"], "#00E5FF"),
                        ])
                    ),
                    info_panel(
                        "Order Analysis",
                        html.Div([
                            html.Div(f"Signal: {signal}", style={"fontSize": "22px", "fontWeight": "800"}),
                            html.Div(f"Support: {plan['support']} | Resistance: {plan['resistance']}", style={"marginTop": "8px", "color": "#A8BBC8"}),
                            html.Div(f"ATR-based levels adapt to volatility for {symbol}.", style={"marginTop": "8px", "color": "#A8BBC8"}),
                            html.Ul([html.Li(r) for r in reasons], style={"marginTop": "10px", "paddingLeft": "18px"}),
                            html.Div(
                                "Use confirmation from price action, volume, and broader index context before live execution.",
                                style={"marginTop": "12px", "fontSize": "12px", "color": "#7E97AA"}
                            )
                        ])
                    )
                ]
            )

        if tab == "tab-news":
            query_links = [
                html.A(name, href=url, target="_blank", rel="noopener noreferrer", style=link_btn())
                for name, url in NEWS_FEEDS
            ]
            symbol_news = [
                html.A(f"{symbol} on Google News", href=f"https://news.google.com/search?q={symbol}%20NSE%20stock", target="_blank", rel="noopener noreferrer", style=link_btn()),
                html.A(f"{symbol} on Economic Times", href=f"https://economictimes.indiatimes.com/topic/{symbol}", target="_blank", rel="noopener noreferrer", style=link_btn()),
                html.A(f"{symbol} on Moneycontrol", href=f"https://www.moneycontrol.com/stocks/cptmarket/compsearchnew.php?search_data={symbol}", target="_blank", rel="noopener noreferrer", style=link_btn()),
            ]
            return html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                children=[
                    info_panel("Market News Deck", html.Div(query_links, style={"display": "flex", "gap": "10px", "flexWrap": "wrap"})),
                    info_panel("Symbol News Search", html.Div(symbol_news, style={"display": "flex", "gap": "10px", "flexWrap": "wrap"})),
                ]
            )

        if tab == "tab-universe":
            dfu = pd.DataFrame(stocks_data)
            if dfu.empty:
                return info_panel("NSE Stock Universe", html.Div("No data loaded.", style={"color": "#FF7B9C"}))

            view = dfu[["SYMBOL", "NAME OF COMPANY", "SERIES", "DATE OF LISTING", "ISIN NUMBER", "FACE VALUE"]].copy().head(1000)
            view.columns = ["Symbol", "Company", "Series", "Listing Date", "ISIN", "Face Value"]

            return info_panel(
                "NSE Stock Universe",
                dash_table.DataTable(
                    data=view.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in view.columns],
                    page_size=20,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "backgroundColor": "#0B1723",
                        "color": "#EAF2F8",
                        "border": "1px solid #1D2B36",
                        "textAlign": "left",
                        "padding": "8px",
                        "fontSize": "13px",
                        "whiteSpace": "normal",
                        "height": "auto",
                    },
                    style_header={
                        "backgroundColor": "#102130",
                        "color": "#8BE9A8",
                        "fontWeight": "700",
                        "border": "1px solid #1D2B36",
                    },
                )
            )

        return html.Div("No tab selected.")

    except Exception as e:
        return info_panel("Error", html.Div(str(e), style={"color": "#FF7B9C"}))

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
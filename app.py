import time
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from dash import Dash, Input, Output, State, dcc, html
from flask_caching import Cache
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


app = Dash(__name__)
server = app.server
app.title = "Blast NSE Lab"

cache = Cache(app.server, config={
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 600
})


@cache.memoize(timeout=600)
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
                time.sleep(5 * (attempt + 1))
            else:
                break

    raise ValueError(f"Yahoo Finance temporarily unavailable for {symbol}: {last_error}")


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
        "display": "inline-block",
        "marginRight": "10px",
        "marginBottom": "10px"
    }


def error_box(message):
    return html.Div(
        [
            html.Div("Data Error", style={
                "fontSize": "18px",
                "fontWeight": "800",
                "marginBottom": "8px",
                "color": "#FFD6D6"
            }),
            html.Div(message, style={
                "fontSize": "14px",
                "lineHeight": "1.6",
                "color": "#FFEAEA"
            })
        ],
        style={
            "background": "rgba(255, 77, 109, 0.14)",
            "border": "1px solid rgba(255, 99, 132, 0.45)",
            "borderRadius": "14px",
            "padding": "14px 16px",
            "marginBottom": "16px"
        }
    )


try:
    stocks_df = load_nse_stocks()
    dropdown_options = make_dropdown_options(stocks_df)
    universe_msg = f"NSE universe loaded: {len(stocks_df)} stocks"
except Exception as e:
    stocks_df = pd.DataFrame(columns=["SYMBOL", "NAME OF COMPANY", "SERIES", "DATE OF LISTING", "ISIN NUMBER", "FACE VALUE"])
    dropdown_options = []
    universe_msg = f"NSE load failed: {e}"


initial_error = None
initial_symbol = DEFAULT_SYMBOL
initial_fig = go.Figure()
initial_signal = html.Div()
initial_summary = html.Div()
initial_trade = html.Div()
initial_metrics = html.Div()

try:
    _df0 = fetch_stock_history(DEFAULT_SYMBOL)
    _df0 = add_indicators(_df0)
    _signal0, _score0, _confidence0, _reasons0 = generate_signal(_df0)
    _plan0 = order_plan(_df0)
    _snap0 = price_snapshot(_df0)
    initial_fig = build_chart(_df0, DEFAULT_SYMBOL)

    signal_color0 = {
        "STRONG BUY": "#22C55E",
        "BUY": "#4ADE80",
        "HOLD": "#FACC15",
        "SELL": "#FB7185",
        "STRONG SELL": "#F43F5E"
    }.get(_signal0, "#EAF2F8")

    initial_signal = html.Div([
        small_title("Signal"),
        big_value(_signal0, color=signal_color0, size="30px"),
        html.Div(f"Score: {_score0} | Confidence: {_confidence0}%", style={
            "marginTop": "8px",
            "color": "#A8BBC8"
        }),
        html.Ul([html.Li(r) for r in _reasons0], style={"marginTop": "12px", "paddingLeft": "18px"})
    ], style=card_style())

    initial_summary = html.Div([
        html.Div([small_title("Last Close"), big_value(f"₹ {_snap0['close']}")]),
        html.Div([small_title("Day Change"), big_value(
            f"{_snap0['change']} ({_snap0['change_pct']}%)",
            "#52D273" if _snap0["change"] >= 0 else "#FF5C7A"
        )], style={"marginTop": "12px"}),
        html.Div([small_title("Day Range"), html.Div(
            f"₹ {_snap0['low']} - ₹ {_snap0['high']}",
            style={"marginTop": "10px", "fontWeight": "700"}
        )], style={"marginTop": "12px"}),
        html.Div([small_title("Volume"), html.Div(
            f"{_snap0['volume']:,}",
            style={"marginTop": "10px", "fontWeight": "700"}
        )], style={"marginTop": "12px"})
    ], style=card_style())

    initial_trade = html.Div([
        section_subtitle("Buy Plan"),
        trade_box("Entry", _plan0["buy_entry"], "#52D273"),
        trade_box("Stop Loss", _plan0["buy_sl"], "#FFB4B4"),
        trade_box("Target 1", _plan0["buy_t1"], "#7EE787"),
        trade_box("Target 2", _plan0["buy_t2"], "#A7F3D0"),
        html.Br(),
        section_subtitle("Sell Plan"),
        trade_box("Entry", _plan0["sell_entry"], "#FF8A8A"),
        trade_box("Stop Loss", _plan0["sell_sl"], "#FFD1D1"),
        trade_box("Target 1", _plan0["sell_t1"], "#FCA5A5"),
        trade_box("Target 2", _plan0["sell_t2"], "#FECACA")
    ], style=card_style())

    latest = _df0.iloc[-1]
    initial_metrics = info_panel("Indicator Snapshot", html.Div([
        metric_row("RSI 14", latest["RSI14"]),
        metric_row("MACD", latest["MACD"]),
        metric_row("Signal", latest["MACD_SIGNAL"]),
        metric_row("ADX 14", latest["ADX14"]),
        metric_row("ATR 14", latest["ATR14"]),
        metric_row("SMA 20", latest["SMA20"]),
        metric_row("SMA 50", latest["SMA50"]),
        metric_row("SMA 200", latest["SMA200"]),
        metric_row("Supertrend", latest["SUPERTREND"]),
    ]))

except Exception as e:
    initial_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#081018",
        plot_bgcolor="#081018",
        font=dict(color="#E8F1F8"),
        height=850,
        title=f"{DEFAULT_SYMBOL} - Data temporarily unavailable"
    )
    msg = str(e)
    if "too many requests" in msg.lower() or "rate limit" in msg.lower():
        initial_error = error_box("Yahoo Finance temporarily rate-limited this request. Please wait a few minutes and try again.")
    else:
        initial_error = error_box(f"Unable to load initial market data. {msg}")


app.layout = html.Div(
    style={
        "background": "linear-gradient(135deg, #061018 0%, #0b1723 40%, #101826 100%)",
        "minHeight": "100vh",
        "color": "#EAF2F8",
        "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
        "padding": "18px",
    },
    children=[
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "gap": "16px",
                "flexWrap": "wrap",
                "marginBottom": "18px"
            },
            children=[
                html.Div([
                    html.H1("Blast NSE Lab", style={"margin": "0", "fontSize": "34px"}),
                    html.Div(universe_msg, style={"color": "#90A9BA", "marginTop": "6px"})
                ]),
                html.Div([
                    dcc.Dropdown(
                        id="stock-dropdown",
                        options=dropdown_options,
                        value=DEFAULT_SYMBOL if dropdown_options else None,
                        placeholder="Select NSE stock",
                        searchable=True,
                        clearable=False,
                        style={
                            "width": "380px",
                            "color": "#111",
                            "borderRadius": "10px"
                        }
                    )
                ])
            ]
        ),

        html.Div(id="error-message", children=initial_error),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr 1fr",
                "gap": "16px",
                "marginBottom": "18px"
            },
            children=[
                html.Div(id="signal-box", children=initial_signal),
                html.Div(id="summary-box", children=initial_summary),
                html.Div(id="trade-plan-box", children=initial_trade),
            ]
        ),

        html.Div(
            style={"marginBottom": "18px"},
            children=[
                dcc.Graph(id="main-chart", figure=initial_fig, config={"displayModeBar": True})
            ]
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1.2fr 0.8fr",
                "gap": "16px",
                "marginBottom": "18px"
            },
            children=[
                html.Div(
                    style={
                        "background": "rgba(255,255,255,0.04)",
                        "border": "1px solid rgba(255,255,255,0.08)",
                        "borderRadius": "18px",
                        "padding": "16px"
                    },
                    children=[
                        html.H4("Quick Links", style={"marginTop": "0"}),
                        html.A("TradingView", id="tv-link", href=f"https://www.tradingview.com/symbols/{make_tv_symbol(DEFAULT_SYMBOL).replace(':', '-')}/", target="_blank", style=link_btn()),
                        html.A("Screener", id="screener-link", href=screener_url(DEFAULT_SYMBOL), target="_blank", style=link_btn()),
                        *[
                            html.A(label, href=url, target="_blank", style=link_btn())
                            for label, url in NEWS_FEEDS
                        ]
                    ]
                ),
                html.Div(id="metrics-box", children=initial_metrics)
            ]
        ),

        html.Div(
            style={
                "background": "rgba(255,255,255,0.04)",
                "border": "1px solid rgba(255,255,255,0.08)",
                "borderRadius": "18px",
                "padding": "16px"
            },
            children=[
                html.H4("Notes", style={"marginTop": "0"}),
                html.Ul([
                    html.Li("Signals are rule-based and meant for educational use only."),
                    html.Li("Data is cached for 10 minutes to reduce rate-limit issues."),
                    html.Li("If Yahoo Finance rate-limits requests, wait a few minutes and retry."),
                ], style={"paddingLeft": "18px", "color": "#B8CAD6"})
            ]
        )
    ]
)


@app.callback(
    Output("main-chart", "figure"),
    Output("signal-box", "children"),
    Output("summary-box", "children"),
    Output("trade-plan-box", "children"),
    Output("metrics-box", "children"),
    Output("error-message", "children"),
    Output("tv-link", "href"),
    Output("screener-link", "href"),
    Input("stock-dropdown", "value"),
    prevent_initial_call=False
)
def update_dashboard(symbol):
    if not symbol:
        symbol = DEFAULT_SYMBOL

    try:
        df = fetch_stock_history(symbol)
        df = add_indicators(df)

        signal, score, confidence, reasons = generate_signal(df)
        plan = order_plan(df)
        snap = price_snapshot(df)
        fig = build_chart(df, symbol)
        latest = df.iloc[-1]

        signal_color = {
            "STRONG BUY": "#22C55E",
            "BUY": "#4ADE80",
            "HOLD": "#FACC15",
            "SELL": "#FB7185",
            "STRONG SELL": "#F43F5E"
        }.get(signal, "#EAF2F8")

        signal_ui = html.Div([
            small_title("Signal"),
            big_value(signal, color=signal_color, size="30px"),
            html.Div(f"Score: {score} | Confidence: {confidence}%", style={
                "marginTop": "8px",
                "color": "#A8BBC8"
            }),
            html.Ul([html.Li(r) for r in reasons], style={"marginTop": "12px", "paddingLeft": "18px"})
        ], style=card_style())

        summary_ui = html.Div([
            html.Div([small_title("Last Close"), big_value(f"₹ {snap['close']}", "#EAF2F8")]),
            html.Div([small_title("Day Change"), big_value(
                f"{snap['change']} ({snap['change_pct']}%)",
                "#52D273" if snap["change"] >= 0 else "#FF5C7A"
            )], style={"marginTop": "12px"}),
            html.Div([small_title("Day Range"), html.Div(
                f"₹ {snap['low']} - ₹ {snap['high']}",
                style={"marginTop": "10px", "fontWeight": "700"}
            )], style={"marginTop": "12px"}),
            html.Div([small_title("Volume"), html.Div(
                f"{snap['volume']:,}",
                style={"marginTop": "10px", "fontWeight": "700"}
            )], style={"marginTop": "12px"})
        ], style=card_style())

        trade_ui = html.Div([
            section_subtitle("Buy Plan"),
            trade_box("Entry", plan["buy_entry"], "#52D273"),
            trade_box("Stop Loss", plan["buy_sl"], "#FFB4B4"),
            trade_box("Target 1", plan["buy_t1"], "#7EE787"),
            trade_box("Target 2", plan["buy_t2"], "#A7F3D0"),
            html.Br(),
            section_subtitle("Sell Plan"),
            trade_box("Entry", plan["sell_entry"], "#FF8A8A"),
            trade_box("Stop Loss", plan["sell_sl"], "#FFD1D1"),
            trade_box("Target 1", plan["sell_t1"], "#FCA5A5"),
            trade_box("Target 2", plan["sell_t2"], "#FECACA")
        ], style=card_style())

        metrics_ui = info_panel("Indicator Snapshot", html.Div([
            metric_row("RSI 14", latest["RSI14"]),
            metric_row("MACD", latest["MACD"]),
            metric_row("Signal", latest["MACD_SIGNAL"]),
            metric_row("ADX 14", latest["ADX14"]),
            metric_row("ATR 14", latest["ATR14"]),
            metric_row("SMA 20", latest["SMA20"]),
            metric_row("SMA 50", latest["SMA50"]),
            metric_row("SMA 200", latest["SMA200"]),
            metric_row("Supertrend", latest["SUPERTREND"]),
        ]))

        tv_href = f"https://www.tradingview.com/symbols/{make_tv_symbol(symbol).replace(':', '-')}/"
        screener_href = screener_url(symbol)

        return fig, signal_ui, summary_ui, trade_ui, metrics_ui, None, tv_href, screener_href

    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#081018",
            plot_bgcolor="#081018",
            font=dict(color="#E8F1F8"),
            height=850,
            title=f"{symbol} - Data temporarily unavailable"
        )

        msg = str(e)
        if "too many requests" in msg.lower() or "rate limit" in msg.lower():
            friendly = "Yahoo Finance temporarily rate-limited this request. Please wait a few minutes and try again."
        else:
            friendly = f"Unable to load market data for {symbol}. {msg}"

        tv_href = f"https://www.tradingview.com/symbols/{make_tv_symbol(symbol).replace(':', '-')}/"
        screener_href = screener_url(symbol)

        return (
            empty_fig,
            html.Div(),
            html.Div(),
            html.Div(),
            html.Div(),
            error_box(friendly),
            tv_href,
            screener_href
        )


if __name__ == "__main__":
    app.run(debug=True)

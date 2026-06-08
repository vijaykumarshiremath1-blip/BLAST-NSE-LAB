from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

app = Dash(__name__)
server = app.server
app.title = "Blast NSE Lab Pro"

THEME = {
    "bg": "#0b1220",
    "bg2": "#111827",
    "bg3": "#172033",
    "panel": "#121a2a",
    "panel2": "#19243a",
    "text": "#e5eefc",
    "muted": "#9fb0cf",
    "accent": "#4ade80",
    "warn": "#f59e0b",
    "danger": "#f87171",
    "border": "#2a3855",
}

STOCKS = [
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

OPTIONS = [{"label": f"{s} - {n}", "value": s} for s, n in STOCKS]
NAME_MAP = {s: n for s, n in STOCKS}
DEFAULT_SYMBOL = "RELIANCE"


def make_series(symbol: str):
    seed = sum(ord(c) for c in symbol)
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=120, freq="B")
    drift = rng.normal(0.18, 1.2, len(dates)).cumsum()
    base = 2000 + (seed % 700)
    close = np.maximum(base + drift * 8, 50)
    open_ = close + rng.normal(0, 10, len(dates))
    high = np.maximum(open_, close) + rng.uniform(5, 25, len(dates))
    low = np.minimum(open_, close) - rng.uniform(5, 25, len(dates))
    volume = rng.integers(100000, 9000000, len(dates))
    df = pd.DataFrame({"Date": dates, "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def compute_signal(df: pd.DataFrame):
    close = float(df["Close"].iloc[-1])
    sma20 = float(df["SMA20"].iloc[-1]) if pd.notna(df["SMA20"].iloc[-1]) else close
    sma50 = float(df["SMA50"].iloc[-1]) if pd.notna(df["SMA50"].iloc[-1]) else close
    rsi = float(df["RSI"].iloc[-1]) if pd.notna(df["RSI"].iloc[-1]) else 50.0
    score = 50
    if close > sma20:
        score += 10
    else:
        score -= 10
    if close > sma50:
        score += 12
    else:
        score -= 12
    if rsi < 35:
        score += 12
    elif rsi > 70:
        score -= 12
    score = max(0, min(100, score))
    if score >= 78:
        sig = "STRONG BUY"
    elif score >= 62:
        sig = "BUY"
    elif score >= 45:
        sig = "HOLD"
    elif score >= 30:
        sig = "SELL"
    else:
        sig = "STRONG SELL"
    confidence = int(min(95, max(35, abs(score - 50) * 2)))
    return sig, score, confidence, rsi, sma20, sma50


def card(title, body):
    return html.Div([
        html.Div(title, style={"fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "1px", "color": THEME["muted"], "marginBottom": "10px"}),
        body,
    ], style={"background": THEME["panel"], "border": f"1px solid {THEME['border']}", "borderRadius": "18px", "padding": "16px", "boxShadow": "0 10px 30px rgba(0,0,0,0.20)"})


def metric(label, value, color=None):
    return html.Div([
        html.Div(label, style={"fontSize": "12px", "color": THEME["muted"], "marginBottom": "8px"}),
        html.Div(value, style={"fontSize": "24px", "fontWeight": "800", "color": color or THEME["text"]}),
    ], style={"background": THEME["panel2"], "border": f"1px solid {THEME['border']}", "borderRadius": "16px", "padding": "14px"})


app.layout = html.Div([
    dcc.Store(id="symbol-store", data=DEFAULT_SYMBOL),
    html.Div([
        html.Div("BLAST NSE LAB PRO", style={"fontSize": "12px", "letterSpacing": "2px", "color": THEME["accent"], "fontWeight": "800"}),
        html.H1("Dark Theme Stock Dashboard", style={"margin": "8px 0 4px 0", "fontSize": "34px", "color": THEME["text"]}),
        html.Div("Safe full dashboard build with search, overview, chart, technicals, financial report, news and confidence sections.", style={"color": THEME["muted"], "maxWidth": "920px", "lineHeight": "1.7"}),
        html.Div(style={"height": "18px"}),
        html.Div([
            card("Stock Search", html.Div([
                dcc.Dropdown(id="stock-dropdown", options=OPTIONS, value=DEFAULT_SYMBOL, searchable=True, clearable=False, style={"color": "#111827"}),
                html.Div(id="stock-label", style={"marginTop": "12px", "fontWeight": "700", "color": THEME["text"]}),
            ])),
            card("Scanner Snapshot", html.Div(id="scanner-table")),
        ], style={"display": "grid", "gridTemplateColumns": "0.95fr 1.35fr", "gap": "16px"}),
        html.Div(style={"height": "18px"}),
        dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="Overview", value="overview"),
            dcc.Tab(label="Chart", value="chart"),
            dcc.Tab(label="Technicals", value="technicals"),
            dcc.Tab(label="Financial Report", value="financials"),
            dcc.Tab(label="News", value="news"),
        ]),
        html.Div(id="tab-content", style={"marginTop": "18px"}),
    ], style={"maxWidth": "1450px", "margin": "0 auto", "padding": "28px 18px 60px 18px"})
], style={"minHeight": "100vh", "background": f"radial-gradient(circle at top left, {THEME['bg3']}, {THEME['bg']})", "color": THEME["text"], "fontFamily": "Inter, Segoe UI, Arial, sans-serif"})


@app.callback(
    Output("symbol-store", "data"),
    Output("stock-label", "children"),
    Input("stock-dropdown", "value"),
)
def update_symbol(symbol):
    name = NAME_MAP.get(symbol, symbol)
    return symbol, f"Selected: {symbol} - {name}"


@app.callback(
    Output("scanner-table", "children"),
    Input("symbol-store", "data"),
)
def update_scanner(_):
    rows = []
    for sym, name in STOCKS:
        df = make_series(sym)
        sig, score, confidence, rsi, *_ = compute_signal(df)
        rows.append({"symbol": sym, "company": name, "signal": sig, "confidence": confidence, "rsi": round(rsi, 2)})
    table = dash_table.DataTable(
        data=rows,
        columns=[
            {"name": "Symbol", "id": "symbol"},
            {"name": "Company", "id": "company"},
            {"name": "Signal", "id": "signal"},
            {"name": "Confidence %", "id": "confidence"},
            {"name": "RSI", "id": "rsi"},
        ],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": THEME["bg3"], "color": THEME["text"], "fontWeight": "700", "border": f"1px solid {THEME['border']}"},
        style_cell={"backgroundColor": THEME["panel"], "color": THEME["text"], "border": f"1px solid {THEME['border']}", "padding": "10px", "textAlign": "left"},
        page_size=8,
    )
    return table


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("symbol-store", "data"),
)
def render_tab(tab, symbol):
    df = make_series(symbol)
    sig, score, confidence, rsi, sma20, sma50 = compute_signal(df)
    close = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2])
    chg = close - prev
    chg_pct = (chg / prev) * 100 if prev else 0
    support = round(df["Low"].tail(20).min(), 2)
    resistance = round(df["High"].tail(20).max(), 2)
    color = THEME["accent"] if sig in ["BUY", "STRONG BUY"] else THEME["danger"] if sig in ["SELL", "STRONG SELL"] else THEME["warn"]

    if tab == "overview":
        return html.Div([
            html.Div([
                metric("Signal", sig, color),
                metric("Score", str(score)),
                metric("Confidence", f"{confidence}%"),
                metric("Close", f"₹ {close:,.2f}"),
                metric("Change", f"{chg:+.2f} ({chg_pct:+.2f}%)", THEME["accent"] if chg >= 0 else THEME["danger"]),
                metric("RSI", f"{rsi:.2f}"),
            ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(180px,1fr))", "gap": "14px"}),
            html.Div(style={"height": "16px"}),
            html.Div([
                card("Applicability", html.Ul([
                    html.Li("Trend-following bias from moving-average structure."),
                    html.Li("Momentum check through RSI level."),
                    html.Li("Support and resistance estimated from recent range."),
                ], style={"paddingLeft": "18px", "lineHeight": "1.8", "color": THEME["text"]})),
                card("Levels", html.Div([
                    html.Div(f"Support: ₹ {support:,.2f}", style={"marginBottom": "10px"}),
                    html.Div(f"Resistance: ₹ {resistance:,.2f}", style={"marginBottom": "10px"}),
                    html.Div(f"SMA20: ₹ {sma20:,.2f}"),
                    html.Div(f"SMA50: ₹ {sma50:,.2f}")
                ], style={"lineHeight": "1.9", "color": THEME["text"]})),
            ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"})
        ])

    if tab == "chart":
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], mode="lines", name="SMA20", line=dict(color="#60a5fa", width=1.5)))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], mode="lines", name="SMA50", line=dict(color="#f59e0b", width=1.5)))
        fig.update_layout(template="plotly_dark", paper_bgcolor=THEME["panel"], plot_bgcolor=THEME["panel"], font_color=THEME["text"], height=560, margin=dict(l=20, r=20, t=30, b=20))
        return card("Chart", dcc.Graph(figure=fig, config={"displaylogo": False}))

    if tab == "technicals":
        return html.Div([
            html.Div([
                metric("RSI", f"{rsi:.2f}"),
                metric("SMA20", f"₹ {sma20:,.2f}"),
                metric("SMA50", f"₹ {sma50:,.2f}"),
                metric("Bias", "Bullish" if close > sma50 else "Bearish", THEME["accent"] if close > sma50 else THEME["danger"]),
            ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(200px,1fr))", "gap":

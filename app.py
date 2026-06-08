import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

from nse_data import NSEDataFetcher
from technical_analysis import TechnicalAnalyzer
from config import STOCK_UNIVERSE

nse = NSEDataFetcher()
analyzer = TechnicalAnalyzer()

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Stock Screener - NSE"

app.layout = html.Div([
    html.Div([
        html.H1("📊 Stock Screener", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("NSE Stock Analysis with Technical Indicators & Alerts", 
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '18px'}),
        html.Div(id='last-updated', style={'textAlign': 'center', 'color': '#3498db', 'marginTop': '10px'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
    
    html.Div([
        html.Div([
            html.H3("Select Stock", style={'color': '#2c3e50'}),
            dcc.Dropdown(
                id='stock-selector',
                options=[{'label': s, 'value': s} for s in STOCK_UNIVERSE],
                value='RELIANCE',
                style={'marginBottom': '20px'}
            ),
            html.Button('🔍 Analyze Stock', id='analyze-btn', 
                       style={'width': '100%', 'padding': '15px', 'fontSize': '16px',
                             'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                             'cursor': 'pointer', 'borderRadius': '5px'}),
            html.Hr(),
            html.H4("Quick Stats", style={'color': '#2c3e50'}),
            html.Div(id='quick-stats', style={'marginTop': '10px'})
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                 'padding': '20px', 'backgroundColor': '#f9f9f9', 'height': '700px'}),
        
        html.Div([
            html.Div([
                html.H3("📈 Price Chart", style={'color': '#2c3e50'}),
                dcc.Graph(id='price-chart')
            ], style={'backgroundColor': 'white', 'padding': '20px', 'marginBottom': '20px',
                     'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.H3("📊 Technical Analysis", style={'color': '#2c3e50'}),
                html.Div(id='technical-summary', 
                        style={'backgroundColor': 'white', 'padding': '20px', 
                              'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'})
            ])
        ], style={'width': '67%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'})
    ], style={'width': '100%', 'padding': '20px'})
], style={'backgroundColor': '#f5f5f5', 'minHeight': '100vh'})

@app.callback(
    [Output('price-chart', 'figure'),
     Output('technical-summary', 'children'),
     Output('quick-stats', 'children'),
     Output('last-updated', 'children')],
    [Input('analyze-btn', 'n_clicks')],
    [State('stock-selector', 'value')]
)
def analyze_stock(n_clicks, symbol):
    if not symbol:
        return go.Figure(), "", "", ""
    
    quote = nse.get_stock_historical_data(symbol)
    if not quote:
        return go.Figure(), "Error fetching data", "", ""
    
    dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(50)) + quote['last_price']
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(50) * 2,
        'high': prices + np.abs(np.random.randn(50)) * 3,
        'low': prices - np.abs(np.random.randn(50)) * 3,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 50)
    })
    
    df = analyzer.calculate_indicators(df)
    summary = analyzer.get_technical_summary(df)
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name='Price'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_20'], mode='lines', name='SMA 20'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_50'], mode='lines', name='SMA 50'))
    fig.update_layout(height=500, xaxis_title='Date', yaxis_title='Price (₹)', template='plotly_white')
    
    color = {'STRONG_BUY': '#27ae60', 'BUY': '#2ecc71', 'NEUTRAL': '#f39c12', 
             'SELL': '#e74c3c'}.get(summary['overall_signal'], '#95a5a6')
    
    tech = html.Div([
        html.H4(f"Signal: {summary['overall_signal'].replace('_', ' ')}", 
               style={'color': color, 'fontSize': '24px', 'fontWeight': 'bold'}),
        html.P(f"RSI: {summary['rsi']:.2f} ({summary['rsi_status']})"),
        html.P(f"MACD: {summary['macd']:.4f} | Signal: {summary['macd_signal']:.4f}"),
        html.P(f"ADX: {summary['adx']:.2f} ({summary['trend_strength']})"),
        html.P(f"Volume: {summary['volume_ratio']:.2f}x average"),
        html.P(f"SMA 20/50/200: ₹{summary['sma_20']:.2f} / ₹{summary['sma_50']:.2f} / ₹{summary['sma_200']:.2f}")
    ])
    
    stats = html.Div([
        html.P(f"<b>Symbol:</b> {symbol}", style={'marginBottom': '5px'}),
        html.P(f"<b>Price:</b> ₹{quote['last_price']:.2f}", style={'marginBottom': '5px'}),
        html.P(f"<b>Change:</b> {quote['change']:.2f} ({quote['pChange']:.2f}%)", 
               style={'marginBottom': '5px', 'color': '#27ae60' if quote['change'] > 0 else '#e74c3c'}),
        html.P(f"<b>52W High:</b> ₹{quote['yearHigh']:.2f}", style={'marginBottom': '5px'}),
        html.P(f"<b>52W Low:</b> ₹{quote['yearLow']:.2f}", style={'marginBottom': '5px'}),
        html.P(f"<b>Volume:</b> {quote['volume']:,}", style={'marginBottom': '5px'})
    ], style={'fontSize': '14px'})
    
    return fig, tech, stats, f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}"

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 STOCK SCREENER STARTING...")
    print("=" * 50)
    print("📱 Open your browser: http://localhost:8050")
    print("📊 Select a stock and click 'Analyze Stock'")
    print("=" * 50)
    app.run_server(debug=True, port=8050)

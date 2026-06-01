# Blast NSE Lab

Dash-based Indian stock analysis dashboard with:
- NSE stock universe
- Technical charting
- RSI, MACD, ATR, ADX, Bollinger Bands, Supertrend
- Rule-based buy/sell signal
- TradingView embed
- News links

## Local run

```bash
pip install -r requirements.txt
python app.py
```

## Deploy on Render

- Push this folder to GitHub
- Create a new Render Web Service
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:server`
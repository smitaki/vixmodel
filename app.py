# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from sklearn.cluster import KMeans
import ta  # Add for RSI; pip install ta-lib or ta

st.set_page_config(page_title="VIX Spike Predictor", layout="wide")

# FULL LIGHT MODE CSS
st.markdown("""
<style>
    /* Main app background and text */
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .css-1d391kg, .css-18e3th9, .css-fblp2m {
        background-color: #ffffff !important;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
    }
    /* Text and headers */
    h1, h2, h3, h4, h5, h6, p, div, span, label, .stMarkdown {
        color: #000000 !important;
    }
    /* Metrics and buttons */
    .css-1xarl3l, .css-1offfwp, .css-1v0mbdj {
        color: black !important;
    }
    /* Expander and divider */
    .streamlit-expanderHeader {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
SPIKE_THRESHOLD = 0.30
ROLLING_WINDOW = 30
START_DATE = "1990-01-01"

FORECAST_HORIZONS = {5: "1 week", 21: "1 month", 63: "3 months"}
MEAN_REVERSION_TOL = 0.10

SEASONAL_DAYS_TO_SPIKE = {
    1: 32, 2: 25, 3: 18, 4: 20, 5: 35, 6: 40,
    7: 38, 8: 22, 9: 15, 10: 16, 11: 30, 12: 28
}

ANALOG_LEVEL_TOL = 0.05
ANALOG_VOL_TOL = 0.10
DAILY_SPIKE_THRESH = 0.10

# ----------------------------------------------------
# DATA & FEATURES - ADDED RSI FOR MATURITY
# ----------------------------------------------------
@st.cache_data(ttl=60)  # Refresh cache every 60s
def load_and_process_data():
    ticker = "^VIX"
    data = yf.download(ticker, start=START_DATE, progress=False)
    if data.empty:
        st.error("Failed to load VIX data.")
        return None
    
    # FIX: Handle potential MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    df = data.reset_index()
    
    # Robust renaming and selection
    df = df.rename(columns={
        'Date': 'DATE',
        'Open': 'OPEN',
        'High': 'HIGH',
        'Low': 'LOW',
        'Close': 'CLOSE',
        'Volume': 'VOLUME'
    })
    
    # Select only needed columns
    df = df[['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values("DATE").reset_index(drop=True)
    
    # Features
    df["daily_return"] = df["CLOSE"].pct_change()
    df["rolling_vol_30d"] = df["daily_return"].rolling(ROLLING_WINDOW).std() * np.sqrt(252)
    
    # NEW: RSI for oversold/overbought (mature math layer)
    df["rsi_14"] = ta.momentum.RSIIndicator(df["CLOSE"], window=14).rsi()
    
    regime_labels = ["calm", "normal", "elevated", "stressed"]
    df["regime"] = pd.qcut(df["CLOSE"], q=4, labels=regime_labels)
    
    for h in FORECAST_HORIZONS:
        df[f"fwd_{h}d_vix_chg"] = df["CLOSE"].shift(-h) / df["CLOSE"] - 1
    
    df["fwd_1d_vix_chg"] = df["CLOSE"].shift(-1) / df["CLOSE"] - 1
    
    # Spike clusters
    df["spike_cluster"] = "no_up_move"
    mask = df["daily_return"] > 0
    if mask.sum() >= 3:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        kmeans.fit(df.loc[mask, ["daily_return"]])
        centers = kmeans.cluster_centers_.flatten()
        order = np.argsort(centers)
        labels = ["small", "medium", "large"]
        cluster_map = {order[i]: labels[i] for i in range(3)}
        df.loc[mask, "spike_cluster"] = pd.Series(kmeans.labels_, index=df.loc[mask].index).map(cluster_map)
    
    # Signals with analogs - ADDED RSI FILTER FOR COMPLEXITY
    df["signal"] = "HOLD"
    df["signal_reason"] = ""
    
    if len(df) > 1:
        latest = df.iloc[-1]
        mean_vix = df["CLOSE"].mean()
        est_days = SEASONAL_DAYS_TO_SPIKE.get(latest["DATE"].month, 28)
        
        # Historical analogs - MATURED: SEASON FILTER
        hist = df.iloc[:-1]
        level_mask = abs((hist["CLOSE"] - latest["CLOSE"]) / latest["CLOSE"]) <= ANALOG_LEVEL_TOL
        vol_mask = abs((hist["rolling_vol_30d"] - latest["rolling_vol_30d"]) / latest["rolling_vol_30d"]) <= ANALOG_VOL_TOL
        regime_mask = hist["regime"] == latest["regime"]
        season_mask = hist["DATE"].dt.month == latest["DATE"].month  # Same month for seasonal maturity
        analogs = hist[level_mask & vol_mask & regime_mask & season_mask]
        prob_spike = (analogs["fwd_1d_vix_chg"] >= DAILY_SPIKE_THRESH).mean() if len(analogs) > 0 else np.nan
        
        if (latest["regime"] == "calm") and (latest["CLOSE"] < mean_vix * 0.95) and (latest["rsi_14"] < 30):  # RSI filter
            signal = "STRONG_BUY_VIX_CALLS" if prob_spike > 0.5 else "BUY_VIX_CALLS"
            reason = f"Calm regime + low level + oversold RSI. Season analogs: {len(analogs)}, prob spike: {prob_spike:.1%}" if not np.isnan(prob_spike) else "Strong complacency"
            df.at[df.index[-1], "signal"] = signal
            df.at[df.index[-1], "signal_reason"] = reason + f" | ~{est_days} days to spike"
        
        elif (latest["regime"] in ["calm", "normal"]) and (latest["rolling_vol_30d"] < df["rolling_vol_30d"].quantile(0.4)) and (latest["rsi_14"] < 30):
            signal = "STRONG_BUY_VIX_CALLS" if prob_spike > 0.5 else "BUY_VIX_CALLS"
            reason = f"Low vol setup + oversold RSI. Season analogs: {len(analogs)}, prob spike: {prob_spike:.1%}" if not np.isnan(prob_spike) else "Complacency building"
            df.at[df.index[-1], "signal"] = signal
            df.at[df.index[-1], "signal_reason"] = reason + f" | ~{est_days} days"
        
        elif (latest["regime"] == "stressed") and (latest["spike_cluster"] in ["medium", "large"]):
            signal = "STRONG_SELL_VIX" if prob_spike < 0.3 else "SELL_VIX"
            df.at[df.index[-1], "signal"] = signal
            df.at[df.index[-1], "signal_reason"] = "Large spike - expect fade"
    
    return df

# ----------------------------------------------------
# STREAMLIT APP
# ----------------------------------------------------
st.title("🛡️ VIX Spike Predictor Dashboard")
st.markdown("Real-time VIX analysis with historical analogs and call-buying signals")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🔄 Refresh Data Now"):
        st.cache_data.clear()
        st.success("Data refreshed!")
    auto = st.checkbox("Auto-refresh every 60s", value=True)

with col2:
    st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if auto:
    import time
    time.sleep(60)
    st.rerun()

df = load_and_process_data()
if df is None:
    st.stop()

latest = df.iloc[-1]
mean_vix = df["CLOSE"].mean()

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("VIX Close", f"{latest['CLOSE']:.2f}", f"{latest['daily_return']:.2%}")
c2.metric("Regime", latest["regime"].capitalize())
c3.metric("Rolling Vol", f"{latest['rolling_vol_30d']:.2f}")
c4.metric("Long-term Mean", f"{mean_vix:.2f}")

# Signal
signal = latest["signal"]
reason = latest.get("signal_reason", "Monitoring")
st.markdown(f"### **SIGNAL: {signal.replace('_', ' ')}**")
st.markdown(f"**Reason:** {reason}")
st.markdown(f"**Seasonal Outlook:** ~{SEASONAL_DAYS_TO_SPIKE.get(latest['DATE'].month, 28)} days to potential spike")

# Options suggestions
if "BUY" in signal:
    with st.expander("📈 VIX Call Options Suggestions", expanded=True):
        st.markdown("""
        - **DTE:** 30–60 days
        - **Strikes:** 20, 22, 25, 30 (OTM for max leverage)
        - **Live Chains:**
          → [Yahoo Finance](https://finance.yahoo.com/quote/%5EVIX/options)
          → [Barchart](https://www.barchart.com/stocks/quotes/$VIX/options)
          → [CBOE](https://www.cboe.com/tradable-products/vix/vix-options)
        """)

# Interactive Plotly Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["DATE"], y=df["CLOSE"], mode='lines', name='VIX Close', line=dict(color='black')))
buy = df[df["signal"].str.contains("BUY_VIX_CALLS")]
sell = df[df["signal"].str.contains("SELL_VIX")]
fig.add_trace(go.Scatter(x=buy["DATE"], y=buy["CLOSE"], mode='markers', name='BUY VIX CALLS',
                         marker=dict(symbol='triangle-up', size=12, color='green')))
fig.add_trace(go.Scatter(x=sell["DATE"], y=sell["CLOSE"], mode='markers', name='SELL VIX',
                         marker=dict(symbol='triangle-down', size=12, color='red')))

qs = df["CLOSE"].quantile([0.25, 0.5, 0.75])
fig.add_hline(y=qs[0.25], line_dash="dash", line_color="green", annotation_text="Calm Threshold")
fig.add_hline(y=qs[0.5], line_dash="dash", line_color="orange", annotation_text="Normal Threshold")
fig.add_hline(y=qs[0.75], line_dash="dash", line_color="red", annotation_text="Stressed Threshold")
fig.add_hline(y=mean_vix, line_dash="dot", line_color="blue", annotation_text="Long-term Mean")
fig.add_hline(y=mean_vix*0.95, line_dash="dashdot", line_color="purple", annotation_text="95% Mean Trigger")

fig.update_layout(title="Interactive VIX with Signals", height=600, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# Historical stats
with st.expander("Historical & Seasonal Stats"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Regime Thresholds**")
        st.write(f"Calm: < {qs[0.25]:.1f}")
        st.write(f"Normal: {qs[0.25]:.1f} – {qs[0.5]:.1f}")
        st.write(f"Elevated: {qs[0.5]:.1f} – {qs[0.75]:.1f}")
        st.write(f"Stressed: > {qs[0.75]:.1f}")
    with col2:
        st.write("**Seasonal Days to Spike**")
        for m in range(1, 13):
            st.write(f"{datetime(2025, m, 1).strftime('%B')}: ~{SEASONAL_DAYS_TO_SPIKE[m]} days")
st.caption("Math-based VIX spike predictor | Not financial advice")

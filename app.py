# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.cluster import KMeans

st.set_page_config(page_title="VIX Volatility Dashboard", layout="wide")

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
# DATA & FEATURES
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
    
    # Signals with analogs
    df["signal"] = "HOLD"
    df["signal_reason"] = ""
    
    if len(df) > 1:
        latest = df.iloc[-1]
        mean_vix = df["CLOSE"].mean()
        est_days = SEASONAL_DAYS_TO_SPIKE.get(latest["DATE"].month, 28)
        
        # Historical analogs
        hist = df.iloc[:-1]
        level_mask = abs((hist["CLOSE"] - latest["CLOSE"]) / latest["CLOSE"]) <= ANALOG_LEVEL_TOL
        vol_mask = abs((hist["rolling_vol_30d"] - latest["rolling_vol_30d"]) / latest["rolling_vol_30d"]) <= ANALOG_VOL_TOL
        regime_mask = hist["regime"] == latest["regime"]
        analogs = hist[level_mask & vol_mask & regime_mask]
        prob_spike = (analogs["fwd_1d_vix_chg"] >= DAILY_SPIKE_THRESH).mean() if len(analogs) > 0 else np.nan
        
        if (latest["regime"] == "calm") and (latest["CLOSE"] < mean_vix * 0.95):
            signal = "STRONG_BUY_VIX_CALLS" if prob_spike > 0.5 else "BUY_VIX_CALLS"
            reason = f"Calm regime + low level. Analogs: {len(analogs)}, prob spike: {prob_spike:.1%}" if not np.isnan(prob_spike) else "Strong complacency"
            df.at[df.index[-1], "signal"] = signal
            df.at[df.index[-1], "signal_reason"] = reason + f" | ~{est_days} days to spike"
        
        elif (latest["regime"] in ["calm", "normal"]) and (latest["rolling_vol_30d"] < df["rolling_vol_30d"].quantile(0.4)):
            signal = "STRONG_BUY_VIX_CALLS" if prob_spike > 0.5 else "BUY_VIX_CALLS"
            reason = f"Low vol setup. Analogs: {len(analogs)}, prob spike: {prob_spike:.1%}" if not np.isnan(prob_spike) else "Complacency building"
            df.at[df.index[-1], "signal"] = signal
            df.at[df.index[-1], "signal_reason"] = reason + f" | ~{est_days} days"
        
        elif (latest["regime"] == "stressed") and (latest["spike_cluster"] in ["medium", "large"]):
            signal = "STRONG_SELL_VIX" if prob_spike < 0.3 else "SELL_VIX"
            df.at[df.index[-1], "signal"] = signal
            df.at[df.index[-1], "signal_reason"] = "Large spike - expect fade"
    
    return df

# ----------------------------------------------------
# STREAMLIT APP - REDESIGNED LIKE FX DASHBOARD
# ----------------------------------------------------
st.sidebar.title("Settings")
st.sidebar.selectbox("VIX View", ["Daily", "Weekly", "Monthly"])  # Placeholder for future
st.sidebar.checkbox("Auto-refresh every 60s", value=True)
st.sidebar.button("🔄 Refresh Now")

latest = df.iloc[-1]
mean_vix = df["CLOSE"].mean()

st.title("VIX Volatility Dashboard")

# Top metrics row (like FX pairs)
col1, col2, col3, col4 = st.columns(4)
col1.metric("VIX Close", f"{latest['CLOSE']:.2f}", f"{latest['daily_return']:.2%}")
col2.metric("Open", f"{latest['OPEN']:.2f}")
col3.metric("High", f"{latest['HIGH']:.2f}")
col4.metric("Low", f"{latest['LOW']:.2f}")

# Chart
st.subheader("VIX Chart (30-day Volatility Overlay)")
fig = go.Figure()

# VIX close (blue line)
fig.add_trace(go.Scatter(x=df["DATE"], y=df["CLOSE"], mode='lines', name='VIX Close', line=dict(color='blue', width=2)))

# Rolling vol (red line)
fig.add_trace(go.Scatter(x=df["DATE"], y=df["rolling_vol_30d"], mode='lines', name='30d Vol', line=dict(color='red', width=2), yaxis="y2"))

# Layout with dual y-axis
fig.update_layout(
    yaxis2=dict(title="Volatility", overlaying="y", side="right", showgrid=False),
    height=400,
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# Table: VIX Regimes Ranked by Volatility (like FX pairs)
st.subheader("VIX Regimes Ranked by Volatility")
regime_vol = df.groupby("regime")["rolling_vol_30d"].mean().reset_index()
regime_vol = regime_vol.sort_values("rolling_vol_30d", ascending=False)
regime_vol.columns = ["Regime", "Avg Vol"]
st.dataframe(regime_vol.style.format({"Avg Vol": "{:.2f}"}))

# Signal and reason
signal = latest["signal"]
reason = latest.get("signal_reason", "Monitoring")
st.subheader("Current Signal")
st.write(signal)
st.write(reason)

st.caption("Math-based VIX predictor | Not advice")

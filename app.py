import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.cluster import KMeans
import time

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VIX Spike Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------
START_DATE = "1990-01-01"
ROLLING_WINDOW = 30
ANALOG_LEVEL_TOL = 0.05
ANALOG_VOL_TOL = 0.10
DAILY_SPIKE_THRESH = 0.10

SEASONAL_DAYS_TO_SPIKE = {
    1: 32, 2: 25, 3: 18, 4: 20, 5: 35, 6: 40,
    7: 38, 8: 22, 9: 15, 10: 16, 11: 30, 12: 28
}

# -----------------------------------------------------------------------------
# 3. DATA LOADING & ADVANCED LOGIC
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_and_process_data():
    ticker = "^VIX"
    # Fetch data
    data = yf.download(ticker, start=START_DATE, progress=False)
    
    if data.empty:
        st.error("Failed to load VIX data.")
        return None
    
    # Handle MultiIndex columns (common in new yfinance versions)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    df = data.reset_index().rename(columns={'Date': 'DATE', 'Close': 'CLOSE'})
    df = df[['DATE', 'CLOSE']]
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values("DATE").reset_index(drop=True)
    
    # --- Feature Engineering ---
    df["daily_return"] = df["CLOSE"].pct_change()
    
    # 1. Volatility & Bands
    df["rolling_mean_20d"] = df["CLOSE"].rolling(20).mean()
    df["rolling_std_20d"] = df["CLOSE"].rolling(20).std()
    
    df['bollinger_upper'] = df['rolling_mean_20d'] + (2 * df['rolling_std_20d'])
    df['bollinger_lower'] = df['rolling_mean_20d'] - (2 * df['rolling_std_20d'])
    
    # 2. NEW: Squeeze Metrics (The "Coiled Spring")
    # BB Width: How tight are the bands? (Tight = Potential Explosion)
    df['bb_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['rolling_mean_20d']
    
    # 3. NEW: RSI (Momentum Divergence)
    delta = df['CLOSE'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 4. Standard Metrics
    df["rolling_vol_30d"] = df["daily_return"].rolling(ROLLING_WINDOW).std() * np.sqrt(252)
    df['z_score'] = (df['CLOSE'] - df['rolling_mean_20d']) / df['rolling_std_20d']
    df['vcr'] = (df['CLOSE'] / df['rolling_mean_20d'] - 1) * 100
    
    # 5. Regimes
    regime_labels = ["calm", "normal", "elevated", "stressed"]
    df["regime"] = pd.qcut(df["CLOSE"], q=4, labels=regime_labels)
    df["fwd_1d_vix_chg"] = df["CLOSE"].shift(-1) / df["CLOSE"] - 1

    # --- Clustering Logic (KMeans for Spike Severity) ---
    df["spike_cluster"] = "no_up_move"
    mask = df["daily_return"] > 0
    if mask.sum() >= 3:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        X = df.loc[mask, ["daily_return"]].values.reshape(-1, 1)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_.flatten()
        order = np.argsort(centers)
        labels = ["small", "medium", "large"]
        cluster_map = {order[i]: labels[i] for i in range(3)}
        cluster_ids = kmeans.predict(X)
        df.loc[mask, "spike_cluster"] = [cluster_map[c] for c in cluster_ids]

    # --- HISTORICAL SIGNAL GENERATION (Vectorized) ---
    # This populates the chart history so you can see past performance
    
    # Define Thresholds
    bb_squeeze_level = df['bb_width'].quantile(0.10) # Bottom 10% tightness
    
    df['chart_signal'] = "HOLD"
    
    # A. Value Buy: Low Price + Low Z-Score (The original logic)
    cond_value_buy = (df['z_score'] < -1.5) & (df['regime'].isin(['calm', 'normal']))
    
    # B. Squeeze Buy: Tight Bands + Price near bottom (NEW: Catch explosions from quiet periods)
    cond_squeeze_buy = (df['bb_width'] < bb_squeeze_level) & (df['CLOSE'] <= df['rolling_mean_20d'])
    
    # C. Sell Signal: Extremes
    cond_sell = (df['z_score'] > 2.0) | (df['rsi'] > 75)
    
    # Apply to dataframe
    df.loc[cond_value_buy, 'chart_signal'] = 'BUY_VALUE'
    df.loc[cond_squeeze_buy, 'chart_signal'] = 'BUY_SQUEEZE' # Overwrites value if both match (prioritize squeeze)
    df.loc[cond_sell, 'chart_signal'] = 'SELL_EXTREME'

    # --- LATEST ROW PREDICTION (Detailed Analog Logic) ---
    # This runs the complex logic only for the "Current Status" box
    df["signal"] = "HOLD"
    df["signal_reason"] = "Monitoring market conditions."
    
    if len(df) > 30:
        latest = df.iloc[-1]
        
        # 1. Analog Search
        hist = df.iloc[:-1]
        level_mask = abs((hist["CLOSE"] - latest["CLOSE"]) / latest["CLOSE"]) <= ANALOG_LEVEL_TOL
        vol_mask = abs((hist["rolling_vol_30d"] - latest["rolling_vol_30d"]) / latest["rolling_vol_30d"]) <= ANALOG_VOL_TOL
        analogs = hist[level_mask & vol_mask]
        prob_spike = (analogs["fwd_1d_vix_chg"] >= DAILY_SPIKE_THRESH).mean() if len(analogs) > 0 else 0
        
        signal = "HOLD"
        reason = "Monitoring"

        # 2. Decision Tree
        # Priority 1: Squeeze (Explosive Potential)
        if latest['bb_width'] < bb_squeeze_level:
            signal = "STRONG_BUY_VOL_SQUEEZE"
            reason = f"⚠️ SQUEEZE DETECTED. BB Width ({latest['bb_width']:.2f}) is in bottom 10%. History suggests explosive move."
            
        # Priority 2: Deep Value (Mean Reversion Up)
        elif (latest["regime"] == "calm") and (latest["z_score"] < -1.5):
            confidence = "STRONG_" if prob_spike > 0.5 else ""
            signal = f"{confidence}BUY_VIX_CALLS"
            reason = f"Deep Value (Z: {latest['z_score']:.2f}). {len(analogs)} Analogs found (Prob Spike: {prob_spike:.1%})"
            
        # Priority 3: Fade the Spike (Mean Reversion Down)
        elif (latest["regime"] == "stressed") and (latest["z_score"] > 2.0):
            signal = "SELL_VIX"
            reason = "Statistical Extreme (>2 Sigma). Expect Mean Reversion."

        # Assign to latest row
        df.at[df.index[-1], "signal"] = signal
        df.at[df.index[-1], "signal_reason"] = reason
        
        # Ensure the chart marker for the latest day matches the specific signal
        chart_tag = "SELL_EXTREME" if "SELL" in signal else ("BUY_SQUEEZE" if "SQUEEZE" in signal else ("BUY_VALUE" if "BUY" in signal else "HOLD"))
        if chart_tag != "HOLD":
            df.at[df.index[-1], 'chart_signal'] = chart_tag

    return df

# Load Data
df = load_and_process_data()

# -----------------------------------------------------------------------------
# 4. SIDEBAR SETTINGS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
        
    st.markdown("---")
    
    # Date Slider
    min_date = df['DATE'].min().date()
    max_date = df['DATE'].max().date()
    default_start = max(min_date, pd.to_datetime("2023-01-01").date())
    
    start_date, end_date = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(default_start, max_date),
        format="YYYY-MM-DD"
    )
    
    st.markdown("### Chart Overlays")
    show_bollinger = st.checkbox("Show Bollinger Bands", value=True)
    show_thresholds = st.checkbox("Show Regime Thresholds", value=True)

# Filter Data
mask = (df['DATE'].dt.date >= start_date) & (df['DATE'].dt.date <= end_date)
df_filtered = df.loc[mask]

# -----------------------------------------------------------------------------
# 5. METRICS HEADER
# -----------------------------------------------------------------------------
st.title("🛡️ VIX Spike Predictor")
st.markdown(f"**Status:** {df.iloc[-1]['signal_reason']}")

latest = df.iloc[-1]
prev = df.iloc[-2]
delta_vix = latest['CLOSE'] - prev['CLOSE']

# Metrics Layout
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("VIX Level", f"{latest['CLOSE']:.2f}", f"{delta_vix:.2f}")
with c2:
    st.metric("Regime", str(latest['regime']).upper(), delta_color="off")
with c3:
    st.metric("Squeeze Metric (BB Width)", f"{latest['bb_width']:.2f}", help="Lower is tighter (more explosive potential)")
with c4:
    st.metric("Z-Score", f"{latest['z_score']:.2f}σ")
with c5:
    sig = latest['signal']
    if "BUY" in sig:
        st.success(f"🔼 {sig}")
    elif "SELL" in sig:
        st.error(f"🔽 {sig}")
    else:
        st.info(f"⏸️ {sig}")

# -----------------------------------------------------------------------------
# 6. CHARTING (Enhanced with Squeeze Signals)
# -----------------------------------------------------------------------------
# Create 2-row subplot
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    row_heights=[0.7, 0.3],
    subplot_titles=("VIX Price Action & Signals", "Z-Score Deviation")
)

# --- TRACE 1: Price ---
fig.add_trace(go.Scatter(
    x=df_filtered['DATE'], y=df_filtered['CLOSE'], 
    mode='lines', name='VIX',
    line=dict(color='#00d4ff', width=2)
), row=1, col=1)

# --- TRACE 2: Bollinger Bands ---
if show_bollinger:
    fig.add_trace(go.Scatter(
        x=df_filtered['DATE'], y=df_filtered['bollinger_upper'],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_filtered['DATE'], y=df_filtered['bollinger_lower'],
        fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)',
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ), row=1, col=1)

# --- TRACE 3: Signals ---
# A. Value Buys (Green Triangles)
buy_val = df_filtered[df_filtered['chart_signal'] == "BUY_VALUE"]
fig.add_trace(go.Scatter(
    x=buy_val['DATE'], y=buy_val['CLOSE'],
    mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00ff00'),
    name='Value Buy'
), row=1, col=1)

# B. Squeeze Alerts (Orange Stars) - NEW
buy_sqz = df_filtered[df_filtered['chart_signal'] == "BUY_SQUEEZE"]
fig.add_trace(go.Scatter(
    x=buy_sqz['DATE'], y=buy_sqz['CLOSE'],
    mode='markers', marker=dict(symbol='star', size=12, color='orange', line=dict(width=1, color='white')),
    name='Squeeze Alert'
), row=1, col=1)

# C. Sells (Red Triangles)
sell_sig = df_filtered[df_filtered['chart_signal'] == "SELL_EXTREME"]
fig.add_trace(go.Scatter(
    x=sell_sig['DATE'], y=sell_sig['CLOSE'],
    mode='markers', marker=dict(symbol='triangle-down', size=10, color='#ff0000'),
    name='Sell Signal'
), row=1, col=1)

# --- TRACE 4: Regime Thresholds ---
if show_thresholds:
    qs = df["CLOSE"].quantile([0.25, 0.75])
    fig.add_hline(y=qs[0.25], line_dash="dot", line_color="green", annotation_text="Calm", row=1, col=1)
    fig.add_hline(y=qs[0.75], line_dash="dot", line_color="red", annotation_text="Stressed", row=1, col=1)

# --- TRACE 5: Z-Score (Bottom Panel) ---
colors = np.where(df_filtered['z_score'] > 2, 'red', np.where(df_filtered['z_score'] < -1.5, 'green', 'gray'))
fig.add_trace(go.Bar(
    x=df_filtered['DATE'], y=df_filtered['z_score'],
    marker_color=colors, name='Z-Score'
), row=2, col=1)

# Add reference lines to Z-Score
fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=-1.5, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(height=650, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. HISTORICAL LEDGER
# -----------------------------------------------------------------------------
st.subheader("📋 Historical Ledger")

# Prepare display dataframe
cols = ['DATE', 'CLOSE', 'chart_signal', 'regime', 'bb_width', 'z_score', 'rsi', 'daily_return']
df_display = df_filtered[cols].sort_values('DATE', ascending=False)

st.dataframe(
    df_display,
    use_container_width=True,
    height=400,
    hide_index=True,
    column_config={
        "DATE": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
        "CLOSE": st.column_config.NumberColumn("VIX", format="%.2f"),
        "daily_return": st.column_config.NumberColumn("1D %", format="%.2f%%"),
        "bb_width": st.column_config.NumberColumn("Squeeze (BBW)", format="%.2f"),
        "z_score": st.column_config.NumberColumn("Z-Score", format="%.2f"),
        "rsi": st.column_config.NumberColumn("RSI", format="%.1f"),
        "chart_signal": st.column_config.TextColumn("Signal Type"),
        "regime": st.column_config.TextColumn("Regime"),
    }
)

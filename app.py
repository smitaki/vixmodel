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
# 2. CONFIG & LOGIC CONSTANTS
# -----------------------------------------------------------------------------
SPIKE_THRESHOLD = 0.30
ROLLING_WINDOW = 30
START_DATE = "1990-01-01"

FORECAST_HORIZONS = {5: "1 week", 21: "1 month", 63: "3 months"}

SEASONAL_DAYS_TO_SPIKE = {
    1: 32, 2: 25, 3: 18, 4: 20, 5: 35, 6: 40,
    7: 38, 8: 22, 9: 15, 10: 16, 11: 30, 12: 28
}

ANALOG_LEVEL_TOL = 0.05
ANALOG_VOL_TOL = 0.10
DAILY_SPIKE_THRESH = 0.10

# -----------------------------------------------------------------------------
# 3. DATA LOADING & PROCESSING (Merged Logic)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_and_process_data():
    ticker = "^VIX"
    data = yf.download(ticker, start=START_DATE, progress=False)
    
    if data.empty:
        st.error("Failed to load VIX data.")
        return None
    
    # Handle MultiIndex if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    df = data.reset_index()
    
    # Standardize columns
    df = df.rename(columns={'Date': 'DATE', 'Close': 'CLOSE'})
    df = df[['DATE', 'CLOSE']] # minimal columns
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values("DATE").reset_index(drop=True)
    
    # --- Feature Engineering ---
    df["daily_return"] = df["CLOSE"].pct_change()
    
    # Rolling Metrics
    df["rolling_mean_20d"] = df["CLOSE"].rolling(20).mean()
    df["rolling_std_20d"] = df["CLOSE"].rolling(20).std()
    
    # Bollinger Bands (for chart visualization)
    df['bollinger_upper'] = df['rolling_mean_20d'] + (2 * df['rolling_std_20d'])
    df['bollinger_lower'] = df['rolling_mean_20d'] - (2 * df['rolling_std_20d'])
    
    # Volatility & Z-Score
    df["rolling_vol_30d"] = df["daily_return"].rolling(ROLLING_WINDOW).std() * np.sqrt(252)
    df['z_score'] = (df['CLOSE'] - df['rolling_mean_20d']) / df['rolling_std_20d']
    df['vcr'] = (df['CLOSE'] / df['rolling_mean_20d'] - 1) * 100
    
    # Regimes (Quartiles)
    regime_labels = ["calm", "normal", "elevated", "stressed"]
    df["regime"] = pd.qcut(df["CLOSE"], q=4, labels=regime_labels)
    
    # Future Look-ahead (for analog training)
    df["fwd_1d_vix_chg"] = df["CLOSE"].shift(-1) / df["CLOSE"] - 1
    
    # --- Clustering Logic (KMeans) ---
    df["spike_cluster"] = "no_up_move"
    mask = df["daily_return"] > 0
    if mask.sum() >= 3:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        # Reshape for sklearn
        X = df.loc[mask, ["daily_return"]].values.reshape(-1, 1)
        kmeans.fit(X)
        
        # Sort clusters by intensity (small -> large)
        centers = kmeans.cluster_centers_.flatten()
        order = np.argsort(centers)
        labels = ["small", "medium", "large"]
        cluster_map = {order[i]: labels[i] for i in range(3)}
        
        # Assign labels
        cluster_ids = kmeans.predict(X)
        df.loc[mask, "spike_cluster"] = [cluster_map[c] for c in cluster_ids]

    # --- Signal Generation (Analog Logic) ---
    df["signal"] = "HOLD"
    df["signal_reason"] = "Monitoring market conditions."
    
    # Only run expensive analog search on the latest row to save time, 
    # or run on full history if backtesting (here we do latest for speed/display)
    if len(df) > 30:
        latest = df.iloc[-1]
        mean_vix = df["CLOSE"].mean()
        est_days = SEASONAL_DAYS_TO_SPIKE.get(latest["DATE"].month, 28)
        
        # Define Historical Search Space
        hist = df.iloc[:-1]
        
        # Analog Matching Masks
        level_mask = abs((hist["CLOSE"] - latest["CLOSE"]) / latest["CLOSE"]) <= ANALOG_LEVEL_TOL
        vol_mask = abs((hist["rolling_vol_30d"] - latest["rolling_vol_30d"]) / latest["rolling_vol_30d"]) <= ANALOG_VOL_TOL
        regime_mask = hist["regime"] == latest["regime"]
        
        analogs = hist[level_mask & vol_mask & regime_mask]
        prob_spike = (analogs["fwd_1d_vix_chg"] >= DAILY_SPIKE_THRESH).mean() if len(analogs) > 0 else 0
        
        # Logic Tree
        signal = "HOLD"
        reason = "Monitoring"
        
        # Case A: Calm/Complacent
        if (latest["regime"] == "calm") and (latest["CLOSE"] < mean_vix * 0.95):
            if prob_spike > 0.5:
                signal = "STRONG_BUY_VIX_CALLS"
            else:
                signal = "BUY_VIX_CALLS"
            reason = f"Calm regime + Low Level. {len(analogs)} Analogs found (Prob Spike: {prob_spike:.1%})"
            
        # Case B: Low Vol Setup
        elif (latest["regime"] in ["calm", "normal"]) and (latest["rolling_vol_30d"] < df["rolling_vol_30d"].quantile(0.4)):
            if prob_spike > 0.5:
                signal = "STRONG_BUY_VIX_CALLS"
            else:
                signal = "BUY_VIX_CALLS"
            reason = f"Vol Compression. {len(analogs)} Analogs found (Prob Spike: {prob_spike:.1%})"
            
        # Case C: Fade the Spike
        elif (latest["regime"] == "stressed") and (latest["spike_cluster"] in ["medium", "large"]):
            signal = "SELL_VIX" if prob_spike < 0.3 else "HOLD"
            reason = "Large spike detected - Mean Reversion likely."

        # Assign to latest row
        df.at[df.index[-1], "signal"] = signal
        df.at[df.index[-1], "signal_reason"] = reason
        
        # --- Backfill signals for visualization (Simplified for chart) ---
        # Note: Real backtesting would require walking forward. 
        # Here we just mark extreme outliers for the chart history.
        df['chart_signal'] = np.where(df['z_score'] < -1.5, 'BUY_VIX_CALLS', 'HOLD')
        df['chart_signal'] = np.where(df['z_score'] > 2.0, 'SELL_VIX', df['chart_signal'])
        # Overwrite latest with the actual smart model signal
        df.at[df.index[-1], 'chart_signal'] = signal

    return df

# Load Data
df = load_and_process_data()

# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Date Filter
    min_date = df['DATE'].min().date()
    max_date = df['DATE'].max().date()
    
    # Default to last 2 years for better view
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

if auto_refresh:
    time.sleep(60)
    st.rerun()

# -----------------------------------------------------------------------------
# 5. METRICS HEADER
# -----------------------------------------------------------------------------
st.title("🛡️ VIX Spike Predictor")
st.markdown(f"**Status:** {df.iloc[-1]['signal_reason']}")

latest = df.iloc[-1]
prev = df.iloc[-2]
delta_vix = latest['CLOSE'] - prev['CLOSE']
mean_vix = df["CLOSE"].mean()

# Calculate Quantiles for display
qs = df["CLOSE"].quantile([0.25, 0.5, 0.75])

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("VIX Level", f"{latest['CLOSE']:.2f}", f"{delta_vix:.2f}")
with c2:
    st.metric("Regime", str(latest['regime']).upper(), delta_color="off")
with c3:
    st.metric("VCR Metric", f"{latest['vcr']:.2f}%", help="Gap between Price and 20d MA")
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
# 6. DYNAMIC CHART
# -----------------------------------------------------------------------------
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    row_heights=[0.7, 0.3],
    subplot_titles=("VIX Price Action & Regimes", "Z-Score Deviation")
)

# --- TRACE 1: VIX Close Price ---
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

# --- TRACE 3: Regime Thresholds (From Original Code) ---
if show_thresholds:
    fig.add_hline(y=qs[0.25], line_dash="dot", line_color="green", annotation_text="Calm", row=1, col=1)
    fig.add_hline(y=qs[0.75], line_dash="dot", line_color="red", annotation_text="Stressed", row=1, col=1)

# --- TRACE 4: Signals (Triangles) ---
# Filter for visualization signals
buy_sig = df_filtered[df_filtered['chart_signal'].str.contains("BUY", na=False)]
sell_sig = df_filtered[df_filtered['chart_signal'].str.contains("SELL", na=False)]

fig.add_trace(go.Scatter(
    x=buy_sig['DATE'], y=buy_sig['CLOSE'],
    mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00ff00'),
    name='Buy Signal'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=sell_sig['DATE'], y=sell_sig['CLOSE'],
    mode='markers', marker=dict(symbol='triangle-down', size=10, color='#ff0000'),
    name='Sell Signal'
), row=1, col=1)

# --- TRACE 5: Z-Score Bar Chart (Bottom Panel) ---
colors = np.where(df_filtered['z_score'] > 2, 'red', np.where(df_filtered['z_score'] < -1.5, 'green', 'gray'))
fig.add_trace(go.Bar(
    x=df_filtered['DATE'], y=df_filtered['z_score'],
    marker_color=colors, name='Z-Score'
), row=2, col=1)

fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=-2.0, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(height=600, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. ENHANCED DATA LEDGER
# -----------------------------------------------------------------------------
st.subheader("📋 Historical Ledger")

# Select relevant columns
cols = ['DATE', 'CLOSE', 'signal', 'regime', 'spike_cluster', 'z_score', 'vcr', 'daily_return', 'signal_reason']
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
        "vcr": st.column_config.NumberColumn("VCR", format="%.2f%%"),
        "z_score": st.column_config.NumberColumn("Z-Score", format="%.2f"),
        "spike_cluster": st.column_config.TextColumn("Cluster"),
        "signal": st.column_config.TextColumn("Model Signal"),
        "signal_reason": st.column_config.TextColumn("Reason", width="large"),
    }
)

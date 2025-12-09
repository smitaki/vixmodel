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
    page_title="VIX Spike Predictor Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------
START_DATE = "2010-01-01" # Shorter start date ensures VVIX data availability (started ~2007)
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
    # Fetch Data: VIX, S&P 500 (Context), VVIX (Smart Money)
    # Note: VVIX history is shorter, so we use inner join logic implicitly by alignment
    tickers = ["^VIX", "^GSPC", "^VVIX"]
    data = yf.download(tickers, start=START_DATE, progress=False)
    
    if data.empty:
        st.error("Failed to load market data.")
        return None, None
    
    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        # Flatten: Close_VIX, Close_GSPC, etc.
        close_df = data['Close'].copy()
    else:
        close_df = data.copy()

    # Rename & Clean
    df = close_df.reset_index().rename(columns={
        'Date': 'DATE', 
        '^VIX': 'CLOSE', 
        '^GSPC': 'SPX', 
        '^VVIX': 'VVIX'
    })
    
    # Ensure Date format
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values("DATE").reset_index(drop=True)
    
    # Fill missing VVIX (it started later than VIX) with ffill or drop
    df['VVIX'] = df['VVIX'].ffill() 
    df.dropna(inplace=True) # Ensure all indicators can be calculated

    # --- A. FEATURE ENGINEERING ---
    df["daily_return"] = df["CLOSE"].pct_change()
    
    # 1. Volatility & Bands (VIX)
    df["rolling_mean_20d"] = df["CLOSE"].rolling(20).mean()
    df["rolling_std_20d"] = df["CLOSE"].rolling(20).std()
    df['bollinger_upper'] = df['rolling_mean_20d'] + (2 * df['rolling_std_20d'])
    df['bollinger_lower'] = df['rolling_mean_20d'] - (2 * df['rolling_std_20d'])
    
    # Squeeze Metric (BB Width)
    df['bb_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['rolling_mean_20d']
    
    # 2. Momentum (RSI)
    delta = df['CLOSE'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. Standard Metrics
    df['z_score'] = (df['CLOSE'] - df['rolling_mean_20d']) / df['rolling_std_20d']
    df['vcr'] = (df['CLOSE'] / df['rolling_mean_20d'] - 1) * 100
    df["rolling_vol_30d"] = df["daily_return"].rolling(ROLLING_WINDOW).std() * np.sqrt(252)

    # 4. Context Layers (New Logic)
    # SPX Trend (Equity Filter)
    df['spx_ma50'] = df['SPX'].rolling(50).mean()
    df['spx_trend'] = np.where(df['SPX'] > df['spx_ma50'], "UPTREND", "FRAGILE")
    
    # VVIX Trend (Veto)
    df['vvix_ma10'] = df['VVIX'].rolling(10).mean()
    
    # Inter-Arrival Time (Recharge Logic)
    # Mark Spike Days (>20 level AND >10% move)
    spike_days = df[(df['CLOSE'] > 20) & (df['daily_return'] > 0.10)].index
    df['days_since_spike'] = np.nan
    if not spike_days.empty:
        last_spike_idx = pd.Series(spike_days, index=spike_days).reindex(df.index, method='ffill')
        df['days_since_spike'] = (df.index - last_spike_idx) 
        # Convert index difference to days (approx, assuming row=day)
        # For precision we should use dates, but index diff is faster for relative "rows ago"
    
    # Regimes
    regime_labels = ["calm", "normal", "elevated", "stressed"]
    df["regime"] = pd.qcut(df["CLOSE"], q=4, labels=regime_labels)
    df["fwd_1d_vix_chg"] = df["CLOSE"].shift(-1) / df["CLOSE"] - 1

    # --- B. SIGNAL GENERATION (With Filters) ---
    
    # Define Thresholds
    bb_squeeze_level = df['bb_width'].quantile(0.10) # Bottom 10%
    
    df['chart_signal'] = "HOLD"
    
    # Logic 1: Squeeze Buy (The Coiled Spring)
    # Filter: Don't buy squeeze if SPX is in a raging Uptrend (Grind Up) AND VVIX is dead
    cond_squeeze_raw = (df['bb_width'] < bb_squeeze_level) & (df['CLOSE'] <= df['rolling_mean_20d'])
    cond_spx_filter = (df['spx_trend'] == "UPTREND") & (df['VVIX'] < df['VVIX'].rolling(50).mean())
    cond_squeeze_buy = cond_squeeze_raw & (~cond_spx_filter) # Only buy if NOT filtered
    
    # Logic 2: Value Buy (Mean Reversion)
    cond_value_buy = (df['z_score'] < -1.5) & (df['regime'].isin(['calm', 'normal']))
    
    # Logic 3: Sell (Extremes)
    cond_sell = (df['z_score'] > 2.0) | (df['rsi'] > 75)
    
    # Apply Signals
    df.loc[cond_value_buy, 'chart_signal'] = 'BUY_VALUE'
    df.loc[cond_squeeze_buy, 'chart_signal'] = 'BUY_SQUEEZE'
    df.loc[cond_sell, 'chart_signal'] = 'SELL_EXTREME'

    # --- C. LATENCY ANALYSIS (Learning from History) ---
    # Calculate stats for the dashboard "Timing" section
    stats = {}
    
    # 1. Median Latency (Days from Squeeze Signal to >10% Spike)
    squeeze_indices = df[df['chart_signal'] == 'BUY_SQUEEZE'].index
    latencies = []
    
    for idx in squeeze_indices[-50:]: # Look at last 50 signals
        # Look forward 30 days
        forward_window = df.loc[idx+1 : idx+30]
        spike_found = forward_window[forward_window['daily_return'] > 0.10]
        if not spike_found.empty:
            days_to_spike = (spike_found.index[0] - idx) # Row delta
            latencies.append(days_to_spike)
            
    stats['median_latency'] = np.median(latencies) if latencies else 5 # Default to 5
    stats['avg_recharge'] = df['days_since_spike'].mean() if 'days_since_spike' in df else 30
    
    # --- D. LATEST PREDICTION ---
    df["signal"] = "HOLD"
    df["signal_reason"] = "Monitoring conditions."
    
    if len(df) > 30:
        latest = df.iloc[-1]
        
        # Check Filters
        is_uptrend = latest['spx_trend'] == "UPTREND"
        vvix_confirm = latest['VVIX'] > latest['vvix_ma10']
        
        signal = "HOLD"
        reason = "Monitoring"
        
        # Priority 1: Squeeze
        if latest['bb_width'] < bb_squeeze_level:
            if is_uptrend and not vvix_confirm:
                reason = "Squeeze Detected BUT filtered by SPX Uptrend (Grind Up Risk)."
                signal = "HOLD (FILTERED)"
            else:
                signal = "STRONG_BUY_VOL_SQUEEZE"
                reason = f"⚠️ SQUEEZE. BB Width ({latest['bb_width']:.2f}) < 10%. Latency: ~{stats['median_latency']:.0f} days."

        # Priority 2: Deep Value
        elif (latest["regime"] == "calm") and (latest["z_score"] < -1.5):
            signal = "BUY_VIX_CALLS"
            reason = f"Deep Value (Z: {latest['z_score']:.2f})."
            
        # Priority 3: Sell
        elif (latest["regime"] == "stressed") and (latest["z_score"] > 2.0):
            signal = "SELL_VIX"
            reason = "Statistical Extreme. Mean Reversion likely."

        df.at[df.index[-1], "signal"] = signal
        df.at[df.index[-1], "signal_reason"] = reason
        
        # Sync chart tag
        tag = "HOLD"
        if "SQUEEZE" in signal: tag = "BUY_SQUEEZE"
        elif "BUY" in signal: tag = "BUY_VALUE"
        elif "SELL" in signal: tag = "SELL_EXTREME"
        if tag != "HOLD": df.at[df.index[-1], 'chart_signal'] = tag

    return df, stats

# Load Data & Stats
df, stats = load_and_process_data()

# -----------------------------------------------------------------------------
# 4. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    
    min_date = df['DATE'].min().date()
    max_date = df['DATE'].max().date()
    default_start = max(min_date, pd.to_datetime("2023-01-01").date())
    
    start_date, end_date = st.slider(
        "Select Date Range", min_value=min_date, max_value=max_date,
        value=(default_start, max_date), format="YYYY-MM-DD"
    )
    
    st.markdown("### Chart Overlays")
    show_bollinger = st.checkbox("Show Bollinger Bands", value=True)
    show_context = st.checkbox("Show SPX/VVIX Context", value=False) # Toggle for extra data

mask = (df['DATE'].dt.date >= start_date) & (df['DATE'].dt.date <= end_date)
df_filtered = df.loc[mask]

# -----------------------------------------------------------------------------
# 5. HEADER & METRICS
# -----------------------------------------------------------------------------
st.title("🛡️ VIX Spike Predictor Pro")
st.caption("Advanced Volatility Modeling: Squeeze Logic, Equity Filters, and Latency Analysis")
st.markdown(f"**Status:** {df.iloc[-1]['signal_reason']}")

latest = df.iloc[-1]
delta_vix = latest['CLOSE'] - df.iloc[-2]['CLOSE']

# Row 1: Market Data
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("VIX Level", f"{latest['CLOSE']:.2f}", f"{delta_vix:.2f}")
with c2: st.metric("Regime", str(latest['regime']).upper(), delta_color="off")
with c3: st.metric("SPX Trend", latest['spx_trend'], help="Uptrends often invalidate VIX signals")
with c4: st.metric("VVIX Level", f"{latest['VVIX']:.2f}", help="Vol of Vol - The 'Smart Money' Indicator")
with c5:
    sig = latest['signal']
    if "BUY" in sig: st.success(f"🔼 {sig}")
    elif "SELL" in sig: st.error(f"🔽 {sig}")
    else: st.info(f"⏸️ {sig}")

# Row 2: Timing Intelligence (New!)
st.markdown("#### ⏳ Timing Intelligence")
t1, t2, t3, t4 = st.columns(4)
with t1:
    st.metric("Est. Latency", f"~{stats['median_latency']:.0f} Days", help="Median days from Signal to Spike")
with t2:
    st.metric("Suggested Option DTE", f"{int(stats['median_latency']*1.5)}-{int(stats['median_latency']*4)} Days", help="Ideal expiration window")
with t3:
    st.metric("Stale Date Limit", f"+{int(stats['median_latency']*1.5)} Days", help="If no move by this time, Exit trade.")
with t4:
    last_spike = df_filtered[df_filtered['daily_return'] > 0.10].index.max()
    days_ago = (df_filtered.index.max() - last_spike) if pd.notna(last_spike) else 0
    st.metric("Days Since Spike", f"{days_ago} Days", help="Market 'Recharge' counter")

# -----------------------------------------------------------------------------
# 6. CHARTING
# -----------------------------------------------------------------------------
fig = make_subplots(
    rows=3, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    row_heights=[0.6, 0.2, 0.2],
    subplot_titles=("VIX Price & Signals", "Z-Score (Deviation)", "Squeeze Metric (BB Width)")
)

# --- TRACE 1: Price ---
fig.add_trace(go.Scatter(x=df_filtered['DATE'], y=df_filtered['CLOSE'], mode='lines', name='VIX', line=dict(color='#00d4ff', width=2)), row=1, col=1)

# Bollinger Bands
if show_bollinger:
    fig.add_trace(go.Scatter(x=df_filtered['DATE'], y=df_filtered['bollinger_upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_filtered['DATE'], y=df_filtered['bollinger_lower'], fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)

# Signals
buy_sqz = df_filtered[df_filtered['chart_signal'] == "BUY_SQUEEZE"]
fig.add_trace(go.Scatter(x=buy_sqz['DATE'], y=buy_sqz['CLOSE'], mode='markers', marker=dict(symbol='star', size=14, color='orange', line=dict(width=1, color='white')), name='Squeeze Buy'), row=1, col=1)

buy_val = df_filtered[df_filtered['chart_signal'] == "BUY_VALUE"]
fig.add_trace(go.Scatter(x=buy_val['DATE'], y=buy_val['CLOSE'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00ff00'), name='Value Buy'), row=1, col=1)

sell_sig = df_filtered[df_filtered['chart_signal'] == "SELL_EXTREME"]
fig.add_trace(go.Scatter(x=sell_sig['DATE'], y=sell_sig['CLOSE'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='#ff0000'), name='Sell'), row=1, col=1)

# --- TRACE 2: Z-Score ---
colors = np.where(df_filtered['z_score'] > 2, 'red', np.where(df_filtered['z_score'] < -1.5, 'green', 'gray'))
fig.add_trace(go.Bar(x=df_filtered['DATE'], y=df_filtered['z_score'], marker_color=colors, name='Z-Score'), row=2, col=1)
fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=-1.5, line_dash="dash", line_color="green", row=2, col=1)

# --- TRACE 3: BB Width (Squeeze) ---
fig.add_trace(go.Scatter(x=df_filtered['DATE'], y=df_filtered['bb_width'], mode='lines', name='BB Width', line=dict(color='yellow', width=1)), row=3, col=1)
# Add Squeeze Threshold Line
squeeze_thresh = df['bb_width'].quantile(0.10)
fig.add_hline(y=squeeze_thresh, line_dash="dot", line_color="orange", annotation_text="Squeeze Zone (10%)", row=3, col=1)

fig.update_layout(height=800, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. HISTORICAL LEDGER
# -----------------------------------------------------------------------------
st.subheader("📋 Historical Ledger")
cols = ['DATE', 'CLOSE', 'chart_signal', 'regime', 'bb_width', 'z_score', 'spx_trend', 'VVIX']
df_display = df_filtered[cols].sort_values('DATE', ascending=False)

st.dataframe(
    df_display,
    use_container_width=True,
    height=400,
    hide_index=True,
    column_config={
        "DATE": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
        "CLOSE": st.column_config.NumberColumn("VIX", format="%.2f"),
        "bb_width": st.column_config.NumberColumn("BB Width", format="%.3f"),
        "z_score": st.column_config.NumberColumn("Z-Score", format="%.2f"),
        "VVIX": st.column_config.NumberColumn("VVIX", format="%.2f"),
        "chart_signal": st.column_config.TextColumn("Signal"),
        "spx_trend": st.column_config.TextColumn("Context"),
    }
)

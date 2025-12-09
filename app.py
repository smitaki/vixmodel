import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
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
START_DATE = "2007-01-01" 
ROLLING_WINDOW = 30

# -----------------------------------------------------------------------------
# 3. DATA LOADING & INTELLIGENT MODELING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_and_process_data():
    # Fetch Data
    tickers = ["^VIX", "^GSPC", "^VVIX"]
    data = yf.download(tickers, start=START_DATE, progress=False)
    
    if data.empty:
        st.error("Failed to load market data.")
        return None, None
    
    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        close_df = data['Close'].copy()
    else:
        close_df = data['Close'].copy()

    # Rename & Clean
    df = close_df.reset_index().rename(columns={
        'Date': 'DATE', 
        '^VIX': 'CLOSE', 
        '^GSPC': 'SPX', 
        '^VVIX': 'VVIX'
    })
    
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values("DATE").reset_index(drop=True)
    df['VVIX'] = df['VVIX'].ffill() 
    df.dropna(inplace=True)

    # --- A. FEATURE ENGINEERING ---
    df["daily_return"] = df["CLOSE"].pct_change()
    
    # 1. Volatility & Bands
    df["rolling_mean_20d"] = df["CLOSE"].rolling(20).mean()
    df["rolling_std_20d"] = df["CLOSE"].rolling(20).std()
    
    df['bollinger_upper'] = df['rolling_mean_20d'] + (2 * df['rolling_std_20d'])
    df['bollinger_lower'] = df['rolling_mean_20d'] - (2 * df['rolling_std_20d'])
    
    # Squeeze Metric (BB Width)
    df['bb_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['rolling_mean_20d']
    
    # 2. Momentum & Value
    delta = df['CLOSE'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['z_score'] = (df['CLOSE'] - df['rolling_mean_20d']) / df['rolling_std_20d']
    
    # 3. Context Layers
    df['spx_ma50'] = df['SPX'].rolling(50).mean()
    df['spx_trend'] = np.where(df['SPX'] > df['spx_ma50'], "UPTREND", "FRAGILE")
    df['vvix_ma10'] = df['VVIX'].rolling(10).mean()
    
    # Regimes
    df["regime"] = pd.qcut(df["CLOSE"], q=4, labels=["calm", "normal", "elevated", "stressed"])

    # -------------------------------------------------------------------------
    # ⚛️ THE "VPE" FORMULA (Master Index)
    # -------------------------------------------------------------------------
    # Formula: Energy = VVIX / (BB_Width * VIX_Price)
    # We add a small epsilon (1e-6) to denominator to avoid division by zero errors
    epsilon = 1e-6
    df['vpe_raw'] = df['VVIX'] / ((df['bb_width'] * df['CLOSE']) + epsilon)
    
    # Normalize to 0-100 Scale (Percentile Rank over last 1 year)
    df['VPE'] = df['vpe_raw'].rolling(252).rank(pct=True) * 100

    # -------------------------------------------------------------------------
    # 🧠 B. CONTINUOUS LEARNING (Walk-Forward Optimization)
    # -------------------------------------------------------------------------
    potential_thresholds = [0.05, 0.10, 0.15] 
    best_thresh = 0.10 
    best_score = -1
    
    training_data = df.tail(252).copy()
    
    for thresh in potential_thresholds:
        squeeze_level = training_data['bb_width'].quantile(thresh)
        signals = (training_data['bb_width'] < squeeze_level) & (training_data['CLOSE'] <= training_data['rolling_mean_20d'])
        
        hits = 0
        signal_indices = training_data[signals].index
        for idx in signal_indices:
            if idx + 10 < len(training_data):
                future_max = training_data.loc[idx+1:idx+10, 'daily_return'].max()
                if future_max > 0.10: 
                    hits += 1
        
        num_signals = len(signal_indices)
        if num_signals > 0:
            score = (hits / num_signals) * np.log(num_signals + 1) 
            if score > best_score:
                best_score = score
                best_thresh = thresh

    learned_squeeze_level = df['bb_width'].quantile(best_thresh)

    # -------------------------------------------------------------------------
    # 🚦 C. SIGNAL GENERATION
    # -------------------------------------------------------------------------
    df['chart_signal'] = "HOLD"
    
    cond_squeeze_raw = (df['bb_width'] < learned_squeeze_level) & (df['CLOSE'] <= df['rolling_mean_20d'])
    cond_spx_filter = (df['spx_trend'] == "UPTREND") & (df['VVIX'] < df['VVIX'].rolling(50).mean())
    cond_squeeze = cond_squeeze_raw & (~cond_spx_filter)
    
    cond_breakout = (
        (df['CLOSE'] > df['rolling_mean_20d']) & 
        (df['CLOSE'].shift(1) <= df['rolling_mean_20d'].shift(1)) & 
        (df['VVIX'] > df['VVIX'].rolling(10).mean()) &
        (df['daily_return'] > 0.05)
    )
    
    cond_value = (df['z_score'] < -1.5) & (df['regime'].isin(['calm', 'normal']))
    cond_sell = (df['z_score'] > 2.0) | (df['rsi'] > 75)
    
    df.loc[cond_value, 'chart_signal'] = 'BUY_VALUE'
    df.loc[cond_squeeze, 'chart_signal'] = 'BUY_SQUEEZE'
    df.loc[cond_breakout, 'chart_signal'] = 'BUY_BREAKOUT'
    df.loc[cond_sell, 'chart_signal'] = 'SELL_EXTREME'

    # -------------------------------------------------------------------------
    # D. STATISTICS
    # -------------------------------------------------------------------------
    stats = {}
    
    buy_indices = df[df['chart_signal'].str.contains("BUY")].index
    latencies = []
    for idx in buy_indices[-50:]: 
        if idx + 30 < len(df):
            forward_window = df.iloc[idx+1 : idx+30]
            spike_found = forward_window[forward_window['daily_return'] > 0.10]
            if not spike_found.empty:
                latencies.append(spike_found.index[0] - idx)
            
    stats['median_latency'] = np.median(latencies) if latencies else 5 
    stats['learned_threshold'] = best_thresh
    
    current_month = df['DATE'].iloc[-1].month
    hist_spikes = df[df['daily_return'] > 0.10]
    month_spikes = hist_spikes[hist_spikes['DATE'].dt.month == current_month]
    if len(month_spikes) > 1:
        stats['seasonal_freq'] = int((month_spikes['DATE'].diff().dt.days).mean())
    else:
        stats['seasonal_freq'] = "N/A"

    if len(df) > 1:
        latest = df.iloc[-1]
        sig = "HOLD"
        reason = "Monitoring."
        
        if latest['VPE'] > 90:
             reason = f"⚠️ CRITICAL ENERGY. VPE Index is {latest['VPE']:.0f}/100."

        if "SQUEEZE" in latest['chart_signal']:
            sig = "BUY_SQUEEZE"
            reason = f"Volatility Compression. (AI Optimal Thresh: {best_thresh*100:.0f}%)"
        elif "BREAKOUT" in latest['chart_signal']:
            sig = "BUY_BREAKOUT"
            reason = "Momentum Breakout confirmed by VVIX."
        elif "VALUE" in latest['chart_signal']:
            sig = "BUY_VALUE"
            reason = "Deep Value (Mean Reversion)."
        elif "SELL" in latest['chart_signal']:
            sig = "SELL_EXTREME"
            reason = "Statistical Extreme."
            
        df.at[df.index[-1], "signal"] = sig
        df.at[df.index[-1], "signal_reason"] = reason

    return df, stats

# ==========================================
# 4. FORECASTING ENGINE
# ==========================================
def generate_monte_carlo_forecast(data, days_ahead=21, num_simulations=1000):
    current_vix = data['CLOSE'].iloc[-1]
    current_regime = data['regime'].iloc[-1]
    
    regime_data = data[data['regime'] == current_regime]
    daily_changes = regime_data['CLOSE'].diff().dropna()
    
    if len(daily_changes) < 10: daily_changes = data['CLOSE'].diff().dropna()

    simulation_results = np.zeros((days_ahead, num_simulations))
    
    for i in range(num_simulations):
        path = [current_vix]
        random_shocks = np.random.choice(daily_changes, days_ahead)
        for shock in random_shocks:
            drift = 0.05 * (19.5 - path[-1]) 
            next_price = max(9, path[-1] + drift + shock)
            path.append(next_price)
        simulation_results[:, i] = path[1:]
        
    last_date = data['DATE'].iloc[-1]
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
    
    return pd.DataFrame({
        'DATE': forecast_dates,
        'Forecast': np.median(simulation_results, axis=1),
        'Upper': np.percentile(simulation_results, 90, axis=1),
        'Lower': np.percentile(simulation_results, 10, axis=1)
    })

# Load Data
df, stats = load_and_process_data()

# -----------------------------------------------------------------------------
# 5. SIDEBAR
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
    show_forecast = st.checkbox("Show Forecast Model", value=True)

mask = (df['DATE'].dt.date >= start_date) & (df['DATE'].dt.date <= end_date)
df_filtered = df.loc[mask]

# -----------------------------------------------------------------------------
# 6. HEADER & METRICS
# -----------------------------------------------------------------------------
st.title("🛡️ VIX Spike Predictor Pro")
st.caption(f"Master Formula (VPE): VVIX / (Width * Price).")

# AI Brain Display
thresh_percent = stats['learned_threshold'] * 100
st.markdown(f"""
<div style="background-color: #0e1117; border: 1px solid #30363d; padding: 12px; border-radius: 8px; margin-bottom: 20px;">
    <small style="color: #8b949e;">🧠 <b>AI Continuous Learning:</b> Based on the last 12 months, the model has optimized the Squeeze Threshold to 
    <span style="color: #00e5ff; font-weight:bold;">{thresh_percent:.0f}%</span> (Percentile).</small>
</div>
""", unsafe_allow_html=True)

st.markdown(f"**Status:** {df.iloc[-1]['signal_reason']}")

latest = df.iloc[-1]
delta_vix = latest['CLOSE'] - df.iloc[-2]['CLOSE']

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("VIX Level", f"{latest['CLOSE']:.2f}", f"{delta_vix:.2f}")
with c2: st.metric("Regime", str(latest['regime']).upper(), delta_color="off")
with c3: st.metric("VPE Index", f"{latest['VPE']:.0f}/100", help="VIX Potential Energy. >90 is Danger Zone.")
with c4: st.metric("VVIX Level", f"{latest['VVIX']:.2f}")
with c5:
    sig = latest['signal']
    if "BUY" in sig: st.success(f"🔼 {sig}")
    elif "SELL" in sig: st.error(f"🔽 {sig}")
    else: st.info(f"⏸️ {sig}")

st.markdown("#### ⏳ Timing Intelligence")
t1, t2, t3, t4 = st.columns(4)
with t1: st.metric("Est. Latency", f"~{stats['median_latency']:.0f} Days")
with t2: st.metric("Suggested Option DTE", f"{int(stats['median_latency']*1.5)}-{int(stats['median_latency']*4)} Days")
with t3: st.metric(f"Seasonality", f"{stats.get('seasonal_freq', 'N/A')} Days")
with t4:
    last_spike = df_filtered[df_filtered['daily_return'] > 0.10].index.max()
    days_ago = (df_filtered.index.max() - last_spike) if pd.notna(last_spike) else 0
    st.metric("Recharge Meter", f"{days_ago} Days Ago")

# -----------------------------------------------------------------------------
# 7. CHARTING
# -----------------------------------------------------------------------------
fig = make_subplots(
    rows=4, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.03, 
    row_heights=[0.5, 0.15, 0.15, 0.2], # Optimized heights
    subplot_titles=("VIX Price & Forecast", "Z-Score (Deviation)", "Squeeze Metric (BB Width)", "⚡ VPE Index (Energy)")
)

# --- TRACE 1: Price ---
fig.add_trace(go.Scatter(x=df_filtered['DATE'], y=df_filtered['CLOSE'], mode='lines', name='VIX', line=dict(color='#00d4ff', width=2)), row=1, col=1)

if show_bollinger:
    fig.add_trace(go.Scatter(x=df_filtered['DATE'], y=df_filtered['bollinger_upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_filtered['DATE'], y=df_filtered['bollinger_lower'], fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)

# Forecast
if show_forecast:
    forecast_df = generate_monte_carlo_forecast(df, days_ahead=21)
    x_comb = list(forecast_df['DATE']) + list(forecast_df['DATE'])[::-1]
    y_comb = list(forecast_df['Upper']) + list(forecast_df['Lower'])[::-1]
    
    fig.add_trace(go.Scatter(x=x_comb, y=y_comb, fill='toself', fillcolor='rgba(0, 229, 255, 0.15)', line=dict(color='rgba(255,255,255,0)'), name='Forecast Range'), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast_df['DATE'], y=forecast_df['Forecast'], mode='lines', line=dict(color='#ffeb3b', width=2, dash='dash'), name='Forecast Median'), row=1, col=1)

# Signals
buy_sqz = df_filtered[df_filtered['chart_signal'] == "BUY_SQUEEZE"]
fig.add_trace(go.Scatter(x=buy_sqz['DATE'], y=buy_sqz['CLOSE'], mode='markers', marker=dict(symbol='star', size=14, color='orange', line=dict(width=1, color='white')), name='Squeeze Buy'), row=1, col=1)

buy_brk = df_filtered[df_filtered['chart_signal'] == "BUY_BREAKOUT"]
fig.add_trace(go.Scatter(x=buy_brk['DATE'], y=buy_brk['CLOSE'], mode='markers', marker=dict(symbol='diamond', size=12, color='#9d00ff', line=dict(width=1, color='white')), name='Breakout Buy'), row=1, col=1)

buy_val = df_filtered[df_filtered['chart_signal'] == "BUY_VALUE"]
fig.add_trace(go.Scatter(x=buy_val['DATE'], y=buy_val['CLOSE'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='#00ff00'), name='Value Buy'), row=1, col=1)

sell_sig = df_filtered[df_filtered['chart_signal'] == "SELL_EXTREME"]
fig.add_trace(go.Scatter(x=sell_sig['DATE'], y=sell_sig['CLOSE'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='#ff0000'), name='Sell'), row=1, col=1)

# --- TRACE 2: Z-Score ---
colors = np.where(df_filtered['z_score'] > 2, 'red', np.where(df_filtered['z_score'] < -1.5, 'green', 'gray'))
fig.add_trace(go.Bar(x=df_filtered['DATE'], y=df_filtered['z_score'], marker_color=colors, name='Z-Score'), row=2, col=1)
fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=2, col=1)

# --- TRACE 3: BB Width (Squeeze) ---
fig.add_trace(go.Scatter(x=df_filtered['DATE'], y=df_filtered['bb_width'], mode='lines', name='BB Width', line=dict(color='yellow', width=1)), row=3, col=1)
learned_val = df['bb_width'].quantile(stats['learned_threshold'])
fig.add_hline(y=learned_val, line_dash="dot", line_color="orange", annotation_text=f"AI Threshold ({stats['learned_threshold']*100:.0f}%)", row=3, col=1)

# --- TRACE 4: VPE INDEX (The New Formula) ---
# We plot it as a filled area chart. 0-100 scale.
fig.add_trace(go.Scatter(x=df_filtered['DATE'], y=df_filtered['VPE'], mode='lines', name='VPE Index', fill='tozeroy', line=dict(color='#d500f9', width=2)), row=4, col=1)
fig.add_hline(y=90, line_dash="dot", line_color="red", annotation_text="Danger", row=4, col=1)
fig.add_hline(y=10, line_dash="dot", line_color="green", annotation_text="Dormant", row=4, col=1)

fig.update_layout(height=1000, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 8. LEDGER
# -----------------------------------------------------------------------------
st.subheader("📋 Historical Ledger")
cols = ['DATE', 'CLOSE', 'chart_signal', 'VPE', 'bb_width', 'z_score']
df_display = df_filtered[cols].sort_values('DATE', ascending=False)

st.dataframe(
    df_display,
    use_container_width=True,
    height=400,
    hide_index=True,
    column_config={
        "DATE": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
        "CLOSE": st.column_config.NumberColumn("VIX", format="%.2f"),
        "VPE": st.column_config.ProgressColumn("VPE Index", min_value=0, max_value=100, format="%d"),
        "bb_width": st.column_config.NumberColumn("BB Width", format="%.3f"),
        "z_score": st.column_config.NumberColumn("Z-Score", format="%.2f"),
    }
)

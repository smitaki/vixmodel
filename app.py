import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VIX Spike Predictor Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background-color: #161b22; border: 1px solid #30363d;
        padding: 15px; border-radius: 8px; color: white;
    }
    .stExpander { border: 1px solid #30363d !important; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & MODELING (DYNAMIC INPUTS)
# -----------------------------------------------------------------------------
START_DATE = "2007-01-01" 

@st.cache_data(ttl=3600) # Cache CNN data for 1 hour
def get_cnn_fear_greed():
    """
    Fetches historical Fear & Greed Index directly from CNN's API.
    """
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        # Parse Historical Data
        history = data['fear_and_greed_historical']['data']
        fng_df = pd.DataFrame(history)
        
        # Convert Timestamp (ms) to DateTime
        fng_df['DATE'] = pd.to_datetime(fng_df['x'], unit='ms').dt.normalize()
        fng_df = fng_df.rename(columns={'y': 'FNG'})
        
        # Aggregate duplicates
        fng_df = fng_df.groupby('DATE')['FNG'].mean().reset_index()
        return fng_df[['DATE', 'FNG']]
        
    except Exception as e:
        print(f"Error fetching CNN data: {e}")
        return pd.DataFrame(columns=['DATE', 'FNG'])

@st.cache_data(ttl=60)
def load_and_process_data(lookback_days=20, fear_mult=1.0):
    # 1. Fetch Market Data
    tickers = ["^VIX", "^GSPC", "^VVIX"]
    data = yf.download(tickers, start=START_DATE, progress=False)
    
    if data.empty: return None, None
    
    # Clean Data
    close_df = data['Close'].copy() if isinstance(data.columns, pd.MultiIndex) else data['Close'].copy()
    df = close_df.reset_index().rename(columns={'Date': 'DATE', '^VIX': 'CLOSE', '^GSPC': 'SPX', '^VVIX': 'VVIX'})
    df['DATE'] = pd.to_datetime(df['DATE']).dt.normalize()
    df = df.sort_values("DATE").reset_index(drop=True)
    df['VVIX'] = df['VVIX'].ffill() 
    df.dropna(inplace=True)

    # 2. Fetch & Merge CNN Fear & Greed
    fng_df = get_cnn_fear_greed()
    if not fng_df.empty:
        df = pd.merge(df, fng_df, on='DATE', how='left')
        df['FNG'] = df['FNG'].ffill()
    else:
        df['FNG'] = 50 # Default neutral

    # 3. Indicators (Using Lookback Input)
    df["daily_return"] = df["CLOSE"].pct_change()
    
    # Dynamic Rolling Window based on user input
    df["rolling_mean"] = df["CLOSE"].rolling(lookback_days).mean()
    df["rolling_std"] = df["CLOSE"].rolling(lookback_days).std()
    
    df['bollinger_upper'] = df['rolling_mean'] + (2 * df['rolling_std'])
    df['bollinger_lower'] = df['rolling_mean'] - (2 * df['rolling_std'])
    df['bb_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['rolling_mean']
    
    # Momentum
    delta = df['CLOSE'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['z_score'] = (df['CLOSE'] - df['rolling_mean']) / df['rolling_std']
    
    # Context
    df['spx_ma50'] = df['SPX'].rolling(50).mean()
    df['spx_trend'] = np.where(df['SPX'] > df['spx_ma50'], "UPTREND", "FRAGILE")
    df['vvix_ma10'] = df['VVIX'].rolling(10).mean()
    df["regime"] = pd.qcut(df["CLOSE"], q=4, labels=["calm", "normal", "elevated", "stressed"])

    # --- VPE Index (Energy) WITH FEAR MULTIPLIER ---
    # Formula: (VVIX * Sensitivity) / (Width * Price)
    epsilon = 1e-6
    df['vpe_raw'] = (df['VVIX'] * fear_mult) / ((df['bb_width'] * df['CLOSE']) + epsilon)
    df['VPE'] = df['vpe_raw'].rolling(252).rank(pct=True) * 100

    # --- AI Learning (Walk-Forward Optimization) ---
    potential_thresholds = [0.05, 0.10, 0.15] 
    best_thresh = 0.10 
    best_score = -1
    training_data = df.tail(252).copy()
    
    for thresh in potential_thresholds:
        squeeze_level = training_data['bb_width'].quantile(thresh)
        signals = (training_data['bb_width'] < squeeze_level) & (training_data['CLOSE'] <= training_data['rolling_mean'])
        hits = 0
        signal_indices = training_data[signals].index
        for idx in signal_indices:
            if idx + 10 < len(training_data):
                if training_data.loc[idx+1:idx+10, 'daily_return'].max() > 0.10: hits += 1
        
        if len(signal_indices) > 0:
            score = (hits / len(signal_indices)) * np.log(len(signal_indices) + 1) 
            if score > best_score:
                best_score = score
                best_thresh = thresh

    stats = {'learned_threshold': best_thresh}
    return df, stats

# --- Signal Generator ---
def apply_signals(df, squeeze_threshold):
    df['chart_signal'] = "HOLD"
    df['confidence'] = 0.0
    
    learned_sqz_val = df['bb_width'].quantile(squeeze_threshold)
    
    cond_sqz = (df['bb_width'] < learned_sqz_val) & (df['CLOSE'] <= df['rolling_mean'])
    cond_spx = (df['spx_trend'] == "UPTREND") & (df['VVIX'] < df['VVIX'].rolling(50).mean())
    cond_sqz_final = cond_sqz & (~cond_spx)
    
    cond_vpe_buy = df['VPE'] > 90
    cond_vpe_sell = df['VPE'] < 10
    
    cond_val = (df['z_score'] < -1.5) & (df['regime'].isin(['calm', 'normal']))
    cond_sell_ext = (df['z_score'] > 2.0) | (df['rsi'] > 75)
    
    # Apply & Score
    df.loc[cond_val, 'chart_signal'] = 'BUY_VALUE'
    df.loc[cond_val, 'confidence'] = ((df.loc[cond_val, 'z_score'].abs() - 1.5) / 1.5).clip(0, 1) * 100
    
    safe_thresh = learned_sqz_val if learned_sqz_val > 0 else 0.001
    df.loc[cond_sqz_final, 'chart_signal'] = 'BUY_SQUEEZE'
    df.loc[cond_sqz_final, 'confidence'] = ((safe_thresh - df.loc[cond_sqz_final, 'bb_width']) / safe_thresh).clip(0, 1) * 100
    
    df.loc[cond_vpe_buy, 'chart_signal'] = 'BUY_VPE'      
    df.loc[cond_vpe_buy, 'confidence'] = ((df.loc[cond_vpe_buy, 'VPE'] - 90) / 10).clip(0, 1) * 100
    
    df.loc[cond_vpe_sell, 'chart_signal'] = 'SELL_VPE'    
    df.loc[cond_vpe_sell, 'confidence'] = ((10 - df.loc[cond_vpe_sell, 'VPE']) / 10).clip(0, 1) * 100
    
    df.loc[cond_sell_ext, 'chart_signal'] = 'SELL_EXTREME'
    df.loc[cond_sell_ext, 'confidence'] = 100
    
    return df

# --- Forecast Engine (Regime Override) ---
def generate_forecast(data, days_ahead=21, num_sims=1000, regime_mode="Auto-Detect"):
    curr_vix = data['CLOSE'].iloc[-1]
    
    # LOGIC: Filter historical data based on User Input
    if "Force Calm" in regime_mode:
        regime_data = data[data['CLOSE'] < 15]
    elif "Force Stressed" in regime_mode:
        regime_data = data[data['CLOSE'] > 25]
    elif "Force Crisis" in regime_mode:
        regime_data = data[data['CLOSE'] > 40]
    else:
        # Auto-Detect (Default)
        curr_regime = data['regime'].iloc[-1]
        regime_data = data[data['regime'] == curr_regime]

    daily_changes = regime_data['CLOSE'].diff().dropna()
    if len(daily_changes) < 10: daily_changes = data['CLOSE'].diff().dropna()

    sim_res = np.zeros((days_ahead, num_sims))
    for i in range(num_sims):
        path = [curr_vix]
        shocks = np.random.choice(daily_changes, days_ahead)
        for s in shocks:
            path.append(max(9, path[-1] + (0.05 * (19.5 - path[-1])) + s))
        sim_res[:, i] = path[1:]
        
    dates = [data['DATE'].iloc[-1] + timedelta(days=i) for i in range(1, days_ahead + 1)]
    return pd.DataFrame({
        'DATE': dates,
        'Forecast': np.median(sim_res, axis=1),
        'Upper': np.percentile(sim_res, 90, axis=1),
        'Lower': np.percentile(sim_res, 10, axis=1)
    })

# -----------------------------------------------------------------------------
# 3. SIDEBAR (CONTROLS)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # 1. Timeline
    default_date = pd.to_datetime("2023-01-01").date()
    today = datetime.now().date()
    date_range = st.slider("Timeline", pd.to_datetime("2020-01-01").date(), today, (default_date, today))
    
    st.markdown("---")
    
    # 2. SCENARIO LAB (NEW INPUTS)
    st.subheader("🧪 Scenario Lab")
    
    # Input A: Regime Override
    regime_override = st.selectbox(
        "Forecast Mode", 
        ["Auto-Detect", "Force Calm (<15)", "Force Stressed (>25)", "Force Crisis (>40)"],
        help="Simulate how the forecast looks if we were in a different market regime."
    )
    
    # Input B: Lookback Window
    lookback = st.slider(
        "Trend Baseline (Days)", 
        10, 50, 20, 
        help="Lower = Faster signals (Scalping). Higher = Slower signals (Investing)."
    )
    
    # Input C: Fear Sensitivity
    fear_mult = st.slider(
        "Fear Sensitivity", 
        0.5, 2.0, 1.0, 0.1, 
        help="Multiplies VVIX impact. Increase to catch early/weak signals."
    )
    
    st.markdown("---")

    # 3. Model Sensitivity (Existing)
    st.subheader("⚙️ Model Settings")
    mode = st.radio("Threshold Strategy", ["AI Auto-Pilot", "Manual Override"], index=0)
    manual_thresh = 0.10
    if mode == "Manual Override":
        manual_thresh = st.slider("Manual Squeeze %", 0.01, 0.20, 0.10, 0.01)
    
    st.markdown("---")
    
    # 4. Visual Aids
    st.subheader("👀 Visuals")
    show_forecast = st.toggle("Show Forecast", value=True)
    show_backtest = st.toggle("Show Backtest", value=False)

# -----------------------------------------------------------------------------
# 4. EXECUTION & DASHBOARD
# -----------------------------------------------------------------------------
# Load Data with User Inputs
raw_df, ai_stats = load_and_process_data(lookback_days=lookback, fear_mult=fear_mult)

if raw_df is None:
    st.stop()

# Determine active threshold
active_thresh = ai_stats['learned_threshold'] if mode == "AI Auto-Pilot" else manual_thresh

# Apply Logic
df = apply_signals(raw_df.copy(), active_thresh)
mask = (df['DATE'].dt.date >= date_range[0]) & (df['DATE'].dt.date <= date_range[1])
df_filtered = df.loc[mask]

# Dashboard Header
st.title("🛡️ VIX Spike Predictor Pro")
st.caption("AI-Driven Volatility Intelligence System")

last = df.iloc[-1]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("VIX Level", f"{last['CLOSE']:.2f}", f"{last['CLOSE'] - df.iloc[-2]['CLOSE']:.2f}")
c2.metric("Regime", str(last['regime']).upper(), delta_color="off")
# Show Fear & Greed in Header
c3.metric("Fear & Greed", f"{last['FNG']:.0f}", help="CNN Index (0-100). <25 Extreme Fear, >75 Extreme Greed.")
c4.metric("VPE Index", f"{last['VPE']:.0f}/100", "Energy")
sig_icon = "🟢" if "BUY" in last['chart_signal'] else "🔴" if "SELL" in last['chart_signal'] else "⚪"
c5.metric("Model Signal", last['chart_signal'].replace("_", " "), sig_icon)

# Charts
with st.expander("ℹ️ How to read the Price & Signal Chart"):
    st.markdown("""
    * **Purple Diamonds (VPE Buy):** Critical Energy (>90). The spring is coiled tight. Spike imminent.
    * **Orange Stars (Squeeze):** Low Volatility. Bands are tight.
    * **Grey Cross (VPE Sell):** Energy Dissipated (<10). Market is dormant or exhausted.
    * **Red Triangles (Sell Extreme):** Statistical Over-extension (Z-Score > 2).
    """)

fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True, 
    row_heights=[0.5, 0.15, 0.15, 0.2], vertical_spacing=0.03,
    subplot_titles=("VIX Price Action", "Z-Score", "Bollinger Width", "CNN Fear & Greed Index")
)

# 1. Price
fig.add_trace(go.Scatter(x=df_filtered['DATE'], y=df_filtered['CLOSE'], mode='lines', line=dict(color='#00d4ff', width=2), name='VIX'), row=1, col=1)

if show_backtest:
    for idx, row in df_filtered.iterrows():
        if "BUY" in row['chart_signal']:
            future = df[df['DATE'] > row['DATE']].head(10)
            color = "rgba(0, 255, 0, 0.1)" if (not future.empty and future['daily_return'].max() > 0.10) else "rgba(255, 0, 0, 0.1)"
            fig.add_vrect(x0=row['DATE'], x1=row['DATE'] + timedelta(days=5), fillcolor=color, layer="below", line_width=0, row=1, col=1)

if show_forecast:
    # Pass the REGIME OVERRIDE to the forecast engine
    f_df = generate_forecast(df, 21, regime_mode=regime_override)
    x = list(f_df['DATE']) + list(f_df['DATE'])[::-1]
    y = list(f_df['Upper']) + list(f_df['Lower'])[::-1]
    fig.add_trace(go.Scatter(x=x, y=y, fill='toself', fillcolor='rgba(0,229,255,0.1)', line=dict(width=0), name='Forecast'), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_df['DATE'], y=f_df['Forecast'], line=dict(dash='dash', color='yellow'), name='Median'), row=1, col=1)

# Markers
sigs = {'BUY_VPE': ('diamond', '#d500f9', 12), 'BUY_SQUEEZE': ('star', 'orange', 14), 
        'SELL_VPE': ('cross', 'gray', 10), 'SELL_EXTREME': ('triangle-down', 'red', 10), 'BUY_VALUE': ('triangle-up', '#00ff00', 10)}

for key, (sym, col, size) in sigs.items():
    d = df_filtered[df_filtered['chart_signal'] == key]
    fig.add_trace(go.Scatter(x=d['DATE'], y=d['CLOSE'], mode='markers', marker=dict(symbol=sym, size=size, color=col, line=dict(width=1, color='white')), name=key), row=1, col=1)

# 2. Z-Score
colors = np.where(df_filtered['z_score'] > 2, 'red', np.where(df_filtered['z_score'] < -1.5, 'green', 'gray'))
fig.add_trace(go.Bar(x=df_filtered['DATE'], y=df_filtered['z_score'], marker_color=colors, name='Z-Score'), row=2, col=1)
fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=2, col=1)

# 3. Squeeze
fig.add_trace(go.Scatter(x=df_filtered['DATE'], y=df_filtered['bb_width'], line=dict(color='yellow', width=1), name='BB Width'), row=3, col=1)
fig.add_hline(y=df['bb_width'].quantile(active_thresh), line_dash="dot", line_color="orange", row=3, col=1)

# 4. CNN Fear & Greed (Chart 4)
# Using a color gradient for Fear (Red) to Greed (Green)
# Since Plotly lines are single color, we plot it cyan but add colored zones
fig.add_trace(go.Scatter(x=df_filtered['DATE'], y=df_filtered['FNG'], fill='tozeroy', line=dict(color='#00e5ff', width=2), name='Fear & Greed'), row=4, col=1)
fig.add_hline(y=25, line_dash="dot", line_color="red", annotation_text="Extreme Fear", row=4, col=1)
fig.add_hline(y=75, line_dash="dot", line_color="green", annotation_text="Extreme Greed", row=4, col=1)

fig.update_layout(height=1000, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# Ledger
st.subheader("📋 Signal Ledger")
st.dataframe(
    df_filtered[['DATE', 'CLOSE', 'chart_signal', 'confidence', 'FNG', 'VPE']].sort_values('DATE', ascending=False), 
    use_container_width=True, height=300, hide_index=True,
    column_config={
        "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%d%%"),
        "FNG": st.column_config.NumberColumn("Fear/Greed", format="%.0f"),
        "VPE": st.column_config.NumberColumn("VPE Energy", format="%.0f")
    }
)

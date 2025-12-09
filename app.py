import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
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

# Custom CSS for "Pro" look
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
# 2. DATA LOADING & MODELING
# -----------------------------------------------------------------------------
START_DATE = "2007-01-01" 

@st.cache_data(ttl=60)
def load_and_process_data():
    tickers = ["^VIX", "^GSPC", "^VVIX"]
    data = yf.download(tickers, start=START_DATE, progress=False)
    
    if data.empty: return None, None
    
    # Clean Data
    close_df = data['Close'].copy() if isinstance(data.columns, pd.MultiIndex) else data['Close'].copy()
    df = close_df.reset_index().rename(columns={'Date': 'DATE', '^VIX': 'CLOSE', '^GSPC': 'SPX', '^VVIX': 'VVIX'})
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values("DATE").reset_index(drop=True)
    df['VVIX'] = df['VVIX'].ffill() 
    df.dropna(inplace=True)

    # --- Indicators ---
    df["daily_return"] = df["CLOSE"].pct_change()
    df["rolling_mean_20d"] = df["CLOSE"].rolling(20).mean()
    df["rolling_std_20d"] = df["CLOSE"].rolling(20).std()
    
    df['bollinger_upper'] = df['rolling_mean_20d'] + (2 * df['rolling_std_20d'])
    df['bollinger_lower'] = df['rolling_mean_20d'] - (2 * df['rolling_std_20d'])
    df['bb_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['rolling_mean_20d']
    
    delta = df['CLOSE'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['z_score'] = (df['CLOSE'] - df['rolling_mean_20d']) / df['rolling_std_20d']
    
    # Context
    df['spx_ma50'] = df['SPX'].rolling(50).mean()
    df['spx_trend'] = np.where(df['SPX'] > df['spx_ma50'], "UPTREND", "FRAGILE")
    df['vvix_ma10'] = df['VVIX'].rolling(10).mean()
    df["regime"] = pd.qcut(df["CLOSE"], q=4, labels=["calm", "normal", "elevated", "stressed"])

    # --- VPE Index (Energy) ---
    epsilon = 1e-6
    df['vpe_raw'] = df['VVIX'] / ((df['bb_width'] * df['CLOSE']) + epsilon)
    # Normalize 0-100
    df['VPE'] = df['vpe_raw'].rolling(252).rank(pct=True) * 100

    # --- AI Learning (Walk-Forward) ---
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
                if training_data.loc[idx+1:idx+10, 'daily_return'].max() > 0.10: hits += 1
        
        if len(signal_indices) > 0:
            score = (hits / len(signal_indices)) * np.log(len(signal_indices) + 1) 
            if score > best_score:
                best_score = score
                best_thresh = thresh

    stats = {'learned_threshold': best_thresh}
    return df, stats

# --- Signal Generator (Dynamic) ---
def apply_signals(df, squeeze_threshold):
    df['chart_signal'] = "HOLD"
    
    learned_sqz_val = df['bb_width'].quantile(squeeze_threshold)
    
    # 1. Squeeze Buy (Standard)
    cond_sqz = (df['bb_width'] < learned_sqz_val) & (df['CLOSE'] <= df['rolling_mean_20d'])
    cond_spx = (df['spx_trend'] == "UPTREND") & (df['VVIX'] < df['VVIX'].rolling(50).mean())
    cond_sqz_final = cond_sqz & (~cond_spx)
    
    # 2. VPE Signals (Master Energy)
    # BUY: High Potential Energy (>90)
    cond_vpe_buy = df['VPE'] > 90
    
    # SELL: Energy Dissipated (<10)
    cond_vpe_sell = df['VPE'] < 10
    
    # 3. Standard Value/Extreme Sell
    cond_val = (df['z_score'] < -1.5) & (df['regime'].isin(['calm', 'normal']))
    cond_sell_ext = (df['z_score'] > 2.0) | (df['rsi'] > 75)
    
    # Apply Signals (Order matters: later overwrites earlier if conflict)
    df.loc[cond_val, 'chart_signal'] = 'BUY_VALUE'
    df.loc[cond_sqz_final, 'chart_signal'] = 'BUY_SQUEEZE'
    df.loc[cond_vpe_buy, 'chart_signal'] = 'BUY_VPE'      # Replaces Breakout
    
    df.loc[cond_vpe_sell, 'chart_signal'] = 'SELL_VPE'    # New Sell Logic
    df.loc[cond_sell_ext, 'chart_signal'] = 'SELL_EXTREME'
    
    return df

# --- Forecast Engine ---
def generate_forecast(data, days_ahead=21, num_sims=1000):
    curr_vix = data['CLOSE'].iloc[-1]
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

# Load Initial Data
raw_df, ai_stats = load_and_process_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR & CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Date Range
    min_d, max_d = raw_df['DATE'].min().date(), raw_df['DATE'].max().date()
    def_start = max(min_d, pd.to_datetime("2023-01-01").date())
    date_range = st.slider("Timeline", min_d, max_d, (def_start, max_d))
    
    st.markdown("---")
    st.subheader("⚙️ Model Sensitivity")
    
    # Override Mode
    mode = st.radio("Threshold Strategy", ["AI Auto-Pilot", "Manual Override"], index=0)
    
    if mode == "AI Auto-Pilot":
        active_thresh = ai_stats['learned_threshold']
        st.info(f"🤖 AI Selected: {active_thresh*100:.0f}% Percentile")
    else:
        active_thresh = st.slider("Manual Squeeze %", 0.01, 0.20, 0.10, 0.01)
    
    st.markdown("---")
    st.subheader("👀 Visual Aids")
    show_forecast = st.toggle("Show Future Forecast", value=True)
    show_backtest = st.toggle("Show Backtest Results", value=False, help="Colors background Green/Red if signal succeeded.")

# Apply Filters & Signals
df = apply_signals(raw_df.copy(), active_thresh)
mask = (df['DATE'].dt.date >= date_range[0]) & (df['DATE'].dt.date <= date_range[1])
df_filtered = df.loc[mask]

# Stats Calculation
buy_idx = df[df['chart_signal'].str.contains("BUY")].index
latencies = []
for idx in buy_idx[-50:]: 
    if idx+30 < len(df):
        fw = df.iloc[idx+1:idx+30]
        if not fw[fw['daily_return'] > 0.10].empty:
            latencies.append(fw[fw['daily_return'] > 0.10].index[0] - idx)
med_latency = np.median(latencies) if latencies else 5

# -----------------------------------------------------------------------------
# 4. DASHBOARD HEADER
# -----------------------------------------------------------------------------
st.title("🛡️ VIX Spike Predictor Pro")
st.caption("AI-Driven Volatility Intelligence System")

# Status Bar
last = df.iloc[-1]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("VIX Level", f"{last['CLOSE']:.2f}", f"{last['CLOSE'] - df.iloc[-2]['CLOSE']:.2f}")
c2.metric("Regime", str(last['regime']).upper(), delta_color="off")
c3.metric("VPE Index", f"{last['VPE']:.0f}/100", "Energy")
c4.metric("VVIX Level", f"{last['VVIX']:.2f}")

sig_icon = "🟢" if "BUY" in last['signal'] else "🔴" if "SELL" in last['signal'] else "⚪"
c5.metric("Model Signal", last['chart_signal'].replace("_", " "), sig_icon)

# -----------------------------------------------------------------------------
# 5. CHARTS WITH "HOW TO READ" EXPANDERS
# -----------------------------------------------------------------------------

# --- CHART 1: MAIN PRICE & SIGNALS ---
with st.expander("ℹ️ How to read the Price & Signal Chart"):
    st.markdown("""
    * **Purple Diamonds (VPE Buy):** Critical Energy (>90). The spring is coiled tight. Spike imminent.
    * **Orange Stars (Squeeze):** Low Volatility. Bands are tight.
    * **Grey Cross (VPE Sell):** Energy Dissipated (<10). Market is dormant or exhausted.
    * **Red Triangles (Sell Extreme):** Statistical Over-extension (Z-Score > 2).
    """)

fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.15, 0.2], vertical_spacing=0.03,
                    subplot_titles=("VIX Price Action", "Z-Score", "Bollinger Width", "VPE Index"))

# Price
fig.add_trace(go.Scatter(x=df_filtered['DATE'], y=df_filtered['CLOSE'], mode='lines', line=dict(color='#00d4ff', width=2), name='VIX'), row=1, col=1)

# Backtest Overlay
if show_backtest:
    for idx, row in df_filtered.iterrows():
        if "BUY" in row['chart_signal']:
            future = df[df['DATE'] > row['DATE']].head(10)
            if not future.empty and future['daily_return'].max() > 0.10:
                color = "rgba(0, 255, 0, 0.1)" # Green Success
            else:
                color = "rgba(255, 0, 0, 0.1)" # Red Fail
            fig.add_vrect(x0=row['DATE'], x1=row['DATE'] + timedelta(days=5), fillcolor=color, layer="below", line_width=0, row=1, col=1)

# Forecast
if show_forecast:
    f_df = generate_forecast(df, 21)
    x = list(f_df['DATE']) + list(f_df['DATE'])[::-1]
    y = list(f_df['Upper']) + list(f_df['Lower'])[::-1]
    fig.add_trace(go.Scatter(x=x, y=y, fill='toself', fillcolor='rgba(0,229,255,0.1)', line=dict(width=0), name='Forecast'), row=1, col=1)
    fig.add_trace(go.Scatter(x=f_df['DATE'], y=f_df['Forecast'], line=dict(dash='dash', color='yellow'), name='Median'), row=1, col=1)

# Markers
sigs = {
    'BUY_VPE': ('diamond', '#d500f9', 12),      # New VPE Buy (Purple)
    'BUY_SQUEEZE': ('star', 'orange', 14),      # Classic Squeeze
    'SELL_VPE': ('cross', 'gray', 10),          # New VPE Sell (Grey)
    'SELL_EXTREME': ('triangle-down', 'red', 10),
    'BUY_VALUE': ('triangle-up', '#00ff00', 10)
}

for key, (sym, col, size) in sigs.items():
    d = df_filtered[df_filtered['chart_signal'] == key]
    fig.add_trace(go.Scatter(x=d['DATE'], y=d['CLOSE'], mode='markers', marker=dict(symbol=sym, size=size, color=col, line=dict(width=1, color='white')), name=key), row=1, col=1)

# --- CHART 2: Z-SCORE ---
colors = np.where(df_filtered['z_score'] > 2, 'red', np.where(df_filtered['z_score'] < -1.5, 'green', 'gray'))
fig.add_trace(go.Bar(x=df_filtered['DATE'], y=df_filtered['z_score'], marker_color=colors, name='Z-Score'), row=2, col=1)
fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=2, col=1)

# --- CHART 3: SQUEEZE ---
fig.add_trace(go.Scatter(x=df_filtered['DATE'], y=df_filtered['bb_width'], line=dict(color='yellow', width=1), name='BB Width'), row=3, col=1)
fig.add_hline(y=df['bb_width'].quantile(active_thresh), line_dash="dot", line_color="orange", annotation_text=f"Trigger: {active_thresh*100:.0f}%", row=3, col=1)

# --- CHART 4: VPE INDEX ---
fig.add_trace(go.Scatter(x=df_filtered['DATE'], y=df_filtered['VPE'], fill='tozeroy', line=dict(color='#d500f9', width=2), name='VPE'), row=4, col=1)
fig.add_hline(y=90, line_dash="dot", line_color="red", annotation_text="Critcial (>90)", row=4, col=1)
fig.add_hline(y=10, line_dash="dot", line_color="gray", annotation_text="Dormant (<10)", row=4, col=1)

fig.update_layout(height=1000, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. EXPLANATION SECTION (VPE)
# -----------------------------------------------------------------------------
with st.expander("⚡ Understanding the VPE (Potential Energy) Index"):
    st.markdown("""
    **Formula:** $VPE = \\frac{VVIX}{Width \\times Price}$
    
    * **BUY (>90):** The VPE signal triggers when Fear (VVIX) is high but Price/Width are suppressed. This is a "coiled spring."
    * **SELL (<10):** The signal drops when the energy dissipates (Price spikes and Bands widen, or Fear drops to zero).
    """)

# -----------------------------------------------------------------------------
# 7. LEDGER
# -----------------------------------------------------------------------------
st.subheader("📋 Signal Ledger")
st.dataframe(df_filtered[['DATE', 'CLOSE', 'chart_signal', 'VPE', 'z_score']].sort_values('DATE', ascending=False), use_container_width=True, height=300)

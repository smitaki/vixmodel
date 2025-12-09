import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# ==========================================
# 1. CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(page_title="VIX Spike Predictor Pro", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS to match your Dark/Cyan theme
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background-color: #161b22;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
        text-align: center;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #ffffff; }
    .metric-label { font-size: 14px; color: #8b949e; }
    .highlight-cyan { color: #00e5ff; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA GENERATION (REPLACE WITH YOUR DATA)
# ==========================================
# NOTE: This function generates fake data so the code is runnable immediately.
# Replace this entire block with your own data loading logic.

@st.cache_data
def load_data():
    # Create 5 years of mock daily dates
    dates = pd.date_range(end=pd.Timestamp.today(), periods=1200)
    
    # Generate Synthetic VIX-like data (Mean reversion + Spikes)
    np.random.seed(42)
    vix_values = [20]
    regimes = []
    
    for i in range(1, len(dates)):
        prev_vix = vix_values[-1]
        
        # Simple Regime Logic for Demo
        if prev_vix < 15: regime = 'COMPLACENT'
        elif prev_vix > 25: regime = 'HIGH_VOL'
        else: regime = 'NORMAL'
        regimes.append(regime)
        
        # Mean reverting random walk
        shock = np.random.normal(0, 1.5 if regime == 'HIGH_VOL' else 0.8)
        change = 0.1 * (19 - prev_vix) + shock
        vix_values.append(max(10, prev_vix + change))
        
    df = pd.DataFrame({'Date': dates, 'VIX': vix_values})
    df['Regime'] = ['NORMAL'] + regimes # Pad first entry
    
    # Calculate indicators (BB Width, Z-Score) mimicking your image
    df['MA20'] = df['VIX'].rolling(20).mean()
    df['STD20'] = df['VIX'].rolling(20).std()
    df['UpperBB'] = df['MA20'] + (2 * df['STD20'])
    df['LowerBB'] = df['MA20'] - (2 * df['STD20'])
    df['BB_Width'] = df['UpperBB'] - df['LowerBB']
    df['Z_Score'] = (df['VIX'] - df['MA20']) / df['STD20']
    
    return df

df = load_data()

# ==========================================
# 3. FORECASTING ENGINE (THE NEW FEATURE)
# ==========================================
def generate_monte_carlo_forecast(data, days_ahead=21, num_simulations=1000):
    """
    Simulates future VIX paths based on historical volatility of the CURRENT Regime.
    """
    current_vix = data['VIX'].iloc[-1]
    current_regime = data['Regime'].iloc[-1]
    
    # 1. Get historical daily changes for this specific regime
    # (If your real data has 'Regime' columns, use them here)
    regime_data = data[data['Regime'] == current_regime]
    
    # Calculate daily returns/changes
    daily_changes = regime_data['VIX'].diff().dropna()
    
    if len(daily_changes) < 10:
        # Fallback if not enough data
        daily_changes = data['VIX'].diff().dropna()

    # 2. Monte Carlo Simulation
    simulation_results = np.zeros((days_ahead, num_simulations))
    
    for i in range(num_simulations):
        path = [current_vix]
        # Randomly sample from history
        random_shocks = np.random.choice(daily_changes, days_ahead)
        
        for shock in random_shocks:
            # Add simple mean reversion drift to prevent exploding paths
            drift = 0.05 * (19.0 - path[-1]) 
            next_price = path[-1] + drift + shock
            # Floor VIX at 9 (historical low)
            next_price = max(9, next_price)
            path.append(next_price)
            
        simulation_results[:, i] = path[1:]
        
    # 3. Aggregate results
    forecast_dates = [data['Date'].iloc[-1] + timedelta(days=i) for i in range(1, days_ahead + 1)]
    
    median_path = np.median(simulation_results, axis=1)
    upper_bound = np.percentile(simulation_results, 90, axis=1) # 90th percentile
    lower_bound = np.percentile(simulation_results, 10, axis=1) # 10th percentile
    
    return pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': median_path,
        'Upper': upper_bound,
        'Lower': lower_bound
    })

# ==========================================
# 4. UI LAYOUT
# ==========================================

# -- HEADER --
st.title("🛡️ VIX Spike Predictor Pro")
st.caption("Mathematical Model: Derived entirely from historical market data.")

# -- TOP METRICS ROW --
latest = df.iloc[-1]
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("VIX Level", f"{latest['VIX']:.2f}", "+1.20")
with c2:
    st.metric("Regime", latest['Regime'], delta_color="off")
with c3:
    st.metric("SPI Level", "UPTREND")
with c4:
    st.metric("IVR Level", "97.35")
with c5:
    # Forecast Toggle
    show_forecast = st.checkbox("Show Forecast Model", value=True)

# -- TIMING INTELLIGENCE ROW --
st.markdown("##### ⏳ Timing Intelligence (Derived from History)")
t1, t2, t3, t4 = st.columns(4)
t1.metric("Est. Latency", "~8 Days")
t2.metric("Suggested Option DTE", "12-32 Days")
t3.metric("Inversion Seasonality", "270 Days")
t4.metric("Recharge Meter", "11 Days Ago")

# ==========================================
# 5. CHARTING (PLOTLY)
# ==========================================
st.markdown("---")

# Filter data for better view (last 180 days)
chart_data = df.tail(252).copy()

fig = go.Figure()

# A. Historical Data
fig.add_trace(go.Scatter(
    x=chart_data['Date'], y=chart_data['VIX'],
    mode='lines', name='VIX History',
    line=dict(color='#00e5ff', width=2)
))

# B. Forecast (If Enabled)
if show_forecast:
    forecast_df = generate_monte_carlo_forecast(df, days_ahead=30)
    
    # 1. The Cone (Confidence Interval)
    # Combine upper and lower bounds for the "filled" area effect
    x_combined = list(forecast_df['Date']) + list(forecast_df['Date'])[::-1]
    y_combined = list(forecast_df['Upper']) + list(forecast_df['Lower'])[::-1]
    
    fig.add_trace(go.Scatter(
        x=x_combined, y=y_combined,
        fill='toself',
        fillcolor='rgba(0, 229, 255, 0.15)', # Transparent Cyan
        line=dict(color='rgba(255,255,255,0)'),
        name='Forecast Range (10-90%)',
        showlegend=True
    ))

    # 2. The Median Projection
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'], y=forecast_df['Forecast'],
        mode='lines',
        line=dict(color='#ffeb3b', width=2, dash='dash'), # Yellow Dashed
        name='Forecast Median'
    ))

# C. Styling (To match your screenshot)
fig.update_layout(
    title="VIX Price & Forecast Signals",
    template="plotly_dark",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    height=500,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='#30363d'),
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
)

st.plotly_chart(fig, use_container_width=True)

# -- SECONDARY CHARTS (Z-Score & Squeeze) --
z_fig = go.Figure()
z_fig.add_trace(go.Bar(
    x=chart_data['Date'], y=chart_data['Z_Score'],
    marker_color=chart_data['Z_Score'].apply(lambda x: '#ef5350' if x > 0 else '#66bb6a'),
    name='Z-Score'
))
z_fig.update_layout(title="Z-Score (Deviation)", template="plotly_dark", height=200, margin=dict(t=30, b=10))
st.plotly_chart(z_fig, use_container_width=True)

sq_fig = go.Figure()
sq_fig.add_trace(go.Scatter(
    x=chart_data['Date'], y=chart_data['BB_Width'],
    line=dict(color='#ffeb3b'), fill='tozeroy',
    fillcolor='rgba(255, 235, 59, 0.1)',
    name='BB Width'
))
sq_fig.update_layout(title="Squeeze Metric (BB Width)", template="plotly_dark", height=200, margin=dict(t=30, b=10))
st.plotly_chart(sq_fig, use_container_width=True)

# -- HISTORICAL LEDGER --
st.markdown("### 📒 Historical Ledger")
st.dataframe(df.tail(10)[::-1], use_container_width=True)

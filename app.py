import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
# 2. DATA LOADING (REPLACE WITH YOUR ACTUAL DATA LOGIC)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # --- MOCK DATA GENERATION (DELETE THIS BLOCK IN PROD) ---
    dates = pd.date_range(start="2024-01-01", end="2025-12-08", freq="B")
    n = len(dates)
    
    # Simulate VIX-like Mean Reverting Series
    np.random.seed(42)
    prices = [15.0]
    for _ in range(n-1):
        change = np.random.normal(0, 0.5) + 0.1 * (15 - prices[-1]) # Mean reversion to 15
        prices.append(max(10, prices[-1] + change))
        
    df = pd.DataFrame({'Date': dates, 'Close': prices})
    
    # Calculate Indicators (Replicating your logic roughly)
    df['rolling_mean_20d'] = df['Close'].rolling(20).mean()
    df['rolling_std_20d'] = df['Close'].rolling(20).std()
    df['bollinger_upper'] = df['rolling_mean_20d'] + (2 * df['rolling_std_20d'])
    df['bollinger_lower'] = df['rolling_mean_20d'] - (2 * df['rolling_std_20d'])
    
    # VCR / Z-Score mock calculation
    df['z_score'] = (df['Close'] - df['rolling_mean_20d']) / df['rolling_std_20d']
    df['vcr'] = (df['Close'] / df['rolling_mean_20d'] - 1) * 100  # Example metric
    
    # Logic for Signal
    df['regime'] = np.where(df['Close'] > 20, 'Elevated', 'Normal')
    df['signal'] = np.where(df['z_score'] > 2.0, 'SPIKE WARNING', 'HOLD')
    df['daily_return'] = df['Close'].pct_change()
    
    return df.sort_values('Date', ascending=True).reset_index(drop=True)

# Load data
df = load_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    
    # Date Filter
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    start_date, end_date = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    
    st.markdown("### Model Parameters")
    st.info("Adjusting these affects the chart visualization.")
    show_bollinger = st.checkbox("Show Bollinger Bands", value=True)
    show_signals = st.checkbox("Show Spike Signals", value=True)

# Filter data based on selection
mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
df_filtered = df.loc[mask]

# -----------------------------------------------------------------------------
# 4. KPI HEADER (METRICS)
# -----------------------------------------------------------------------------
st.title("🛡️ VIX Spike Predictor")
st.markdown("### Standardized Volatility Analysis & Historical Analogs")

# Get latest values
latest = df.iloc[-1]
prev = df.iloc[-2]
delta_vix = latest['Close'] - prev['Close']

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("VIX Level", f"{latest['Close']:.2f}", f"{delta_vix:.2f}")
with col2:
    regime_color = "normal" if latest['regime'] == 'Normal' else "inverse"
    st.metric("Regime", latest['regime'], delta_color="off")
with col3:
    st.metric("VCR Metric", f"{latest['vcr']:.2f}%", help="Volatility Convergence Ratio")
with col4:
    st.metric("Z-Score", f"{latest['z_score']:.2f}σ")
with col5:
    # Highlight signal if active
    if latest['signal'] != 'HOLD':
        st.error(f"⚠️ {latest['signal']}")
    else:
        st.success(latest['signal'])

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. DYNAMIC CHART (PLOTLY)
# -----------------------------------------------------------------------------
# Create subplots: Row 1 = Price, Row 2 = Z-Score/Indicator
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    row_heights=[0.7, 0.3], # Main chart takes 70% space
    subplot_titles=("VIX Price Action & Bollinger Bands", "Z-Score (Deviation Metric)")
)

# --- TRACE 1: VIX Close Price ---
fig.add_trace(go.Scatter(
    x=df_filtered['Date'], 
    y=df_filtered['Close'], 
    mode='lines', 
    name='VIX',
    line=dict(color='#00d4ff', width=2)
), row=1, col=1)

# --- TRACE 2: Bollinger Bands (Optional) ---
if show_bollinger:
    # Upper Band
    fig.add_trace(go.Scatter(
        x=df_filtered['Date'], y=df_filtered['bollinger_upper'],
        mode='lines', line=dict(width=1, color='rgba(150, 150, 150, 0.5)'),
        showlegend=False, name='Upper BB'
    ), row=1, col=1)
    
    # Lower Band (Filled)
    fig.add_trace(go.Scatter(
        x=df_filtered['Date'], y=df_filtered['bollinger_lower'],
        mode='lines', line=dict(width=1, color='rgba(150, 150, 150, 0.5)'),
        fill='tonexty', # Fills area between Upper and Lower
        fillcolor='rgba(150, 150, 150, 0.1)',
        showlegend=False, name='Lower BB'
    ), row=1, col=1)

# --- TRACE 3: Buy/Spike Signals ---
if show_signals:
    # Filter only rows where signal is NOT 'HOLD'
    signals = df_filtered[df_filtered['signal'] != 'HOLD']
    fig.add_trace(go.Scatter(
        x=signals['Date'], y=signals['Close'],
        mode='markers', 
        marker=dict(symbol='triangle-up', color='red', size=12, line=dict(width=2, color='white')),
        name='Spike Alert'
    ), row=1, col=1)

# --- TRACE 4: Z-Score (Bottom Panel) ---
fig.add_trace(go.Bar(
    x=df_filtered['Date'], y=df_filtered['z_score'],
    name='Z-Score',
    marker_color=np.where(df_filtered['z_score'] > 2, 'red', 'gray') # Color spikes red
), row=2, col=1)

# Add "Danger Zone" line on subplot
fig.add_hline(y=2.0, line_dash="dot", line_color="red", annotation_text="Danger (>2σ)", row=2, col=1)

# --- LAYOUT POLISH ---
fig.update_layout(
    height=600,
    margin=dict(l=20, r=20, t=30, b=20),
    template="plotly_dark",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02, xanchor="right", x=1)
)

# Render Chart
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. ENHANCED DATA TABLE
# -----------------------------------------------------------------------------
st.subheader("📋 Historical Data Ledger")

# Select clean columns for display
display_cols = ['Date', 'Close', 'signal', 'regime', 'vcr', 'z_score', 'daily_return', 'rolling_std_20d']

# Sort by newest first
df_display = df_filtered[display_cols].sort_values('Date', ascending=False)

st.dataframe(
    df_display,
    use_container_width=True,
    height=400,
    hide_index=True,
    column_config={
        "Date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
        "Close": st.column_config.NumberColumn("VIX Level", format="%.2f"),
        "daily_return": st.column_config.NumberColumn("1D Chg", format="%.2f%%"),
        "vcr": st.column_config.NumberColumn("VCR Metric", format="%.2f%%"),
        "z_score": st.column_config.NumberColumn("Z-Score", format="%.2f"),
        "signal": st.column_config.TextColumn("Signal", help="Model Trigger"),
        "regime": st.column_config.TextColumn("Regime", width="small"),
    }
)

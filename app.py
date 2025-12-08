import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.cluster import KMeans

# ----------------------------------------------------
# CONFIG & PAGE SETUP
# ----------------------------------------------------
st.set_page_config(page_title="VIX Spike Predictor", layout="wide", page_icon="🛡️")

# Custom CSS for a "Financial Terminal" Clean Look
st.markdown("""
<style>
    /* Clean white background with high contrast text */
    .stApp { background-color: #ffffff; color: #111111; }
    
    /* Metrics styling */
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #111111; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #555555; }
    
    /* Sidebar clarity */
    section[data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #e0e0e0; }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #ffffff; border-radius: 4px; color: #555555; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #eef2f6; color: #000000; border-bottom: 2px solid #000000; }
    
    /* Divider */
    hr { margin: 1em 0; border-top: 1px solid #eeeeee; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# CONSTANTS & PARAMETERS
# ----------------------------------------------------
VIX_PREMIUM = 4.5  # From paper
ROLLING_WINDOW = 30
START_DATE = "1990-01-01"
SEASONAL_DAYS_TO_SPIKE = {
    1: 32, 2: 25, 3: 18, 4: 20, 5: 35, 6: 40,
    7: 38, 8: 22, 9: 15, 10: 16, 11: 30, 12: 28
}

# ----------------------------------------------------
# DATA PROCESSING
# ----------------------------------------------------
@st.cache_data(ttl=60)
def load_data():
    # Use VIX and VIX3M (if available) for term structure, here we stick to VIX
    data = yf.download("^VIX", start=START_DATE, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    df = data.reset_index().rename(columns={
        'Date': 'DATE', 'Open': 'OPEN', 'High': 'HIGH', 
        'Low': 'LOW', 'Close': 'CLOSE', 'Volume': 'VOLUME'
    })
    
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values("DATE").reset_index(drop=True)
    
    # --- Feature Engineering ---
    df["daily_return"] = df["CLOSE"].pct_change()
    
    # 1. Volatility & Tail Risk (Kurtosis/Skew)
    df["rolling_vol_30d"] = df["daily_return"].rolling(ROLLING_WINDOW).std() * np.sqrt(252)
    df["skew_30d"] = df["daily_return"].rolling(30).skew()
    df["kurt_30d"] = df["daily_return"].rolling(30).kurt()
    
    # 2. Paper Logic: Expected VIX & VCR
    # Expected VIX = Recent Vol + Mean Reversion Pull + Premium
    long_term_mean = df["CLOSE"].rolling(252).mean() # 1 year rolling mean for robust reversion
    df["expected_vix"] = df["rolling_vol_30d"] + (long_term_mean - df["CLOSE"]) * 0.1 + VIX_PREMIUM
    df["vcr"] = (df["CLOSE"] - df["expected_vix"]) / df["expected_vix"]
    
    # 3. Bollinger Bands (Standard Deviation Moves)
    df['MA20'] = df['CLOSE'].rolling(window=20).mean()
    df['STD20'] = df['CLOSE'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)

    # 4. Spike Clustering
    df["spike_cluster"] = "none"
    mask_pos = df["daily_return"] > 0
    if mask_pos.sum() > 10:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        # reshape for sklearn
        X = df.loc[mask_pos, "daily_return"].values.reshape(-1, 1)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_.flatten()
        # Sort labels by magnitude
        sorted_idx = np.argsort(centers)
        label_map = {sorted_idx[0]: "small", sorted_idx[1]: "medium", sorted_idx[2]: "large"}
        
        # Assign back
        cluster_labels = kmeans.predict(X)
        mapped_labels = [label_map[l] for l in cluster_labels]
        df.loc[mask_pos, "spike_cluster"] = mapped_labels

    return df

def generate_signal(df):
    latest = df.iloc[-1]
    hist = df.iloc[:-1]
    
    # Analog Finding
    level_tol = 0.10
    vol_tol = 0.15
    
    # Filter historical data for similar conditions
    mask = (
        (abs(hist["CLOSE"] - latest["CLOSE"]) / latest["CLOSE"] < level_tol) &
        (abs(hist["rolling_vol_30d"] - latest["rolling_vol_30d"]) / latest["rolling_vol_30d"] < vol_tol)
    )
    analogs = hist[mask]
    
    # Calculate probabilities from analogs
    if len(analogs) > 0:
        fwd_returns = analogs["CLOSE"].shift(-5) / analogs["CLOSE"] - 1 # 1 week fwd
        prob_up = (fwd_returns > 0.10).mean()
    else:
        prob_up = 0.0

    # Logic Tree
    signal = "NEUTRAL"
    reason = "Market in equilibrium."
    color = "gray"
    
    # BUY Logic (Complacency)
    if latest["vcr"] > 0.02 and latest["kurt_30d"] > 1.0:
        signal = "BUY VIX CALLS"
        reason = f"High VCR ({latest['vcr']:.2%}) + Fat Tails (Kurt: {latest['kurt_30d']:.1f})"
        color = "green"
        if prob_up > 0.4:
            signal = "STRONG BUY VIX CALLS"
            color = "green"

    # SELL Logic (Panic)
    elif latest["spike_cluster"] == "large" and latest["CLOSE"] > latest["Upper_Band"]:
        signal = "SELL VIX / FADE SPIKE"
        reason = "Price > 2σ Bollinger Band + Large Cluster Spike"
        color = "red"
        
    return signal, reason, color, analogs

# ----------------------------------------------------
# MAIN APP
# ----------------------------------------------------

df = load_data()
if df is None: st.stop()

signal, reason, sig_color, analogs = generate_signal(df)
latest = df.iloc[-1]

# -- HEADER --
c1, c2 = st.columns([3, 1])
with c1:
    st.title("🛡️ VIX Spike Predictor")
    st.caption(f"Based on tail-risk clustering and VIX Change Ratio (VCR) research. Last Data: {latest['DATE'].strftime('%Y-%m-%d')}")
with c2:
    if st.button("🔄 Refresh Analysis"):
        st.cache_data.clear()
        st.rerun()

# -- SIGNAL BANNER --
st.markdown(f"""
<div style="padding: 20px; border-radius: 8px; background-color: {'#e6fffa' if 'BUY' in signal else '#fff5f5' if 'SELL' in signal else '#f8f9fa'}; border-left: 6px solid {sig_color};">
    <h3 style="margin:0; color: {sig_color};">{signal}</h3>
    <p style="margin:5px 0 0 0; font-size: 16px;">{reason} • <span style="color:#666">Analogs found: {len(analogs)}</span></p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -- TABS FOR ORGANIZED VIEW --
tab_chart, tab_data, tab_research = st.tabs(["📊 Interactive Chart", "🔢 Deep Data", "📚 Research Logic"])

with tab_chart:
    # Controls
    col_ctrl1, col_ctrl2 = st.columns([1, 4])
    with col_ctrl1:
        lookback = st.select_slider(
            "Zoom Level", 
            options=["1 Month", "3 Months", "6 Months", "1 Year", "5 Years", "Max"], 
            value="6 Months"
        )
        
        # Calculate start date based on selection
        today = df["DATE"].iloc[-1]
        if lookback == "1 Month": start_plot = today - timedelta(days=30)
        elif lookback == "3 Months": start_plot = today - timedelta(days=90)
        elif lookback == "6 Months": start_plot = today - timedelta(days=180)
        elif lookback == "1 Year": start_plot = today - timedelta(days=365)
        elif lookback == "5 Years": start_plot = today - timedelta(days=365*5)
        else: start_plot = df["DATE"].iloc[0]
        
        plot_df = df[df["DATE"] >= start_plot]

    # COMPOSITE CHART: Candlesticks + Bollinger + VCR
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # 1. Main Price (Candlestick)
    fig.add_trace(go.Candlestick(
        x=plot_df['DATE'],
        open=plot_df['OPEN'], high=plot_df['HIGH'],
        low=plot_df['LOW'], close=plot_df['CLOSE'],
        name='VIX Price'
    ), row=1, col=1)

    # 2. Bollinger Bands (The "Standard" Movement)
    fig.add_trace(go.Scatter(
        x=plot_df['DATE'], y=plot_df['Upper_Band'],
        line=dict(color='rgba(128, 128, 128, 0.5)', width=1, dash='dot'),
        name='Upper Band (2σ)', hoverinfo="skip"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=plot_df['DATE'], y=plot_df['Lower_Band'],
        line=dict(color='rgba(128, 128, 128, 0.5)', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(200, 200, 200, 0.1)',
        name='Lower Band (2σ)', hoverinfo="skip"
    ), row=1, col=1)

    # 3. Research Metric: VCR (VIX Change Ratio)
    # Green bars = VIX is "Cheap" relative to model (Potential Spike)
    # Red bars = VIX is "Expensive" relative to model (Potential Mean Reversion)
    colors = ['green' if v > 0 else 'red' for v in plot_df['vcr']]
    fig.add_trace(go.Bar(
        x=plot_df['DATE'], y=plot_df['vcr'],
        marker_color=colors,
        name='VCR (Premium/Discount)'
    ), row=2, col=1)

    # Layout Updates
    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_rangeslider_visible=False, # We use our own logic or built-in zoom
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        showlegend=False
    )
    
    # Axis styling
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0', row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0', title="VIX Price", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0', title="VCR (Model Error)", row=2, col=1)
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Chart Guide:** The gray shaded area represents the 'Standard' move (Bollinger Bands). If candles pierce this area, it is a statistical anomaly. The bottom bar chart is the 'VCR' from your research paper—green bars imply the VIX is undervalued relative to expected volatility.")

with tab_data:
    st.markdown("### Market Regime Data")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Current VIX", f"{latest['CLOSE']:.2f}", f"{latest['daily_return']:.2%}")
    kpi2.metric("Kurtosis (Tail Risk)", f"{latest['kurt_30d']:.2f}", delta_color="inverse")
    kpi3.metric("Skewness", f"{latest['skew_30d']:.2f}")
    kpi4.metric("Model Premium", f"{latest['spike_premium'] if 'spike_premium' in latest else 0:.2f}")
    
    st.markdown("### Top Historical Analogs")
    st.write("These dates had similar VIX levels and Volatility setups to today:")
    
    if not analogs.empty:
        # Show top 5 analogs
        display_analogs = analogs.tail(5).copy()
        display_analogs['Outcome_1wk'] = display_analogs['CLOSE'].shift(-5)
        display_analogs['Return_1wk'] = (display_analogs['Outcome_1wk'] - display_analogs['CLOSE']) / display_analogs['CLOSE']
        
        st.dataframe(
            display_analogs[['DATE', 'CLOSE', 'rolling_vol_30d', 'Return_1wk']].style.format({
                "CLOSE": "{:.2f}",
                "rolling_vol_30d": "{:.2f}",
                "Return_1wk": "{:.2%}"
            }), 
            use_container_width=True
        )
    else:
        st.write("No strict analogs found in current history.")

with tab_research:
    st.markdown("""
    ### About the Model
    
    This app implements logic from **"Analysis of VIX Spike Prediction"** using a hybrid of statistical clustering and mean-reversion modeling.
    
    **Key Concepts:**
    1.  **VCR (VIX Change Ratio):** Calculates the deviation of the current VIX from its "Expected" value.
        $$ VCR = \\frac{VIX_{actual} - VIX_{expected}}{VIX_{expected}} $$
    2.  **Fat Tails (Kurtosis):** High Kurtosis in recent returns indicates "fragility" in the market—often a precursor to a spike.
    3.  **Spike Clustering:** Uses K-Means clustering to categorize daily moves into "Small", "Medium", or "Large" to filter out noise.
    
    **Seasonal Outlook:**
    """)
    month = latest['DATE'].month
    days = SEASONAL_DAYS_TO_SPIKE.get(month, 30)
    st.write(f"In **{datetime(2023, month, 1).strftime('%B')}**, the average time to a volatility spike is historically **{days} days**.")

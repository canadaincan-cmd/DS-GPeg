import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_components_plotly
import plotly.graph_objects as go

st.set_page_config(
    page_title="Gold Price Dashboard - Egypt",
    layout="wide"
)

# ------------------------
# Load Data
# ------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/canadaincan-cmd/repo/main/gold_egp_gram_first_last_2021_2026.csv"
    df = pd.read_csv(url)

    df = df[df["type"] == "close price"]
    df["ds"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    df["y"] = df["price_per_gram"]
    return df[["ds", "y"]].sort_values("ds")

df = load_data()

# ------------------------
# Sidebar Controls
# ------------------------
st.sidebar.title("⚙️ التحكم")

forecast_months = st.sidebar.slider(
    "عدد أشهر التوقع",
    min_value=3,
    max_value=36,
    value=12
)

show_components = st.sidebar.checkbox("عرض التحليل الموسمي", value=True)

# ------------------------
# Train Prophet
# ------------------------
model = Prophet(
    yearly_seasonality=True,
    seasonality_mode="multiplicative"
)
model.fit(df)

future = model.make_future_dataframe(periods=forecast_months, freq="M")
forecast = model.predict(future)

# ------------------------
# KPIs
# ------------------------
last_price = df.iloc[-1]["y"]
prev_year_price = df.iloc[-13]["y"] if len(df) > 13 else last_price
yoy_change = ((last_price - prev_year_price) / prev_year_price) * 100
trend_arrow = "📈" if yoy_change > 0 else "📉"

col1, col2, col3 = st.columns(3)

col1.metric("آخر سعر جرام (EGP)", f"{last_price:,.2f}")
col2.metric("التغير السنوي %", f"{yoy_change:.2f}%")
col3.metric("الاتجاه", trend_arrow)

# ------------------------
# Actual vs Forecast Chart
# ------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["ds"],
    y=df["y"],
    mode="lines",
    name="Actual"
))

fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat"],
    mode="lines",
    name="Forecast"
))

fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_upper"],
    mode="lines",
    line=dict(width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_lower"],
    mode="lines",
    fill="tonexty",
    fillcolor="rgba(255,215,0,0.2)",
    line=dict(width=0),
    name="Confidence Interval"
))

fig.update_layout(
    title="Gold Price Forecast (EGP / gram)",
    xaxis_title="Date",
    yaxis_title="Price",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Components
# ------------------------
if show_components:
    st.subheader("📊 التحليل الموسمي")
    fig_comp = plot_components_plotly(model, forecast)
    st.plotly_chart(fig_comp, use_container_width=True)

# ------------------------
# Download Forecast
# ------------------------
st.subheader("⬇️ تحميل البيانات")
csv = forecast[["ds","yhat","yhat_lower","yhat_upper"]].to_csv(index=False)
st.download_button(
    "تحميل ملف التوقعات CSV",
    csv,
    "gold_forecast.csv",
    "text/csv"
)


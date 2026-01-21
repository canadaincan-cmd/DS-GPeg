
import streamlit as st
import pandas as pd
from prophet import Prophet

st.set_page_config(page_title="Gold Price Forecast Egypt", layout="centered")

@st.cache_data
def load_data():
    df = pd.read_csv("gold_egp_gram_first_last_2021_2026.csv")
    df = df[df["type"] == "close price"]
    df["ds"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    df["y"] = df["price_per_gram"]
    return df[["ds", "y"]]

df = load_data()

model = Prophet(yearly_seasonality=True)
model.fit(df)

future = model.make_future_dataframe(periods=12, freq="M")
forecast = model.predict(future)

st.title("Gold Price Forecast in Egypt (EGP / gram)")
st.line_chart(forecast.set_index("ds")["yhat"])

st.subheader("Forecast Data")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())


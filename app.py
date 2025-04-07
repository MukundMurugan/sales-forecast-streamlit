import pandas as pd
from prophet import Prophet
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit app
st.title("🛍️ Retail Sales Forecasting with Prophet")
st.markdown("This dashboard forecasts total daily sales using Facebook Prophet.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your sales CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show data
    st.subheader("Raw Data")
    st.write(df.head())

    # Rename and prepare
    df['data'] = pd.to_datetime(df['data'])
    df_prophet = df.groupby('data')['venda'].sum().reset_index()
    df_prophet.rename(columns={'data': 'ds', 'venda': 'y'}, inplace=True)

    # Forecast horizon
    periods = st.slider("Select forecast period (days)", 30, 180, 90)

    # Prophet model
    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Plot forecast
    st.subheader("Forecast Plot")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Plot components
    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # Show forecast table
    st.subheader("Forecast Data")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))


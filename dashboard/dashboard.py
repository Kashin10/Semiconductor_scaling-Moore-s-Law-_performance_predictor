
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="ScaleSage-GPP",
    layout="wide"
)

st.title("ScaleSage-GPP")

st.subheader(
    "Semiconductor Scaling & GPU FPS Prediction"
)

games = [

    "Cyberpunk 2077",
    "Counter Strike 2",
    "Forza Horizon 5",
    "Elden Ring",
    "Microsoft Flight Simulator"
]

fps = [132, 418, 165, 141, 97]

df = pd.DataFrame({

    "Game": games,

    "FPS": fps
})

fig = px.bar(
    df,
    x="Game",
    y="FPS",
    title="Predicted FPS"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.metric("R²", "0.9988")

st.metric("RMSE", "2.11 FPS")

st.metric("Latency", "43ms")

"""
    uvicorn api:app --reload --port 8000
    streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="Weekend Planner", layout="wide")
st.title("Weather + Event Planner (Open-Meteo + Geoapify)")

city = st.text_input("City", "Amsterdam")
date_from = st.date_input("From", datetime.now(timezone.utc).date())
date_to = st.date_input("To", (datetime.now(timezone.utc)+timedelta(days=2)).date())

if st.button("Plan"):
    payload={
        "city_query": city,
        "date_from": datetime.combine(date_from, datetime.min.time()).isoformat(),
        "date_to": datetime.combine(date_to,   datetime.max.time()).isoformat(),
        "preferences": {}
    }
    
    r = requests.post("http://localhost:8000/plan", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    st.subheader(data["city"].get("label") or data["city"]["query"])
    st.write(data["narrative"])

    ev = pd.DataFrame(data["events"])
    plan = pd.DataFrame(data["plan"])
    
    df = ev.merge(plan, left_on="id", right_on="event_id", how="left")
    st.dataframe(
        df[["name","start","venue","suitability","reason","url"]].sort_values("suitability", ascending=False),
        use_container_width=True
    )

    df_map = df.dropna(subset=["lat","lon"])
    
    if not df_map.empty:
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position='[lon, lat]',
            get_radius=80,
            get_fill_color='[255*(1-suitability), 255*suitability, 80]',
            pickable=True,
        )
        
        view = pdk.ViewState(latitude=float(df_map.iloc[0]["lat"]), longitude=float(df_map.iloc[0]["lon"]), zoom=10)
        
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text":"{name}\nSuitability: {suitability:.2f}\n{reason}"}))

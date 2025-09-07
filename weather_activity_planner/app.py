"""
    uvicorn api:app --reload --port 8000
    streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="Event Planner", layout="wide")
st.title("Weather + Event Planner (Open-Meteo + Geoapify + Ticketmaster)")

# Available place categories for selection
PLACE_CATEGORIES = {
    "Museums": "entertainment.museum",
    "Zoos": "entertainment.zoo", 
    "Theme Parks": "entertainment.theme_park",
    "Aquariums": "entertainment.aquarium",
    "Parks": "leisure.park",
    "Tourist Sights": "tourism.sights",
    "Shopping Malls": "commercial.shopping_mall",
    "Restaurants": "catering.restaurant",
    "Cafes": "catering.cafe",
    "Bars": "catering.bar",
    "Hotels": "accommodation.hotel",
    "Attractions": "tourism.attraction",
    "Historic Sites": "heritage.historic",
    "Art Galleries": "entertainment.gallery"
}

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    city = st.text_input("City", "Amsterdam")
    date_from = st.date_input("From", datetime.now(timezone.utc).date())
    date_to = st.date_input("To", (datetime.now(timezone.utc)+timedelta(days=2)).date())

with col2:
    radius = st.slider(
        "Search radius (km)", 
        min_value=1, 
        max_value=20, 
        value=10, 
        help="Radius in kilometers to search for events around the city (max 20km)"
    )
    
    activity_types = st.multiselect(
        "Preferred activity types",
        ["Outdoor", "Indoor", "Cultural", "Sports", "Music", "Food & Drink", "Family", "Nightlife"],
        default=["Outdoor", "Cultural"],
        help="Select your preferred types of activities"
    )

# Place categories section
st.subheader("Place Categories")
place_categories = st.multiselect(
    "Select place categories to include",
    list(PLACE_CATEGORIES.keys()),
    default=["Museums", "Parks", "Tourist Sights", "Restaurants"],
    help="Choose what types of places to search for"
)

# Preferences section
st.subheader("Additional Preferences")
preferences_text = st.text_area(
    "Tell us more about your preferences (optional)",
    placeholder="E.g., 'I prefer morning activities', 'Avoid crowded places', 'Looking for romantic spots', 'Family-friendly venues only', etc.",
    height=100,
    help="The AI will consider these preferences when planning your activities"
)

if st.button("Plan"):
    # Convert selected category names to API category strings
    selected_category_codes = [PLACE_CATEGORIES[cat] for cat in place_categories]
    
    payload={
        "city_query": city,
        "date_from": datetime.combine(date_from, datetime.min.time()).isoformat(),
        "date_to": datetime.combine(date_to,   datetime.max.time()).isoformat(),
        "preferences": {
            "radius_km": radius,
            "activity_types": activity_types,
            "place_categories": selected_category_codes,
            "additional_preferences": preferences_text.strip() if preferences_text.strip() else None
        }
    }
    
    try:
        with st.spinner("Planning your activities..."):
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
            width="stretch"
        )

        df_map = df.dropna(subset=["lat","lon"])
        
        if not df_map.empty:
            # Convert DataFrame to records for PyDeck
            df_map_records = df_map.to_dict('records')
            
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_map_records,
                get_position='[lon, lat]',
                get_radius=80,
                get_fill_color='[255*(1-suitability), 255*suitability, 80]',
                pickable=True,
            )
            
            view = pdk.ViewState(latitude=float(df_map.iloc[0]["lat"]), longitude=float(df_map.iloc[0]["lon"]), zoom=10)
            
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text":"{name}\n{reason}"}))
            
        pois = pd.DataFrame(data.get("places", []))
        if not pois.empty:
            poi_layer = pdk.Layer(
                "IconLayer",
                data=pois.assign(icon="marker"),
                get_icon="icon",
                get_size=3,
                get_position='[lon, lat]',
                pickable=True,
            )
            
            st.pydeck_chart(pdk.Deck(
                layers=[layer, poi_layer] if not df_map.empty else [poi_layer],
                initial_view_state=view,
                tooltip={"text":"{name}\n{category}\n{address}"}
            ))
    
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e}")
        
        if hasattr(e, 'response') and e.response:
            try:
                error_detail = e.response.json()
                st.error(f"Error details: {error_detail}")
            except Exception:
                st.error(f"Response: {e.response.text}")
                
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        st.info("Make sure the API server is running on http://localhost:8000")
        
    except Exception as e:
        st.error(f"Unexpected Error: {e}")

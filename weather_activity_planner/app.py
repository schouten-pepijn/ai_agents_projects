"""
    uvicorn api:app --reload --port 8000
    streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="Event Planner", layout="wide", page_icon="🌤️")

st.title("🌤️ Weather + Event Planner")
st.markdown("*Powered by Open-Meteo, Geoapify & Ticketmaster APIs*")

st.markdown("""
**Plan your perfect activities based on weather conditions and local events!**

This tool helps you discover:
- 🎪 **Events & Shows** from Ticketmaster
- 📍 **Local Places** from Geoapify (restaurants, museums, parks, etc.)
- 🌤️ **Weather-aware recommendations** using AI planning

Simply enter your destination, dates, and preferences below to get started.
""")

st.divider()

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
        max_value=100, 
        value=10, 
        help="Radius in kilometers to search for events and places"
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

st.divider()

# Plan button with some styling
col_button, col_info = st.columns([1, 2])
with col_button:
    plan_button = st.button("🎯 Plan My Activities", type="primary", width="stretch")

with col_info:
    st.info("**Note**: The AI will analyze weather conditions and your preferences to recommend the best activities for each day!")

if plan_button:
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

        # Process events and plan data
        events_df = pd.DataFrame(data["events"])
        plan_df = pd.DataFrame(data["plan"])
        places_df = pd.DataFrame(data.get("places", []))
        
        # Create tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Activity Plan", "🎪 All Events", "📍 All Places", "🗺️ Map View"])
        
        with tab1:
            if not events_df.empty and not plan_df.empty:
                # Merge events with plan for recommended activities
                recommended_df = events_df.merge(plan_df, left_on="id", right_on="event_id", how="inner")
                
                if not recommended_df.empty:
                    st.subheader("Recommended Activities")
                    st.dataframe(
                        recommended_df[["name","start","venue","suitability","reason","url"]].sort_values("suitability", ascending=False),
                        width="stretch",
                        column_config={
                            "name": "Activity",
                            "start": st.column_config.DatetimeColumn("Start Time"),
                            "venue": "Venue",
                            "suitability": st.column_config.ProgressColumn("Suitability", min_value=0, max_value=1),
                            "reason": "Why Recommended",
                            "url": st.column_config.LinkColumn("More Info")
                        }
                    )
                else:
                    st.info("No activities matched your preferences for the selected dates.")
            else:
                st.info("No activity plan available.")
        
        with tab2:
            if not events_df.empty:
                st.subheader("All Available Events")
                st.dataframe(
                    events_df[["name", "start", "end", "venue", "category", "description", "url"]],
                    width="stretch",
                    column_config={
                        "name": "Event Name",
                        "start": st.column_config.DatetimeColumn("Start"),
                        "end": st.column_config.DatetimeColumn("End"),
                        "venue": "Venue",
                        "category": "Category",
                        "description": "Description",
                        "url": st.column_config.LinkColumn("Event Link")
                    }
                )
                st.caption(f"Total events found: {len(events_df)}")
            else:
                st.info("No events found for your search criteria.")
        
        with tab3:
            if not places_df.empty:
                st.subheader("All Places Found")
                st.dataframe(
                    places_df[["name", "category", "address", "url"]],
                    width="stretch",
                    column_config={
                        "name": "Place Name",
                        "category": "Category",
                        "address": "Address",
                        "url": st.column_config.LinkColumn("Website")
                    }
                )
                st.caption(f"Total places found: {len(places_df)}")
            else:
                st.info("No places found for your selected categories.")
        
        with tab4:
            st.subheader("Interactive Map")
            
            # Prepare map layers
            layers = []
            
            # Events layer (recommended activities)
            if not events_df.empty and not plan_df.empty:
                events_with_plan = events_df.merge(plan_df, left_on="id", right_on="event_id", how="inner")
                events_map_data = events_with_plan.dropna(subset=["lat", "lon"])
                
                if not events_map_data.empty:
                    # Convert to dict for PyDeck
                    events_records = events_map_data.to_dict('records')
                    
                    events_layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=events_records,
                        get_position='[lon, lat]',
                        get_radius=120,
                        get_fill_color='[255*(1-suitability), 255*suitability, 80, 180]',  # Green to red based on suitability
                        pickable=True,
                    )
                    layers.append(events_layer)
            
            # Places layer
            if not places_df.empty:
                places_map_data = places_df.dropna(subset=["lat", "lon"])
                
                if not places_map_data.empty:
                    # Convert to dict for PyDeck
                    places_records = places_map_data.to_dict('records')
                    
                    places_layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=places_records,
                        get_position='[lon, lat]',
                        get_radius=60,
                        get_fill_color='[70, 130, 180, 160]',  # Steel blue for places
                        pickable=True,
                    )
                    layers.append(places_layer)
            
            # Set up map view
            if layers:
                # Use city coordinates as center, or first available point
                city_lat = data["city"].get("lat", 52.370216)
                city_lon = data["city"].get("lon", 4.895168)
                
                view_state = pdk.ViewState(
                    latitude=city_lat,
                    longitude=city_lon,
                    zoom=11,
                    pitch=0
                )
                
                # Create the map
                st.pydeck_chart(pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    tooltip={
                        "html": """
                        <b>{name}</b><br/>
                        <i>{category}</i><br/>
                        {venue}<br/>
                        {address}<br/>
                        {reason}
                        """,
                        "style": {
                            "backgroundColor": "steelblue",
                            "color": "white",
                            "border": "1px solid white",
                            "borderRadius": "5px",
                            "padding": "10px"
                        }
                    }
                ))
                
                # Add legend
                st.markdown("""
                **Map Legend:**
                - 🟢 **Green circles**: High suitability recommended activities
                - 🔴 **Red circles**: Lower suitability recommended activities  
                - 🔵 **Blue circles**: Available places/points of interest
                """)
                
                # Map statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    recommended_count = len(events_with_plan) if 'events_with_plan' in locals() else 0
                    st.metric("Recommended Activities", recommended_count)
                with col2:
                    places_count = len(places_df) if not places_df.empty else 0
                    st.metric("Places Found", places_count)
                with col3:
                    total_events = len(events_df) if not events_df.empty else 0
                    st.metric("Total Events", total_events)
            else:
                st.info("No map data available. Make sure events or places have location coordinates.")
    
    except requests.exceptions.HTTPError as e:
        st.error("**API Error Occurred**")
        st.write(f"**Status Code**: {e.response.status_code if hasattr(e, 'response') else 'Unknown'}")
        
        if hasattr(e, 'response') and e.response:
            try:
                error_detail = e.response.json()
                st.json(error_detail)
            except Exception:
                with st.expander("View Raw Response"):
                    st.code(e.response.text)
        
        st.info("**Troubleshooting Tips:**\n- Check if your search location is valid\n- Try a different date range\n- Verify API keys are properly configured")
                
    except requests.exceptions.RequestException as e:
        st.error("**Connection Error**")
        st.write("Unable to connect to the planning service.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Check if the API server is running:**\n- Server should be on `http://localhost:8000`\n- Try restarting the API service")
        with col2:
            if st.button("Retry Connection"):
                st.rerun()
                
    except Exception as e:
        st.error("**Unexpected Error**")
        st.write("Something unexpected happened during planning.")
        
        with st.expander("View Error Details"):
            st.code(str(e))
        
        st.info("**What you can try:**\n- Refresh the page and try again\n- Simplify your search criteria\n- Check the API server logs for more details")

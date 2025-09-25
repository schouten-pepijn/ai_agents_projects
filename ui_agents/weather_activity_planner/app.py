"""
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import pydeck as pdk
from datetime import datetime, timedelta, timezone
import asyncio

from models import Query
from agents import build_graph

# Cache the compiled graph so it's built only once per Streamlit session
@st.cache_resource(show_spinner=False)
def get_graph():
    return build_graph()

graph = get_graph()

st.set_page_config(page_title="Event Planner", layout="wide", page_icon="🌤️")

st.title("🌤️ Weather + Event Planner")
st.markdown("*Powered by Open-Meteo, Geoapify & Ticketmaster APIs*")
st.markdown("""
**Plan your perfect activities based on weather conditions and local events!**

This tool helps you discover:
- 🎪 **Events & Shows** from Ticketmaster & Google Events
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

    # Ensure timezone-aware datetimes
    date_from_dt = datetime.combine(date_from, datetime.min.time()).replace(tzinfo=timezone.utc)
    date_to_dt = datetime.combine(date_to, datetime.max.time()).replace(tzinfo=timezone.utc)

    # Build Query object
    query = Query(
        city_query=city,
        date_from=date_from_dt,
        date_to=date_to_dt,
        preferences={
            "radius_km": radius,
            "activity_types": activity_types,
            "place_categories": selected_category_codes,
            "additional_preferences": preferences_text.strip() if preferences_text.strip() else None,
        },
    )

    try:
        with st.spinner("Planning your activities..."):
            # Debug: Show the query being used
            with st.expander("🔍 Debug: Query Input", expanded=False):
                st.json(query.model_dump())

            # Run the graph asynchronously
            result = asyncio.run(graph.ainvoke({"query": query}))

            # Transform result
            data = {
                "city": result["city"].model_dump(),
                "events": [e.model_dump() for e in result.get("events", [])],
                "places": [p.model_dump() for p in result.get("places", [])],
                "plan": [i.model_dump() for i in result.get("plan", []).items],
                "narrative": result.get("narrative", ""),
            }

        st.subheader(data["city"].get("label") or data["city"].get("query", "Unknown City"))
        st.write(data.get("narrative", ""))

        # Process events and plan data
        events_df = pd.DataFrame(data["events"])
        plan_df = pd.DataFrame(data["plan"])
        places_df = pd.DataFrame(data.get("places", []))
        
        # Create tabs
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
                st.subheader("🎪 All Available Events")
                
                # Add source filter if the column exists
                if "source" in events_df.columns:
                    sources = events_df["source"].unique().tolist()
                    selected_sources = st.multiselect(
                        "Filter by event source",
                        sources,
                        default=sources,
                        help="Select which event sources to display"
                    )
                    
                    filtered_events_df = events_df[events_df["source"].isin(selected_sources)] if selected_sources else events_df
                else:
                    filtered_events_df = events_df
                
                # Display events with source information
                display_columns = ["name", "source", "start", "end", "venue", "category", "description", "url"]
                available_columns = [col for col in display_columns if col in filtered_events_df.columns]
                
                st.dataframe(
                    filtered_events_df[available_columns],
                    width="stretch",
                    column_config={
                        "name": "Event Name",
                        "source": st.column_config.TextColumn("Source", help="Event data source (ticketmaster, google_events, etc.)"),
                        "start": st.column_config.DatetimeColumn("Start"),
                        "end": st.column_config.DatetimeColumn("End"),
                        "venue": "Venue",
                        "category": "Category", 
                        "description": "Description",
                        "url": st.column_config.LinkColumn("Event Link")
                    }
                )
                
                st.caption(f"Showing {len(filtered_events_df)} of {len(events_df)} total events found.")
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
    
    except Exception as e:
        st.error("**Planning Error**")
        st.write("Something went wrong while generating your plan.")
        with st.expander("View Error Details"):
            st.exception(e)
        st.info("Try adjusting your inputs or verifying external API keys (Geoapify, Ticketmaster, SerpAPI).")

    # Debug expander to view raw result
    with st.expander("🕵️‍♂️ Debug: Raw Planner Output"):
        if 'data' in locals() and data:
            st.json(data)
        else:
            st.info("No data available to display.")

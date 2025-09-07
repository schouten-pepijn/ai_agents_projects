import asyncio
import httpx
from datetime import datetime, timezone
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from models import City, WeatherSlot, Event, Place
from config import config
import urls
from helpers import to_zulu, safe_get, safe_parse_dt, clean_city_label

# GEOAPIFY - GEOCODING
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5))
async def geoapify_autocomplete_city(query: str) -> List[dict]:    
    url = urls.GEOAPIFY_AUTOCOMPLETE_URL
    params = {
        "text": query,
        "type": "city",
        "limit": 5,
        "apiKey": config.GEOAPP_KEY
    }
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
    
        return r.json().get("features", [])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5))
async def geoapify_forward_geocode(query: str) -> City:
    url = urls.GEOAPIFY_SEARCH_URL
    params = {
        "text": query,
        "limit": 1,
        "apiKey": config.GEOAPP_KEY
    }
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        
        f = r.json()["features"][0]
        lon, lat = f["geometry"]["coordinates"]
        prop = f["properties"]

    return City(
        query=query,
        lat=float(lat),
        lon=float(lon),
        country=prop.get("country_code"),
        label=prop.get("formatted")
    )


# OPENMETEO - WEATHER
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5))
async def open_meteo_forecast(lat: float, lon: float) -> List[WeatherSlot]:
    url = urls.OPENMETEO_FORECAST_URL
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "UTC",
        "hourly": ["temperature_2m", "precipitation", "precipitation_probability", "wind_speed_10m", "cloud_cover_low"],
	    "models": "best_match",
    }
    
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        
        h = r.json()["hourly"]
        
    out=[]
    for i, iso_ts in enumerate(h["time"]):
        out.append(
            WeatherSlot(
                dt=datetime.fromisoformat(iso_ts.replace("Z","+00:00")),
                temp=float(h["temperature_2m"][i]),
                precip_mm=float(h["precipitation"][i]),
                pop=float(h["precipitation_probability"][i]),
                wind_ms=float(h["wind_speed_10m"][i]),
                clouds_pct=int(h["cloud_cover_low"][i]),
            )
        )
        
    return out


# GEOAPIFY - PLACES
def _parse_geoapify_place(f: dict) -> Place:
    props = f.get("properties", {})
    lon, lat = f["geometry"]["coordinates"]
    
    return Place(
        id=str(props.get("place_id") or props.get("osm_id") or props.get("name")),
        name=props.get("name") or props.get("formatted") or "Unknown",
        category=props.get("categories", ["poi"])[0] if props.get("categories") else "poi",
        lat=float(lat), lon=float(lon),
        address=props.get("formatted"),
        url=props.get("website")
    )
    
async def geoapify_places(lat: float, lon: float, categories: str, radius_km: int = 5) -> List[dict]:
    url = urls.GEOAPIFY_PLACES_URL

    # Ensure radius is within reasonable limits (Geoapify supports up to 10km)
    radius_km = min(radius_km, 10)
    radius_meters = int(radius_km * 1000)
    
    params = {
        "categories": categories,
        "filter": f"circle:{lon},{lat},{radius_meters}",
        "limit": 20,
        "apiKey": config.GEOAPP_KEY
    }
    
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(url, params=params)
            r.raise_for_status()
            return r.json().get("features", [])

    except Exception as e:
        print(f"Geoapify places API error: {e}")
        
        if hasattr(e, 'response') and e.response:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        # Return empty list instead of failing the entire request
        raise e


# Serp Google Events API
def _parse_serp_event(e: dict) -> Event:
    try:
        title = e.get("title") or ""
        desc = e.get("description") or e.get("snippet") or None
        link = e.get("link")
        venue = None
        
        if isinstance(e.get("venue"), dict):
            venue = e["venue"].get("name") or e["venue"].get("address")
            
        else:
            venue = e.get("address")

        lat = lon = None
        gps = e.get("gps_coordinates") or {}
        if isinstance(gps, dict):
            lat = gps.get("latitude")
            lon = gps.get("longitude")

        d = e.get("date") or {}
        start = safe_parse_dt(d.get("start_date") or d.get("when"))
        end   = safe_parse_dt(d.get("end_date"))

        if not start and d.get("start_time"):
            start = safe_parse_dt(d["start_time"])
            
        if not end and d.get("end_time"):
            end = safe_parse_dt(d["end_time"])

        return Event(
            id=str(e.get("event_id") or e.get("inline_entity_id") or link or title),
            source="google_events",
            name=title,
            description=desc,
            category=e.get("type") or None,
            start=start,
            end=end,
            venue=venue,
            indoor=None,
            lat=float(lat) if lat is not None else None,
            lon=float(lon) if lon is not None else None,
            url=link,
        )
    except Exception:
        # Return a minimal event to avoid completely dropping the event
        return Event(
            id=f"serp-error-{hash(str(e))}",
            source="google_events",
            name=e.get("title", "Unknown Event"),
            description=None,
            category="event",
            start=None,
            end=None,
            venue=None,
            indoor=None,
            lat=None,
            lon=None,
            url=e.get("link"),
        )
 
 
async def _do_serpapi_request(params: dict) -> dict:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(urls.SERPAPI_SEARCH_URL, params=params, headers={"Accept": "application/json"})
        r.raise_for_status()
        
        return r.json()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.6))
async def serpapi_google_events(city: City, start_iso: str, end_iso: str, query: str = "events", max_results: int = 20) -> List[Event]:
    # Skip SerpAPI if disabled or no key is configured
    if not getattr(config, 'SERPAPI_ENABLED', True) or not config.SERPAPI_KEY:
        return []

    try:
        city_label = clean_city_label(city)
        # Strategy A (preferred): only q, hl, google_domain (avoid 'location' and 'gl')
        params_a = {
            "engine": "google_events",
            "q": f"{query} in {city_label}",
            "hl": "en",
            "google_domain": "google.nl",  # NL bias without 'gl'
            "api_key": config.SERPAPI_KEY,
        }

        # Strategy B: simpler query text (sometimes too-specific queries 400)
        params_b = {
            "engine": "google_events",
            "q": f"events {city_label}",
            "hl": "en",
            "google_domain": "google.nl",
            "api_key": config.SERPAPI_KEY,
        }

        # Strategy C: as a last resort, try adding 'location' and removing google_domain
        params_c = {
            "engine": "google_events",
            "q": "events",
            "location": city_label,
            "hl": "en",
            "api_key": config.SERPAPI_KEY,
        }

        js = None
        
        for i, params in enumerate((params_a, params_b, params_c), 1):
            try:
                js = await _do_serpapi_request(params)
                break
            
            except httpx.HTTPStatusError as e:
                
                # If 400/422, try next strategy; re-raise for other errors
                if e.response is not None and e.response.status_code in (400, 422):
                    continue
                
                # For other HTTP errors, don't try more strategies
                break
            
            except Exception:
                break

        if not js:
            return []

        raw = js.get("events_results", []) or []
        
        if not raw:
            return []
            
        # Parse events with error handling
        events = []
        for event_data in raw:
            try:
                event = _parse_serp_event(event_data)
                events.append(event)
            except Exception:
                # Continue with other events instead of failing completely
                pass

        # Filter by time window
        try:
            start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
            end_dt   = datetime.fromisoformat(end_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
            
        except Exception:
            start_dt = end_dt = None

        if start_dt and end_dt:
            def _in_range(ev: Event) -> bool:
                if not ev.start:
                    return True
                
                s = ev.start.astimezone(timezone.utc)
                
                return (s >= start_dt) and (s <= end_dt)
            
            events = [e for e in events if _in_range(e)]

        result = events[:max_results]
        return result
        
    except Exception:
        return []

# TICKETMASTER
def _parse_ticketmaster_event(e)->Event:
    venues = safe_get(e, "_embedded", "venues", default=[]) or [{}]
    v = venues[0] if isinstance(venues, list) else venues
    
    loc = v.get("location", {}) or {}
    lat = loc.get("latitude")
    lon = loc.get("longitude")

    start_iso = safe_get(e, "dates", "start", "dateTime")
    start = (
        datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        if start_iso
        else None
    )

    return Event(
        id=e["id"],
        source="ticketmaster",
        name=e.get("name", ""),
        description=e.get("info") or e.get("pleaseNote"),
        category=safe_get(e, "classifications", default=[{}])[0]
                 .get("segment", {})
                 .get("name"),
        start=start,
        end=None,
        venue=v.get("name"),
        indoor=None,
        lat=float(lat) if lat else None,
        lon=float(lon) if lon else None,
        url=e.get("url"),
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5))
async def ticketmaster_events(lat: float, lon: float, start_iso: str, end_iso: str, radius_km: int = 10, max_pages: int = 5) -> List[Event]:
    # Skip Ticketmaster API if disabled or no API key
    if not getattr(config, 'TICKETMASTER_ENABLED', True) or not config.TMK:
        return []

    url = urls.TICKETMASTER_EVENTS_URL
    params = {
        "apikey": config.TMK,
        "latlong": f"{lat},{lon}",
        "radius": radius_km,  # Use user-specified radius
        "unit": "km",
        "startDateTime": to_zulu(start_iso),
        "endDateTime": to_zulu(end_iso),
        "size": 20,  # Increased size to get more events
        "sort": "date,asc",
        "CountryCode": "NL",
        # "classificationName": "music"
    }
    
    events: List[Event] = []
    page = 0
    
    async with httpx.AsyncClient(timeout=15) as c:
        while page < max_pages:
            cur_params = dict(params, page=page)
            r = await c.get(url, params=cur_params, headers={"Accept": "application/json"})
            
            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", "1"))
                
                await asyncio.sleep(retry_after)
                
                raise httpx.HTTPStatusError("Rate limit exceeded", request=r.request, response=r)
            
            r.raise_for_status()
            js = r.json()
            
            embedded = js.get("_embedded", {})
            raw_events = embedded.get("events", [])
            
            if not raw_events:
                break
            
            events.extend(_parse_ticketmaster_event(e) for e in raw_events)
            
            links = js.get("_links", {})
            has_next = "next" in links
            
            if not has_next:
                break
            
            page += 1
    
    return events
                
            

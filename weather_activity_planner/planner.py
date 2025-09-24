import re
from models import Event, Place, WeatherSlot, Plan, PlanItem, City
from typing import List
from datetime import datetime, timezone


def nearest_weather(slot_df: datetime, wx: List[WeatherSlot]) -> WeatherSlot:
    if slot_df.tzinfo is None:
        slot_df = slot_df.replace(tzinfo=timezone.utc)

    return min(
        wx,
        key=lambda w: abs(
            (
                w.dt.replace(tzinfo=timezone.utc) if w.dt.tzinfo is None else w.dt
            ).astimezone(timezone.utc)
            - slot_df.astimezone(timezone.utc)
        ).total_seconds(),
    )


def outdoor_suitability(w: WeatherSlot) -> float:
    temp_score = max(0, 1 - (abs(w.temp - 21) / 15))
    rain_score = 1 - min(1, (w.precip_mm / 3) + w.pop)
    wind_score = 1 - min(1, w.wind_ms / 12)
    clouds_penalty = max(0, (w.cloud_pct - 80) / 20 * 0.2)

    return max(
        0.0,
        min(
            1.0,
            0.5 * temp_score + 0.35 * rain_score + 0.15 * wind_score - clouds_penalty,
        ),
    )


def indoor_suitability(w: WeatherSlot) -> float:
    comfort = max(0, 1 - (abs(w.temp - 22) / 20))
    bad_weather = min(1, (w.precip_mm / 2) + w.pop + (w.wind_ms / 15))

    return max(0.0, min(1.0, 0.4 * comfort + 0.6 * bad_weather))


# AI-driven suitability functions with fallback to original formulas
async def ai_outdoor_suitability(w: WeatherSlot) -> float:
    """AI-driven outdoor suitability assessment with fallback to formula-based calculation."""
    try:
        from agents import get_llm

        llm = get_llm()

        prompt = (
            f"Rate outdoor activity suitability (0.0-1.0) for these conditions:\n"
            f"• Temperature: {w.temp}°C\n"
            f"• Precipitation: {w.precip_mm}mm\n"
            f"• Wind speed: {w.wind_ms} m/s\n"
            f"• Cloud coverage: {w.cloud_pct}%\n"
            f"• Precipitation probability: {w.pop:.1%}\n\n"
            f"Consider factors like:\n"
            f"- Comfort for walking/outdoor activities\n"
            f"- Safety (wind, rain)\n"
            f"- General pleasantness\n\n"
            f"Return ONLY a decimal number between 0.0 and 1.0."
        )

        result = await llm.ainvoke(prompt)
        score_str = result.content.strip()

        # Extract numeric value from response
        try:
            score = float(score_str)
            # Clamp to valid range
            return max(0.0, min(1.0, score))
        except ValueError:
            # Try to extract first number from response
            import re

            numbers = re.findall(r"0?\.\d+|[01]\.?\d*", score_str)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            else:
                raise ValueError("No valid score found")

    except Exception:
        # Fallback to original formula
        return outdoor_suitability(w)


async def ai_indoor_suitability(w: WeatherSlot) -> float:
    """AI-driven indoor suitability assessment with fallback to formula-based calculation."""
    try:
        from agents import get_llm

        llm = get_llm()

        prompt = (
            f"Rate indoor activity suitability (0.0-1.0) for these conditions:\n"
            f"• Temperature: {w.temp}°C\n"
            f"• Precipitation: {w.precip_mm}mm\n"
            f"• Wind speed: {w.wind_ms} m/s\n"
            f"• Cloud coverage: {w.cloud_pct}%\n"
            f"• Precipitation probability: {w.pop:.1%}\n\n"
            f"Consider factors like:\n"
            f"- Bad weather makes indoor activities more appealing\n"
            f"- Moderate temperatures are comfortable for any activity\n"
            f"- Indoor activities are weather-independent but benefit from contrast\n\n"
            f"Return ONLY a decimal number between 0.0 and 1.0."
        )

        result = await llm.ainvoke(prompt)
        score_str = result.content.strip()

        # Extract numeric value from response
        try:
            score = float(score_str)
            # Clamp to valid range
            return max(0.0, min(1.0, score))

        except ValueError:
            # Try to extract first number from response
            numbers = re.findall(r"0?\.\d+|[01]\.?\d*", score_str)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))

            else:
                raise ValueError("No valid score found")

    except Exception:
        # Fallback to original formula
        return indoor_suitability(w)


async def build_plan(city: City, events: List[Event], wx: List[WeatherSlot]) -> Plan:
    items = []

    for e in events:
        if not e.start:
            continue

        # Use the indoor classification from the AI node, default to indoor if not set
        ind = e.indoor if e.indoor is not None else True
        w = nearest_weather(e.start.astimezone(timezone.utc), wx)

        # Use AI-driven suitability assessment
        s = await (ai_indoor_suitability(w) if ind else ai_outdoor_suitability(w))

        reason = f"{'Indoor' if ind else 'Outdoor'} · clouds {w.cloud_pct}%, {round(w.temp)}°C, precip {w.precip_mm:.1f}mm"

        items.append(
            PlanItem(event_id=e.id, suitability=s, reason=reason, slot_dt=w.dt)
        )

    items.sort(key=lambda x: (-x.suitability, x.slot_dt))

    return Plan(city=city, items=items)


def append_pois_when_sparse(
    plan: Plan, places: list[Place], wx: list[WeatherSlot], k: int = 6
) -> None:

    good = [it for it in plan.items if it.suitability >= 0.6]
    if len(good) >= k:
        return

    add_n = max(0, k - len(good))

    w = wx[0]
    base = 0.6 - min(0.2, w.precip_mm / 5 + w.wind_ms / 25)
    base = max(0.45, min(0.75, base))

    for p in places[:add_n]:
        plan.items.append(
            PlanItem(
                event_id=f"poi::{p.id}",
                suitability=base,
                reason=f"POI · {p.category.replace('.', ' ')}",
                slot_dt=wx[0].dt,
            )
        )

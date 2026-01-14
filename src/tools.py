import json
import hashlib
from datetime import datetime, timezone
from functools import wraps
import requests
from pymongo import MongoClient
from langchain.tools import tool

from src.config import (
    MONGO_URI, MONGO_DB_NAME, MONGO_API_CACHE_COLLECTION,
    FREE_ASTROLOGY_API_KEY, ASTRO_OBSERVATION_POINT, ASTRO_AYANAMSHA, ASTRO_LANGUAGE,
)
from src.utils import get_lat_lon_offset
from src.vector_store import get_pinecone_retriever


# -------------------- MongoDB Caching --------------------

def _cache_key(dob: str, tob: str, lat: float, lon: float, chart_type: str):
    payload = {"dob": dob, "tob": tob, "lat": lat, "lon": lon, "chart_type": chart_type}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def mongo_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        dob = kwargs["dob"]; tob = kwargs["tob"]; lat = kwargs["lat"]; lon = kwargs["lon"]; chart_type = kwargs["chart_type"]
        cache_id, payload = _cache_key(dob, tob, lat, lon, chart_type)

        client = MongoClient(MONGO_URI)
        col = client[MONGO_DB_NAME][MONGO_API_CACHE_COLLECTION]
        hit = col.find_one({"_id": cache_id})
        if hit:
            client.close()
            return hit["api_response"]

        result = func(*args, **kwargs)
        if result:
            col.insert_one({
                "_id": cache_id,
                **payload,
                "api_response": result,
                "created_at": datetime.now(timezone.utc),
            })
        client.close()
        return result
    return wrapper


# -------------------- External API --------------------

def _build_payload(dob: str, tob: str, lat: float, lon: float, tz: float) -> dict:
    """Builds the JSON payload for FreeAstrologyAPI endpoints from inputs."""
    # dob: YYYY-MM-DD, tob: HH:MM[:SS]
    year, month, day = [int(x) for x in dob.split("-")]
    time_parts = tob.split(":")
    hours = int(time_parts[0])
    minutes = int(time_parts[1]) if len(time_parts) > 1 else 0
    seconds = int(time_parts[2]) if len(time_parts) > 2 else 0
    return {
        "year": year,
        "month": month,
        "date": day,
        "hours": hours,
        "minutes": minutes,
        "seconds": seconds,
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,
        "config": {
            "observation_point": ASTRO_OBSERVATION_POINT,
            "ayanamsha": ASTRO_AYANAMSHA,
        },
    }


def _post(url: str, payload: dict, expect_json: bool = True):
    headers = {
        "Content-Type": "application/json",
        "x-api-key": FREE_ASTROLOGY_API_KEY,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json() if expect_json else resp.text
    except requests.exceptions.RequestException as e:
        return {"error": "API request failed", "details": str(e)}


@mongo_cache
def _fetch_chart(dob: str, tob: str, lat: float, lon: float, tz: float, chart_type: str):
    payload = _build_payload(dob=dob, tob=tob, lat=lat, lon=lon, tz=tz)
    if chart_type == "D10":
        # Dasamsa chart info
        result = _post("https://api.freeastrologyapi.com/d10-chart-info", payload, expect_json=True)
        return result
    elif chart_type == "D9":
        # Navamsa URL response (SVG hosted URL)
        url_text = _post("https://json.freeastrologyapi.com/navamsa-chart-url", {**payload, "language": ASTRO_LANGUAGE}, expect_json=False)
        if isinstance(url_text, dict):
            return url_text
        return {"url": url_text}
    else:
        # Default to D1 (Rasi) chart URL
        url_text = _post("https://api.freeastrologyapi.com/horoscope-chart-url", payload, expect_json=False)
        if isinstance(url_text, dict):
            return url_text
        return {"url": url_text}


# -------------------- LangChain Tools --------------------


def _tool_impl(dob: str, tob: str, city: str, chart_type: str):
    try:
        date_obj = datetime.strptime(dob, "%Y-%m-%d")
        lat, lon, tz = get_lat_lon_offset(city, date_obj)
        if lat is None or lon is None or tz is None:
            return {"error": "Geocoding failed for provided city/date."}
        return _fetch_chart(dob=dob, tob=tob, lat=lat, lon=lon, tz=tz, chart_type=chart_type)
    except Exception as e:
        return {"error": f"Unexpected failure: {e}"}


@tool("get_d10_chart")
def get_d10_chart(dob: str, tob: str, city: str) -> dict:
    """Fetches Dasamsa (D10) chart. Use ONLY for CAREER queries."""
    return _tool_impl(dob=dob, tob=tob, city=city, chart_type="D10")


@tool("get_d9_chart")
def get_d9_chart(dob: str, tob: str, city: str) -> dict:
    """Fetches Navamsa (D9) chart. Use ONLY for MARRIAGE/RELATIONSHIP queries."""
    return _tool_impl(dob=dob, tob=tob, city=city, chart_type="D9")


@tool("get_d1_chart")
def get_d1_chart(dob: str, tob: str, city: str) -> dict:
    """Fetches Rasi (D1) chart. Use for GENERAL HEALTH/BODY/PERSONALITY queries."""
    return _tool_impl(dob=dob, tob=tob, city=city, chart_type="D1")


@tool("search_bphs")
def search_bphs(query: str) -> str:
    """Search BPHS in Pinecone and return concatenated relevant passages."""
    retriever = get_pinecone_retriever(top_k=4)
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant passages found."
    return "\n\n".join([d.page_content for d in docs])

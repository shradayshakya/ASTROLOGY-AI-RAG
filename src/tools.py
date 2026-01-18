import json
import hashlib
from datetime import datetime, timezone
from functools import wraps
import requests
from pymongo import MongoClient
from langchain.tools import tool
from src.logging_utils import get_logger, log_call

from src.config import (
    MONGO_URI, MONGO_DB_NAME, MONGO_API_CACHE_COLLECTION,
    FREE_ASTROLOGY_API_KEY, ASTRO_OBSERVATION_POINT, ASTRO_AYANAMSHA
)
from src.utils import get_lat_lon_offset
from src.vector_store import get_pinecone_retriever

logger = get_logger(__name__)

# -------------------- Configuration Mapping --------------------

# This dictionary maps the internal "Chart Code" to the specific data API endpoints.
CHART_CONFIG = {
    # --- The Big Three ---
    "D1": {
        # "data": "https://json.freeastrologyapi.com/planets-extended", 
        "data": "https://json.freeastrologyapi.com/planets", 
    },
    "D9": {
        "data": "https://json.freeastrologyapi.com/navamsa-chart-info", 
    },
    "D10": {
        "data": "https://json.freeastrologyapi.com/d10-chart-info", 
    },
    
    # --- Wealth & Family ---
    "D3": { "data": "https://json.freeastrologyapi.com/d3-chart-info"},
    "D4": { "data": "https://json.freeastrologyapi.com/d4-chart-info"},
    
    # --- Progeny & Health ---
    "D7": { "data": "https://json.freeastrologyapi.com/d7-chart-info"},
    "D6": { "data": "https://json.freeastrologyapi.com/d6-chart-info"},
    "D8": { "data": "https://json.freeastrologyapi.com/d8-chart-info"},
    
    # --- Advanced / Spiritual ---
    "D5":  { "data": "https://json.freeastrologyapi.com/d5-chart-info"},
    "D11": { "data": "https://json.freeastrologyapi.com/d11-chart-info"},
    "D12": { "data": "https://json.freeastrologyapi.com/d12-chart-info"},
    "D16": { "data": "https://json.freeastrologyapi.com/d16-chart-info"},
    "D20": { "data": "https://json.freeastrologyapi.com/d20-chart-info"},
    "D24": { "data": "https://json.freeastrologyapi.com/d24-chart-info"},
    "D27": { "data": "https://json.freeastrologyapi.com/d27-chart-info"},
    "D30": { "data": "https://json.freeastrologyapi.com/d30-chart-info"},
    
    # --- Esoteric / Karmic ---
    "D40": { "data": "https://json.freeastrologyapi.com/d40-chart-info"},
    "D45": { "data": "https://json.freeastrologyapi.com/d45-chart-info"},
    "D60": { "data": "https://json.freeastrologyapi.com/d60-chart-info"},
}


# -------------------- Internal Helpers (Caching & Fetching) --------------------

def _cache_key(dob, tob, lat, lon, chart_type):
    """Generates a consistent SHA-256 hash."""
    payload = {"dob": dob, "tob": tob, "lat": lat, "lon": lon, "chart_type": chart_type}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload

def mongo_cache(func):
    """Caching Decorator."""
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
        if result and "error" not in result:
            col.insert_one({
                "_id": cache_id, **payload,
                "api_response": result, "created_at": datetime.now(timezone.utc)
            })
        client.close()
        return result
    return wrapper

def _build_payload(dob, tob, lat, lon, tz):
    """Builds standard API payload."""
    year, month, day = [int(x) for x in dob.split("-")]
    h, m, s = (tob.split(":") + ["0", "0"])[:3]
    return {
        "year": year, "month": month, "date": day,
        "hours": int(h), "minutes": int(m), "seconds": int(s),
        "latitude": lat, "longitude": lon, "timezone": tz,
        "config": { "observation_point": ASTRO_OBSERVATION_POINT, "ayanamsha": ASTRO_AYANAMSHA }
    }

def _post(url, payload):
    headers = {"Content-Type": "application/json", "x-api-key": FREE_ASTROLOGY_API_KEY}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError as e:
            logger.warning(f"Non-JSON response from {url}: {e}")
            return resp.text  # Handle raw text responses
    except Exception as e:
        logger.exception(f"HTTP POST error to {url}: {e}")
        return {"error": str(e)}

@mongo_cache
def _fetch_chart(dob, tob, lat, lon, tz, chart_type):
    """Dynamic fetcher that looks up the endpoint in CHART_CONFIG."""
    config = CHART_CONFIG.get(chart_type)
    if not config:
        return {"error": f"Chart type '{chart_type}' is not supported."}

    payload = _build_payload(dob, tob, lat, lon, tz)
    
    # 1. Fetch Data
    data_response = _post(config["data"], payload)
    logger.info(f"Fetched {chart_type} chart data for DOB: {dob}, TOB: {tob}, Lat: {lat}, Lon: {lon}")

    return {
        "chart_type": chart_type,
        "chart_data": data_response
    }


# -------------------- LangChain Tools (Exposed to Agent) --------------------

def _tool_impl(dob, tob, city, chart_type):
    """Common logic for all chart tools."""
    try:
        date_obj = datetime.strptime(dob, "%Y-%m-%d")
        lat, lon, tz = get_lat_lon_offset(city, date_obj)
        if lat is None:
            return {"error": f"Could not geocode city: {city}"}
        return _fetch_chart(dob=dob, tob=tob, lat=lat, lon=lon, tz=tz, chart_type=chart_type)
    except Exception as e:
        logger.exception(f"Tool implementation error for chart_type={chart_type}: {e}")
        return {"error": f"Tool error: {e}"}

# --- PRIMARY TOOLS (The Big 3) ---

@tool("chart_d10_career")
def get_d10_chart(dob: str, tob: str, city: str) -> dict:
    """
    Fetches Dasamsa (D10) chart. 
    USE CASE: Career, Profession, Status, Fame, Promotion, Business.
    ARGS: dob (YYYY-MM-DD), tob (HH:MM), city (str).
    """
    return _tool_impl(dob, tob, city, "D10")

@tool("chart_d9_marriage")
def get_d9_chart(dob: str, tob: str, city: str) -> dict:
    """
    Fetches Navamsa (D9) chart. 
    USE CASE: Marriage, Spouse, Relationships, Inner Strength, Partnership.
    ARGS: dob (YYYY-MM-DD), tob (HH:MM), city (str).
    """
    return _tool_impl(dob, tob, city, "D9")

@tool("chart_d1_general_health")
def get_d1_chart(dob: str, tob: str, city: str) -> dict:
    """
    Fetches Rasi (D1) / Planetary Chart.
    USE CASE: General Health, Body, Personality, Life Direction, or fallback.
    ARGS: dob (YYYY-MM-DD), tob (HH:MM), city (str).
    """
    return _tool_impl(dob, tob, city, "D1")

# --- SECONDARY TOOLS (Wealth, Progeny, etc.) ---

@tool("chart_d2_wealth_hora")
def get_d2_chart(dob: str, tob: str, city: str) -> dict:
    """
    Fetches Hora (D2) chart. 
    USE CASE: Wealth, Assets, Money, Family Resources.
    ARGS: dob (YYYY-MM-DD), tob (HH:MM), city (str).
    """
    return _tool_impl(dob, tob, city, "D2")

@tool("chart_d7_progeny_saptamsa")
def get_d7_chart(dob: str, tob: str, city: str) -> dict:
    """
    Fetches Saptamsa (D7) chart. 
    USE CASE: Children, Progeny, Pregnancy, Creative Output.
    ARGS: dob (YYYY-MM-DD), tob (HH:MM), city (str).
    """
    return _tool_impl(dob, tob, city, "D7")

@tool("chart_d24_education_siddhamsa")
def get_d24_chart(dob: str, tob: str, city: str) -> dict:
    """
    Fetches Siddhamsa (D24) chart. 
    USE CASE: Education, Learning, Degrees, Knowledge.
    ARGS: dob (YYYY-MM-DD), tob (HH:MM), city (str).
    """
    return _tool_impl(dob, tob, city, "D24")

# --- GENERAL FETCHER (Advanced) ---

@tool("chart_varga_specific")
def get_specific_varga_chart(dob: str, tob: str, city: str, chart_code: str) -> dict:
    """
    Fetches any specific Divisional Chart by its code.
    
    USE CASE: Use this ONLY if the user specifically asks for a chart not covered by other tools.
    Examples: "Show me my D60 chart" or "Check my D4 chart for property".
    
    ARGS:
    - dob (str): YYYY-MM-DD
    - tob (str): HH:MM
    - city (str): City Name
    - chart_code (str): One of [D3, D4, D5, D6, D8, D11, D12, D16, D20, D27, D30, D40, D45, D60]
    """
    valid_codes = list(CHART_CONFIG.keys())
    if chart_code.upper() not in valid_codes:
        return {"error": f"Invalid Chart Code. Supported: {valid_codes}"}
    
    return _tool_impl(dob, tob, city, chart_code.upper())

@tool("bphs_search_pinecone")
def search_bphs(query: str) -> str:
    """Search BPHS in Pinecone for interpretation rules."""
    retriever = get_pinecone_retriever(top_k=4)
    try:
        # Use standard retriever API for compatibility across LangChain versions
        # VectorStoreRetriever implements BaseRunnable; prefer public invoke()
        docs = retriever.invoke(query)
        logger.info(f"BPHS search returned {len(docs) if docs else 0} documents with query: {query}")
        return "\n\n".join([d.page_content for d in docs]) if docs else "No relevant passages found."
    except Exception as e:
        logger.exception(f"BPHS search error: {e}")
        return "No relevant passages found."
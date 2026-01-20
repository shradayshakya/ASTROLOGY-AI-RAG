from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut
from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
import time
from src.logging_utils import get_logger, log_call

logger = get_logger(__name__)

# Best-effort static fallbacks for common Nepal cities to avoid geocoding outages
CITY_COORD_FALLBACKS = {
    "Kathmandu, Nepal": (27.7172, 85.3240),
    "Kathmandu": (27.7172, 85.3240),
    "Lalitpur, Nepal": (27.6667, 85.3333),  # Lalitpur (Patan)
    "Lalitpur": (27.6667, 85.3333),
}


@log_call
def get_lat_lon_offset(city_name: str, date_object: datetime):
    """
    Returns (lat, lon, utc_offset_hours) for a city on the given historical date.
    Handles DST and historical timezone changes.
    """
    try:
        logger.info(f"Geocoding city '{city_name}' for date {date_object}")
        geolocator = Nominatim(user_agent="vedic-astro-bot", timeout=10)

        # Retry geocoding a few times to handle transient timeouts
        location = None
        for attempt in range(3):
            try:
                location = geolocator.geocode(city_name, exactly_one=True, timeout=10)
                if location:
                    break
            except (GeocoderUnavailable, GeocoderTimedOut):
                # Exponential backoff
                sleep_for = 1.0 * (2 ** attempt)
                logger.warning(f"Geocoding attempt {attempt+1} failed; retrying in {sleep_for:.1f}s")
                time.sleep(sleep_for)
            except Exception:
                # Other errors: log and break
                logger.exception("Unexpected geocoding error")
                break

        if not location:
            # Try static fallback coordinates
            fallback = CITY_COORD_FALLBACKS.get(city_name)
            if not fallback:
                # Also try normalized forms (strip country punctuation and extra spaces)
                normalized = city_name.strip()
                fallback = CITY_COORD_FALLBACKS.get(normalized)

            if fallback:
                lat, lon = fallback
                logger.warning(f"Using fallback coordinates for '{city_name}': (lat={lat}, lon={lon})")
            else:
                logger.warning("City geocode not found and no fallback available")
                return None, None, None
        else:
            lat, lon = location.latitude, location.longitude
        tz_name = TimezoneFinder().timezone_at(lng=lon, lat=lat)
        if not tz_name:
            logger.warning("Timezone not found; returning lat/lon without offset")
            return lat, lon, None

        tz = pytz.timezone(tz_name)
        # Localize to compute historical offset (DST-aware)
        localized = tz.localize(date_object, is_dst=None)
        offset_hours = localized.utcoffset().total_seconds() / 3600.0
        return lat, lon, offset_hours
    except Exception:
        logger.exception("Failed to compute lat/lon/offset")
        return None, None, None

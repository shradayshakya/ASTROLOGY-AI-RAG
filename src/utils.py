from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
from src.logging_utils import get_logger, log_call

logger = get_logger(__name__)


@log_call
def get_lat_lon_offset(city_name: str, date_object: datetime):
    """
    Returns (lat, lon, utc_offset_hours) for a city on the given historical date.
    Handles DST and historical timezone changes.
    """
    try:
        logger.info(f"Geocoding city '{city_name}' for date {date_object}")
        geolocator = Nominatim(user_agent="vedic-astro-bot")
        location = geolocator.geocode(city_name)
        if not location:
            logger.warning("City geocode not found")
            return None, None, None

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

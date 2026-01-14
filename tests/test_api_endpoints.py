import os
import json
import pytest
import requests

API_KEY = os.environ.get("FREE_ASTROLOGY_API_KEY")
LANGUAGE = os.environ.get("ASTRO_LANGUAGE", "en")
OBS_POINT = os.environ.get("ASTRO_OBSERVATION_POINT", "topocentric")
AYANAMSHA = os.environ.get("ASTRO_AYANAMSHA", "lahiri")

# Sample payload values (from API docs examples)
BASE_PAYLOAD = {
    "year": 2022,
    "month": 8,
    "date": 11,
    "hours": 6,
    "minutes": 0,
    "seconds": 0,
    "latitude": 17.38333,
    "longitude": 78.4666,
    "timezone": 5.5,
}

HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY or "",
}

skip_if_no_key = pytest.mark.skipif(not API_KEY, reason="FREE_ASTROLOGY_API_KEY not set")


@skip_if_no_key
@pytest.mark.integration
def test_navamsa_chart_url_status_and_output():
    url = "https://json.freeastrologyapi.com/navamsa-chart-url"
    payload = {
        **BASE_PAYLOAD,
        "language": LANGUAGE,
        "config": {"observation_point": OBS_POINT, "ayanamsha": AYANAMSHA},
        "settings": {"observation_point": OBS_POINT, "ayanamsha": AYANAMSHA},
    }
    resp = requests.post(url, headers=HEADERS, data=json.dumps(payload), timeout=30)
    assert resp.status_code == 200
    assert isinstance(resp.text, str) and len(resp.text) > 0
    assert "http" in resp.text.lower()


@skip_if_no_key
@pytest.mark.integration
def test_navamsa_chart_info_status_and_output():
    url = "https://json.freeastrologyapi.com/navamsa-chart-info"
    payload = {
        **BASE_PAYLOAD,
        "language": LANGUAGE,
        "config": {"observation_point": OBS_POINT, "ayanamsha": AYANAMSHA},
        "settings": {"observation_point": OBS_POINT, "ayanamsha": AYANAMSHA},
    }
    resp = requests.post(url, headers=HEADERS, data=json.dumps(payload), timeout=30)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert data.get("statusCode") == 200
    assert "output" in data and isinstance(data["output"], dict)


@skip_if_no_key
@pytest.mark.integration
def test_d10_chart_info_status_and_output():
    url = "https://api.freeastrologyapi.com/d10-chart-info"
    payload = {
        **BASE_PAYLOAD,
        "config": {"observation_point": OBS_POINT, "ayanamsha": AYANAMSHA},
        "settings": {"observation_point": OBS_POINT, "ayanamsha": AYANAMSHA},
    }
    resp = requests.post(url, headers=HEADERS, data=json.dumps(payload), timeout=30)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert data.get("statusCode") == 200
    assert "output" in data and isinstance(data["output"], dict)


@skip_if_no_key
@pytest.mark.integration
def test_horoscope_chart_url_status_and_output():
    url = "https://api.freeastrologyapi.com/horoscope-chart-url"
    payload = {
        **BASE_PAYLOAD,
        "config": {"observation_point": OBS_POINT, "ayanamsha": AYANAMSHA},
        "settings": {"observation_point": OBS_POINT, "ayanamsha": AYANAMSHA},
    }
    resp = requests.post(url, headers=HEADERS, data=json.dumps(payload), timeout=30)
    assert resp.status_code == 200
    assert isinstance(resp.text, str) and len(resp.text) > 0
    assert "http" in resp.text.lower()

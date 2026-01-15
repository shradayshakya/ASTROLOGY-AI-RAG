import time
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

# --- Unified Varga Chart API Tests ---

@skip_if_no_key
@pytest.mark.integration
@pytest.mark.parametrize("chart_code, data_url", [
    ("D1", "https://json.freeastrologyapi.com/planets"),
    ("D9", "https://json.freeastrologyapi.com/navamsa-chart-info"),
    ("D10", "https://json.freeastrologyapi.com/d10-chart-info"),
    ("D2", "https://json.freeastrologyapi.com/d2-chart-info"),
    ("D3", "https://json.freeastrologyapi.com/d3-chart-info"),
    ("D4", "https://json.freeastrologyapi.com/d4-chart-info"),
    ("D5", "https://json.freeastrologyapi.com/d5-chart-info"),
    ("D6", "https://json.freeastrologyapi.com/d6-chart-info"),
    ("D7", "https://json.freeastrologyapi.com/d7-chart-info"),
    ("D8", "https://json.freeastrologyapi.com/d8-chart-info"),
    ("D11", "https://json.freeastrologyapi.com/d11-chart-info"),
    ("D12", "https://json.freeastrologyapi.com/d12-chart-info"),
    ("D16", "https://json.freeastrologyapi.com/d16-chart-info"),
    ("D20", "https://json.freeastrologyapi.com/d20-chart-info"),
    ("D24", "https://json.freeastrologyapi.com/d24-chart-info"),
    ("D27", "https://json.freeastrologyapi.com/d27-chart-info"),
    ("D30", "https://json.freeastrologyapi.com/d30-chart-info"),
    ("D40", "https://json.freeastrologyapi.com/d40-chart-info"),
    ("D45", "https://json.freeastrologyapi.com/d45-chart-info"),
    ("D60", "https://json.freeastrologyapi.com/d60-chart-info"),
])
def test_varga_chart_data(chart_code, data_url):
    payload = {
        **BASE_PAYLOAD,
        "config": {"observation_point": OBS_POINT, "ayanamsha": AYANAMSHA}
    }
    resp = requests.post(data_url, headers=HEADERS, data=json.dumps(payload), timeout=30)
    assert resp.status_code == 200, f"{chart_code} data endpoint failed: {resp.text}"
    data = resp.json()
    assert isinstance(data, dict)
    assert data.get("statusCode") == 200
    assert "output" in data
    if chart_code == "D1":  # /planets endpoint
        assert isinstance(data["output"], list), f"/planets output should be a list, got {type(data['output'])}"
    else:
        assert isinstance(data["output"], dict), f"{chart_code} output should be a dict, got {type(data['output'])}"
    time.sleep(5)

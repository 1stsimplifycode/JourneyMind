"""Test-wide setup.

The geocoder is switched OFF for the whole suite. It is a real network call to
donated infrastructure, and a test run that depends on it is both slow and
somebody else's rate limit. `tests/test_geocoding.py` exercises it directly,
against a cache, and skips when there is no network.
"""

import os

os.environ.setdefault("JM_GEOCODER", "0")

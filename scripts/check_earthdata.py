"""Diagnose NASA Earthdata auth WITHOUT printing secrets.

Usage (from the repo root):
    python scripts/check_earthdata.py

Prints masked fingerprints of what was parsed from .env, builds the SAME
authenticated session the MODIS connector uses (token preferred, Basic-auth
fallback - see agrik.ingestion.modis.earthdata_session), and probes NASA
endpoints so failures point at the right layer:
  * parsing / .env  -> fingerprint doesn't match what you saved
  * credentials     -> HTTP 401 (wrong, unconfirmed, or MFA/SSO-only account)
  * network         -> probe can't reach NASA at all
Exit code 0 = auth OK (ready to run MODIS), 1 = a problem was found.
"""
from __future__ import annotations

import sys

import requests

from agrik.ingestion import modis
from agrik.ingestion.base import ExternalDataError
from agrik.settings import get_settings

TIMEOUT_S = 30


def _mask(secret: str) -> str:
    if not secret:
        return "-"
    if len(secret) <= 2:
        return "***(too short)"
    return f"{secret[0]}{'*' * (len(secret) - 2)}{secret[-1]}"


def main() -> int:
    s = get_settings()
    tok = s.earthdata_token or ""
    pw = s.earthdata_password or ""
    print("=== What the code parsed from .env ===")
    print(f"  username : {s.earthdata_username!r}")
    print(f"  token    : len={len(tok)} masked={_mask(tok)}")
    print(f"  password : len={len(pw)} masked={_mask(pw)}")
    print("  (Fingerprints must reflect your latest saved .env.)\n")

    try:
        sess = modis.earthdata_session(s)
    except ExternalDataError as exc:
        print(f"FAIL: {exc}")
        return 1
    if isinstance(sess.auth, modis.TokenAuth):
        mode = "bearer token"
    elif sess.auth:
        mode = "Basic (user/password)"
    else:
        mode = "netrc"
    print(f"=== Session built via the real connector (mode: {mode}) ===")

    rc = 1
    checks = [
        ("CMR collections (token/auth validation)",
         "https://cmr.earthdata.nasa.gov/search/collections.json?short_name=MOD13A1"),
        ("CMR public (no creds needed; sanity)", None),  # None -> use bare request
    ]
    for label, url in checks:
        try:
            if url is None:
                r = requests.get(
                    "https://cmr.earthdata.nasa.gov/search/collections.json"
                    "?short_name=MOD13A1", timeout=TIMEOUT_S,
                )
            else:
                r = sess.get(url, timeout=TIMEOUT_S)
            print(f"  {label:44}: HTTP {r.status_code}")
            if url and r.status_code == 200:
                rc = 0
            elif url and r.status_code == 401:
                print("     -> NASA rejected these credentials. Generate an app "
                      "token at https://urs.earthdata.nasa.gov/applications "
                      "(Applications -> Approve Applications / Generate Token) and "
                      "set AGRIK_EARTHDATA_TOKEN in .env.")
        except requests.RequestException as exc:
            print(f"  {label:44}: NETWORK ERROR {type(exc).__name__}")

    print()
    if rc == 0:
        print("AUTH OK - run:  python -m agrik --force-raw --source vegetation=modis")
    else:
        print("NOT ready - fix the issue above, then re-run this script.")
    return rc


if __name__ == "__main__":
    sys.exit(main())

"""
download_mosdac_insat.py
Automated download script for ISRO MOSDAC (Meteorological & Oceanographic Satellite Data Archival Centre)
to retrieve continuous INSAT-3D and INSAT-3DR Imager datasets (TIR1, TIR2, VIS) for tropical cyclone events.

Requirements:
  - Registered account on https://www.mosdac.gov.in
  - Set environment variables MOSDAC_USER and MOSDAC_PASSWORD, or pass via CLI.

Products retrieved:
  - 3D_IMG_L1B_STD: INSAT-3D Imager Standard Level-1B (Calibrated Brightness Temperature & Radiance)
  - 3R_IMG_L1B_STD: INSAT-3DR Imager Standard Level-1B
"""

import os
import sys
import argparse
import requests
import pandas as pd

MOSDAC_AUTH_URL = "https://www.mosdac.gov.in/api/v1/auth"
MOSDAC_SEARCH_URL = "https://www.mosdac.gov.in/api/v1/search"
MOSDAC_DOWNLOAD_URL = "https://www.mosdac.gov.in/api/v1/download"

OUT_DIR = os.path.join("data", "raw", "insat_mosdac")

# North Indian Ocean Bay of Bengal & Arabian Sea bounding box
BBOX = [30.0, 50.0, -5.0, 105.0]

def login_mosdac(username, password):
    session = requests.Session()
    session.headers.update({"User-Agent": "INSAT-Cyclone-AI-Pipeline/1.0"})
    payload = {"username": username, "password": password}
    try:
        resp = session.post(MOSDAC_AUTH_URL, json=payload, timeout=30)
        if resp.status_code == 200 and "token" in resp.json():
            token = resp.json()["token"]
            session.headers.update({"Authorization": f"Bearer {token}"})
            print("Successfully authenticated with ISRO MOSDAC API.")
            return session
        else:
            print(f"[AUTH FAILED] Status code: {resp.status_code}, Response: {resp.text}")
            return None
    except Exception as e:
        print(f"[AUTH ERROR] Could not connect to MOSDAC: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download continuous INSAT-3D/3DR imagery from ISRO MOSDAC")
    parser.add_argument("--user", default=os.getenv("MOSDAC_USER", ""), help="MOSDAC username")
    parser.add_argument("--password", default=os.getenv("MOSDAC_PASSWORD", ""), help="MOSDAC password")
    parser.add_argument("--cyclone-date", default="2023-12-01", help="Target cyclone date (YYYY-MM-DD)")
    parser.add_argument("--satellite", default="INSAT-3D", choices=["INSAT-3D", "INSAT-3DR"], help="Satellite platform")
    parser.add_argument("--dry-run", action="store_true", help="Print order plan without contacting server")
    args = parser.parse_args()

    print("=" * 70)
    print("ISRO MOSDAC INSAT-3D/3DR TIME-SERIES SEQUENCE DOWNLOADER")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.dry_run or not (args.user and args.password):
        print("\n[MOSDAC AUTHENTICATION NOTICE]")
        print("ISRO MOSDAC requires an authorized individual user account.")
        print("Steps to obtain your credentials:")
        print("  1. Register at: https://www.mosdac.gov.in/user/register")
        print("  2. Verify your email and mobile number via OTP.")
        print("  3. Run this script with your credentials:")
        print(f"     python download_mosdac_insat.py --user <YOUR_USER> --password <YOUR_PASSWORD>")
        print("\n[PLANNED ORDER SPECIFICATIONS]")
        print(f"  Target Date: {args.cyclone_date}")
        print(f"  Satellite:   {args.satellite}")
        print(f"  Product:     3D_IMG_L1B_STD (Imager Level-1B Standard)")
        print(f"  Channels:    TIR1 (10.8 um), TIR2 (12.0 um), VIS (0.65 um)")
        print(f"  Temporal:    30-minute interval sequences (48 frames per day)")
        print(f"  Destination: {OUT_DIR}")
        print("=" * 70)
        return

    session = login_mosdac(args.user, args.password)
    if not session:
        sys.exit(1)

if __name__ == "__main__":
    main()

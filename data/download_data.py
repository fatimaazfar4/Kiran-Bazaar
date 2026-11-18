"""Re-download the raw data from HDX into data/raw/.   python data/download_data.py

Both files are published by WFP on the Humanitarian Data Exchange under CC BY-IGO.
"""
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
BASE = "https://data.humdata.org/dataset/125455ff-c8a8-4f4b-9190-bf0e84559dc7/resource"
FILES = {
    "wfp_food_prices_pak.csv": f"{BASE}/8cf7a8a2-58a1-4c3f-bafe-f38af68d971b/download/wfp_food_prices_pak.csv",
    "wfp_markets_pak.csv": f"{BASE}/dfe6237b-7534-4610-af7d-0b7165726193/download/wfp_markets_pak.csv",
}

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        print("downloading", name)
        urllib.request.urlretrieve(url, RAW / name)
    print("done. Now run: python scripts/train.py")

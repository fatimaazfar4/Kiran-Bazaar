"""Project-wide constants for Kiran Bazaar."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

PRICES_CSV = RAW_DIR / "wfp_food_prices_pak.csv"
MARKETS_CSV = RAW_DIR / "wfp_markets_pak.csv"
FEATURES_CSV = PROCESSED_DIR / "features.csv"

# A "spike" = next-month retail price rises by more than this fraction.
SPIKE_THRESHOLD = 0.05

# Food staples we predict for (targets). Fuel and wage series are used as
# explanatory features only. Salt (ends 2018) and gram lentils (9 months of
# data) are dropped because they are too short to model.
TARGET_COMMODITIES = [
    "Wheat flour", "Wheat", "Rice (basmati, broken)", "Rice (coarse)",
    "Sugar", "Oil (cooking)", "Ghee (artificial)", "Milk", "Eggs", "Poultry",
    "Lentils (masur)", "Lentils (moong)", "Beans(mash)",
]
DIESEL = "Fuel (diesel)"
PETROL = "Fuel (petrol-gasoline)"
WAGE = "Wage (non-qualified labour, non-agricultural)"

# Friendly names for the apps (atta = wheat flour, etc.)
DISPLAY_NAMES = {
    "Wheat flour": "Wheat flour (atta)",
    "Wheat": "Wheat (grain)",
    "Rice (basmati, broken)": "Rice - basmati (broken)",
    "Rice (coarse)": "Rice - coarse",
    "Sugar": "Sugar (cheeni)",
    "Oil (cooking)": "Cooking oil",
    "Ghee (artificial)": "Ghee (banaspati)",
    "Milk": "Milk (doodh)",
    "Eggs": "Eggs (anday)",
    "Poultry": "Chicken (murghi)",
    "Lentils (masur)": "Daal masoor",
    "Lentils (moong)": "Daal moong",
    "Beans(mash)": "Daal mash",
}

CATEGORICAL_FEATURES = ["market", "commodity", "category", "province"]
NUMERIC_FEATURES = [
    "month", "latitude", "longitude",
    "ret_1", "ret_2", "ret_3", "ret_6", "ret_12",
    "vol_6", "ma_ratio_3", "ma_ratio_12", "max12_ratio", "spikes_12",
    "nat_gap", "nat_ret_1", "basket_ret_1",
    "diesel_ret_1", "diesel_ret_3", "petrol_ret_1", "wage_ret_3",
    "fx_ret_1", "fx_ret_3",
]
# Level features such as log(price) or the PKR/USD level are deliberately NOT used: they trend upward
# for 20 years, so a model trained on 2014-2021 would have to extrapolate on 2022+ data.

# Modelling window: before 2014 only rice/wheat exist and fuel/wage series are empty.
MODEL_START = "2014-01-01"
TARGET_CLF = "spike_next"
TARGET_REG = "ret_next"
RANDOM_STATE = 42

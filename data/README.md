# Data sources

| File | Source | Publisher | Licence | Coverage | Retrieved |
|---|---|---|---|---|---|
| `raw/wfp_food_prices_pak.csv` | [HDX: WFP Food Prices for Pakistan](https://data.humdata.org/dataset/wfp-food-prices-for-pakistan), resource *Pakistan - Food Prices* (`8cf7a8a2-58a1-4c3f-bafe-f38af68d971b`) | World Food Programme (WFP); underlying sources: FAO GIEWS / FPMA, Pakistan Bureau of Statistics | CC BY-IGO | 2004-01 to 2026-10, 5 cities, 18 series, 14,430 rows × 16 columns | 2026-11-22 |
| `raw/wfp_markets_pak.csv` | Same HDX dataset, resource *Pakistan - Markets* (`dfe6237b-7534-4610-af7d-0b7165726193`) | WFP | CC BY-IGO | 234 markets with province, district, latitude, longitude | 2026-11-18 (resource updated 2026-02-18) |

Re-download with `python data/download_data.py`.

## Collection method
WFP compiles monthly retail prices from the Pakistan Bureau of Statistics (via FAO GIEWS FPMA). Prices are stamped
on the 15th of each month, in PKR, with a USD conversion (`usdprice`) computed by WFP using the monthly exchange rate.

## How the sources are combined
1. **Food price series** (13 staples): the prediction targets and their lag/rolling features.
2. **Energy and labour series** from the same table (diesel, petrol, unskilled daily wage): joined by market and month as cost-push drivers.
3. **Implied USD/PKR exchange rate**: `sum(price) / sum(usdprice)` per month, which recovers WFP's conversion rate as a macro driver.
4. **Markets table**: joined on `market` to add province and latitude/longitude.

## Processing
`scripts/train.py` → `src/kiranbazaar/features.py` builds `processed/features.csv`, one row per *(city, staple, month)*,
with 4 categorical and 22 numeric features (5 more are derived inside the Pipeline) and the targets
`spike_next` (next-month price rise > 5 %) and `ret_next` (next-month % change).
Modelling window: **2014-01 to 2026-09**, giving **9,633 labelled rows** (2026-10 has no outcome yet and is used for live forecasts).

## Personal data
None. All values are market-level aggregate prices; no individuals are identified.

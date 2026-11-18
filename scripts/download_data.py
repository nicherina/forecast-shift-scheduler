"""Get the Kaggle 'Recruit Restaurant Visitor Forecasting' competition data
and prepare the processed CSVs the pipeline expects.

Three ways to get the raw data:

A. API (automated, needs a Kaggle API token):
     python scripts/download_data.py
   Prerequisites: pip install kaggle, ~/.kaggle/kaggle.json, accepted rules.

B. Manual (no token): log into kaggle.com once, accept the competition rules,
   download the zip from the Data tab, save it as
     data/raw/<competition>.zip
   then run:
     python scripts/download_data.py --prepare-zip data/raw/<name>.zip

C. No Kaggle at all: skip the download and run the pipeline on synthetic
   data (python -m src.demo) — the repo works end-to-end without it.

Produces:
  data/processed/visitors.csv   store_id, date, visitors
  data/processed/holidays.csv   date  (Japanese public holidays)

NOTE: competition data is licensed under the competition rules — the
.gitignore excludes data/ so nothing is redistributed via this repo.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

COMPETITION = "recruit-restaurant-visitor-forecasting"
RAW = Path("data/raw")
OUT = Path("data/processed")


def download_api() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"downloading {COMPETITION} ...")
    api.competition_download_files(COMPETITION, path=RAW, quiet=False)
    for z in RAW.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW)
    print("extracted:", [f.name for f in RAW.glob('*')][:12])


def unzip_local(zip_path: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW)
    print(f"extracted {zip_path} -> {RAW}/ :",
          [f.name for f in RAW.glob('*')][:12])


def prepare() -> None:
    visits = pd.read_csv(RAW / "air_visit_data.csv")
    visits = visits.rename(columns={"visit_date": "date"})
    OUT.mkdir(parents=True, exist_ok=True)
    visits.to_csv(OUT / "visitors.csv", index=False)

    cal = pd.read_csv(RAW / "date_info.csv")
    cal.loc[cal["holiday_flg"] == 1, ["calendar_date"]] \
        .rename(columns={"calendar_date": "date"}) \
        .to_csv(OUT / "holidays.csv", index=False)
    print(f"visitors: {len(visits)} rows, "
          f"{visits['store_id'].nunique()} stores, "
          f"{pd.to_datetime(visits['date']).min().date()} .. "
          f"{pd.to_datetime(visits['date']).max().date()}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-zip", metavar="ZIP",
                   help="extract an already-downloaded competition zip "
                        "(manual path B, no Kaggle token needed)")
    args = p.parse_args()
    if args.prepare_zip:
        unzip_local(args.prepare_zip)
    else:
        download_api()
    prepare()

# forecast-shift-scheduler

**Demand forecast → staffing requirements → MILP shift scheduling**, demonstrated
end-to-end on real restaurant data.

The case it mirrors: workforce management platforms for restaurants/retail turn a
*nested revenue forecast* into a *legal, cost-optimal shift plan* (coverage vs
labour cost vs employee preferences). This repo implements the two core engine
steps from scratch with open data.

## Pipeline

```
visitors.csv (per store, per day)          ── real: Kaggle Recruit Restaurant
        │  HistGradientBoosting                Visitor Forecasting (POS data,
        │  lags · rolling · calendar           829 stores, ~2016-04..2017-04)
        ▼
visitors forecast (test days)
        │  stated assumptions (src/config.py):
        │  hourly profile · staff per guests
        ▼
per-slot staffing requirements  (date · hour · role)
        │  MIP: x[employee, role, shift, day] ∈ {0,1}
        │  min  labour cost + under/over coverage penalties
        │        + evening-preference surcharges
        │  s.t. coverage · 1 shift/day · 40h·5-shift weekly caps
        ▼
shift schedule  +  KPI comparison vs greedy baseline
```

Everything tunable lives in **`src/config.py`** — the modelling assumptions are
stated, not hidden in code.

## Two solvers, same model

| Solver | Licence | Notes |
|---|---|---|
| **Gurobi MIP** | free size-limited license via pip (~2k vars) | first choice for small instances; commercial-grade LP/MIP engine |
| **OR-Tools CP-SAT** | fully free | identical model, used as fallback and to compare runtime |

Run `--solver cpsat` / `--solver gurobi` / `--solver auto` on identical inputs
to compare objective and runtime. Abstraction point: `src/optimization/solve.py`.

## Quickstart (no data download needed)

```bash
pip install -r requirements.txt
python -m src.demo                       # synthetic 100 days, 10 stores
```

The demo trains the forecaster, plans one store's week, solves the MIP and
prints the schedule-vs-baseline KPI table; outputs land in `reports/`.

## Real data

```bash
# one-time: kaggle API token at ~/.kaggle/kaggle.json + accept comp rules
python scripts/download_data.py          # -> data/processed/{visitors,holidays}.csv
python -m src.demo --source kaggle --store <store_id>
```
`data/` is git-ignored: competition data may not be redistributed.

## Tests

```bash
pytest -q        # coverage/shift-cap/constraint smoke tests on CP-SAT
```

## Roadmap / honest TODOs

- [ ] leak-free rolling aggregates in `src/forecast/features.py` (currently
      store_mean uses the full series)
- [ ] LightGBM + reservation-time features (the Kaggle data has them)
- [ ] multi-store scheduling (currently one location per run)
- [ ] Gurobi benchmark table auto-generated for both solvers
- [ ] Streamlit demo: sliders for demand → schedule calendar view

## Dev workflow

Optimisation models and notebooks developed interactively with **Claude Code**
(prompt-driven model drafting, test-first fixes) — the same AI-assisted coding
workflow these solver teams use daily.

## Data licence

Recruit Restaurant Visitor Forecasting data © Recruit Holdings, distributed via
Kaggle under the competition rules (download, do not redistribute). Benchmark
ideas for a second, harder phase: nurse-rostering instances (psmet1/nrp).

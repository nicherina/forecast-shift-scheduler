"""Compare solver schedules vs the greedy baseline, printed as a table."""
from __future__ import annotations

import pandas as pd

from ..optimization.common import build_shifts
from .baseline import baseline_greedy
from .metrics import compute_kpis


def compare(demand: pd.DataFrame, employees, result_mip: dict) -> pd.DataFrame:
    emp_by_id = {e.eid: e for e in employees}
    base = baseline_greedy(demand, employees)
    rows = {}
    mip = compute_kpis(result_mip["schedule"], emp_by_id, demand)
    rows[f"MIP ({result_mip['solver']})"] = {**mip,
        "runtime_s": result_mip["runtime_s"], "status": result_mip["status"]}
    rows["greedy baseline"] = {
        **compute_kpis(base["schedule"], emp_by_id, demand),
        "runtime_s": None, "status": "heuristic"}
    out = pd.DataFrame(rows).T
    cols = ["labour_cost", "under_hours", "over_hours",
            "demand_met", "shifts", "avg_hours_per_shift", "runtime_s", "status"]
    for c in cols:
        if c not in out.columns:
            out[c] = None
    return out[cols]


def save_report(compared: pd.DataFrame, schedule: pd.DataFrame, out_dir: str) -> str:
    from pathlib import Path
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    compared.to_csv(p / "comparison.csv")
    schedule.to_csv(p / "schedule.csv", index=False)
    return str(p)

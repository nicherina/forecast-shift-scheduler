"""Shared KPI computation from a *schedule* (rows of committed shifts).

Used identically for the MIP solution and the greedy baseline so the cost /
coverage comparison in evaluate.py is apples-to-apples.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .. import config
from ..optimization.common import build_shifts, requirement_lookup

SHIFTS = build_shifts()
SHIFT_BY_NAME = {sh.id: sh for sh in SHIFTS}


def compute_kpis(schedule: pd.DataFrame, emp_by_id: dict, demand: pd.DataFrame) -> dict:
    req = requirement_lookup(demand)
    covered = defaultdict(float)
    labour = 0.0
    for r in schedule.itertuples():
        sh = SHIFT_BY_NAME[f"{int(r.hours)}h_from_{int(r.start):02d}"]
        e = emp_by_id[r.employee]
        for h in range(sh.start, sh.end):
            if sh.covers(h):
                covered[(r.date, h, r.role)] += 1
                labour += config.WAGE[r.role] * e.wage_mult

    under = over = 0.0
    for k, need in req.items():
        got = covered.get(k, 0)
        under += max(0.0, need - got)
        over += max(0.0, got - need)

    total = float(sum(req.values()))
    return {
        "labour_cost": round(labour, 2),
        "under_hours": round(under, 1),
        "over_hours": round(over, 1),
        "demand_met": round(1 - under / max(total, 1e-9), 4),
        "shifts": len(schedule),
        "avg_hours_per_shift": round(float(schedule["hours"].mean()), 2) if len(schedule) else 0.0,
    }

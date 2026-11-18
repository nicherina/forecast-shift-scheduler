"""Naive reactive staffing baseline — what a manager plausibly does without
optimisation: walk through each slot chronologically and commit staff to the
shortest covering shift until demand is met. It over-commits late in the day
(a shift taken for one busy hour covers 8 quiet hours) and stalls when the
pool is exhausted — exactly the failure modes the MIP is supposed to fix."""
from __future__ import annotations

import pandas as pd

from .. import config
from ..optimization.common import build_shifts, requirement_lookup
from ..optimization.employees import Employee


def baseline_greedy(demand: pd.DataFrame, employees: list[Employee]) -> dict:
    req = requirement_lookup(demand)
    shifts = build_shifts()
    dates = sorted({d for (d, _, _) in req})
    hours = sorted({h for (_, h, _) in req})
    week_hours: dict[tuple[str, str], float] = {}

    rows = []
    for d in dates:
        week = pd.Timestamp(d).strftime("%G-W%V")
        assigned_today: set[str] = set()
        covered: dict[tuple[str, int], int] = {}

        for h in hours:
            for role in config.ROLES:
                need = int(req.get((d, h, role), 0))
                while covered.get((role, h), 0) < need:
                    placed = False
                    candidates = [
                        e for e in employees
                        if e.eid not in assigned_today
                        and role in e.roles
                        and d in e.availability
                        and week_hours.get((e.eid, week), 0.0) < e.weekly_max_hours
                    ]
                    # prefer staff whose speciality is the needed role? keep it
                    # naive on purpose; candidate order == pool order
                    for e in candidates:
                        feasible = [
                            sh for sh in shifts
                            if sh.covers(h) and sh.start <= h
                            and week_hours.get((e.eid, week), 0.0) + sh.length
                            <= e.weekly_max_hours
                        ]
                        if not feasible:
                            continue
                        sh = min(feasible, key=lambda s: (s.length, s.start))
                        rows.append({
                            "employee": e.eid, "role": role, "date": d,
                            "start": sh.start, "end": sh.end, "hours": sh.length,
                        })
                        assigned_today.add(e.eid)
                        week_hours[(e.eid, week)] = week_hours.get((e.eid, week), 0.0) + sh.length
                        for hh in sh.covers_range():
                            covered[(role, hh)] = covered.get((role, hh), 0) + 1
                        placed = True
                        break
                    if not placed:
                        break  # nobody qualified & free → demand stays unmet

    schedule = pd.DataFrame(rows)
    if schedule.empty:
        schedule = pd.DataFrame(columns=["employee", "role", "date", "start", "end", "hours"])
    return {"solver": "greedy_baseline", "status": "heuristic",
            "runtime_s": None, "objective": None,
            "schedule": schedule, "under_hours": None, "over_hours": None,
            "demand_met": None, "labour_cost": None}

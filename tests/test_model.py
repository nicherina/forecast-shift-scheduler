"""Smoke tests for the optimisation core (no solver licence needed — uses
CP-SAT, which is free). Run from the project root:  pytest -q"""
from __future__ import annotations

import pandas as pd

from src.optimization.demand import visitors_to_demand
from src.optimization.employees import make_employee_pool
from src.optimization.solve import solve
from src.evaluation.metrics import compute_kpis


def _tiny_demand() -> pd.DataFrame:
    visitors = pd.Series(
        {pd.Timestamp("2026-11-20"): 60.0, pd.Timestamp("2026-11-21"): 90.0})
    return visitors_to_demand(visitors)


def test_solve_respects_one_shift_per_day_and_coverage():
    demand = _tiny_demand()
    dates = sorted(demand["date"].unique())
    emps = make_employee_pool(6, dates, seed=1)
    res = solve(demand, emps, solver="cpsat", time_limit=10)
    assert res["status"] in ("optimal", "feasible")

    sched = res["schedule"]
    # at most one shift per employee per day
    per_day = sched.groupby(["employee", "date"]).size()
    assert (per_day <= 1).all()

    # weekly hour cap not exceeded (single week here)
    hrs = sched.groupby(["employee", "date"])["hours"].sum().groupby("employee").sum()
    emap = {e.eid: e for e in emps}
    for eid, h in hrs.items():
        assert h <= emap[eid].weekly_max_hours + 1e-9

    # KPI recompute is consistent: demand mostly met for such small demand
    kpis = compute_kpis(sched, emap, demand)
    assert kpis["demand_met"] > 0.9
    assert len(sched) > 0


def test_demand_builder_caps_and_scales():
    demand = _tiny_demand()
    assert set(demand["role"]) == {"service", "kitchen"}
    assert demand["required"].between(0, 10).all()

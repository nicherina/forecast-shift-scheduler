"""CP-SAT (OR-Tools, free) implementation — same model as the Gurobi MIP,
so the pipeline runs without any commercial licence and the two solvers can
be compared on runtime and objective for the same instance.
"""
from __future__ import annotations

import time

import pandas as pd
from ortools.sat.python import cp_model

from .. import config
from .common import allowed_vars, pref_penalty, requirement_lookup, weekly_key


def solve_cpsat(demand: pd.DataFrame, employees, time_limit: float = 30.0) -> dict:
    req = requirement_lookup(demand)
    pairs, dates, shifts = allowed_vars(employees, demand)
    emp_by_id = {e.eid: e for e in employees}
    sh_by_id = {sh.id: sh for sh in shifts}

    m = cp_model.CpModel()
    x = {(e.eid, r, sh.id, d): m.NewBoolVar(f"x_{e.eid}_{sh.id}_{d}_{r}")
         for (e, r, sh, d) in pairs}
    cap = float(max(req.values(), default=0)) + 8.0
    u = {k: m.NewIntVar(0, int(cap + 1), f"u_{k}") for k in req}
    o = {k: m.NewIntVar(0, int(cap + 1) + 20, f"o_{k}") for k in req}

    terms = []
    for (eid, r, sid, _d), v in x.items():
        terms.append(int(config.WAGE[r] * sh_by_id[sid].length
                         * emp_by_id[eid].wage_mult) * v)
        terms.append(int(pref_penalty(emp_by_id[eid], sh_by_id[sid])) * v)
    terms += [int(config.PENALTY_UNDER) * v for v in u.values()]
    terms += [-int(config.PENALTY_OVER) * v for v in o.values()]
    m.Minimize(sum(terms))

    # (1) coverage
    per_shift = {}
    for (eid, r, sid, d), v in x.items():
        per_shift.setdefault((d, r), []).append((sh_by_id[sid], v))
    for (d, h, role), need in req.items():
        covering = [v for sh, v in per_shift.get((d, role), []) if sh.covers(h)]
        m.Add(sum(covering) + u[(d, h, role)] - o[(d, h, role)] == int(need))

    # (2) one shift per employee per day
    for e in employees:
        for d in dates:
            day_vars = [v for (eid, _r, _sid, dd), v in x.items()
                        if eid == e.eid and dd == d]
            if day_vars:
                m.Add(sum(day_vars) <= 1)

    # (3)+(4) weekly caps
    for e in employees:
        weeks = sorted({weekly_key(d) for d in dates})
        for wk in weeks:
            in_wk = [(sh_by_id[sid], v) for (eid, _r, sid, dd), v in x.items()
                     if eid == e.eid and weekly_key(dd) == wk]
            if not in_wk:
                continue
            m.Add(sum(sh.length * v for sh, v in in_wk) <= e.weekly_max_hours)
            m.Add(sum(v for _, v in in_wk) <= e.weekly_max_shifts)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    t0 = time.time()
    status = solver.Solve(m)
    runtime = time.time() - t0

    ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    status_str = {cp_model.OPTIMAL: "optimal", cp_model.FEASIBLE: "feasible"}.get(
        status, "no_solution")

    rows = []
    if ok:
        for (eid, r, sid, d), v in x.items():
            if solver.Value(v):
                sh = sh_by_id[sid]
                rows.append({"employee": eid, "role": r, "date": d,
                             "start": int(sh.start), "end": sh.end,
                             "hours": sh.length})
    schedule = pd.DataFrame(rows).sort_values(["date", "start", "employee"]) if rows else pd.DataFrame()

    under_hours = float(sum(solver.Value(v) for v in u.values())) if ok else float("nan")
    over_hours = float(sum(solver.Value(v) for v in o.values())) if ok else float("nan")
    total_req = float(sum(req.values()))
    labour = 0.0
    if ok:
        for (eid, r, sid, _d), v in x.items():
            if solver.Value(v):
                labour += config.WAGE[r] * sh_by_id[sid].length * emp_by_id[eid].wage_mult
    return {
        "solver": "cpsat",
        "status": status_str,
        "runtime_s": round(runtime, 2),
        "objective": round(solver.ObjectiveValue()) if ok else None,
        "schedule": schedule,
        "under_hours": under_hours,
        "over_hours": over_hours,
        "demand_met": 1 - under_hours / max(total_req, 1e-9) if ok else None,
        "labour_cost": labour,
    }

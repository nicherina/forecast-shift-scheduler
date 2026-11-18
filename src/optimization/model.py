"""Gurobi MIP implementation of the weekly shift-scheduling problem.

Variables
    x[e, r, s, d] in {0,1}  employee e works role r, shift s, on date d
    u[d, h, r]   >= 0       under-covered slot-hours (demand missed)
    o[d, h, r]   >= 0       over-covered slot-hours (idle staff)

Objective
    min  labour cost + PENALTY_UNDER * u - PENALTY_OVER * o
         + evening preference surcharges

Constraints
    (1) coverage:      sum of covering assignments == required + u - o
        per (date, hour, role)
    (2) one shift per employee per day
    (3) weekly hours cap per employee (ISO calendar week)
    (4) weekly shift-count cap per employee
"""
from __future__ import annotations

import time

import pandas as pd


def solve_gurobi(demand: pd.DataFrame, employees, time_limit: float = 30.0) -> dict:
    import gurobipy as gp
    from gurobipy import GRB

    from .. import config
    from ..optimization.employees import Employee  # noqa: F401 (typing clarity)
    from .common import allowed_vars, pref_penalty, requirement_lookup, weekly_key

    req = requirement_lookup(demand)
    pairs, dates, shifts = allowed_vars(employees, demand)
    emp_by_id = {e.eid: e for e in employees}
    sh_by_id = {sh.id: sh for sh in shifts}

    m = gp.Model("shift_scheduling")
    x = {(e.eid, r, sh.id, d): m.addVar(vtype=GRB.BINARY, name=f"x_{e.eid}_{sh.id}_{d}_{r}")
         for (e, r, sh, d) in pairs}
    u = m.addVars(req.keys(), lb=0.0, name="undercover")
    o = m.addVars(req.keys(), lb=0.0, name="overcover")

    # Objective (x keys are (eid, role, shift_id, date))
    labour = gp.quicksum(
        config.WAGE[r] * sh_by_id[sid].length * emp_by_id[eid].wage_mult * v
        for (eid, r, sid, _d), v in x.items())
    evening = gp.quicksum(
        pref_penalty(emp_by_id[eid], sh_by_id[sid]) * v
        for (eid, _r, sid, _d), v in x.items())
    under = gp.quicksum(config.PENALTY_UNDER * v for v in u.values())
    over = gp.quicksum(config.PENALTY_OVER * v for v in o.values())
    m.setObjective(labour + evening + under - over, GRB.MINIMIZE)

    # (1) coverage per date/hour/role
    per_emp_day = {}
    for (eid, r, sid, d), v in x.items():
        per_emp_day.setdefault((eid, d), []).append(v)
    per_slot_cover = {}
    for (eid, r, sid, d), v in x.items():
        per_slot_cover.setdefault((d, r), []).append((sh_by_id[sid], v))

    for (d, h, role), need in req.items():
        covering = gp.quicksum(
            v for sh, v in per_slot_cover.get((d, role), []) if sh.covers(h))
        m.addConstr(covering + u[(d, h, role)] - o[(d, h, role)] == need)

    # (2) one shift per employee per day
    for e in employees:
        for d in dates:
            day_vars = per_emp_day.get((e.eid, d), [])
            if day_vars:
                m.addConstr(gp.quicksum(day_vars) <= 1)

    # (3)+(4) weekly hour and shift-count caps
    for e in employees:
        weeks = sorted({weekly_key(d) for d in dates})
        for wk in weeks:
            in_wk = [(sh_by_id[sid], v) for (eid, _r, sid, dd), v in x.items()
                     if eid == e.eid and weekly_key(dd) == wk]
            if not in_wk:
                continue
            m.addConstr(gp.quicksum(sh.length * v for sh, v in in_wk)
                        <= e.weekly_max_hours)
            m.addConstr(gp.quicksum(v for _, v in in_wk)
                        <= e.weekly_max_shifts)

    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = 0.01
    t0 = time.time()
    m.optimize()
    runtime = time.time() - t0

    status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time-limit"}.get(
        m.Status, f"status_{m.Status}")

    rows = []
    for (eid, r, sid, d), v in x.items():
        if v.X > 0.5:
            sh = sh_by_id[sid]
            rows.append({
                "employee": eid, "role": r, "date": d,
                "start": sh.start, "end": sh.end, "hours": sh.length,
            })
    schedule = pd.DataFrame(rows).sort_values(["date", "start", "employee"])

    under_hours = float(sum(v.X for v in u.values()))
    over_hours = float(sum(v.X for v in o.values()))
    total_req = float(sum(req.values()))
    return {
        "solver": "gurobi",
        "status": status,
        "runtime_s": round(runtime, 2),
        "objective": float(m.ObjVal) if status == "optimal" else None,
        "schedule": schedule,
        "under_hours": under_hours,
        "over_hours": over_hours,
        "demand_met": 1 - under_hours / max(total_req, 1e-9),
        "labour_cost": float(sum(
            config.WAGE[r] * sh_by_id[sid].length * emp_by_id[eid].wage_mult * v.X
            for (eid, r, sid, _d), v in x.items() if v.X > 0.5)),
    }

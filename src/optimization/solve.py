"""Solver dispatcher: 'gurobi', 'cpsat' or 'auto' (Gurobi if installed/licensed,
CP-SAT otherwise). Same interface and result dict for both."""
from __future__ import annotations

import pandas as pd


def solve(demand: pd.DataFrame, employees, solver: str = "auto",
          time_limit: float = 30.0) -> dict:
    if solver in ("auto", "gurobi"):
        try:
            import gurobipy  # noqa: F401
            from .model import solve_gurobi
            return solve_gurobi(demand, employees, time_limit)
        except ImportError:
            if solver == "gurobi":
                raise
    if solver == "cpsat" or solver == "auto":
        from .cpsat_model import solve_cpsat
        return solve_cpsat(demand, employees, time_limit)
    raise ValueError(f"unknown solver: {solver!r}")

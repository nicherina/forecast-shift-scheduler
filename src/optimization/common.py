"""Shared index/structure builders used by both solver implementations."""
from dataclasses import dataclass

import pandas as pd

from .. import config
from .employees import Employee


@dataclass
class Shift:
    start: int
    length: int
    id: str

    @property
    def end(self) -> int:
        return self.start + self.length

    def covers(self, hour: int) -> bool:
        return self.start <= hour < self.end

    def covers_range(self):
        return range(self.start, self.end)


def build_shifts() -> list[Shift]:
    out = []
    for s in config.SHIFT_STARTS:
        for length in config.SHIFT_LENGTHS:
            if s + length <= config.CLOSING_HOUR:
                out.append(Shift(start=s, length=length, id=f"{length}h_from_{s:02d}"))
    return out


def weekly_key(date_str: str) -> str:
    """ISO calendar week ('2026-W47') so weekly hour caps are legal weeks."""
    return pd.Timestamp(date_str).strftime("%G-W%V")


def allowed_vars(employees: list[Employee], demand: pd.DataFrame):
    """Enumerate feasible (employee, role, shift, date) combos."""
    dates = sorted(demand["date"].unique())
    shifts = build_shifts()
    pairs = []
    for e in employees:
        for d in dates:
            if d not in e.availability:
                continue
            for r in config.ROLES:
                if r not in e.roles:
                    continue
                for sh in shifts:
                    pairs.append((e, r, sh, d))
    return pairs, dates, shifts


def pref_penalty(e: Employee, sh: Shift) -> float:
    return e.evening_penalty if sh.start >= config.EVENING_START_HOUR else 0.0


def requirement_lookup(demand: pd.DataFrame) -> dict[tuple[str, int, str], float]:
    return {
        (r.date, int(r.hour), r.role): float(r.required)
        for r in demand.itertuples()
    }

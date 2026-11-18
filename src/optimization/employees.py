"""Employee dataclass shared by the models and the baseline."""
from dataclasses import dataclass, field


@dataclass
class Employee:
    eid: str
    roles: set            # subset of config.ROLES this person is qualified for
    availability: set     # set of ISO date strings ("2026-11-20") they can work
    wage_mult: float = 1.0
    weekly_max_hours: int = 40
    weekly_max_shifts: int = 5
    evening_penalty: float = 0.0   # extra EUR cost for shifts starting >= EVENING_START_HOUR


def make_employee_pool(n: int, dates_iso: list[str], seed: int = 42) -> list[Employee]:
    """Small deterministic pool: mixed skill sets, staggered rest days,
    a few employees with an evening preference surcharge.

    The first three employees never rest — guards small pools against
    accidentally starving a whole role on one date (the 1-shift/day cap makes
    staffing very tight at this size)."""
    import random
    rng = random.Random(seed)
    emps = []
    for i in range(n):
        day_off = dates_iso[i % len(dates_iso)] if (dates_iso and i >= 3) else None
        avail = {d for d in dates_iso} - ({day_off} if day_off else set())
        roles = {"service", "kitchen"} if i % 4 == 0 else (
            {"service"} if i % 2 == 0 else {"kitchen"})
        evening = 15.0 if i % 5 == 3 else 0.0
        emps.append(Employee(
            eid=f"emp_{i:02d}",
            roles=roles,
            availability=avail,
            evening_penalty=evening,
        ))
    # make sure every date has both roles covered by *someone always available*
    if dates_iso:
        for role in ("service", "kitchen"):
            if not any(role in e.roles for e in emps[:3]):
                emps[0].roles.add(role)
    return emps

"""Central assumptions of the forecast -> staffing -> scheduling pipeline.

Every number here is a *stated, adjustable business assumption* — the point of
keeping them in one file is that interview reviewers can see (and challenge)
the modeling choices explicitly.
"""

# --- Restaurant opening hours -------------------------------------------------
OPEN_HOURS = list(range(10, 23))   # 10:00 .. 22:00 start slots
CLOSING_HOUR = 23                  # shifts must end by CLOSING_HOUR (exclusive end)

# Hourly share of daily visitors (index 0 == 10:00). Sums to 1 after
# normalisation in demand.py — lunch peak around 12-14, dinner peak 18-21.
HOURLY_PROFILE = [
    0.02, 0.04, 0.06, 0.07, 0.06, 0.05, 0.04, 0.06,
    0.10, 0.13, 0.13, 0.10, 0.06,
]

# --- Roles --------------------------------------------------------------------
ROLES = ["service", "kitchen"]

# How many guests each role must serve per slot-hour before another member of
# that role is required ("staffing factor").
VISITORS_PER_STAFF_PER_SLOT = {"service": 20.0, "kitchen": 30.0}

# Fraction of guests attributable to each role (must sum to 1.0).
ROLE_VISITOR_SHARE = {"service": 0.6, "kitchen": 0.4}

MAX_STAFF_PER_SLOT = 8  # physical cap (floor space, kitchen size)

# --- Shift model ---------------------------------------------------------------
SHIFT_STARTS = list(range(9, 20))   # 09:00 .. 19:00 starts
SHIFT_LENGTHS = [4, 6, 8]           # hours

# --- Employees ------------------------------------------------------------------
WAGE = {"service": 15.0, "kitchen": 17.0}   # EUR per hour per role
WEEKLY_HOURS_MAX = 40
WEEKLY_SHIFTS_MAX = 5
EVENING_START_HOUR = 16  # shifts starting >= this hour count as "evening"

# --- Solver tuning ----------------------------------------------------------------
PENALTY_UNDER = 500.0   # per uncovered slot-hour (demand missed)
PENALTY_OVER = 10.0     # per over-covered slot-hour (idle staff)

N_EMPLOYEES_DEFAULT = 12
FORECAST_DAYS_DEFAULT = 7

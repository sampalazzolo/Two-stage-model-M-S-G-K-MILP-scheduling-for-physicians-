import math

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

# ---------------------------------------------------------------------------
# Stage 1: M/G/s/K priority queue -> hourly staffing requirement s*(t)
# ---------------------------------------------------------------------------

N_annual = 100687
lam_daily = N_annual / 365.0

w = np.array([
    0.0227, 0.0180, 0.0104, 0.0114, 0.0099, 0.0151,
    0.0137, 0.0279, 0.0407, 0.0643, 0.0676, 0.0728,
    0.0624, 0.0572, 0.0591, 0.0601, 0.0553, 0.0525,
    0.0482, 0.0473, 0.0478, 0.0544, 0.0478, 0.0331,
])
w = w / w.sum()

# 3-hour centered moving average, cyclic over 24 h, repeated for 7 days
lam_raw = lam_daily * w
ext = np.concatenate(([lam_raw[-1]], lam_raw, [lam_raw[0]]))
lam_24 = np.convolve(ext, np.ones(3) / 3, mode='valid')
lam_168 = np.tile(lam_24, 7)

p1, p2 = 0.15, 0.85                 # ESI 1-2 / ESI 3-5
kappa = 4.0                         # multitasking factor
ES1 = 317.346310 / kappa / 60       # mean service time, hours
ES2 = 148.472331 / kappa / 60
cs1_sq, cs2_sq = 1.18787, 0.73154
ES_pool = p1 * ES1 + p2 * ES2

K_sys = 70                          # total system capacity
W1_target, W2_target = 0.25, 1.50   # hours (15 and 90 min)
RHO_MAX = 0.98


def blocking_prob(s, K, lam, es):
    """Blocking probability P_K of an M/M/s/K queue."""
    a = lam * es
    rho = a / s
    head = sum(a**n / math.factorial(n) for n in range(s))
    term_s = a**s / math.factorial(s)
    if abs(rho - 1.0) < 1e-7:
        tail = term_s * (K - s + 1)
    else:
        tail = term_s * (1 - rho**(K - s + 1)) / (1 - rho)
    return term_s * rho**(K - s) / (head + tail)


def queue_delays(s, lam1, lam2):
    """Mean waiting times (hours) for the two classes:
    Sakasegawa for the multiserver delay, Cobham for the priorities."""
    rho1, rho2 = lam1 * ES1 / s, lam2 * ES2 / s
    rho = rho1 + rho2
    if rho >= RHO_MAX:
        return math.inf, math.inf
    lam = lam1 + lam2
    es = (lam1 * ES1 + lam2 * ES2) / lam
    es2 = (lam1 * ES1**2 * (cs1_sq + 1) + lam2 * ES2**2 * (cs2_sq + 1)) / lam
    cs_sq = es2 / es**2 - 1
    wq_fcfs = (1 + cs_sq) / 2 * rho**(math.sqrt(2 * (s + 1)) - 1) / (s * (1 - rho)) * es
    w0 = wq_fcfs * (1 - rho)
    return w0 / (1 - rho1), w0 / ((1 - rho1) * (1 - rho))


T = list(range(1, 169))
s_star = {}
for t in T:
    lam = lam_168[t - 1]
    for s in range(1, 61):
        lam_eff = lam * (1 - blocking_prob(s, K_sys, lam, ES_pool))
        wq1, wq2 = queue_delays(s, p1 * lam_eff, p2 * lam_eff)
        if wq1 <= W1_target and wq2 <= W2_target:
            s_star[t] = s
            break
    else:
        raise RuntimeError(f'No feasible number of servers for hour {t}')

# ---------------------------------------------------------------------------
# Stage 2: MILP rostering
# ---------------------------------------------------------------------------

DAYS = list(range(1, 8))
EXEMPT = [5, 12, 23, 41, 55]        # night-exempt physicians

# shift -> (start hour, end hour); end > 24 means it ends the next day
SHIFTS_RIGID = {'M': (8, 14), 'P': (14, 20), 'N': (20, 32)}
SHIFTS_SPLIT = {**SHIFTS_RIGID, 'N1': (20, 26), 'N2': (2, 8)}
COST = {'M': 210.0, 'P': 210.0, 'N': 540.0, 'N1': 270.0, 'N2': 270.0, 'R': 0.0}
NIGHT = ('N', 'N1', 'N2')

C_ot = 60.0
H_target, H_min, H_max = 38.0, 32.0, 48.0
P_under, P_over = 10000.0, 50.0
NIGHT_HOURS_CAP = 24.0              # = 2 twelve-hour nights per week
MIN_REST = 11                       # hours between consecutive shifts


def solve_roster(n_doc=60, split_night=False, verbose=False, time_limit=120):
    shifts = SHIFTS_SPLIT if split_night else SHIFTS_RIGID
    K = list(shifts) + ['R']
    h = {k: shifts[k][1] - shifts[k][0] for k in shifts}
    h['R'] = 0.0
    I = list(range(1, n_doc + 1))
    exempt = [i for i in EXEMPT if i <= n_doc]
    night = [k for k in NIGHT if k in shifts]

    def nxt(d):
        return d % 7 + 1

    # covers[t] = (day, shift) pairs that cover hour t
    covers = {t: [] for t in T}
    for d in DAYS:
        for k, (s, e) in shifts.items():
            for hr in range(s, e):
                covers[((d - 1) * 24 + hr) % 168 + 1].append((d, k))

    m = gp.Model('ED_roster')
    m.Params.OutputFlag = int(verbose)
    m.Params.TimeLimit = time_limit

    x = m.addVars(I, DAYS, K, vtype=GRB.BINARY, name='x')
    y = m.addVars(I, lb=0.0, name='y')
    v_minus = m.addVars(T, lb=0.0, name='v_minus')
    v_plus = m.addVars(T, lb=0.0, name='v_plus')

    staff_cost = (
        gp.quicksum(COST[k] * x[i, d, k] for i in I for d in DAYS for k in K)
        + gp.quicksum(C_ot * y[i] for i in I)
    )
    penalty = gp.quicksum(P_under * v_minus[t] + P_over * v_plus[t] for t in T)
    m.setObjective(staff_cost + penalty, GRB.MINIMIZE)

    # coverage
    for t in T:
        m.addConstr(
            gp.quicksum(x[i, d, k] for i in I for d, k in covers[t])
            + v_minus[t] - v_plus[t] == s_star[t]
        )

    for i in I:
        for d in DAYS:
            # one shift (or rest) per day
            m.addConstr(gp.quicksum(x[i, d, k] for k in K) == 1)
            # minimum rest between a shift on day d and one on day d+1
            for k, (s, e) in shifts.items():
                for k2, (s2, e2) in shifts.items():
                    if 24 + s2 - e < MIN_REST:
                        m.addConstr(x[i, d, k] + x[i, nxt(d), k2] <= 1)

        # at most 2 night shifts in any 3 consecutive days
        for d in DAYS:
            span = [d, nxt(d), nxt(nxt(d))]
            m.addConstr(gp.quicksum(x[i, dd, k] for dd in span for k in night) <= 2)

        # weekly rest day
        m.addConstr(gp.quicksum(x[i, d, 'R'] for d in DAYS) >= 1)

        # night exemption / weekly cap on night hours
        if i in exempt:
            for d in DAYS:
                for k in night:
                    m.addConstr(x[i, d, k] == 0)
        else:
            m.addConstr(
                gp.quicksum(h[k] * x[i, d, k] for d in DAYS for k in night)
                <= NIGHT_HOURS_CAP
            )

        # weekly hours and overtime
        hours = gp.quicksum(h[k] * x[i, d, k] for d in DAYS for k in K)
        m.addConstr(hours <= H_max)
        m.addConstr(hours >= H_min)
        m.addConstr(y[i] >= hours - H_target)

    m.optimize()
    if m.SolCount == 0:
        return None

    # blocking probability at the staffing actually delivered by the roster
    cover = {t: round(sum(x[i, d, k].X for i in I for d, k in covers[t])) for t in T}
    pk = [
        blocking_prob(cover[t], K_sys, lam_168[t - 1], ES_pool) if cover[t] >= 1 else 1.0
        for t in T
    ]
    night_t = [t for t in T if (t - 1) % 24 >= 20 or (t - 1) % 24 < 8]

    schedule = pd.DataFrame(
        [{'doc': i, 'day': d, 'shift': k}
         for i in I for d in DAYS for k in K if x[i, d, k].X > 0.5]
    ).pivot(index='doc', columns='day', values='shift')

    return {
        'n_doc': n_doc,
        'night': 'split' if split_night else 'rigid',
        'status': m.Status,
        'staff_cost': staff_cost.getValue(),
        'understaff_h': sum(v_minus[t].X for t in T),
        'overtime_h': sum(y[i].X for i in I),
        'surplus_h': sum(v_plus[t].X for t in T),
        'surplus_night_h': sum(v_plus[t].X for t in night_t),
        'pK_mean_%': 100 * np.mean(pk),
        'pK_max_%': 100 * np.max(pk),
        'schedule': schedule,
    }


if __name__ == '__main__':
    print(
        f's*(t): min {min(s_star.values())}, max {max(s_star.values())}, '
        f'total {sum(s_star.values())} physician-hours/week'
    )

    base = []
    for split in (False, True):
        r = solve_roster(60, split)
        base.append(r)
        print(f"\n{r['night']} nights - first 15 physicians")
        print(r['schedule'].head(15))

    cols = [c for c in base[0] if c != 'schedule']
    print('\n', pd.DataFrame([{c: r[c] for c in cols} for r in base]).round(2).to_string(index=False))

    # sensitivity to the number of physicians
    rows = []
    for n in range(60, 43, -2):
        for split in (False, True):
            r = solve_roster(n, split)
            if r is not None:
                rows.append({c: r[c] for c in cols})
    print('\n', pd.DataFrame(rows).round(2).to_string(index=False))

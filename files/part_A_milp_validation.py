# ================================================================
# part_A_milp_validation.py
# Exact algorithmic validation: 10-customer instance, MILP
# (epsilon-constraint method) vs. NSGA-II, on the final cost-emission
# last-mile fleet (bike, e-van; no diesel -- Proposed scenario)
# Requires: model_config.py, hfvrp_model.py, network_utils.py, stats_utils.py
# in the same folder
# ================================================================
import random
import numpy as np
import osmnx as ox
import pulp
from shapely.geometry import Point
from pyproj import Transformer
from pymoo.indicators.igd import IGD
import matplotlib.pyplot as plt

from model_config import build_fleet, generate_customer_demand
from hfvrp_model import HFVRPProblem
from network_utils import build_street_graphs, build_network_distance_matrix, build_drone_distance_matrix
from stats_utils import run_multiple_seeds

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ================================================================
# Part 1: generate 50 customers (same seed as main experiments) and take first 10
# ================================================================
place = "London Borough of Hackney, United Kingdom"
boundary = ox.geocode_to_gdf(place)
polygon = boundary.iloc[0].geometry
minx, miny, maxx, maxy = polygon.bounds

N_cust_full = 50
customers_latlon = []
while len(customers_latlon) < N_cust_full:
    p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
    if polygon.contains(p):
        customers_latlon.append((p.y, p.x))

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
def project_point(lat, lon):
    x, y = transformer.transform(lon, lat)
    return (x, y)

depot_proj = project_point(51.5830, -0.0198)
ucc_proj = project_point(51.5473, -0.0558)
customers_proj = [project_point(lat, lon) for lat, lon in customers_latlon]
all_points_proj = [depot_proj, ucc_proj] + customers_proj

weights_all, volumes_all, cats_all = generate_customer_demand(N_cust_full, seed=SEED)

G_drive, G_bike = build_street_graphs(place, extra_latlon_points=[(51.5830, -0.0198)])
dist_drive_km, _ = build_network_distance_matrix(G_drive, all_points_proj)
dist_bike_km, _ = build_network_distance_matrix(G_bike, all_points_proj)
dist_drone_km = build_drone_distance_matrix(all_points_proj)
print("Real network distance matrices computed.")

N_VAL = 10
idx_map = [1] + list(range(2, 2 + N_VAL))  # 0=UCC, 1..10=customers
dist_bike_10 = dist_bike_km[np.ix_(idx_map, idx_map)]
dist_drive_10 = dist_drive_km[np.ix_(idx_map, idx_map)]
dist_drone_10 = dist_drone_km[np.ix_(idx_map, idx_map)]
weights_10 = weights_all[:N_VAL]
volumes_10 = volumes_all[:N_VAL]

_ref_fleet = build_fleet(N_VAL, last_mile_only=True)
_n_bike = sum(1 for v in _ref_fleet if v['type'] == 'bike')
_n_evan = sum(1 for v in _ref_fleet if v['type'] == 'evan')
_n_drone = sum(1 for v in _ref_fleet if v['type'] == 'drone')
print(f"Last-mile fleet (sized for N={N_VAL}, 20% safety margin): "
      f"{_n_bike} Bike + {_n_evan} EVan + {_n_drone} Drone (no diesel)")

# ================================================================
# Part 2: MILP with epsilon-constraint — built directly from build_fleet
# to avoid any risk of duplicated/inconsistent numbers vs NSGA-II
# ================================================================
_dist_by_key = {'bike': dist_bike_10, 'drive': dist_drive_10, 'drone': dist_drone_10}
fleet_val = []
for v in _ref_fleet:
    fleet_val.append({
        'type': v['type'], 'cap': v['cap_w'], 'cap_v': v['cap_v'],
        'range': v['range'], 'max_stops': v['max_stops'],
        'fixed_cost_per_stop': v['fixed_cost_per_stop'],
        'fuel_cost_per_km': v['fuel_cost_per_km'],
        'em_km': v['em_km'], 'dist': _dist_by_key[v['dist_key']],
    })
K_val = list(range(len(fleet_val)))

C = list(range(1, N_VAL + 1))
q = {i: weights_10[i - 1] for i in C}
vol = {i: volumes_10[i - 1] for i in C}
eligible = {k: [i for i in C if q[i] <= fleet_val[k]['cap'] and vol[i] <= fleet_val[k]['cap_v']] for k in K_val}


def build_and_solve(epsilon=None, objective='cost', time_limit=300):
    prob = pulp.LpProblem("EpsilonVRP", pulp.LpMinimize)
    x = {}
    for k in K_val:
        allowed_from = [0] + eligible[k]
        allowed_to = eligible[k] + [0]
        for i in allowed_from:
            for j in allowed_to:
                if i != j:
                    x[i, j, k] = pulp.LpVariable(f"x_{i}_{j}_{k}", cat='Binary')
    u = {}
    u_vol = {}
    for k in K_val:
        for i in eligible[k]:
            u[i, k] = pulp.LpVariable(f"u_{i}_{k}", lowBound=0, upBound=fleet_val[k]['cap'])
            u_vol[i, k] = pulp.LpVariable(f"uvol_{i}_{k}", lowBound=0, upBound=fleet_val[k]['cap_v'])

    def arcs(k):
        return [(i, j) for i in ([0] + eligible[k]) for j in (eligible[k] + [0]) if i != j]

    # Cost: distance x fuel-cost-per-km + number of stops x fixed-cost-per-stop
    cost_expr = (
        pulp.lpSum(fleet_val[k]['dist'][i][j] * fleet_val[k]['fuel_cost_per_km'] * x[i, j, k]
                   for k in K_val for (i, j) in arcs(k) if (i, j, k) in x)
        + pulp.lpSum(fleet_val[k]['fixed_cost_per_stop'] * x[i, j, k]
                     for k in K_val for (i, j) in arcs(k) if (i, j, k) in x and j != 0)
    )
    emission_expr = pulp.lpSum(
        fleet_val[k]['dist'][i][j] * fleet_val[k]['em_km'] * x[i, j, k]
        for k in K_val for (i, j) in arcs(k) if (i, j, k) in x)

    prob += cost_expr if objective == 'cost' else emission_expr
    if epsilon is not None:
        prob += emission_expr <= epsilon

    for j in C:
        prob += pulp.lpSum(x[i, j, k] for k in K_val if j in eligible[k]
                            for i in ([0] + eligible[k]) if i != j and (i, j, k) in x) == 1

    for k in K_val:
        nodes_k = [0] + eligible[k]
        for h in nodes_k:
            prob += (pulp.lpSum(x[i, h, k] for i in nodes_k if i != h and (i, h, k) in x) ==
                     pulp.lpSum(x[h, j, k] for j in nodes_k if j != h and (h, j, k) in x))
        prob += pulp.lpSum(x[0, j, k] for j in eligible[k] if (0, j, k) in x) <= 1
        prob += (pulp.lpSum(x[i, j, k] for i in nodes_k for j in eligible[k] if i != j and (i, j, k) in x)
                 <= fleet_val[k]['max_stops'])
        prob += (pulp.lpSum(fleet_val[k]['dist'][i][j] * x[i, j, k]
                             for i in nodes_k for j in nodes_k if i != j and (i, j, k) in x)
                 <= fleet_val[k]['range'])

    for k in K_val:
        Qk = fleet_val[k]['cap']
        Qk_vol = fleet_val[k]['cap_v']
        for i in eligible[k]:
            for j in eligible[k]:
                if i != j and (i, j, k) in x:
                    prob += u[i, k] - u[j, k] + Qk * x[i, j, k] <= Qk - q[j]
                    # Volume MTZ constraint (critical bug fix: previously missing)
                    prob += u_vol[i, k] - u_vol[j, k] + Qk_vol * x[i, j, k] <= Qk_vol - vol[j]
        if eligible[k]:
            y_k = pulp.LpVariable(f"y_{k}", cat='Binary')
            prob += y_k == pulp.lpSum(x[0, j, k] for j in eligible[k] if (0, j, k) in x)
            for i in eligible[k]:
                prob += u[i, k] >= q[i] * y_k
                prob += u_vol[i, k] >= vol[i] * y_k

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    if prob.status != pulp.LpStatusOptimal:
        return None
    return pulp.value(cost_expr), pulp.value(emission_expr)


print("\nSolving anchor points with the cost-emission formula...")
cost_min, emission_at_cost_min = build_and_solve(epsilon=None, objective='cost')
_, emission_min = build_and_solve(epsilon=None, objective='emission')
print(f"Anchor cost_min:     cost={cost_min:.3f} GBP, emission={emission_at_cost_min:.1f} g")
print(f"Anchor emission_min: emission={emission_min:.1f} g")

n_steps = 10
epsilons = np.linspace(emission_min, emission_at_cost_min, n_steps)
exact_pareto = []
n_failed = 0
for eps in epsilons:
    res = build_and_solve(epsilon=eps, objective='cost')
    if res is not None:
        exact_pareto.append(res)
    else:
        n_failed += 1
exact_pareto = np.array(sorted(set(exact_pareto)))
if n_failed > 0:
    print(f"\n⚠️ {n_failed}/{n_steps} epsilon-constraint sub-problems did not reach proven "
          f"optimality within time_limit and were dropped. If this number is large, increase "
          f"time_limit further and re-run.")
print(f"\nExact MILP Pareto front: {len(exact_pareto)} non-dominated points.")
print(exact_pareto)

# ================================================================
# Part 3: NSGA-II on the same 10 customers, same fleet and cost formula
# ================================================================
fleet_nsga = build_fleet(N_VAL, last_mile_only=True)
dist_mats_10 = {'drive': dist_drive_10, 'bike': dist_bike_10, 'drone': dist_drone_10}
# Note: at this small scale, node indexing is +1 (UCC + 10 customers only), not +2
from pymoo.core.problem import Problem
from model_config import route_cost_time, two_opt, two_opt_route_length

class HFVRPProblem10(Problem):
    def __init__(self, N_cust, fleet, weights, volumes, dist_mats, start_node, depot_return_node):
        self.N_cust, self.fleet, self.K = N_cust, fleet, len(fleet)
        self.weights, self.volumes, self.dist_mats = weights, volumes, dist_mats
        self.start_node, self.depot_return_node = start_node, depot_return_node
        super().__init__(n_var=2 * N_cust, n_obj=2, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        N = x.shape[0]
        costs, emissions = np.zeros(N), np.zeros(N)
        for ind in range(N):
            chrom = x[ind]
            veh_choice = np.clip(np.floor(chrom[:self.N_cust] * self.K).astype(int), 0, self.K - 1)
            route_priority = np.argsort(chrom[self.N_cust:])
            veh_custs = {k: [] for k in range(self.K)}
            for cust in route_priority:
                veh_custs[veh_choice[cust]].append(cust)
            total_cost = total_emission = penalty = 0.0
            for k in range(self.K):
                if not veh_custs[k]:
                    continue
                v = self.fleet[k]
                dist_mat = self.dist_mats[v['dist_key']]
                current, unvisited = self.start_node, veh_custs[k][:]
                total_dist = load_w = load_v = 0.0
                stops = 0
                order = []
                while unvisited and stops < v['max_stops']:
                    next_cust, min_dist = None, np.inf
                    for c in unvisited:
                        d = dist_mat[current][c + 1]
                        if not np.isfinite(d):
                            continue
                        return_d = dist_mat[c + 1][self.depot_return_node]
                        if (load_w + self.weights[c] <= v['cap_w'] and load_v + self.volumes[c] <= v['cap_v']
                                and total_dist + d + return_d <= v['range'] and d < min_dist):
                            min_dist, next_cust = d, c
                    if next_cust is None:
                        break
                    total_dist += min_dist
                    load_w += self.weights[next_cust]; load_v += self.volumes[next_cust]
                    unvisited.remove(next_cust); current = next_cust + 1; stops += 1
                    order.append(next_cust + 1)
                if len(order) >= 3:
                    order = two_opt(order, dist_mat, self.start_node, self.depot_return_node)
                total_dist = two_opt_route_length(order, dist_mat, self.start_node, self.depot_return_node)
                route_cost, route_emission = route_cost_time(total_dist, len(order), v)
                total_cost += route_cost; total_emission += route_emission
                penalty += 1000 * len(unvisited)
            costs[ind] = total_cost + penalty; emissions[ind] = total_emission + penalty
        out["F"] = np.column_stack([costs, emissions])


problem_10 = HFVRPProblem10(N_VAL, fleet_nsga, weights_10, volumes_10, dist_mats_10,
                             start_node=0, depot_return_node=0)

N_SEEDS_VAL = 20
print(f"\nRunning {N_SEEDS_VAL} independent NSGA-II seeds on the same 10 customers...")
results_10 = run_multiple_seeds(problem_10, n_seeds=N_SEEDS_VAL, pop_size=250, n_gen=350, print_progress=False)
print(f"  {N_SEEDS_VAL}/{N_SEEDS_VAL} seeds completed.")
all_F_nsga = np.vstack([r.F for r in results_10])

igd_indicator = IGD(exact_pareto)
igd_per_seed = np.array([igd_indicator(r.F) for r in results_10])
igd_combined = igd_indicator(all_F_nsga)
print(f"\nIGD relative to exact MILP front (cost-emission formula):")
print(f"   Mean over {N_SEEDS_VAL} seeds: {igd_per_seed.mean():.4f} +/- {igd_per_seed.std():.4f}")
print(f"   Combined-front IGD: {igd_combined:.4f}")

plt.figure(figsize=(9, 6))
plt.scatter(all_F_nsga[:, 0], all_F_nsga[:, 1], s=10, alpha=0.3, color='green', label='NSGA-II (all seeds)')
plt.plot(exact_pareto[:, 0], exact_pareto[:, 1], 'ro-', label='Exact MILP front (epsilon-constraint)')
plt.xlabel("Cost (GBP)"); plt.ylabel("Emission (g CO2)")
plt.title(f"NSGA-II vs Exact MILP — {N_VAL} customers, cost-emission model (IGD={igd_combined:.3f})")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("milp_vs_nsga2_validation_EN.png", dpi=300, bbox_inches="tight")
plt.show()

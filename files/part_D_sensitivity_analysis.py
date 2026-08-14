# ================================================================
# part_D_sensitivity_analysis.py
# One-at-a-time (OAT) sensitivity analysis on documented uncertain
# parameters, at N=50: cost (+/-20%), emission (+/-20%), drone labour
# scenario (realistic vs. optimistic), drone payload capacity
# (2.0/3.63/5.0 kg), and emission-accounting scope (full life-cycle vs.
# operational-only, per Santiago-Montano, Silva, & Smith, 2024).
# Requires: model_config.py, hfvrp_model.py, network_utils.py, stats_utils.py
#
# NOTE: the drone-payload and emission-scope dimensions were re-run at
# full experimental rigour (20 seeds, 250 generations) in
# part_D_reverify_flagged_v2.py after showing high run-to-run
# variability at the reduced budget used below (5 seeds, 150
# generations); see that script and the manuscript's Results section
# for the final, stable values on those two dimensions.
# ================================================================
import random
import numpy as np
import matplotlib.pyplot as plt

from model_config import build_fleet, generate_customer_demand, compute_trunk_haul, TRUNK_FLEET
from hfvrp_model import HFVRPProblem
from network_utils import build_street_graphs, build_network_distance_matrix, build_drone_distance_matrix
from stats_utils import run_multiple_seeds, shared_reference_point, compute_hypervolume_stats

plt.rcParams['savefig.dpi'] = 300

SEED = 42
N_CUST = 50
N_SEEDS_SENS = 5     # reduced from 8 to fit time budget
POP_SIZE, N_GEN = 100, 150   # N_GEN reduced from 200

np.random.seed(SEED)
random.seed(SEED)

# ================================================================
# Base data (built once, reused for all sensitivity scenarios)
# ================================================================
place = "London Borough of Hackney, United Kingdom"
import osmnx as ox
from pyproj import Transformer
from network_utils import generate_customer_locations_realistic

customers_latlon, location_method = generate_customer_locations_realistic(N_CUST, place, seed=SEED)
print(f"Location method: {location_method}")

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
def project_point(lat, lon):
    x, y = transformer.transform(lon, lat)
    return (x, y)

depot_proj = project_point(51.5830, -0.0198)
ucc_proj = project_point(51.5473, -0.0558)
customers_proj = [project_point(lat, lon) for lat, lon in customers_latlon]
all_points_proj = [depot_proj, ucc_proj] + customers_proj

weights, volumes, categories = generate_customer_demand(N_CUST, seed=SEED)
G_drive, G_bike = build_street_graphs(place, extra_latlon_points=[(51.5830, -0.0198)])
dist_drive_km, _ = build_network_distance_matrix(G_drive, all_points_proj)
dist_bike_km, _ = build_network_distance_matrix(G_bike, all_points_proj)
dist_drone_km = build_drone_distance_matrix(all_points_proj)
dist_mats = {'drive': dist_drive_km, 'bike': dist_bike_km, 'drone': dist_drone_km}
print("Base data prepared once for all scenarios.")

# Quick check: how many customers are drone-eligible under each payload option?
for cap in [2.0, 3.63, 5.0]:
    n_elig = int(((weights <= cap) & (volumes <= 0.03)).sum())
    print(f"Drone-eligible customers (weight<={cap}kg AND volume<=0.03 m3): {n_elig}/{N_CUST}")


def run_proposed_scenario(fleet_override=None):
    """Run the Proposed scenario with a (possibly perturbed) fleet; return raw results."""
    fleet = fleet_override if fleet_override is not None else build_fleet(N_CUST, last_mile_only=True)
    problem = HFVRPProblem(N_CUST, fleet, weights, volumes, dist_mats, start_node=1, depot_return_node=1)
    return run_multiple_seeds(problem, n_seeds=N_SEEDS_SENS, pop_size=POP_SIZE, n_gen=N_GEN, print_progress=False)


# ================================================================
# Step 1: run all scenarios and collect raw results (no HV yet —
# HV across scenarios is only comparable with one shared reference point)
# ================================================================
scenario_results = {}

print("\n[1/9] Baseline scenario (drone=realistic, payload=3.63kg)...")
scenario_results['Baseline (drone=realistic)'] = run_proposed_scenario(
    build_fleet(N_CUST, last_mile_only=True, drone_scenario='realistic'))

print("\n[2/9] Sensitivity: drone labour scenario = optimistic...")
scenario_results['Drone labour=optimistic'] = run_proposed_scenario(
    build_fleet(N_CUST, last_mile_only=True, drone_scenario='optimistic'))

for pct, label in [(1.2, 'Cost +20%'), (0.8, 'Cost -20%')]:
    print(f"\n[{'3' if pct>1 else '4'}/9] Sensitivity: {label}...")
    fleet_cost_alt = build_fleet(N_CUST, last_mile_only=True)
    for v in fleet_cost_alt:
        v['fixed_cost_per_stop'] *= pct
        v['fuel_cost_per_km'] *= pct
    scenario_results[label] = run_proposed_scenario(fleet_cost_alt)

for pct, label in [(1.2, 'Emission +20%'), (0.8, 'Emission -20%')]:
    print(f"\n[{'5' if pct>1 else '6'}/9] Sensitivity: {label}...")
    fleet_em_alt = build_fleet(N_CUST, last_mile_only=True)
    for v in fleet_em_alt:
        v['em_km'] *= pct
    scenario_results[label] = run_proposed_scenario(fleet_em_alt)

# --- Drone payload capacity: conservative (2.0kg) vs growth-scenario (5.0kg) ---
for cap, label in [(2.0, 'Drone payload=2.0kg (conservative)'), (5.0, 'Drone payload=5.0kg (growth scenario)')]:
    print(f"\n[7/9] Sensitivity: {label}...")
    scenario_results[label] = run_proposed_scenario(
        build_fleet(N_CUST, last_mile_only=True, drone_payload_kg=cap))

# --- Emission accounting scope: full life-cycle (base) vs operational-only ---
# Operational-only values: Santiago-Montaño, Silva, & Smith (2024), International
# Journal of Sustainable Transportation, 18(10), 887-902 (via Eun et al., 2019 for
# drones and Lee et al., 2019 for e-bikes). Evan has no matching alternative source
# and is left unchanged.
print("\n[8-9/9] Sensitivity: emission scope = operational-only (bike, drone, diesel)...")
fleet_operational_only = build_fleet(N_CUST, last_mile_only=True)
for v in fleet_operational_only:
    if v['type'] == 'bike':
        v['em_km'] = 8.4
    elif v['type'] == 'drone':
        v['em_km'] = 0.8
scenario_results['Emission scope=operational-only'] = run_proposed_scenario(fleet_operational_only)

# ================================================================
# Step 2: ONE shared reference point across ALL scenarios (critical fix —
# using separate reference points per scenario previously caused a
# spurious +4000%-style artifact)
# ================================================================
shared_ref = shared_reference_point(*scenario_results.values())
print(f"\nShared reference point (across all scenarios): {shared_ref}")

sensitivity_results = {}
for name, results in scenario_results.items():
    hv_stats = compute_hypervolume_stats(results, shared_ref)
    sensitivity_results[name] = (hv_stats['mean'], hv_stats['std'])

base_name = 'Baseline (drone=realistic)'
hv_base, std_base = sensitivity_results[base_name]
print(f"\nBaseline HV (shared reference) = {hv_base:.2f} +/- {std_base:.2f}")
for name, (hv, std) in sensitivity_results.items():
    if name == base_name:
        continue
    print(f"  {name}: HV={hv:.2f} +/- {std:.2f}  (change vs baseline: {(hv-hv_base)/hv_base*100:+.1f}%)")

# ================================================================
# Trunk-haul fixed-cost sensitivity (direct calculation, no NSGA-II needed)
# ================================================================
import copy
print("\n[Extra] Trunk-haul vehicle-choice sensitivity to fixed cost per trip...")
dist_depot_ucc = dist_drive_km[0][1]
for pct, label in [(1.0, 'Baseline'), (1.2, 'Fixed cost +20%'), (0.8, 'Fixed cost -20%')]:
    trunk_fleet_alt = copy.deepcopy(TRUNK_FLEET)
    for v in trunk_fleet_alt.values():
        v['fixed_cost_per_stop'] *= pct
    trunk_results = compute_trunk_haul(weights.sum(), volumes.sum(), dist_depot_ucc, trunk_fleet_alt)
    best_name, best = min(trunk_results.items(), key=lambda kv: kv[1]['cost'])
    print(f"   {label}: best option={best_name}, cost={best['cost']:.2f} GBP, emission={best['emission']:.1f} g")

# ================================================================
# Summary table + Tornado chart (English, 300 dpi)
# ================================================================
print(f"\n{'='*70}\nSensitivity analysis summary (relative to baseline HV = {hv_base:.2f})\n{'='*70}")
print(f"{'Scenario':30s} {'HV':>14s} {'Change %':>10s}")
names, changes = [], []
for name, (hv, std) in sensitivity_results.items():
    if name == base_name:
        continue
    pct_change = (hv - hv_base) / hv_base * 100
    print(f"{name:30s} {hv:>14.2f} {pct_change:>+9.1f}%")
    names.append(name)
    changes.append(pct_change)

order = np.argsort(np.abs(changes))[::-1]
plt.figure(figsize=(10, 6))
colors = ['crimson' if changes[i] < 0 else 'seagreen' for i in order]
plt.barh([names[i] for i in order], [changes[i] for i in order], color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel("Change in Hypervolume relative to baseline (%)")
plt.title(f"Sensitivity Analysis Tornado Chart (N={N_CUST})")
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig("sensitivity_tornado.png", dpi=300, bbox_inches="tight")
plt.show()

np.savez("sensitivity_results.npz", **{k: v[0] for k, v in sensitivity_results.items()})
print("\nSaved: sensitivity_results.npz, sensitivity_tornado.png")

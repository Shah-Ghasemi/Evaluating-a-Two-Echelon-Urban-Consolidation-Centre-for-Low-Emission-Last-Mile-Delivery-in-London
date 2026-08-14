# ================================================================
# part_D_reverify_flagged.py
# Full-rigour version: since only the drone-payload dimension (not
# the emission-scope dimension, already stabilised) remained unstable,
# both seed count and generation budget are raised to the standard
# used throughout the main experiments (20 seeds, 250 generations) --
# not just more seeds, but a full opportunity to converge, since the
# mechanistic explanation for the earlier instability (a weak penalty
# signal for dominated-but-permitted options) implicated insufficient
# *generations*, not only seed count. Only 3 scenarios (baseline + two
# drone-payload variants) are re-run here; the emission-scope scenario
# is omitted, having already stabilised at 10 seeds (CV = 10.6%).
# ================================================================
import numpy as np
import pandas as pd

from model_config import build_fleet, generate_customer_demand
from hfvrp_model import HFVRPProblem
from network_utils import build_street_graphs, build_network_distance_matrix, build_drone_distance_matrix, generate_customer_locations_realistic
from stats_utils import run_multiple_seeds, shared_reference_point, compute_hypervolume_stats

SEED = 42
N_CUST = 50
N_SEEDS_SENS = 20   # matches the standard used in the main experiments
POP_SIZE, N_GEN = 100, 250   # matches the standard used in the main experiments

place = "London Borough of Hackney, United Kingdom"
G_drive, G_bike = build_street_graphs(place, extra_latlon_points=[(51.5830, -0.0198)])

customers_latlon, location_method = generate_customer_locations_realistic(N_CUST, place, seed=SEED)
print(f"Location method: {location_method}")

from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
def project_point(lat, lon):
    x, y = transformer.transform(lon, lat)
    return (x, y)
depot_proj = project_point(51.5830, -0.0198)
ucc_proj = project_point(51.5473, -0.0558)
customers_proj = [project_point(lat, lon) for lat, lon in customers_latlon]
all_points_proj = [depot_proj, ucc_proj] + customers_proj

weights, volumes, categories = generate_customer_demand(N_CUST, seed=SEED)
dist_drive_km, _ = build_network_distance_matrix(G_drive, all_points_proj)
dist_bike_km, _ = build_network_distance_matrix(G_bike, all_points_proj)
dist_drone_km = build_drone_distance_matrix(all_points_proj)
dist_mats = {'drive': dist_drive_km, 'bike': dist_bike_km, 'drone': dist_drone_km}
print("Base data prepared.")


def run_proposed_scenario(fleet_override=None):
    fleet = fleet_override if fleet_override is not None else build_fleet(N_CUST, last_mile_only=True)
    problem = HFVRPProblem(N_CUST, fleet, weights, volumes, dist_mats, start_node=1, depot_return_node=1)
    return run_multiple_seeds(problem, n_seeds=N_SEEDS_SENS, pop_size=POP_SIZE, n_gen=N_GEN, print_progress=False)


scenario_results = {}

print("\n[1/3] Baseline (drone=realistic, payload=3.63kg) -- full rigor re-run...")
scenario_results['Baseline'] = run_proposed_scenario(
    build_fleet(N_CUST, last_mile_only=True, drone_scenario='realistic'))

print("\n[2/3] Drone payload=2.0kg (conservative)...")
scenario_results['Drone payload=2.0kg (conservative)'] = run_proposed_scenario(
    build_fleet(N_CUST, last_mile_only=True, drone_payload_kg=2.0))

print("\n[3/3] Drone payload=5.0kg (growth scenario)...")
scenario_results['Drone payload=5.0kg (growth scenario)'] = run_proposed_scenario(
    build_fleet(N_CUST, last_mile_only=True, drone_payload_kg=5.0))

# Shared reference point, computed only across these scenarios (for a fair within-batch comparison)
shared_ref = shared_reference_point(*scenario_results.values())
print(f"\nShared reference point (this batch): {shared_ref}")

results_table = {}
for name, results in scenario_results.items():
    hv_stats = compute_hypervolume_stats(results, shared_ref)
    results_table[name] = (hv_stats['mean'], hv_stats['std'])

hv_base, std_base = results_table['Baseline']
print(f"\n{'='*70}\nRe-verified results (N_SEEDS_SENS={N_SEEDS_SENS}, N_GEN={N_GEN} -- full experimental rigor)\n{'='*70}")
print(f"Baseline: HV={hv_base:.2f} +/- {std_base:.2f}")
rows = []
for name, (hv, std) in results_table.items():
    pct = (hv - hv_base) / hv_base * 100
    cv = std / hv * 100
    rows.append({'Scenario': name, 'HV': hv, 'SD': std, 'CV (%)': cv,
                 'Change vs baseline (%)': pct if name != 'Baseline' else 0.0})
    if name != 'Baseline':
        print(f"  {name}: HV={hv:.2f} +/- {std:.2f} (CV={cv:.1f}%)  change vs baseline: {pct:+.1f}%")

df = pd.DataFrame(rows)
df.to_csv("sensitivity_reverified_flagged_v2.csv", index=False)
print("\nSaved: sensitivity_reverified_flagged_v2.csv")

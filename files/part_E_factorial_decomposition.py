# ================================================================
# part_E_factorial_decomposition.py
# Decomposes the Proposed-vs-Baseline improvement into two separate
# main effects -- consolidation (Depot vs UCC origin) and fleet
# composition (diesel-inclusive vs green-only last-mile fleet) --
# via a 2x2 factorial design at N = 50. Addresses the reviewer
# critique that the original comparison changes both factors at once.
#
# The four cells:
#   A. Depot + full fleet (bike,evan,drone,diesel)   <- original "Baseline"
#   B. Depot + green fleet only (bike,evan,drone)     <- NEW
#   C. UCC   + full fleet (bike,evan,drone,diesel)    <- NEW
#   D. UCC   + green fleet only (bike,evan,drone)      <- original "Proposed"
#
# All four are run fresh, at the SAME budget (20 seeds, pop 100,
# 250 gens), and compared under ONE shared Hypervolume reference
# point across all four -- do not reuse HV values from earlier runs
# computed under a different (2-scenario) reference point; mixing
# reference points was exactly the mistake that distorted the
# sensitivity-analysis tornado chart earlier in this project.
#
# Requires: model_config.py, hfvrp_model.py, network_utils.py, stats_utils.py
# ================================================================
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import mannwhitneyu

from model_config import build_fleet, generate_customer_demand
from hfvrp_model import HFVRPProblem
from network_utils import (build_street_graphs, build_network_distance_matrix,
                            build_drone_distance_matrix, generate_customer_locations_realistic)
from stats_utils import run_multiple_seeds_checkpointed, shared_reference_point, compute_hypervolume_stats

plt.rcParams['savefig.dpi'] = 300

SEED = 42
N_CUST = 50
N_SEEDS, POP_SIZE, N_GEN = 20, 100, 250

np.random.seed(SEED)
random.seed(SEED)

# ================================================================
# Base data -- built once, identical to the N=50 case used elsewhere
# in this paper, so results are directly comparable
# ================================================================
place = "London Borough of Hackney, United Kingdom"
from pyproj import Transformer

print("Building/loading street graphs...")
G_drive, G_bike = build_street_graphs(place, extra_latlon_points=[(51.5830, -0.0198)])

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
dist_drive_km, _ = build_network_distance_matrix(G_drive, all_points_proj)
dist_bike_km, _ = build_network_distance_matrix(G_bike, all_points_proj)
dist_drone_km = build_drone_distance_matrix(all_points_proj)
dist_mats = {'drive': dist_drive_km, 'bike': dist_bike_km, 'drone': dist_drone_km}
print("Base data prepared.")

# ================================================================
# The 2x2 factorial: (origin) x (fleet composition)
# ================================================================
CELLS = {
    'A_Depot_FullFleet':  dict(last_mile_only=False, start_node=0, label='Depot + full fleet (incl. diesel)'),
    'B_Depot_GreenFleet': dict(last_mile_only=True,  start_node=0, label='Depot + green fleet only'),
    'C_UCC_FullFleet':    dict(last_mile_only=False, start_node=1, label='UCC + full fleet (incl. diesel)'),
    'D_UCC_GreenFleet':   dict(last_mile_only=True,  start_node=1, label='UCC + green fleet only'),
}

results_by_cell = {}
t_start = time.time()

for cell_id, cfg in CELLS.items():
    print(f"\n{'='*70}\nCELL {cell_id}: {cfg['label']}\n{'='*70}")
    t0 = time.time()
    fleet = build_fleet(N_CUST, last_mile_only=cfg['last_mile_only'])
    problem = HFVRPProblem(N_CUST, fleet, weights, volumes, dist_mats,
                            start_node=cfg['start_node'], depot_return_node=cfg['start_node'])
    results = run_multiple_seeds_checkpointed(
        problem, f"checkpoints_factorial_{cell_id}",
        n_seeds=N_SEEDS, pop_size=POP_SIZE, n_gen=N_GEN, save_history=False)
    results_by_cell[cell_id] = results
    print(f"[{cell_id}] completed in {(time.time()-t0)/60:.1f} minutes.")

# ================================================================
# ONE shared reference point across all FOUR cells
# ================================================================
shared_ref = shared_reference_point(*results_by_cell.values())
print(f"\nShared reference point (across all 4 cells): {shared_ref}")

hv = {}
for cell_id, results in results_by_cell.items():
    stats = compute_hypervolume_stats(results, shared_ref)
    hv[cell_id] = stats
    print(f"  {cell_id}: HV = {stats['mean']:.2f} +/- {stats['std']:.2f}")

# ================================================================
# Main effects and interaction (in log-HV, since HV spans orders of
# magnitude and ratios -- not raw differences -- are the meaningful
# comparison here, consistent with how HV ratios are reported
# elsewhere in this paper)
# ================================================================
log_hv = {c: np.log(hv[c]['mean']) for c in hv}

consolidation_effect_fullfleet = log_hv['C_UCC_FullFleet']  - log_hv['A_Depot_FullFleet']   # UCC effect, holding fleet=full
consolidation_effect_greenfleet = log_hv['D_UCC_GreenFleet'] - log_hv['B_Depot_GreenFleet']  # UCC effect, holding fleet=green

fleet_effect_depot = log_hv['B_Depot_GreenFleet'] - log_hv['A_Depot_FullFleet']   # green-fleet effect, holding origin=Depot
fleet_effect_ucc   = log_hv['D_UCC_GreenFleet']   - log_hv['C_UCC_FullFleet']     # green-fleet effect, holding origin=UCC

total_effect = log_hv['D_UCC_GreenFleet'] - log_hv['A_Depot_FullFleet']  # the original headline Proposed-vs-Baseline comparison
additive_prediction = consolidation_effect_fullfleet + fleet_effect_depot
interaction = total_effect - additive_prediction

print(f"\n{'='*70}\nDECOMPOSITION (in log-Hypervolume units)\n{'='*70}")
print(f"Consolidation effect (holding fleet=full fleet):   {consolidation_effect_fullfleet:+.3f}")
print(f"Consolidation effect (holding fleet=green fleet):  {consolidation_effect_greenfleet:+.3f}")
print(f"Fleet-composition effect (holding origin=Depot):   {fleet_effect_depot:+.3f}")
print(f"Fleet-composition effect (holding origin=UCC):     {fleet_effect_ucc:+.3f}")
print(f"Total observed effect (A -> D, the headline result): {total_effect:+.3f}")
print(f"Additive prediction (consolidation + fleet, from A): {additive_prediction:+.3f}")
print(f"Interaction (total - additive prediction):          {interaction:+.3f}")
pct_from_consolidation = consolidation_effect_fullfleet / total_effect * 100
pct_from_fleet = fleet_effect_depot / total_effect * 100
print(f"\nShare of total effect attributable to consolidation alone: {pct_from_consolidation:.1f}%")
print(f"Share of total effect attributable to fleet composition alone: {pct_from_fleet:.1f}%")
print(f"Remaining share attributable to interaction: {100 - pct_from_consolidation - pct_from_fleet:.1f}%")

# ================================================================
# Pairwise significance tests (Mann-Whitney, one-sided) for each
# main-effect comparison
# ================================================================
def mw_test(cell_a, cell_b):
    stat, p = mannwhitneyu(hv[cell_b]['values'], hv[cell_a]['values'], alternative='greater')
    return stat, p

print(f"\n{'='*70}\nSIGNIFICANCE TESTS\n{'='*70}")
for label, (a, b) in {
    'Consolidation, full fleet held (A -> C)':  ('A_Depot_FullFleet', 'C_UCC_FullFleet'),
    'Consolidation, green fleet held (B -> D)': ('B_Depot_GreenFleet', 'D_UCC_GreenFleet'),
    'Fleet composition, Depot held (A -> B)':   ('A_Depot_FullFleet', 'B_Depot_GreenFleet'),
    'Fleet composition, UCC held (C -> D)':     ('C_UCC_FullFleet', 'D_UCC_GreenFleet'),
    'Total effect, headline result (A -> D)':   ('A_Depot_FullFleet', 'D_UCC_GreenFleet'),
}.items():
    stat, p = mw_test(a, b)
    print(f"  {label}: p = {p:.2e}")

# ================================================================
# Save summary CSV + decomposition bar chart
# ================================================================
summary = pd.DataFrame([
    {'Cell': cid, 'Label': cfg['label'], 'HV_mean': hv[cid]['mean'], 'HV_std': hv[cid]['std']}
    for cid, cfg in CELLS.items()
])
summary.to_csv("factorial_decomposition_N50.csv", index=False)
print("\nSaved: factorial_decomposition_N50.csv")

plt.figure(figsize=(8, 5))
bars = ['Consolidation\n(fleet=full)', 'Consolidation\n(fleet=green)',
        'Fleet composition\n(origin=Depot)', 'Fleet composition\n(origin=UCC)',
        'Interaction']
values = [consolidation_effect_fullfleet, consolidation_effect_greenfleet,
          fleet_effect_depot, fleet_effect_ucc, interaction]
colors = ['seagreen' if v >= 0 else 'crimson' for v in values]
plt.bar(bars, values, color=colors)
plt.axhline(0, color='black', linewidth=0.8)
plt.ylabel("Effect on log(Hypervolume)")
plt.title(f"Decomposition of the Consolidation Effect (N={N_CUST})")
plt.xticks(rotation=15, ha='right')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("figure_factorial_decomposition.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: figure_factorial_decomposition.png")

print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} minutes.")

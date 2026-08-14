# ================================================================
# part_B_demand_realizations.py
# Five independent demand realisations at N=50 -- checks how stable
# the key findings (HV, e-van overflow, drone-eligible customer
# count) are when the customer sample itself changes (not only the
# NSGA-II seed). A direct response to the "static/single-instance
# data" limitation.
# ================================================================
import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from model_config import build_fleet, generate_customer_demand, compute_trunk_haul, TRUNK_FLEET
from hfvrp_model import HFVRPProblem
from network_utils import build_street_graphs, build_network_distance_matrix, build_drone_distance_matrix, generate_customer_locations_realistic
from stats_utils import run_multiple_seeds_checkpointed, shared_reference_point, compute_hypervolume_stats

plt.rcParams['savefig.dpi'] = 300

N_CUST = 50
DEMAND_SEEDS = [42, 101, 202, 303, 404]   # 5 independent demand realisations
N_NSGA_SEEDS = 20
POP_SIZE, N_GEN = 100, 250

# Checkpointing operates at two levels: each full demand realisation
# (once complete, never recomputed) and each individual NSGA-II seed
# within it (via run_multiple_seeds_checkpointed). If execution is
# interrupted, simply re-run this script -- it resumes from where it left off.
CHECKPOINT_BASE_DIR = "checkpoints_demand_realizations"
os.makedirs(CHECKPOINT_BASE_DIR, exist_ok=True)

place = "London Borough of Hackney, United Kingdom"
print("Building/loading street graphs (once for all realizations)...")
G_drive, G_bike = build_street_graphs(place, extra_latlon_points=[(51.5830, -0.0198)])

all_results = {}
for demand_seed in DEMAND_SEEDS:
    t0 = time.time()
    print(f"\n{'='*70}\nDEMAND REALIZATION seed = {demand_seed}\n{'='*70}")

    realization_ckpt = os.path.join(CHECKPOINT_BASE_DIR, f"realization_seed{demand_seed}.pkl")
    if os.path.exists(realization_ckpt):
        with open(realization_ckpt, "rb") as f:
            all_results[demand_seed] = pickle.load(f)
        print(f"OK: this realisation was already completed -- loaded from checkpoint, skipping.")
        continue

    customers_latlon, location_method = generate_customer_locations_realistic(
        N_CUST, place, seed=demand_seed)
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

    weights, volumes, categories = generate_customer_demand(N_CUST, seed=demand_seed)
    print(f"Total demand: {weights.sum():.1f} kg, {volumes.sum():.3f} m3")

    dist_drive_km, _ = build_network_distance_matrix(G_drive, all_points_proj)
    dist_bike_km, _ = build_network_distance_matrix(G_bike, all_points_proj)
    dist_drone_km = build_drone_distance_matrix(all_points_proj)
    dist_mats = {'drive': dist_drive_km, 'bike': dist_bike_km, 'drone': dist_drone_km}

    # --- Drone eligibility count (weight<=capacity AND volume<=0.03) for this realization ---
    drone_fleet_ref = build_fleet(N_CUST, last_mile_only=True)
    drone_cap = next(v['cap_w'] for v in drone_fleet_ref if v['type'] == 'drone')
    drone_eligible = int(((weights <= drone_cap) & (volumes <= 0.03)).sum())

    # --- Trunk-haul ---
    dist_depot_ucc = dist_drive_km[0][1]
    trunk_results = compute_trunk_haul(weights.sum(), volumes.sum(), dist_depot_ucc, TRUNK_FLEET)
    best_trunk_name, best_trunk = min(trunk_results.items(), key=lambda kv: kv[1]['cost'])

    # --- Proposed & Baseline (checkpointed: each seed saved individually) ---
    fleet_proposed = build_fleet(N_CUST, last_mile_only=True)
    problem_proposed = HFVRPProblem(N_CUST, fleet_proposed, weights, volumes, dist_mats,
                                     start_node=1, depot_return_node=1)
    ckpt_proposed = os.path.join(CHECKPOINT_BASE_DIR, f"seed{demand_seed}_proposed")
    results_proposed = run_multiple_seeds_checkpointed(problem_proposed, ckpt_proposed,
                                                        n_seeds=N_NSGA_SEEDS, pop_size=POP_SIZE,
                                                        n_gen=N_GEN, print_progress=False)

    fleet_baseline = build_fleet(N_CUST, last_mile_only=False)
    problem_baseline = HFVRPProblem(N_CUST, fleet_baseline, weights, volumes, dist_mats,
                                     start_node=0, depot_return_node=0)
    ckpt_baseline = os.path.join(CHECKPOINT_BASE_DIR, f"seed{demand_seed}_baseline")
    results_baseline = run_multiple_seeds_checkpointed(problem_baseline, ckpt_baseline,
                                                        n_seeds=N_NSGA_SEEDS, pop_size=POP_SIZE,
                                                        n_gen=N_GEN, print_progress=False)

    ref_point = shared_reference_point(results_proposed, results_baseline)
    hv_proposed = compute_hypervolume_stats(results_proposed, ref_point)
    hv_baseline = compute_hypervolume_stats(results_baseline, ref_point)
    stat, p_value = mannwhitneyu(hv_proposed['values'], hv_baseline['values'], alternative='greater')

    # --- Overflow to e-van (Proposed, best-cost individual per seed) ---
    def evan_share(problem, results):
        shares = []
        for res in results:
            chrom = res.X[np.argmin(res.F[:, 0])]
            veh_choice = np.clip(np.floor(chrom[:problem.N_cust] * problem.K).astype(int), 0, problem.K - 1)
            evan_idx = [k for k, v in enumerate(problem.fleet) if v['type'] == 'evan']
            shares.append(np.mean([1 if c in evan_idx else 0 for c in veh_choice]) * 100)
        return np.mean(shares), np.std(shares)

    overflow_mean, overflow_std = evan_share(problem_proposed, results_proposed)

    elapsed = time.time() - t0
    all_results[demand_seed] = {
        'location_method': location_method,
        'total_weight_kg': weights.sum(), 'total_volume_m3': volumes.sum(),
        'drone_eligible_count': drone_eligible,
        'hv_proposed_mean': hv_proposed['mean'], 'hv_proposed_std': hv_proposed['std'],
        'hv_baseline_mean': hv_baseline['mean'], 'hv_baseline_std': hv_baseline['std'],
        'hv_ratio': hv_proposed['mean'] / hv_baseline['mean'],
        'mannwhitney_p': p_value,
        'overflow_evan_pct_mean': overflow_mean, 'overflow_evan_pct_std': overflow_std,
        'trunk_best_option': best_trunk_name,
        'runtime_minutes': elapsed / 60,
    }
    with open(realization_ckpt, "wb") as f:
        pickle.dump(all_results[demand_seed], f)
    print(f"Realization seed={demand_seed} completed in {elapsed/60:.1f} min. "
          f"HV ratio={all_results[demand_seed]['hv_ratio']:.2f}, "
          f"drone-eligible={drone_eligible}, evan-share={overflow_mean:.1f}%")

# ================================================================
# Summary table of stability across demand realisations
# ================================================================
df = pd.DataFrame.from_dict(all_results, orient='index')
df.index.name = 'demand_seed'
df.to_csv("demand_realizations_summary.csv")
print(f"\n{'='*70}\nSTABILITY ACROSS {len(DEMAND_SEEDS)} INDEPENDENT DEMAND REALIZATIONS\n{'='*70}")
print(df.to_string())
print(f"\nHV ratio: mean={df['hv_ratio'].mean():.2f}, std={df['hv_ratio'].std():.2f}, "
      f"range=[{df['hv_ratio'].min():.2f}, {df['hv_ratio'].max():.2f}]")
print(f"Evan share (%): mean={df['overflow_evan_pct_mean'].mean():.1f}, "
      f"std={df['overflow_evan_pct_mean'].std():.1f}")
print(f"Drone-eligible count: {df['drone_eligible_count'].tolist()} "
      f"(mean={df['drone_eligible_count'].mean():.1f})")

# --- Stability plot (300dpi) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].bar(range(len(DEMAND_SEEDS)), df['hv_ratio'], color='steelblue')
axes[0].axhline(df['hv_ratio'].mean(), color='red', linestyle='--', label='Mean')
axes[0].set_xticks(range(len(DEMAND_SEEDS))); axes[0].set_xticklabels(DEMAND_SEEDS)
axes[0].set_xlabel("Demand realization seed"); axes[0].set_ylabel("HV ratio (Proposed/Baseline)")
axes[0].set_title("Stability of Proposed/Baseline HV ratio"); axes[0].legend()

axes[1].bar(range(len(DEMAND_SEEDS)), df['overflow_evan_pct_mean'], color='orange')
axes[1].set_xticks(range(len(DEMAND_SEEDS))); axes[1].set_xticklabels(DEMAND_SEEDS)
axes[1].set_xlabel("Demand realization seed"); axes[1].set_ylabel("E-van share (%)")
axes[1].set_title("Stability of overflow to e-van")

axes[2].bar(range(len(DEMAND_SEEDS)), df['drone_eligible_count'], color='purple')
axes[2].set_xticks(range(len(DEMAND_SEEDS))); axes[2].set_xticklabels(DEMAND_SEEDS)
axes[2].set_xlabel("Demand realization seed"); axes[2].set_ylabel("Drone-eligible customers")
axes[2].set_title("Stability of drone eligibility count")

plt.tight_layout()
plt.savefig("demand_realizations_stability.png", dpi=300, bbox_inches="tight")
print("\nSaved: demand_realizations_summary.csv, demand_realizations_stability.png")

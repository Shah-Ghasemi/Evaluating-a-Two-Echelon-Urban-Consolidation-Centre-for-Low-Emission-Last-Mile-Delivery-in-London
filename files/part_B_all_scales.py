# ================================================================
# part_B_all_scales.py
# Automated execution of Proposed vs. Baseline for N = [10, 50, 100, 200]
# Final output: summary table (CSV) + trend plots (300dpi)
# Requires: model_config.py, hfvrp_model.py, network_utils.py, stats_utils.py
# ================================================================
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from model_config import (build_fleet, generate_customer_demand, compute_trunk_haul, TRUNK_FLEET,
                          route_cost_time, two_opt, two_opt_route_length)
from hfvrp_model import HFVRPProblem
from network_utils import build_street_graphs, build_network_distance_matrix, build_drone_distance_matrix, report_unreachable
from stats_utils import (run_multiple_seeds, run_multiple_seeds_checkpointed, shared_reference_point,
                         compute_hypervolume_stats, compute_mean_convergence, check_convergence)


# ================================================================
# Representative-solution extraction -- reuses the same
# results_proposed/results_baseline already computed for the main
# statistics (HV/Mann-Whitney), without re-running NSGA-II (which an
# earlier, separate script -- part_C_representative_extraction.py --
# had mistakenly done, doubling computational cost).
# ================================================================
def count_unserved(chrom, N_cust, fleet, weights, volumes, dist_mats, start_node, depot_return_node):
    K = len(fleet)
    veh_choice = np.clip(np.floor(chrom[:N_cust] * K).astype(int), 0, K - 1)
    route_priority = np.argsort(chrom[N_cust:])
    veh_custs = {k: [] for k in range(K)}
    for cust in route_priority:
        veh_custs[veh_choice[cust]].append(cust)
    total_unserved = 0
    for k in range(K):
        if not veh_custs[k]:
            continue
        v = fleet[k]
        dist_mat = dist_mats[v['dist_key']]
        current, unvisited = start_node, veh_custs[k][:]
        total_dist = load_w = load_v = 0.0
        stops = 0
        while unvisited and stops < v['max_stops']:
            next_cust, min_dist = None, np.inf
            for c in unvisited:
                d = dist_mat[current][c + 2]
                if not np.isfinite(d):
                    continue
                return_d = dist_mat[c + 2][depot_return_node]
                if (load_w + weights[c] <= v['cap_w'] and load_v + volumes[c] <= v['cap_v']
                        and total_dist + d + return_d <= v['range'] and d < min_dist):
                    min_dist, next_cust = d, c
            if next_cust is None:
                break
            total_dist += min_dist
            load_w += weights[next_cust]; load_v += volumes[next_cust]
            unvisited.remove(next_cust); current = next_cust + 2; stops += 1
        total_unserved += len(unvisited)
    return total_unserved


def decode_solution(chrom, N_cust, fleet, weights, volumes, dist_mats, start_node, depot_return_node):
    K = len(fleet)
    veh_choice = np.clip(np.floor(chrom[:N_cust] * K).astype(int), 0, K - 1)
    route_priority = np.argsort(chrom[N_cust:])
    veh_custs = {k: [] for k in range(K)}
    for cust in route_priority:
        veh_custs[veh_choice[cust]].append(cust)
    routes_detail = []
    for k in range(K):
        if not veh_custs[k]:
            continue
        v = fleet[k]
        dist_mat = dist_mats[v['dist_key']]
        current, unvisited = start_node, veh_custs[k][:]
        total_dist = load_w = load_v = 0.0
        stops = 0
        order = []
        while unvisited and stops < v['max_stops']:
            next_cust, min_dist = None, np.inf
            for c in unvisited:
                d = dist_mat[current][c + 2]
                if not np.isfinite(d):
                    continue
                return_d = dist_mat[c + 2][depot_return_node]
                if (load_w + weights[c] <= v['cap_w'] and load_v + volumes[c] <= v['cap_v']
                        and total_dist + d + return_d <= v['range'] and d < min_dist):
                    min_dist, next_cust = d, c
            if next_cust is None:
                break
            total_dist += min_dist
            load_w += weights[next_cust]; load_v += volumes[next_cust]
            unvisited.remove(next_cust); current = next_cust + 2; stops += 1
            order.append(next_cust + 2)
        if len(order) >= 3:
            order = two_opt(order, dist_mat, start_node, depot_return_node)
        total_dist = two_opt_route_length(order, dist_mat, start_node, depot_return_node)
        route_cost, route_emission = route_cost_time(total_dist, len(order), v)
        routes_detail.append({
            'vehicle_type': v['type'], 'vehicle_idx': k,
            'customers_served': [o - 2 for o in order],
            'n_customers': len(order), 'total_dist_km': total_dist,
            'cost': route_cost, 'emission': route_emission, 'unserved': unvisited,
        })
    return routes_detail


def extract_representative(N_cust, fleet, weights, volumes, dist_mats, results,
                            start_node, depot_return_node, config_name):
    all_F_raw = np.vstack([r.F for r in results if r.F is not None])
    all_X_raw = np.vstack([r.X for r in results if r.X is not None])
    unserved_counts = np.array([
        count_unserved(all_X_raw[i], N_cust, fleet, weights, volumes, dist_mats,
                       start_node, depot_return_node)
        for i in range(all_X_raw.shape[0])
    ])
    mask = unserved_counts == 0
    print(f"[{config_name}] {mask.sum()}/{len(unserved_counts)} candidates serve 100% of customers.")
    if mask.sum() == 0:
        print(f"⚠️ [{config_name}] No fully-served solution found at N={N_cust} — skipped.")
        return None
    all_F, all_X = all_F_raw[mask], all_X_raw[mask]

    idx_min_cost = np.argmin(all_F[:, 0])
    idx_min_em = np.argmin(all_F[:, 1])
    F_norm = (all_F - all_F.min(axis=0)) / (all_F.max(axis=0) - all_F.min(axis=0) + 1e-12)
    idx_knee = np.argmin(np.sqrt((F_norm ** 2).sum(axis=1)))
    representative_idx = {'Min-Cost': idx_min_cost, 'Min-Emission': idx_min_em, 'Knee-Point': idx_knee}

    decoded = {name: decode_solution(all_X[idx], N_cust, fleet, weights, volumes, dist_mats,
                                      start_node, depot_return_node)
               for name, idx in representative_idx.items()}

    summary_rows, detail_rows = [], []
    for name, routes in decoded.items():
        veh_counts = {}
        for r in routes:
            veh_counts[r['vehicle_type']] = veh_counts.get(r['vehicle_type'], 0) + 1
        total_served = sum(r['n_customers'] for r in routes)
        summary_rows.append({
            'Solution': name,
            'Total Cost (GBP)': round(sum(r['cost'] for r in routes), 2),
            'Total Emission (g CO2)': round(sum(r['emission'] for r in routes), 1),
            'Bikes Used': veh_counts.get('bike', 0), 'E-Vans Used': veh_counts.get('evan', 0),
            'Drones Used': veh_counts.get('drone', 0), 'Diesel Used': veh_counts.get('diesel', 0),
            'Customers Served': total_served, 'Customers Served (%)': round(total_served / N_cust * 100, 1),
        })
        for r in routes:
            detail_rows.append({
                'Solution': name, 'Vehicle': f"{r['vehicle_type']}_{r['vehicle_idx']}",
                'N_Customers': r['n_customers'], 'Customers': r['customers_served'],
                'Distance (km)': round(r['total_dist_km'], 2),
                'Cost (GBP)': round(r['cost'], 3), 'Emission (g CO2)': round(r['emission'], 1),
            })
    suffix = "" if config_name == "Proposed" else "_BASELINE"
    pd.DataFrame(summary_rows).to_csv(f"representative_solutions_summary_N{N_cust}{suffix}.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(f"representative_solutions_detail_N{N_cust}{suffix}.csv", index=False)
    print(f"[{config_name}] Saved representative_solutions_(summary|detail)_N{N_cust}{suffix}.csv")
    return decoded


def plot_representative_maps(decoded, N_cust, all_points_proj, G_drive, G_bike, config_name):
    """
    Route maps drawn on Hackney's real street network -- restores
    functionality from the earlier, standalone part_C_solution_maps.py
    that was inadvertently dropped during consolidation.
    """
    import osmnx as ox
    import networkx as nx

    def get_node_ids(G, points_xy):
        xs = [p[0] for p in points_xy]
        ys = [p[1] for p in points_xy]
        return ox.distance.nearest_nodes(G, X=xs, Y=ys)

    drive_node_ids = get_node_ids(G_drive, all_points_proj)
    bike_node_ids = get_node_ids(G_bike, all_points_proj)
    start_node = 1 if config_name == "Proposed" else 0

    for name, routes in decoded.items():
        fig, ax = ox.plot_graph(G_drive, show=False, close=False, node_size=0,
                                 edge_color="#dddddd", edge_linewidth=0.5, figsize=(12, 12),
                                 bgcolor="white")
        colors = plt.cm.tab10.colors
        for i, r in enumerate(routes):
            color = colors[i % len(colors)]
            node_seq = [start_node] + r['customers_served'] + [start_node]
            node_seq = [n if n == start_node else n + 2 for n in node_seq]
            if r['vehicle_type'] == 'drone':
                xs = [all_points_proj[n][0] for n in node_seq]
                ys = [all_points_proj[n][1] for n in node_seq]
                ax.plot(xs, ys, '--', color=color, linewidth=2,
                        label=f"{r['vehicle_type']}_{r['vehicle_idx']}")
            else:
                G_use = G_bike if r['vehicle_type'] == 'bike' else G_drive
                node_ids_use = bike_node_ids if r['vehicle_type'] == 'bike' else drive_node_ids
                full_path = []
                for a, b in zip(node_seq[:-1], node_seq[1:]):
                    try:
                        path = nx.shortest_path(G_use, node_ids_use[a], node_ids_use[b], weight='length')
                    except nx.NetworkXNoPath:
                        continue
                    full_path.extend(path if not full_path else path[1:])
                if full_path:
                    ox.plot_graph_route(G_use, full_path, ax=ax, route_color=color,
                                         route_linewidth=2.5, show=False, close=False,
                                         orig_dest_size=0)
        origin_label = 'UCC' if config_name == "Proposed" else 'Depot'
        ox.scatter = ax.scatter([all_points_proj[start_node][0]], [all_points_proj[start_node][1]],
                                 c='red', s=150, marker='*', zorder=5, label=origin_label)
        for c in range(N_cust):
            cx, cy = all_points_proj[c + 2]
            ax.scatter([cx], [cy], c='black', s=20, zorder=4)
        ax.legend(loc='upper left', fontsize=8)
        ax.set_title(f"{config_name} — {name} Solution (N={N_cust})")
        fname = f"routes_map_{config_name.lower()}_{name.lower().replace('-', '_')}_N{N_cust}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[{config_name}] Saved map: {fname}")

# ================================================================
# Main settings -- reduce these if time is short (a warning is printed at the end)
# ================================================================
SCALES = [10, 50, 100, 200]
SEED = 42
N_SEEDS = 20
POP_SIZE = 100
N_GEN = 250

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 150

place = "London Borough of Hackney, United Kingdom"

all_results = {}  # N -> dict of summary stats
t_start_total = time.time()

# Street graphs are built/cached only once (independent of N)
print("Building/loading street graphs (once for all scales)...")
G_drive, G_bike = build_street_graphs(place, extra_latlon_points=[(51.5830, -0.0198)])

for N_CUST in SCALES:
    t0 = time.time()
    print(f"\n{'='*70}\nSCALE N = {N_CUST}\n{'='*70}")

    np.random.seed(SEED)
    random.seed(SEED)

    import osmnx as ox
    from pyproj import Transformer
    from network_utils import generate_customer_locations_realistic

    customers_latlon, location_method = generate_customer_locations_realistic(
        N_CUST, place, seed=SEED)
    print(f"Customer location generation method: {location_method}")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
    def project_point(lat, lon):
        x, y = transformer.transform(lon, lat)
        return (x, y)

    depot_proj = project_point(51.5830, -0.0198)
    ucc_proj = project_point(51.5473, -0.0558)
    customers_proj = [project_point(lat, lon) for lat, lon in customers_latlon]
    all_points_proj = [depot_proj, ucc_proj] + customers_proj
    point_labels = ["Depot", "UCC"] + [f"Cust_{i}" for i in range(N_CUST)]

    weights, volumes, categories = generate_customer_demand(N_CUST, seed=SEED)
    print(f"Generated {N_CUST} customers. Total demand: {weights.sum():.1f} kg, {volumes.sum():.3f} m3")
    print(f"  Category mix: letterbox={np.mean(categories=='letterbox')*100:.0f}%, "
          f"shoebox={np.mean(categories=='shoebox')*100:.0f}%, larger={np.mean(categories=='larger')*100:.0f}%")

    dist_drive_km, _ = build_network_distance_matrix(G_drive, all_points_proj)
    dist_bike_km, _ = build_network_distance_matrix(G_bike, all_points_proj)
    dist_drone_km = build_drone_distance_matrix(all_points_proj)
    dist_mats = {'drive': dist_drive_km, 'bike': dist_bike_km, 'drone': dist_drone_km}
    report_unreachable(dist_drive_km, "drive", point_labels)
    report_unreachable(dist_bike_km, "bike", point_labels)

    # --- Trunk-haul (Depot -> UCC) ---
    dist_depot_ucc = dist_drive_km[0][1]
    trunk_results = compute_trunk_haul(weights.sum(), volumes.sum(), dist_depot_ucc, TRUNK_FLEET)
    best_trunk_name, best_trunk = min(trunk_results.items(), key=lambda kv: kv[1]['cost'])
    print(f"Trunk-haul (Depot->UCC): best option={best_trunk_name}, "
          f"cost={best_trunk['cost']:.2f} GBP, emission={best_trunk['emission']:.1f} g")

    # --- Proposed scenario (checkpointed: if you copy the Colab
    # checkpoint files into the checkpoints_N{N}_proposed folder, this
    # resumes from there) ---
    fleet_proposed = build_fleet(N_CUST, last_mile_only=True)
    problem_proposed = HFVRPProblem(N_CUST, fleet_proposed, weights, volumes, dist_mats,
                                     start_node=1, depot_return_node=1)
    print(f"Running {N_SEEDS} independent seeds for Proposed scenario...")
    results_proposed = run_multiple_seeds_checkpointed(
        problem_proposed, f"checkpoints_N{N_CUST}_proposed",
        n_seeds=N_SEEDS, pop_size=POP_SIZE, n_gen=N_GEN, save_history=True)

    # --- Baseline scenario (also checkpointed) ---
    fleet_baseline = build_fleet(N_CUST, last_mile_only=False)
    problem_baseline = HFVRPProblem(N_CUST, fleet_baseline, weights, volumes, dist_mats,
                                     start_node=0, depot_return_node=0)
    print(f"Running {N_SEEDS} independent seeds for Baseline scenario...")
    results_baseline = run_multiple_seeds_checkpointed(
        problem_baseline, f"checkpoints_N{N_CUST}_baseline",
        n_seeds=N_SEEDS, pop_size=POP_SIZE, n_gen=N_GEN, save_history=True)

    # --- Extract representative solutions from these same results (no NSGA-II re-run) ---
    decoded_proposed = extract_representative(N_CUST, fleet_proposed, weights, volumes, dist_mats, results_proposed,
                                               start_node=1, depot_return_node=1, config_name="Proposed")
    decoded_baseline = extract_representative(N_CUST, fleet_baseline, weights, volumes, dist_mats, results_baseline,
                                               start_node=0, depot_return_node=0, config_name="Baseline")
    if decoded_proposed is not None:
        plot_representative_maps(decoded_proposed, N_CUST, all_points_proj, G_drive, G_bike, "Proposed")
    if decoded_baseline is not None:
        plot_representative_maps(decoded_baseline, N_CUST, all_points_proj, G_drive, G_bike, "Baseline")

    # --- Statistical validation ---
    ref_point = shared_reference_point(results_proposed, results_baseline)
    hv_stats_proposed = compute_hypervolume_stats(results_proposed, ref_point)
    hv_stats_baseline = compute_hypervolume_stats(results_baseline, ref_point)

    mean_curve_proposed, std_curve_proposed = compute_mean_convergence(results_proposed, ref_point)
    mean_curve_baseline, std_curve_baseline = compute_mean_convergence(results_baseline, ref_point)
    print("Convergence check:")
    conv_proposed = check_convergence(mean_curve_proposed, "Proposed")
    conv_baseline = check_convergence(mean_curve_baseline, "Baseline")

    stat, p_value = mannwhitneyu(hv_stats_proposed['values'], hv_stats_baseline['values'], alternative='greater')

    # --- Overflow / unserved diagnostics ---
    def compute_overflow_unserved(problem, results):
        overflow_pcts, unserved_pcts = [], []
        for res in results:
            chrom = res.X[np.argmin(res.F[:, 0])]
            veh_choice = np.clip(np.floor(chrom[:problem.N_cust] * problem.K).astype(int), 0, problem.K - 1)
            route_priority = np.argsort(chrom[problem.N_cust:])
            veh_custs = {k: [] for k in range(problem.K)}
            for cust in route_priority:
                veh_custs[veh_choice[cust]].append(cust)
            n_served = 0
            for k in range(problem.K):
                if not veh_custs[k]:
                    continue
                v = problem.fleet[k]
                dist_mat = problem.dist_mats[v['dist_key']]
                current, unvisited = problem.start_node, veh_custs[k][:]
                total_dist = load_w = load_v = 0.0
                stops = 0
                while unvisited and stops < v['max_stops']:
                    next_cust, min_dist = None, np.inf
                    for c in unvisited:
                        d = dist_mat[current][c + 2]
                        if not np.isfinite(d):
                            continue
                        return_d = dist_mat[c + 2][problem.depot_return_node]
                        if (load_w + problem.weights[c] <= v['cap_w'] and load_v + problem.volumes[c] <= v['cap_v']
                                and total_dist + d + return_d <= v['range'] and d < min_dist):
                            min_dist, next_cust = d, c
                    if next_cust is None:
                        break
                    total_dist += min_dist
                    load_w += problem.weights[next_cust]; load_v += problem.volumes[next_cust]
                    unvisited.remove(next_cust); current = next_cust + 2; stops += 1
                    n_served += 1
            evan_idx = [k for k, v in enumerate(problem.fleet) if v['type'] == 'evan']
            n_evan = sum(1 for c in veh_choice if c in evan_idx)
            overflow_pcts.append(n_evan / problem.N_cust * 100)
            unserved_pcts.append((problem.N_cust - n_served) / problem.N_cust * 100)
        return np.mean(overflow_pcts), np.std(overflow_pcts), np.mean(unserved_pcts), np.std(unserved_pcts)

    overflow_mean, overflow_std, unserved_mean, unserved_std = compute_overflow_unserved(problem_proposed, results_proposed)
    _, _, unserved_mean_b, unserved_std_b = compute_overflow_unserved(problem_baseline, results_baseline)

    elapsed = time.time() - t0
    print(f"N={N_CUST} completed in {elapsed/60:.1f} minutes.")

    all_results[N_CUST] = {
        'hv_proposed_mean': hv_stats_proposed['mean'], 'hv_proposed_std': hv_stats_proposed['std'],
        'hv_baseline_mean': hv_stats_baseline['mean'], 'hv_baseline_std': hv_stats_baseline['std'],
        'mannwhitney_u': stat, 'mannwhitney_p': p_value,
        'overflow_evan_pct_mean': overflow_mean, 'overflow_evan_pct_std': overflow_std,
        'unserved_pct_proposed_mean': unserved_mean, 'unserved_pct_proposed_std': unserved_std,
        'unserved_pct_baseline_mean': unserved_mean_b, 'unserved_pct_baseline_std': unserved_std_b,
        'trunk_best_option': best_trunk_name, 'trunk_cost': best_trunk['cost'], 'trunk_emission': best_trunk['emission'],
        'converged_proposed': conv_proposed, 'converged_baseline': conv_baseline,
        'runtime_minutes': elapsed / 60, 'location_method': location_method,
    }

    # --- Per-scale convergence plot (English, 300dpi) ---
    plt.figure(figsize=(9, 6))
    gens = np.arange(1, len(mean_curve_proposed) + 1)
    plt.plot(gens, mean_curve_proposed, label='Proposed (with UCC)', color='green')
    plt.fill_between(gens, mean_curve_proposed - std_curve_proposed, mean_curve_proposed + std_curve_proposed,
                      color='green', alpha=0.2)
    gens_b = np.arange(1, len(mean_curve_baseline) + 1)
    plt.plot(gens_b, mean_curve_baseline, label='Baseline (no UCC)', color='gray')
    plt.fill_between(gens_b, mean_curve_baseline - std_curve_baseline, mean_curve_baseline + std_curve_baseline,
                      color='gray', alpha=0.2)
    plt.xlabel("Generation"); plt.ylabel("Hypervolume (shared reference point)")
    plt.title(f"Convergence (mean ± std over {N_SEEDS} seeds) — N={N_CUST}")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"convergence_N{N_CUST}.png", dpi=300, bbox_inches="tight")
    plt.close()

    np.savez(f"results_N{N_CUST}.npz",
             hv_proposed=hv_stats_proposed['values'], hv_baseline=hv_stats_baseline['values'], ref_point=ref_point)

# ================================================================
# Final summary table (CSV) -- exactly what the manuscript's Results section needs
# ================================================================
df = pd.DataFrame.from_dict(all_results, orient='index')
df.index.name = 'N_customers'
df.to_csv("summary_all_scales.csv")
print(f"\n{'='*70}\nFINAL SUMMARY TABLE (also saved to summary_all_scales.csv)\n{'='*70}")
print(df.to_string())

# ================================================================
# Trend plots -- 300dpi
# ================================================================
Ns = list(all_results.keys())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].errorbar(Ns, [all_results[n]['hv_proposed_mean'] for n in Ns],
                  yerr=[all_results[n]['hv_proposed_std'] for n in Ns],
                  marker='o', label='Proposed (with UCC)', color='green', capsize=4)
axes[0].errorbar(Ns, [all_results[n]['hv_baseline_mean'] for n in Ns],
                  yerr=[all_results[n]['hv_baseline_std'] for n in Ns],
                  marker='s', label='Baseline (no UCC)', color='gray', capsize=4)
axes[0].set_xlabel("Number of customers (N)"); axes[0].set_ylabel("Hypervolume")
axes[0].set_title("Hypervolume vs. Scale"); axes[0].legend(); axes[0].grid(True)
axes[0].set_yscale('log')

axes[1].errorbar(Ns, [all_results[n]['overflow_evan_pct_mean'] for n in Ns],
                  yerr=[all_results[n]['overflow_evan_pct_std'] for n in Ns],
                  marker='o', color='orange', capsize=4)
axes[1].set_xlabel("Number of customers (N)"); axes[1].set_ylabel("Customers served by e-van (%)")
axes[1].set_title("Overflow to E-Van vs. Scale"); axes[1].grid(True)

axes[2].errorbar(Ns, [all_results[n]['unserved_pct_proposed_mean'] for n in Ns],
                  yerr=[all_results[n]['unserved_pct_proposed_std'] for n in Ns],
                  marker='o', label='Proposed (with UCC)', color='green', capsize=4)
axes[2].errorbar(Ns, [all_results[n]['unserved_pct_baseline_mean'] for n in Ns],
                  yerr=[all_results[n]['unserved_pct_baseline_std'] for n in Ns],
                  marker='s', label='Baseline (no UCC)', color='gray', capsize=4)
axes[2].set_xlabel("Number of customers (N)"); axes[2].set_ylabel("Unserved customers (%)")
axes[2].set_title("Service Failure Rate vs. Scale"); axes[2].legend(); axes[2].grid(True)

plt.tight_layout()
plt.savefig("trend_all_scales.png", dpi=300, bbox_inches="tight")
plt.close()

total_elapsed = time.time() - t_start_total
print(f"\n✅ All scales completed in {total_elapsed/60:.1f} minutes total.")
print("✅ Files saved: summary_all_scales.csv, trend_all_scales.png, "
      "convergence_N{10,50,100,200}.png, results_N{10,50,100,200}.npz")

for n in Ns:
    if not all_results[n]['converged_proposed'] or not all_results[n]['converged_baseline']:
        print(f"⚠️ WARNING: N={n} did not fully converge — consider increasing N_GEN and re-running for this scale.")

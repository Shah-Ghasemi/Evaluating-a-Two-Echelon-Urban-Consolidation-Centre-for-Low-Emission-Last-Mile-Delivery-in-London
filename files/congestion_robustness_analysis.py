# ================================================================
# congestion_robustness_analysis.py
# Post-hoc robustness analysis: how do the already-computed Pareto
# routes (Proposed vs Baseline, N=50 representative solutions) hold
# up under realistic day-to-day road-traffic congestion variability?
# NSGA-II is NOT re-run; only stored routes are re-evaluated.
#
# Inputs required (same folder):
#   representative_solutions_detail_N50.csv           (Proposed)
#   representative_solutions_detail_N50_BASELINE.csv  (Baseline)
# ================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

plt.rcParams['savefig.dpi'] = 300

# NOTE: only change this number for a different scale (the files
# representative_solutions_detail_N{N_CUST}(.csv/_BASELINE.csv) must
# already have been produced by part_B_all_scales.py for that scale)
N_CUST = 50

# ================================================================
# Fixed model parameters (Table 1 + congestion calibration)
# ================================================================
SPEED_KMH = {'bike': 16.0, 'evan': 7.0, 'drone': 38.0, 'diesel': 7.0}
DWELL_MIN = {'bike': 0.0, 'evan': 4.1, 'drone': 0.0, 'diesel': 4.1}
MOTORISED = {'evan', 'diesel'}   # only these are subject to road congestion
TAU_HOURS = 8.0                  # shift-duration failure threshold

MEDIAN_C = 1.0                   # congestion multiplier is centred on Table 1's
                                  # own typical operating speed (not free-flow) —
                                  # see accompanying methodology note
SIGMA_C = 0.0986                 # derived from DfT CGN0504a monthly urban-road
                                  # delay variability (2019-2025, excluding 2020)
K_SCENARIOS = 500
SEED = 123

# ================================================================
# Step 1: load the previously-computed representative-solution routes
# ================================================================
proposed_df = pd.read_csv(f"representative_solutions_detail_N{N_CUST}.csv")
baseline_df = pd.read_csv(f"representative_solutions_detail_N{N_CUST}_BASELINE.csv")

for df in (proposed_df, baseline_df):
    df['vehicle_type'] = df['Vehicle'].str.split('_').str[0]

# ================================================================
# Step 2: generate the K congestion draws ONCE — common random numbers,
# shared identically between Proposed and Baseline
# ================================================================
rng = np.random.default_rng(SEED)
mu = np.log(MEDIAN_C)
c_draws = rng.lognormal(mean=mu, sigma=SIGMA_C, size=K_SCENARIOS)  # shape (K,)


def compute_durations(df, c_draws):
    """
    Returns a (n_vehicles, K) array of trip duration (hours) for every
    vehicle route in df, under every one of the K congestion scenarios.
    """
    n = len(df)
    K = len(c_draws)
    durations = np.zeros((n, K))
    for i, row in df.iterrows():
        vtype = row['vehicle_type']
        speed = SPEED_KMH[vtype]
        dwell_h = row['N_Customers'] * DWELL_MIN[vtype] / 60.0
        base_drive_h = row['Distance (km)'] / speed
        if vtype in MOTORISED:
            drive_h_scenarios = base_drive_h * c_draws          # (K,)
        else:
            drive_h_scenarios = np.full(K, base_drive_h)        # unaffected
        durations[i, :] = drive_h_scenarios + dwell_h
    return durations


proposed_durations = compute_durations(proposed_df, c_draws)   # (n_proposed, K)
baseline_durations = compute_durations(baseline_df, c_draws)   # (n_baseline, K)

proposed_df_expanded = proposed_df.copy()
baseline_df_expanded = baseline_df.copy()

# ================================================================
# Step 3: failure indicator and per-(solution, vehicle-type) statistics
# ================================================================
def summarise(df, durations, config_name):
    rows = []
    for (solution, vtype), idx in df.groupby(['Solution', 'vehicle_type']).groups.items():
        d = durations[df.index.get_indexer(idx)]   # (n_vehicles_of_type, K)
        flat = d.flatten()
        rows.append({
            'Configuration': config_name,
            'Solution': solution,
            'Vehicle type': vtype,
            'N vehicles': d.shape[0],
            'Mean duration (h)': flat.mean(),
            'SD duration (h)': flat.std(),
            'P(F=1) [duration > 8h]': (flat > TAU_HOURS).mean(),
        })
    return pd.DataFrame(rows)


summary_proposed = summarise(proposed_df, proposed_durations, 'Proposed')
summary_baseline = summarise(baseline_df, baseline_durations, 'Baseline')
summary_table = pd.concat([summary_proposed, summary_baseline], ignore_index=True)
summary_table = summary_table.sort_values(['Solution', 'Configuration', 'Vehicle type'])

print("=" * 90)
print("Congestion robustness summary (mean +/- SD duration, P(F=1)) by configuration and vehicle type")
print("=" * 90)
print(summary_table.to_string(index=False))
summary_table.to_csv(f"congestion_robustness_summary_N{N_CUST}.csv", index=False)

# ================================================================
# Step 4: Wilcoxon signed-rank test — paired on the K shared congestion
# scenarios, comparing the WORST-CASE (max) vehicle duration in the fleet
# for Proposed vs Baseline, separately for each representative solution.
# (Common random numbers make this a valid paired test.)
# ================================================================
print("\n" + "=" * 90)
print("Wilcoxon signed-rank test: max fleet duration per scenario, Proposed vs Baseline")
print("=" * 90)

wilcoxon_results = []
max_duration_series = {}   # for plotting
for solution in proposed_df['Solution'].unique():
    idx_p = proposed_df.index[proposed_df['Solution'] == solution]
    idx_b = baseline_df.index[baseline_df['Solution'] == solution]
    max_p = proposed_durations[proposed_df.index.get_indexer(idx_p)].max(axis=0)   # (K,)
    max_b = baseline_durations[baseline_df.index.get_indexer(idx_b)].max(axis=0)   # (K,)
    max_duration_series[solution] = (max_p, max_b)

    diff = max_p - max_b
    if np.allclose(diff, 0):
        stat, p_value = np.nan, 1.0
    else:
        stat, p_value = wilcoxon(max_p, max_b)
    wilcoxon_results.append({
        'Solution': solution,
        'Mean max-duration Proposed (h)': max_p.mean(),
        'Mean max-duration Baseline (h)': max_b.mean(),
        'Wilcoxon statistic': stat,
        'p-value': p_value,
    })
    print(f"  {solution}: Proposed max-duration={max_p.mean():.2f}h, "
          f"Baseline max-duration={max_b.mean():.2f}h, "
          f"Wilcoxon stat={stat:.1f}, p={p_value:.4f}")

wilcoxon_df = pd.DataFrame(wilcoxon_results)
wilcoxon_df.to_csv(f"congestion_robustness_wilcoxon_N{N_CUST}.csv", index=False)

# ================================================================
# Step 5: figure — boxplot of max fleet duration per scenario, by solution
# ================================================================
fig, axes = plt.subplots(1, len(max_duration_series), figsize=(6 * len(max_duration_series), 5),
                          sharey=True)
if len(max_duration_series) == 1:
    axes = [axes]
for ax, (solution, (max_p, max_b)) in zip(axes, max_duration_series.items()):
    ax.boxplot([max_p, max_b], tick_labels=['Proposed', 'Baseline'])
    ax.axhline(TAU_HOURS, color='red', linestyle='--', linewidth=1, label='8-hour threshold')
    ax.set_title(solution)
    ax.set_ylabel("Max fleet duration (h)")
    ax.legend(fontsize=8)
plt.suptitle(f"Worst-case vehicle duration under {K_SCENARIOS} congestion scenarios (N={N_CUST})")
plt.tight_layout()
plt.savefig(f"congestion_robustness_boxplot_N{N_CUST}.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved: congestion_robustness_summary_N{N_CUST}.csv, "
      f"congestion_robustness_wilcoxon_N{N_CUST}.csv, congestion_robustness_boxplot_N{N_CUST}.png")

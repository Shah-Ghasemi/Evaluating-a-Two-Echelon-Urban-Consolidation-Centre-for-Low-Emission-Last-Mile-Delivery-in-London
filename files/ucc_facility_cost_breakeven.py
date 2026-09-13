# ================================================================
# ucc_facility_cost_breakeven.py
# Addresses the reviewer critique that this study's cost comparison
# omits UCC facility operating costs (rental, sorting, handling
# labour). Adds a per-parcel facility-handling charge, sourced from
# real UCC case-study literature, to the Proposed scenario's cost,
# and asks: at what facility charge does the Proposed scenario's
# cost advantage over the Baseline disappear?
#
# Pure post-hoc calculation -- a per-parcel facility charge is an
# additive constant applied equally to every solution's cost, so it
# shifts the Pareto front uniformly along the cost axis without
# changing which routes are chosen. No NSGA-II re-run required.
#
# Requires: representative_solutions_summary_N50.csv (Proposed) and
# representative_solutions_summary_N50_BASELINE.csv (Baseline,
# produced by extract_baseline_representative_from_factorial.py),
# both in the same directory as this script.
# ================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 300

N_CUST = 50

# ================================================================
# Real-world UCC facility handling-cost range, from the dedicated
# financial-viability literature (not assumed or fitted):
#   Janjevic, M., & Ndiaye, A. B. (2017). Development and application
#   of a transferable framework for evaluating urban consolidation
#   centre extensions. Research in Transportation Business & Management.
#   Reports per-parcel UCC handling costs of EUR 2.50-5.00 across
#   real case studies (Brussels, Padova).
# ================================================================
EUR_TO_GBP = 0.86  # 2026 rate
FACILITY_COST_LOW_EUR, FACILITY_COST_HIGH_EUR = 2.50, 5.00
FACILITY_COST_LOW_GBP = FACILITY_COST_LOW_EUR * EUR_TO_GBP
FACILITY_COST_HIGH_GBP = FACILITY_COST_HIGH_EUR * EUR_TO_GBP

# Known trunk-haul cost, Proposed scenario, N=50 (depot -> UCC, electric van;
# from summary_all_scales_FINAL.csv, already reported in Table 2/Section 4.3)
TRUNK_COST_N50 = 4.861333646

# ================================================================
# Load representative-solution costs (last-mile only, as extracted)
# ================================================================
proposed = pd.read_csv("representative_solutions_summary_N50.csv").set_index("Solution")
baseline = pd.read_csv("representative_solutions_summary_N50_BASELINE.csv").set_index("Solution")

print("Loaded representative solutions.")
print("\nProposed (last-mile only):")
print(proposed[["Total Cost (GBP)", "Total Emission (g CO2)"]])
print("\nBaseline (last-mile only, no trunk leg -- dispatches direct from depot):")
print(baseline[["Total Cost (GBP)", "Total Emission (g CO2)"]])

# ================================================================
# Full-system cost comparison at the Min-Cost representative solution:
#   Proposed_full = last-mile cost + trunk-haul cost + N x facility charge
#   Baseline_full = last-mile cost  (no UCC, no trunk leg)
# ================================================================
proposed_lastmile = proposed.loc["Min-Cost", "Total Cost (GBP)"]
baseline_lastmile = baseline.loc["Min-Cost", "Total Cost (GBP)"]
proposed_full_no_facility = proposed_lastmile + TRUNK_COST_N50

raw_gap = baseline_lastmile - proposed_full_no_facility  # advantage of Proposed BEFORE facility cost
print(f"\nProposed (last-mile + trunk, before facility cost): GBP {proposed_full_no_facility:.2f}")
print(f"Baseline (last-mile only):                          GBP {baseline_lastmile:.2f}")
print(f"Proposed's cost advantage before facility cost:      GBP {raw_gap:.2f}  "
      f"({raw_gap / N_CUST:.3f} GBP/parcel)")

breakeven_per_parcel = raw_gap / N_CUST
print(f"\nBreak-even facility handling charge (where Proposed's advantage disappears): "
      f"GBP {breakeven_per_parcel:.2f} per parcel")

print(f"\nReal-world UCC facility handling-cost range (Janjevic & Ndiaye, 2017): "
      f"GBP {FACILITY_COST_LOW_GBP:.2f}-{FACILITY_COST_HIGH_GBP:.2f} per parcel")

for label, cost in [("Low estimate", FACILITY_COST_LOW_GBP), ("High estimate", FACILITY_COST_HIGH_GBP)]:
    total_facility = cost * N_CUST
    proposed_full = proposed_full_no_facility + total_facility
    net_advantage = baseline_lastmile - proposed_full
    survives = "SURVIVES" if net_advantage > 0 else "DOES NOT SURVIVE"
    print(f"  {label} (GBP {cost:.2f}/parcel): total facility cost GBP {total_facility:.2f} -> "
          f"Proposed full cost GBP {proposed_full:.2f} vs. Baseline GBP {baseline_lastmile:.2f} "
          f"-> net advantage GBP {net_advantage:+.2f}/day -> Proposed's cost advantage {survives}")

# ================================================================
# Figure: Proposed's net cost advantage as a function of the assumed
# facility handling charge, with the real-world literature range shaded
# ================================================================
facility_range = np.linspace(0, max(breakeven_per_parcel * 1.5, FACILITY_COST_HIGH_GBP * 1.3), 200)
net_advantage_curve = raw_gap - facility_range * N_CUST

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(facility_range, net_advantage_curve, color='seagreen', linewidth=2,
        label="Proposed's net cost advantage over Baseline")
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(breakeven_per_parcel, color='red', linestyle=':',
           label=f'Break-even (GBP {breakeven_per_parcel:.2f}/parcel)')
ax.axvspan(FACILITY_COST_LOW_GBP, FACILITY_COST_HIGH_GBP, color='orange', alpha=0.2,
           label=f'Literature range (Janjevic & Ndiaye, 2017): '
                 f'GBP {FACILITY_COST_LOW_GBP:.2f}-{FACILITY_COST_HIGH_GBP:.2f}')
ax.set_xlabel("Assumed UCC facility handling charge (GBP per parcel)")
ax.set_ylabel("Proposed's net cost advantage over Baseline (GBP/day, N=50)")
ax.set_title("UCC Facility Cost Break-Even Analysis (N=50)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ucc_facility_cost_breakeven.png", dpi=300, bbox_inches="tight")
plt.show()
print("\nSaved: ucc_facility_cost_breakeven.png")

# Supplementary Code — Two-Echelon Low-Emission Last-Mile Delivery Network

This repository contains the full computational pipeline used to produce the
results reported in the manuscript. All scripts are written in Python 3 and
require `numpy`, `pandas`, `scipy`, `matplotlib`, `pymoo`, `osmnx`,
`geopandas`, `pulp`, `pyproj`, and `shapely`.

## Shared modules (required by every script below)

| File | Purpose |
|---|---|
| `model_config.py` | Fleet composition, cost/emission parameters, trunk-haul selection, congestion chance-constraint (Section 3.6), and demand generation. |
| `hfvrp_model.py` | The NSGA-II problem class (encoding, greedy route construction, 2-opt, embedded chance constraint). |
| `network_utils.py` | Real street-network graph construction (OSMnx), Dijkstra distance matrices, and population-weighted customer location generation (ONS/Nomis Census 2021 LSOA data). |
| `stats_utils.py` | Multi-seed NSGA-II execution (plain and crash-resilient/checkpointed), Hypervolume computation, and convergence diagnostics. |

## Experiment scripts

| File | Corresponds to | Purpose |
|---|---|---|
| `part_A_milp_validation.py` | Section 4.1 | Exact MILP (epsilon-constraint) validation of NSGA-II on a 10-customer instance. |
| `part_B_all_scales.py` | Sections 4.2–4.3 | Main experiment: Proposed vs. Baseline across N = 10, 50, 100, 200; produces the summary table, representative solutions, route maps, and convergence plots. |
| `part_B_colab_checkpointed.py` | Sections 4.2–4.3 | Google Colab (free-tier)–friendly variant of `part_B_all_scales.py` for the heavier scales (N = 100, 200), with per-seed checkpointing to Google Drive. |
| `part_B_demand_realizations.py` | Section 4.7 | Robustness check: repeats the full pipeline across five independent demand realisations at N = 50. |
| `part_D_sensitivity_analysis.py` | Section 4.4 | One-at-a-time sensitivity analysis (cost, emission, drone labour scenario, drone payload, emission-accounting scope) at N = 50. |
| `part_D_reverify_flagged_v2.py` | Section 4.4 | Re-evaluates the drone-payload sensitivity dimension at full experimental rigour (20 seeds, 250 generations), after the reduced-budget run in `part_D_sensitivity_analysis.py` showed high variability. |
| `congestion_robustness_analysis.py` | Section 4.6 | Post-hoc congestion-robustness check on the representative solutions (paired Wilcoxon test, 500 scenarios, common random numbers). |
| `drone_breakeven_analysis.py` | Section 4.5 | Closed-form calculation of the operator-to-drone ratio at which drone delivery reaches cost parity with the cargo e-bike. |

## Typical run order

1. `part_A_milp_validation.py` (quick; validates the algorithm)
2. `part_B_all_scales.py`, `part_B_demand_realizations.py` (or the Colab
   variant for N = 100/200)
3. `congestion_robustness_analysis.py` (requires the representative-solution
   CSVs produced by step 2)
4. `part_D_sensitivity_analysis.py`, then `part_D_reverify_flagged_v2.py`
5. `drone_breakeven_analysis.py` (independent of all other scripts)

## Notes on reproducibility

- All customer locations are generated via `generate_customer_locations_realistic()`
  in `network_utils.py`, which weights sampling by real 2021 Census LSOA
  population and automatically falls back to uniform random sampling (with a
  printed warning) if the ONS/Nomis data cannot be retrieved.
- Street-network graphs are cached locally (`osm_cache/`) after the first
  download; delete this folder if the query area changes (e.g., a different
  depot location).
- Scripts using `run_multiple_seeds_checkpointed()` (the Colab variant and
  the demand-realisation script) can be safely interrupted and re-run; they
  resume from the last completed seed rather than starting over.

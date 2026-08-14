# Two-Echelon Urban Consolidation Centre Framework for Low-Emission Last-Mile Delivery

![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)
![Last-Mile](https://img.shields.io/badge/Last--Mile-Delivery-blue?style=flat-square)
![NSGA-II](https://img.shields.io/badge/Optimization-NSGA--II-red?style=flat-square)
![OSMnx](https://img.shields.io/badge/OSMnx-1.9.4-orange?style=flat-square)
![MILP](https://img.shields.io/badge/Validation-MILP-green?style=flat-square)


Supplementary code for the paper **"Two-Echelon Urban Consolidation for Low-Emission Last-Mile Delivery: A Chance-Constrained Multi-Objective Approach under Real-World Traffic Uncertainty"** (manuscript under review, *International Journal of Sustainable Transportation*).

This repository contains the full computational pipeline used to produce every result reported in the paper: a two-echelon urban consolidation centre (UCC) framework for last-mile delivery, evaluated on the real street network of the London Borough of Hackney, UK, using a heterogeneous fleet (cargo e-bike, electric van, delivery drone, diesel van) and a chance-constrained NSGA-II formulation validated against an exact MILP benchmark.

## Requirements

Python 3.10+, with:

```
numpy pandas scipy matplotlib pymoo osmnx geopandas pulp pyproj shapely
```

Install via:

```bash
pip install numpy pandas scipy matplotlib pymoo osmnx geopandas pulp pyproj shapely
```

## Repository structure

### Shared modules (required by every script below)

| File | Purpose |
|---|---|
| `model_config.py` | Fleet composition, cost/emission parameters, trunk-haul selection, congestion chance-constraint (Section 3.6), and demand generation. |
| `hfvrp_model.py` | The NSGA-II problem class (encoding, greedy route construction, 2-opt, embedded chance constraint). |
| `network_utils.py` | Real street-network graph construction (OSMnx), Dijkstra distance matrices, and population-weighted customer location generation (ONS/Nomis Census 2021 LSOA data). |
| `stats_utils.py` | Multi-seed NSGA-II execution (plain and crash-resilient/checkpointed), Hypervolume computation, and convergence diagnostics. |

### Experiment scripts

| File | Corresponds to | Purpose |
|---|---|---|
| `part_A_milp_validation.py` | Section 4.1 | Exact MILP (epsilon-constraint) validation of NSGA-II on a 10-customer instance. |
| `part_B_all_scales.py` | Sections 4.2–4.3 | Main experiment: Proposed vs. Baseline across N = 10, 50, 100, 200; produces the summary table, representative solutions, route maps, and convergence plots. Uses disk-based checkpointing (see below), so it can be safely interrupted and resumed on any machine, including a remote/cloud notebook — just point `checkpoint_dir` at a persistent folder (e.g., a mounted cloud-drive path). |
| `part_B_demand_realizations.py` | Section 4.7 | Robustness check: repeats the full pipeline across five independent demand realisations at N = 50. |
| `part_D_sensitivity_analysis.py` | Section 4.4 | One-at-a-time sensitivity analysis (cost, emission, drone labour scenario, drone payload, emission-accounting scope) at N = 50. |
| `part_D_reverify_flagged_v2.py` | Section 4.4 | Re-evaluates the drone-payload sensitivity dimension at full experimental rigour (20 seeds, 250 generations), after the reduced-budget run in `part_D_sensitivity_analysis.py` showed high variability. |
| `congestion_robustness_analysis.py` | Section 4.6 | Post-hoc congestion-robustness check on the representative solutions (paired Wilcoxon test, 500 scenarios, common random numbers). |
| `drone_breakeven_analysis.py` | Section 4.5 | Closed-form calculation of the operator-to-drone ratio at which drone delivery reaches cost parity with the cargo e-bike. |

## Usage

Typical run order:

1. `part_A_milp_validation.py` (quick; validates the algorithm)
2. `part_B_all_scales.py`, `part_B_demand_realizations.py`
3. `congestion_robustness_analysis.py` (requires the representative-solution CSVs produced by step 2)
4. `part_D_sensitivity_analysis.py`, then `part_D_reverify_flagged_v2.py`
5. `drone_breakeven_analysis.py` (independent of all other scripts)

`part_B_all_scales.py` is computationally the heaviest step (N = 100 and N = 200 in particular can take several hours). Because it checkpoints every completed seed to disk, it is safe to run it in short sessions or on a remote/free-tier notebook — if the runtime disconnects, simply re-run the same command and it resumes from the last completed seed rather than starting over.

## Notes on reproducibility

- All customer locations are generated via `generate_customer_locations_realistic()` in `network_utils.py`, which weights sampling by real 2021 Census LSOA population and automatically falls back to uniform random sampling (with a printed warning) if the ONS/Nomis data cannot be retrieved.
- Street-network graphs are cached locally (`osm_cache/`) after the first download; delete this folder if the query area changes (e.g., a different depot location).
- `run_multiple_seeds_checkpointed()` (used throughout `part_B_all_scales.py` and the demand-realisation script) can be safely interrupted and re-run; it resumes from the last completed seed rather than starting over.

## Citation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on this repository or contact the corresponding author at: shshrokh Ghasemi — shahrokh.qsmi@gmail.com — [github.com/Shah-Ghasemi]

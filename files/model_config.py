# ================================================================
# model_config.py
# Fleet, cost, emission, and chance-constraint parameters for the
# two-echelon last-mile delivery model (Hackney case study).
#
# Parameter sources:
#   1) Just Economics (2024), "Delivering Value: A quantitative model
#      for estimating the true cost of freight via three transport
#      modes." Impact on Urban Health & Team London Bridge.
#      https://www.justeconomics.co.uk/delivering-value
#      -> cargo e-bike, e-van, diesel van (cost + full life-cycle emission)
#   2) Drone capacity: Zipline P2 platform specification.
#      Drone labour cost: McKinsey & Company (2023).
#      Drone emission: Aster, Zeilerbauer, & Lindorfer (2026), a
#      dedicated drone life-cycle assessment.
#
# Cost formula: (number of stops x fixed cost per delivery) +
#               (distance x fuel/energy cost per km).
# This reflects the Just Economics costing convention, in which cost
# is reported as an all-in "fair price per delivery" (already
# combining labour, vehicle, and overhead), rather than being decomposed
# into separate distance and time/labour components.
# ================================================================
import numpy as np
from scipy.stats import lognorm


def build_fleet(N_cust=50, last_mile_only=True, safety_margin=1.2,
                drone_scenario='realistic', drone_payload_kg=3.63):
    """
    Builds a last-mile fleet sized proportionally to N_cust (each demand
    scale represents a different business/growth stage, with its own
    appropriately sized fleet plus a 20% capacity safety margin).

    drone_scenario:
      'realistic'   -> one remote operator per drone (current practice,
                        McKinsey 2023)
      'optimistic'  -> one operator managing a fleet of ~20 drones
                        (a near-term fleet-management scenario)

    drone_payload_kg: baseline = 3.63 kg (Zipline P2, the most
      operationally mature commercial drone platform as of 2026).
      For sensitivity analysis: 2.0 kg (Matternet M2, a conservative
      lower bound) or 5.0 kg (Matternet M3, reflecting near-term
      industry growth).

    Drone emission (31.7 g CO2/km) is taken from Aster, Zeilerbauer,
    & Lindorfer (2026), under that study's high-replacement scenario
    (four battery replacements and ten propeller-set replacements per
    year) -- judged more representative of a high-volume commercial
    duty cycle than the same study's lower-turnover baseline scenario.
    """
    bike_share, evan_share = 0.42, 0.58
    bike_stops, evan_stops, drone_stops = 10, 12, 1

    n_bike = max(2, int(np.ceil(N_cust * bike_share * safety_margin / bike_stops)))
    n_evan = max(2, int(np.ceil(N_cust * evan_share * safety_margin / evan_stops)))
    n_drone = max(1, int(np.ceil(N_cust * 0.05 * safety_margin / drone_stops)))

    drone_fixed_cost = {'realistic': 10.67, 'optimistic': 1.38}[drone_scenario]

    fleet = []
    for _ in range(n_bike):
        fleet.append({
            'type': 'bike', 'cap_w': 50, 'cap_v': 0.1,
            'fixed_cost_per_stop': 3.596, 'fuel_cost_per_km': 0.017, 'em_km': 23.2,
            'speed_kmh': 16.0, 'dwell_min': 0.0,
            'range': 15, 'max_stops': bike_stops, 'dist_key': 'bike',
        })
    for _ in range(n_evan):
        fleet.append({
            'type': 'evan', 'cap_w': 500, 'cap_v': 4.0,
            'fixed_cost_per_stop': 4.056, 'fuel_cost_per_km': 0.066, 'em_km': 107.7,
            'speed_kmh': 7.0, 'dwell_min': 4.1,
            'range': 30, 'max_stops': evan_stops, 'dist_key': 'drive',
        })
    for _ in range(n_drone):
        fleet.append({
            'type': 'drone', 'cap_w': drone_payload_kg, 'cap_v': 0.03,
            'fixed_cost_per_stop': drone_fixed_cost, 'fuel_cost_per_km': 0.025, 'em_km': 31.7,
            'speed_kmh': 38.0, 'dwell_min': 0.0,
            'range': 25, 'max_stops': drone_stops, 'dist_key': 'drone',
        })
    if not last_mile_only:
        n_diesel = max(2, int(np.ceil(N_cust * evan_share * safety_margin / 15)))
        for _ in range(n_diesel):
            fleet.append({
                'type': 'diesel', 'cap_w': 800, 'cap_v': 20.0,
                'fixed_cost_per_stop': 4.506, 'fuel_cost_per_km': 0.150, 'em_km': 280.6,
                'speed_kmh': 7.0, 'dwell_min': 4.1,
                'range': 1000, 'max_stops': 15, 'dist_key': 'drive',
            })
    return fleet


# Trunk-haul fleet (Depot -> UCC). The e-van entry uses cap_v=15 (not the
# 4 m3 of the compact last-mile e-van above) to represent a large
# commercial electric van (e.g., Mercedes eSprinter / Ford E-Transit,
# ~14-15 m3), for a size-matched comparison with the diesel van at the
# trunk leg. Its fixed_cost_per_stop / fuel_cost_per_km / em_km rates are
# still taken from the compact last-mile e-van (Just Economics, 2024),
# since no separately itemised cost/emission data for a large electric
# van were available; only the capacity (cap_v) was adjusted. This
# residual limitation is discussed in the manuscript's Limitations
# section.
TRUNK_FLEET = {
    'diesel': {'cap_w': 800, 'cap_v': 20, 'fixed_cost_per_stop': 4.506,
               'fuel_cost_per_km': 0.150, 'em_km': 280.6},
    'evan':   {'cap_w': 500, 'cap_v': 15, 'fixed_cost_per_stop': 4.056,
               'fuel_cost_per_km': 0.066, 'em_km': 107.7},
}


def route_cost_time(total_dist_km, n_stops, v):
    cost = n_stops * v['fixed_cost_per_stop'] + total_dist_km * v['fuel_cost_per_km']
    emission = total_dist_km * v['em_km']
    return cost, emission


# ================================================================
# Congestion chance constraint -- embedded directly in the NSGA-II
# fitness function (Section 3.6 of the manuscript), not only evaluated
# after optimisation. Because the congestion multiplier follows a
# log-normal distribution with known parameters (median = 1.0,
# sigma = 0.0986; calibrated from the monthly variability of DfT
# CGN0504a delay statistics, excluding 2020), the probability that a
# given route's duration exceeds the shift-length threshold has a
# closed-form solution, requiring no Monte Carlo simulation.
# ================================================================
CONGESTION_SIGMA = 0.0986
CONGESTION_TAU_HOURS = 8.0
RISK_THRESHOLD = 0.05         # acceptable risk level (5% exceedance probability)
RISK_PENALTY_WEIGHT = 1000.0  # same order of magnitude as the unserved-customer penalty
MOTORISED_TYPES = {'evan', 'diesel'}  # only these vehicle types are subject to road congestion


def route_duration_risk(total_dist_km, n_stops, v):
    """
    Probability that this route's duration, under congestion uncertainty,
    exceeds the shift-length threshold (CONGESTION_TAU_HOURS). For
    non-motorised vehicles (cargo bike, drone), this probability is
    deterministic (0 or 1), not stochastic, since these are not subject
    to road-traffic congestion.
    """
    dwell_hours = n_stops * v['dwell_min'] / 60.0
    base_drive_hours = total_dist_km / v['speed_kmh']
    if v['type'] not in MOTORISED_TYPES:
        return 1.0 if (base_drive_hours + dwell_hours) > CONGESTION_TAU_HOURS else 0.0
    if base_drive_hours <= 0:
        return 0.0
    c_threshold = (CONGESTION_TAU_HOURS - dwell_hours) / base_drive_hours
    if c_threshold <= 0:
        return 1.0
    return float(lognorm.sf(c_threshold, s=CONGESTION_SIGMA, scale=1.0))


def compute_trunk_haul(total_weight_kg, total_volume_m3, dist_depot_ucc_km,
                        trunk_fleet=TRUNK_FLEET):
    """Deterministic, cost-minimising trunk-haul vehicle choice (Depot -> UCC)."""
    results = {}
    for name, v in trunk_fleet.items():
        n_trips = max(int(np.ceil(total_weight_kg / v['cap_w'])),
                      int(np.ceil(total_volume_m3 / v['cap_v'])), 1)
        dist_total = n_trips * dist_depot_ucc_km * 2
        cost = n_trips * v['fixed_cost_per_stop'] + dist_total * v['fuel_cost_per_km']
        emission = dist_total * v['em_km']
        results[name] = {'n_trips': n_trips, 'dist_total_km': dist_total,
                          'cost': cost, 'emission': emission}
    return results


def generate_customer_demand(N_cust, seed=None):
    """
    Generates parcel weight (log-normal, mean 5 kg) and volume (three-
    category mix: letterbox, shoebox, larger; shares follow Allen et al.,
    2018) for N_cust customers.
    """
    rng = np.random.default_rng(seed)
    weights = np.clip(rng.lognormal(mean=np.log(5), sigma=0.6, size=N_cust), 0.5, 30)

    p = np.array([0.079, 0.336, 0.588]); p = p / p.sum()
    cats = rng.choice(['letterbox', 'shoebox', 'larger'], size=N_cust, p=p)
    volumes = np.zeros(N_cust)
    volumes[cats == 'letterbox'] = rng.uniform(0.001, 0.003, (cats == 'letterbox').sum())
    volumes[cats == 'shoebox'] = rng.uniform(0.005, 0.015, (cats == 'shoebox').sum())
    volumes[cats == 'larger'] = rng.uniform(0.03, 0.331, (cats == 'larger').sum())
    return weights, volumes, cats


def two_opt_route_length(order, dist_mat, start_node, depot_return_node):
    if not order:
        return 0.0
    td = dist_mat[start_node][order[0]]
    for i in range(len(order) - 1):
        td += dist_mat[order[i]][order[i + 1]]
    td += dist_mat[order[-1]][depot_return_node]
    return td


def two_opt(order, dist_mat, start_node, depot_return_node):
    """Standard 2-opt local search refinement of a route's customer order."""
    if len(order) < 3:
        return order
    order = order[:]
    improved = True
    while improved:
        improved = False
        best_len = two_opt_route_length(order, dist_mat, start_node, depot_return_node)
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                new_order = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                new_len = two_opt_route_length(new_order, dist_mat, start_node, depot_return_node)
                if new_len < best_len - 1e-9:
                    order, best_len = new_order, new_len
                    improved = True
    return order

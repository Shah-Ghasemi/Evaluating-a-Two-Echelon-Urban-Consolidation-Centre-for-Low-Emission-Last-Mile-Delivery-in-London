# ================================================================
# hfvrp_model.py -- NSGA-II problem class (imports from model_config)
# Includes the congestion chance constraint, embedded directly in the
# fitness function rather than evaluated only after the fact. The
# probability of exceeding the shift-length threshold for each route is
# computed via a closed-form log-normal formula (no simulation), and a
# penalty is added when it exceeds the acceptable risk threshold.
# ================================================================
import numpy as np
from pymoo.core.problem import Problem
from model_config import (route_cost_time, two_opt, two_opt_route_length,
                           route_duration_risk, RISK_THRESHOLD, RISK_PENALTY_WEIGHT)


class HFVRPProblem(Problem):
    def __init__(self, N_cust, fleet, weights, volumes, dist_mats, start_node, depot_return_node):
        self.N_cust, self.fleet, self.K = N_cust, fleet, len(fleet)
        self.weights, self.volumes, self.dist_mats = weights, volumes, dist_mats
        self.start_node, self.depot_return_node = start_node, depot_return_node
        super().__init__(n_var=2 * N_cust, n_obj=2, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        N = x.shape[0]
        costs, emissions = np.zeros(N), np.zeros(N)
        for ind in range(N):
            chrom = x[ind]
            veh_choice = np.clip(np.floor(chrom[:self.N_cust] * self.K).astype(int), 0, self.K - 1)
            route_priority = np.argsort(chrom[self.N_cust:])
            veh_custs = {k: [] for k in range(self.K)}
            for cust in route_priority:
                veh_custs[veh_choice[cust]].append(cust)
            total_cost = total_emission = penalty = 0.0
            for k in range(self.K):
                if not veh_custs[k]:
                    continue
                v = self.fleet[k]
                dist_mat = self.dist_mats[v['dist_key']]
                current, unvisited = self.start_node, veh_custs[k][:]
                total_dist = load_w = load_v = 0.0
                stops = 0
                order = []
                while unvisited and stops < v['max_stops']:
                    next_cust, min_dist = None, np.inf
                    for c in unvisited:
                        d = dist_mat[current][c + 2]
                        if not np.isfinite(d):
                            continue
                        return_d = dist_mat[c + 2][self.depot_return_node]
                        if (load_w + self.weights[c] <= v['cap_w'] and load_v + self.volumes[c] <= v['cap_v']
                                and total_dist + d + return_d <= v['range'] and d < min_dist):
                            min_dist, next_cust = d, c
                    if next_cust is None:
                        break
                    total_dist += min_dist
                    load_w += self.weights[next_cust]; load_v += self.volumes[next_cust]
                    unvisited.remove(next_cust); current = next_cust + 2; stops += 1
                    order.append(next_cust + 2)
                if len(order) >= 3:
                    order = two_opt(order, dist_mat, self.start_node, self.depot_return_node)
                total_dist = two_opt_route_length(order, dist_mat, self.start_node, self.depot_return_node)
                route_cost, route_emission = route_cost_time(total_dist, len(order), v)
                total_cost += route_cost
                total_emission += route_emission
                penalty += 1000 * len(unvisited)
                # Chance constraint: if the shift-length exceedance risk
                # is above the acceptable threshold, a penalty proportional
                # to the degree of violation is added
                risk = route_duration_risk(total_dist, len(order), v)
                if risk > RISK_THRESHOLD:
                    penalty += RISK_PENALTY_WEIGHT * (risk - RISK_THRESHOLD)
            costs[ind] = total_cost + penalty
            emissions[ind] = total_emission + penalty
        out["F"] = np.column_stack([costs, emissions])

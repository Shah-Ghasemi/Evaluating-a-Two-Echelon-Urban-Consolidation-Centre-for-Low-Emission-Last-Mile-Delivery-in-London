# ================================================================
# drone_breakeven_analysis.py
# Threshold sensitivity analysis: at what operator-to-drone ratio
# does drone delivery reach cost parity with the cargo e-bike?
# Pure algebraic calculation — no NSGA-II re-run required.
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 300

# Fixed parameters (Table 1)
BIKE_FIXED, BIKE_FUEL, BIKE_EM = 3.596, 0.017, 23.2
DRONE_FUEL, DRONE_EM = 0.025, 25.0

# Two McKinsey (2023) anchor points for drone fixed cost per delivery:
#   N=1 drone/operator (current practice): $13.50 -> GBP 10.67
#   N=20 drones/operator (fleet-managed):  $1.50-2.00, midpoint $1.75 -> GBP 1.38
# Interpolating cost(N) = a + b/N between these two anchors.
N1, cost1 = 1, 10.67
N2, cost2 = 20, 1.38
b = (cost1 - cost2) / (1 / N1 - 1 / N2)
a = cost1 - b / N1

def drone_fixed_cost(n_drones_per_operator):
    return a + b / n_drones_per_operator

N_break_even = b / (BIKE_FIXED - a)
print(f"Fitted model: drone fixed cost(N) = {a:.3f} + {b:.3f}/N  (GBP per delivery)")
print(f"Cost break-even vs. cargo bike (fixed cost = GBP {BIKE_FIXED:.3f}): "
      f"N ~= {N_break_even:.1f} drones per operator")
print(f"Emission break-even: NEVER reached under current parameters "
      f"(drone {DRONE_EM} vs. bike {BIKE_EM} g CO2/km, independent of N)")

# ================================================================
# Figure: drone fixed cost vs. N, with bike's fixed cost as reference
# ================================================================
N_range = np.linspace(1, 20, 200)
drone_costs = drone_fixed_cost(N_range)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(N_range, drone_costs, color='purple', label='Drone fixed cost per delivery (fitted)')
ax.axhline(BIKE_FIXED, color='green', linestyle='--', label=f'Cargo bike fixed cost (GBP {BIKE_FIXED:.2f})')
ax.axvline(N_break_even, color='red', linestyle=':', label=f'Break-even (N = {N_break_even:.1f})')
ax.scatter([N1, N2], [cost1, cost2], color='black', zorder=5, label='McKinsey (2023) anchor points')
ax.set_xlabel("Drones managed per operator (N)")
ax.set_ylabel("Fixed cost per delivery (GBP)")
ax.set_title("Drone cost break-even vs. cargo e-bike, by operator-to-drone ratio")
ax.legend(fontsize=9)
ax.grid(True)
plt.tight_layout()
plt.savefig("drone_breakeven_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved: drone_breakeven_analysis.png")

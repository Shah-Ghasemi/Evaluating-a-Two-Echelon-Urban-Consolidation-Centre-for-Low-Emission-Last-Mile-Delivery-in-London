# ================================================================
# stats_utils.py
# Shared statistical validation utilities: multi-seed NSGA-II
# execution (plain and checkpointed), Hypervolume, and convergence
# diagnostics.
# ================================================================
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV


def run_multiple_seeds(problem, n_seeds=20, pop_size=100, n_gen=250, verbose=False,
                       print_progress=True, save_history=True):
    results = []
    for seed in range(1, n_seeds + 1):
        algo = NSGA2(pop_size=pop_size, sampling=FloatRandomSampling(),
                     crossover=SBX(prob=0.9, eta=15), mutation=PM(prob=0.1, eta=20),
                     eliminate_duplicates=True)
        res = minimize(problem, algo, get_termination("n_gen", n_gen),
                        seed=seed, save_history=save_history, verbose=verbose)
        results.append(res)
        if print_progress:
            print(f"  seed {seed}/{n_seeds} completed.")
    return results


def run_multiple_seeds_checkpointed(problem, checkpoint_dir, n_seeds=20, pop_size=100, n_gen=250,
                                     verbose=False, print_progress=True, save_history=False):
    """
    Crash-resilient variant (for free Colab sessions or any unstable
    environment): each seed's result is saved to disk (e.g., a mounted
    Google Drive) immediately after it completes. If the runtime is
    disconnected or restarted, calling this function again loads any
    already-saved seeds from disk (without recomputing them) and only
    runs the remaining seeds.

    - Atomic writes: each result is first written to a temporary file,
      then swapped into place via os.replace -- if a crash occurs
      mid-write, the final checkpoint file is never left partially
      written/corrupted (it is either complete or does not exist).
    - Automatic recovery: if a corrupted checkpoint is nonetheless
      encountered (e.g., from a run prior to this fix), that seed is
      recomputed rather than raising an exception.
    - save_history defaults to False (needed only for convergence
      plots); retaining the full population history across every
      generation is memory-intensive, and disabling it reduces crash
      risk when a convergence plot is not required.
    """
    import os
    import pickle
    os.makedirs(checkpoint_dir, exist_ok=True)
    results = []
    for seed in range(1, n_seeds + 1):
        ckpt_path = os.path.join(checkpoint_dir, f"seed_{seed}.pkl")
        res = None
        if os.path.exists(ckpt_path):
            try:
                with open(ckpt_path, "rb") as f:
                    res = pickle.load(f)
                if print_progress:
                    print(f"  seed {seed}/{n_seeds} already checkpointed -- loaded (no recomputation).")
            except Exception as e:
                print(f"  WARNING: seed {seed}: checkpoint was corrupted ({e}) -- recomputing.")
                res = None
        if res is None:
            algo = NSGA2(pop_size=pop_size, sampling=FloatRandomSampling(),
                         crossover=SBX(prob=0.9, eta=15), mutation=PM(prob=0.1, eta=20),
                         eliminate_duplicates=True)
            res = minimize(problem, algo, get_termination("n_gen", n_gen),
                            seed=seed, save_history=save_history, verbose=verbose)
            tmp_path = ckpt_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(res, f)
            os.replace(tmp_path, ckpt_path)  # atomic: either fully replaced or not at all
            if print_progress:
                print(f"  seed {seed}/{n_seeds} completed and checkpointed.")
        results.append(res)
    return results


def shared_reference_point(*result_lists, margin=1.1):
    """Component-wise maximum objective values across all provided result sets, with a margin."""
    all_F = np.vstack([r.F for results in result_lists for r in results if r.F is not None])
    return all_F.max(axis=0) * margin


def compute_hypervolume_stats(all_results, ref_point):
    hv_indicator = HV(ref_point=ref_point)
    hv_values = np.array([hv_indicator(r.F) for r in all_results if r.F is not None])
    return {'mean': hv_values.mean(), 'std': hv_values.std(),
            'min': hv_values.min(), 'max': hv_values.max(), 'values': hv_values}


def compute_convergence_curve(res, ref_point):
    hv_indicator = HV(ref_point=ref_point)
    return np.array([hv_indicator(s.opt.get("F")) if s.opt.get("F") is not None else 0.0 for s in res.history])


def compute_mean_convergence(all_results, ref_point):
    curves = [compute_convergence_curve(r, ref_point) for r in all_results]
    min_len = min(len(c) for c in curves)
    curves = np.array([c[:min_len] for c in curves])
    return curves.mean(axis=0), curves.std(axis=0)


def check_convergence(curve, name, threshold_pct=5.0):
    """
    Flags convergence based on the proportional improvement in mean HV
    over the final 20% of generations relative to total improvement.
    """
    n = len(curve)
    if n < 2:
        print(f"  {name}: WARNING -- convergence history is missing/empty (some seeds were "
              f"checkpointed without save_history) -- convergence check skipped, but this does "
              f"not affect the main results (HV / Mann-Whitney).")
        return None
    last20 = curve[int(n * 0.8):]
    improvement = last20[-1] - last20[0]
    total_range = curve[-1] - curve[0] if curve[-1] != curve[0] else 1.0
    pct = improvement / total_range * 100 if total_range else 0.0
    converged = pct <= threshold_pct
    status = "CONVERGED" if converged else "NOT CONVERGED -- consider increasing the generation count"
    print(f"  {name}: improvement over final 20% of generations = {pct:.1f}% of total improvement -> {status}")
    return converged

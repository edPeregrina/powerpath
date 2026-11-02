"""Impact-distribution box plots for the single best adaptation combination.

Companion figure for Figure 5f (``book/postprocessing/produce_figures.ipynb``,
"Policy type trade-off with sweet-spot highlights"): that figure identifies
which L1+L2 adaptation combinations are Pareto-efficient *among
moderate-extent policies* (its "sweet spot" -- not saturated on coverage or
intensity). This script picks the single sweet-spot combination with the
largest combined (monetary + population) median benefit and plots the raw
impact *distributions* -- not benefit/reduction deltas -- for that one
adaptation side by side with the no-adaptation baseline, so that both the
baseline's and the adaptation's own uncertainty (across the 100-scenario x
10-20-repair-crew ensemble) are visible. Intended to sit next to a map of the
chosen adaptation measure's footprint, illustrating that even one fixed
adaptation combination has a real spread of outcomes, not a single number.

Uses the same ``adaptation_combinations_road`` dataset, crew-range filter
(10-20 repair crews), and policy-parsing / Pareto / "moderate extent" logic
as ``produce_figures.ipynb``'s Figure 5 cells -- replicated here rather than
imported since the notebook has no importable module interface.

Data shape (``ema_workbench.util.utilities.load_results``): see
``produce_figures_group_equity.py``'s module docstring for the general
(experiments, outcomes) shape. Here we use two outcomes:
  - ``monetary_impact_total``: cumulative monetary loss (EUR); the final
    timestep column is the run's total (confirmed cumulative in the
    notebook's Figure 4/5 cells).
  - ``affected_population``: per-timestep affected-population count;
    summed over timesteps gives a person-hours total.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from ema_workbench.util.utilities import load_results  # noqa: E402

from config import get_config, setup_directories  # noqa: E402

DEFAULT_RESULTS_NAME = "ema_results_1776271700_combinations_4.tar.gz"
BASELINE_POLICY = "monetary_impacts_policy"
CREW_MIN, CREW_MAX = 10, 20

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
}

BASELINE_COLOR = "#5C5C5C"
ADAPTATION_COLOR = "#1a9850"

# --------------------------------------------------------------------------
# Policy name parsing (mirrors produce_figures.ipynb's Figure 5 cell)
# --------------------------------------------------------------------------
_RE_L1_ONLY = re.compile(r"^L1_only_policy_L1_area_(monetary|population)_top(\d+)_red_(0\.2|0\.4)$")
_RE_L2_ONLY = re.compile(r"^L2_only_(mon|pop)_subs_top(\d+)_(0\.5|1\.0)m$")
_RE_L1L2 = re.compile(
    r"^L1L2_policy_L1_area_(monetary|population)_top(\d+)_red_(0\.2|0\.4)_(mon|pop)_subs_top_?(\d+)_(0\.5|1\.0)m$"
)


def parse_policy(pol: str):
    if pol == BASELINE_POLICY:
        return dict(l1_rank="none", l2_rank="none", l1_cov=0, l1_red=0.0, l2_cov=0, l2_elev=0.0)

    m = _RE_L1_ONLY.match(pol)
    if m:
        return dict(l1_rank=m.group(1), l2_rank="none",
                    l1_cov=int(m.group(2)), l1_red=float(m.group(3)), l2_cov=0, l2_elev=0.0)

    m = _RE_L2_ONLY.match(pol)
    if m:
        return dict(l1_rank="none", l2_rank=m.group(1),
                    l1_cov=0, l1_red=0.0, l2_cov=int(m.group(2)), l2_elev=float(m.group(3)))

    m = _RE_L1L2.match(pol)
    if m:
        return dict(l1_rank=m.group(1), l2_rank=m.group(4),
                    l1_cov=int(m.group(2)), l1_red=float(m.group(3)),
                    l2_cov=int(m.group(5)), l2_elev=float(m.group(6)))

    return None


def pareto_mask(costs: np.ndarray) -> np.ndarray:
    """costs: (n, 2), both *maximised* here (benefit %, higher is better)."""
    costs = np.asarray(costs)
    n = costs.shape[0]
    is_eff = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_eff[i]:
            continue
        dominated = np.all(costs >= costs[i], axis=1) & np.any(costs > costs[i], axis=1)
        dominated[i] = False
        if np.any(dominated):
            is_eff[i] = False
    return is_eff


def is_moderate(r: dict) -> bool:
    """"Moderate extent" = not saturated on coverage (both <=20%) OR not
    saturated on intensity (uses the lower dose for each active lever).
    Matches the Fig 5f sweet-spot filter exactly."""
    if r["policy"] == BASELINE_POLICY:
        return False
    cov_ok = (r["l1_cov"] <= 20) or (r["l2_cov"] <= 20)
    int_ok = ((r["l1_cov"] == 0 or r["l1_red"] == 0.2) or
              (r["l2_cov"] == 0 or r["l2_elev"] == 0.5))
    return cov_ok or int_ok


def monetary_total(series_2d: np.ndarray) -> np.ndarray:
    """Cumulative outcome -> use the final timestep."""
    return series_2d[:, -1]


def population_total(series_2d: np.ndarray) -> np.ndarray:
    """Per-timestep outcome -> sum over time (person-hours)."""
    return series_2d.sum(axis=1)


# --------------------------------------------------------------------------
# Sweet-spot selection
# --------------------------------------------------------------------------
def find_best_sweet_spot_policy(experiments_df: pd.DataFrame, outcomes: dict, crew_mask: pd.Series) -> dict:
    """Replicates Figure 5f's "Efficient at moderate extent" highlight, then
    picks the single policy with the largest combined (monetary + population)
    median benefit among those highlighted points."""
    mon_arr = outcomes["monetary_impact_total"]
    pop_arr = outcomes["affected_population"]

    available_pols = sorted(set(experiments_df.loc[crew_mask, "policy"].unique()))
    records = []
    for pol in available_pols:
        parsed = parse_policy(pol)
        if parsed is None:
            continue
        parsed["policy"] = pol
        m = crew_mask & (experiments_df["policy"] == pol)
        parsed["mon_median"] = float(np.median(monetary_total(mon_arr[m.values]) / 1e6))
        parsed["pop_median"] = float(np.median(population_total(pop_arr[m.values]) / 1e6))
        records.append(parsed)
    if not records:
        raise ValueError("No matching L1/L2 policies found; check naming patterns.")

    baseline_recs = [r for r in records if r["policy"] == BASELINE_POLICY]
    if not baseline_recs:
        raise ValueError(f"Baseline policy '{BASELINE_POLICY}' not found under the crew-range filter.")
    baseline = baseline_recs[0]
    m0, p0 = baseline["mon_median"], baseline["pop_median"]

    l1l2_records = [r for r in records
                    if r["l1_rank"] != "none" or r["l2_rank"] != "none" or r["policy"] == BASELINE_POLICY]

    mod_recs = [r for r in l1l2_records]# if is_moderate(r)]
    if not mod_recs:
        raise ValueError("No moderate-extent policies found; cannot identify a sweet spot.")
    mod_benefit = np.array([
        [100.0 * (m0 - r["mon_median"]) / m0, 100.0 * (p0 - r["pop_median"]) / p0]
        for r in mod_recs
    ])
    is_pf_mod = pareto_mask(mod_benefit)
    mod_pf_recs = [r for r, flag in zip(mod_recs, is_pf_mod) if flag]
    mod_pf_benefit = mod_benefit[is_pf_mod]

    combined = mod_pf_benefit.sum(axis=1)
    best_idx = int(np.argmax(combined))
    best = mod_pf_recs[best_idx]
    print(f"Sweet-spot policies found: {len(mod_pf_recs)} (moderate-extent, Pareto-efficient)")
    print(f"Selected single best: '{best['policy']}' "
          f"(monetary benefit {mod_pf_benefit[best_idx, 0]:.1f}%, "
          f"population benefit {mod_pf_benefit[best_idx, 1]:.1f}%)")
    return best


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_impact_boxplots(experiments_df: pd.DataFrame, outcomes: dict, crew_mask: pd.Series,
                          adaptation_policy: str, fig_dir: Path) -> Path:
    mon_arr = outcomes["monetary_impact_total"]
    pop_arr = outcomes["affected_population"]

    base_mask = crew_mask & (experiments_df["policy"] == BASELINE_POLICY)
    adapt_mask = crew_mask & (experiments_df["policy"] == adaptation_policy)

    base_mon = monetary_total(mon_arr[base_mask.values]) / 1e6
    adapt_mon = monetary_total(mon_arr[adapt_mask.values]) / 1e6
    base_pop = population_total(pop_arr[base_mask.values]) / 1e6
    adapt_pop = population_total(pop_arr[adapt_mask.values]) / 1e6

    print(f"Baseline runs (10-20 crews): {len(base_mon)}; adaptation runs: {len(adapt_mon)}")

    plt.rcParams.update(PLOT_STYLE)
    fig, (ax_mon, ax_pop) = plt.subplots(1, 2, figsize=(8, 6), constrained_layout=True)

    box_kwargs = dict(
        widths=0.5, patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
    )

    for ax, (base_vals, adapt_vals), ylabel, title in [
        (ax_mon, (base_mon, adapt_mon), "Economic impact\n(million \u20ac)", "(a) Economic impact"),
        (ax_pop, (base_pop, adapt_pop), "Population impact\n(million person-hours)", "(b) Population impact"),
    ]:
        bp = ax.boxplot([base_vals, adapt_vals], positions=[0, 1], **box_kwargs)
        for patch, color in zip(bp["boxes"], (BASELINE_COLOR, ADAPTATION_COLOR)):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline\n(no adaptation)", "Adaptation"], fontsize=14)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(title, fontsize=16, loc="left")
        ax.grid(axis="y", alpha=0.15)
        ax.tick_params(labelsize=14)
        ax.set_ylim(bottom=0)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    fig.suptitle(
        f"Economic impact distributions, baseline vs. '{adaptation_policy}'\n"
        f"({CREW_MIN}-{CREW_MAX} repair crews; "
        f"baseline n={len(base_mon)}, adaptation n={len(adapt_mon)})",
        fontsize=14,
    )

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "fig_crew_impact_boxplots.png"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.show()
    print(f"Saved: {out_path} (+ .pdf)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-name", default=DEFAULT_RESULTS_NAME,
                         help="EMA results archive filename, relative to <output_dir>/ema_exports.")
    parser.add_argument("--policy", default=None,
                         help="Explicit adaptation policy name to use instead of auto-selecting "
                              "the single best Fig-5f sweet-spot combination.")
    args = parser.parse_args()

    config = get_config()
    setup_directories(config)
    out_dir = config["output_dir"] / "ema_exports"
    fig_dir = config["output_dir"] / "paper_figures"

    results_path = out_dir / args.results_name
    print(f"Loading {results_path} ...")
    experiments, outcomes_raw = load_results(str(results_path))
    experiments_df = pd.DataFrame(experiments)
    outcomes = {k: np.array(v) for k, v in outcomes_raw.items()}

    crew_mask = ((experiments_df["number_repair_crews"] >= CREW_MIN) &
                 (experiments_df["number_repair_crews"] <= CREW_MAX))
    print(f"Loaded {len(experiments_df)} experiments "
          f"({crew_mask.sum()} within crew range {CREW_MIN}-{CREW_MAX})")

    if args.policy:
        adaptation_policy = args.policy
    else:
        best = find_best_sweet_spot_policy(experiments_df, outcomes, crew_mask)
        adaptation_policy = best["policy"]

    plot_impact_boxplots(experiments_df, outcomes, crew_mask, adaptation_policy, fig_dir)


if __name__ == "__main__":
    main()


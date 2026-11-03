"""Population-group differentiation in disruption and adaptation preference.

Post-processes a full-extent EMA workbench adaptation run (see
``book/interdependent_ci_societal_access.py``) to answer the three questions
structuring the paper's Results section:

  3.1 Population-group differences in loss of access (who loses access?)
  3.2 Adaptation benefits differentiated by population group (who benefits?)
  3.3 Consequences of group differentiation for adaptation preference (does
      distinguishing beneficiaries change which adaptation is preferable?)

Vocabulary follows the paper's introduction: "population access to
functioning service providers", "societal functions", "population groups",
"adaptation benefits/preference" -- not generic EMA-workbench jargon.

Data shape (``ema_workbench.util.utilities.load_results``):
  - ``experiments``: one row per (scenario x policy), columns are the
    sampled uncertainties (``number_repair_crews__<asset_type>``,
    ``fragility_param_k_divby2__msls``) plus ``scenario``/``policy``/``model``.
  - ``outcomes``: ``{name: ndarray(n_experiments, n_timesteps)}``. Every
    societal-access metric is a ``TimeSeriesOutcome`` named
    ``societal_<metric>__<function>__<population_group>`` (see
    ``src.societal_access.list_societal_metric_names``).
  - Each of the ``n_timesteps`` columns is **one hour** (confirmed from the
    ``timesteps`` outcome: unit increments of 1). This is the simulation's
    internal bookkeeping step, *not* the 6-hourly hazard map spacing used
    elsewhere (``MAJOR_TS`` in ``book/postprocessing/produce_figures.ipynb``)
    -- don't reuse that constant here when converting to "people-days".

Policies: 171 total = 3 repair-crew-assignment strategies (no L1/L2
adaptation: ``baseline_no_prioritisation``, ``monetary_impacts_policy``,
``monetary_impacts_unconstrained``) + 168 adaptation measures (12 L1-only +
12 L2-only + 144 L1xL2 combinations), all built on top of
``repair_crew_assignment_method="monetary impacts islands"`` (prioritised
repair, constrained to shared road islands). ``monetary_impacts_policy`` is
used as the "no adaptation" baseline throughout, since it is the
islands-constrained policy without any L1/L2 measure (the fairest
counterfactual for isolating adaptation effects from repair strategy).

Every figure's underlying statistics are computed over the FULL policy set
(all 168 adaptation policies x 100 scenarios); only the *plots* subsample for
legibility (documented per-figure below). Each figure's underlying stats
table is written as a CSV alongside its PNG in ``data/output/
paper_figures_group_equity/``:

  - ``figure2_loss_of_access_stats.csv``: median/p05/p95 loss-of-access (%)
    and median equity gap, per function x population group (baseline
    policy only).
  - ``figure3_policy_benefit_stats.csv``: median/p05/p95 access-pct-gain,
    median people-days-avoided and equity-gap-closed, per policy (all 168)
    x function (incl. the ``all_functions`` aggregate) x population group.
  - ``figure4_policy_rankings.csv``: rank + median benefit per policy (all
    168, not just the plotted subsample) x population group.
  - ``figure4_top1_per_group.csv``: the rank-1 policy per group and whether
    the preferred adaptation is unchanged across groups.
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from ema_workbench.util.utilities import load_results  # noqa: E402

DEFAULT_RESULTS_PATH = Path(
    r"C:\Users\peregrin\OneDrive - Stichting Deltares\Documents\Save runs\Powerpath"
    r"\1791817945\ema_results_1791817945_full_extent_adaptation.tar.gz"
)
CACHE_DIR = REPO_ROOT / "data" / "interim" / "paper_figures_group_equity_cache"
FIG_DIR = REPO_ROOT / "data" / "output" / "paper_figures_group_equity"

HOURS_PER_TIMESTEP = 1.0  # see module docstring -- do not confuse with the
                          # 6-hourly hazard map spacing used elsewhere.
REFERENCE_GROUP = "total"
# Canonical display order (reference group first); discovered groups are
# filtered/reordered against this rather than used in alphabetical order.
GROUP_DISPLAY_ORDER = ["total", "elderly", "children", "working_age"]

# Preferred no-adaptation baseline candidates, newest-naming first (current
# default repair_priority is "population impacts islands" ->
# "population_impacts_islands_policy"); falls back through legacy names and
# finally "baseline_no_prioritisation" (plain "islands", always present).
BASELINE_POLICY_CANDIDATES = [
    "population_impacts_islands_policy",
    "monetary_impacts_policy",
    "baseline_no_prioritisation",
]
NON_ADAPTATION_POLICIES = {
    "baseline_no_prioritisation",
    "population_impacts_islands_policy",
    "population_impacts_unconstrained",
    "monetary_impacts_policy",
    "monetary_impacts_unconstrained",
}

GROUP_LABELS = {
    "total": "Total population",
    "elderly": "Elderly",
    "children": "Children",
    "working_age": "Working age",
}
FUNCTION_LABELS = {
    "electricity": "Electricity",
    "hospital": "Hospital",
    "primary_care": "Primary care",
    "pharmacy": "Pharmacy",
    "ems": "EMS",
    "supermarket": "Supermarket",
    "all_functions": "All functions (mean)",
}
CATEGORY_LABELS = {
    "L1-only": "L1 (area-based)",
    "L2-only": "L2 (asset barriers)",
    "L1+L2": "L1 + L2 combined",
}
CATEGORY_COLORS = {
    "L1-only": "#4C72B0",
    "L2-only": "#DD8452",
    "L1+L2": "#55A868",
}
GROUP_COLORS = {
    "total": "#333333",
    "elderly": "#C44E52",
    "children": "#8172B2",
    "working_age": "#4C72B0",
}

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


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _cache_key(results_path: Path) -> str:
    stat = results_path.stat()
    digest = hashlib.sha1(f"{results_path}-{stat.st_size}-{stat.st_mtime}".encode()).hexdigest()
    return digest[:16]


def load_or_cache_results(results_path: Path, refresh: bool = False):
    """Load EMA results, caching the parsed (experiments, outcomes) tuple as
    a pickle keyed on the archive's path/size/mtime so repeated figure
    iterations don't re-parse the ~43MB tar.gz (~50s) every time."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"results_{_cache_key(results_path)}.pkl"
    if cache_path.exists() and not refresh:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    experiments, outcomes = load_results(str(results_path))
    with open(cache_path, "wb") as f:
        pickle.dump((experiments, outcomes), f, protocol=4)
    return experiments, outcomes


def discover_functions_and_groups(outcome_keys: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Parse ``societal_<metric>__<function>__<group>`` outcome names to
    recover the actual function/group vocabulary present in this run
    (rather than hardcoding the full taxonomy, since not every asset type
    is necessarily present in every run)."""
    functions, groups = set(), set()
    for key in outcome_keys:
        if not key.startswith("societal_"):
            continue
        parts = key.split("__")
        if len(parts) == 3:
            functions.add(parts[1])
            groups.add(parts[2])
        elif len(parts) == 2 and parts[0] == "societal_total_population":
            groups.add(parts[1])
    ordered_groups = [g for g in GROUP_DISPLAY_ORDER if g in groups]
    ordered_groups += sorted(groups - set(ordered_groups))  # any unexpected extras, appended
    return sorted(functions), ordered_groups


def classify_policy(policy_name: str) -> str:
    if policy_name in NON_ADAPTATION_POLICIES:
        return "baseline/other"
    if policy_name.startswith("L1L2_"):
        return "L1+L2"
    if policy_name.startswith("L1_only"):
        return "L1-only"
    if policy_name.startswith("L2_only"):
        return "L2-only"
    return "other"


# --------------------------------------------------------------------------
# Metric frame: one row per (experiment, function, group)
# --------------------------------------------------------------------------
def build_metric_frame(
    experiments_df: pd.DataFrame,
    outcomes: Dict[str, np.ndarray],
    functions: Sequence[str],
    groups: Sequence[str],
) -> pd.DataFrame:
    """Build a long-format frame with, per experiment x function x group:

    - ``mean_access_pct``: event-average % of the group with access
      (0-100; comparable across groups of different sizes).
    - ``people_days_no_access``: time-integrated people without access,
      in person-days (``sum(no_access_population) * 1h / 24``); policy-
      relevant absolute magnitude, NOT comparable across groups of
      different sizes.
    - ``equity_abs_gap``: event-average ``societal_equity_absolute_gap``
      (= total_access_pct - group_access_pct; positive => group worse off
      than the total population), NaN for the reference group itself.

    Also appends a synthetic ``function == "all_functions"`` block (simple
    mean across the available functions) for the cross-function benefit
    ranking used in Figures 3-4.
    """
    n = len(experiments_df)
    base_cols = {
        "policy": experiments_df["policy"].to_numpy(),
        "scenario": experiments_df["scenario"].to_numpy(),
    }

    frames = []
    per_function_access = {}
    per_function_people_days = {}
    per_function_abs_gap = {}

    for func in functions:
        access_pct = np.full((n, len(groups)), np.nan)
        people_days = np.full((n, len(groups)), np.nan)
        abs_gap = np.full((n, len(groups)), np.nan)
        for gi, group in enumerate(groups):
            access_key = f"societal_access_pct__{func}__{group}"
            no_access_key = f"societal_no_access_population__{func}__{group}"
            if access_key in outcomes:
                access_pct[:, gi] = np.nanmean(outcomes[access_key], axis=1)
            if no_access_key in outcomes:
                people_days[:, gi] = (
                    np.nansum(outcomes[no_access_key], axis=1) * HOURS_PER_TIMESTEP / 24.0
                )
            if group != REFERENCE_GROUP:
                gap_key = f"societal_equity_absolute_gap__{func}__{group}"
                if gap_key in outcomes:
                    abs_gap[:, gi] = np.nanmean(outcomes[gap_key], axis=1)

        per_function_access[func] = access_pct
        per_function_people_days[func] = people_days
        per_function_abs_gap[func] = abs_gap

        for gi, group in enumerate(groups):
            frames.append(pd.DataFrame({
                **base_cols,
                "function": func,
                "population_group": group,
                "mean_access_pct": access_pct[:, gi],
                "people_days_no_access": people_days[:, gi],
                "equity_abs_gap": abs_gap[:, gi],
            }))

    # Synthetic cross-function aggregate (unweighted mean of access_pct
    # across functions; sum of people_days across functions).
    stacked_access = np.nanmean(np.stack([per_function_access[f] for f in functions], axis=0), axis=0)
    stacked_people_days = np.nansum(np.stack([per_function_people_days[f] for f in functions], axis=0), axis=0)
    stacked_abs_gap = np.nanmean(np.stack([per_function_abs_gap[f] for f in functions], axis=0), axis=0)
    for gi, group in enumerate(groups):
        frames.append(pd.DataFrame({
            **base_cols,
            "function": "all_functions",
            "population_group": group,
            "mean_access_pct": stacked_access[:, gi],
            "people_days_no_access": stacked_people_days[:, gi],
            "equity_abs_gap": stacked_abs_gap[:, gi],
        }))

    metric_df = pd.concat(frames, ignore_index=True)
    metric_df["policy_category"] = metric_df["policy"].map(classify_policy)
    return metric_df


# --------------------------------------------------------------------------
# Figure 2 -- population-group differences in loss of access
# --------------------------------------------------------------------------
def figure2_loss_of_access(metric_df: pd.DataFrame, baseline_policy: str, functions: Sequence[str],
                            groups: Sequence[str], fig_dir: Path) -> None:
    """3.1 -- who loses access? Baseline (no-adaptation) policy only.

    (a) Overall disruption by asset type (total population).
    (b) Same metric broken down by population group (do losses differ?).
    (c) Equity gap by function/group (where are the biggest differences?).
    """
    base = metric_df[(metric_df["policy"] == baseline_policy) & (metric_df["function"] != "all_functions")]
    base = base.copy()
    base["loss_pct"] = 100.0 - base["mean_access_pct"]

    func_order = sorted(functions, key=lambda f: -base.loc[
        (base["function"] == f) & (base["population_group"] == REFERENCE_GROUP), "loss_pct"
    ].median())

    fig, axes = plt.subplots(1, 3, figsize=(22, 9), constrained_layout=True)

    # (a) overall disruption by asset type -- total population only
    ax = axes[0]
    totals = base[base["population_group"] == REFERENCE_GROUP]
    medians = [totals.loc[totals["function"] == f, "loss_pct"].median() for f in func_order]
    lo = [totals.loc[totals["function"] == f, "loss_pct"].quantile(0.05) for f in func_order]
    hi = [totals.loc[totals["function"] == f, "loss_pct"].quantile(0.95) for f in func_order]
    x = np.arange(len(func_order))
    ax.bar(x, medians, color=GROUP_COLORS["total"], width=0.6)
    ax.errorbar(x, medians, yerr=[np.array(medians) - lo, np.array(hi) - medians],
                fmt="none", ecolor="black", capsize=4, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([FUNCTION_LABELS.get(f, f) for f in func_order], rotation=35, ha="right")
    ax.set_ylabel("Mean loss of access (%)")
    ax.set_title("(a) Overall disruption by societal function")

    # (b) broken down by population group
    ax = axes[1]
    width = 0.8 / len(groups)
    for gi, group in enumerate(groups):
        sub = base[base["population_group"] == group]
        meds = [sub.loc[sub["function"] == f, "loss_pct"].median() for f in func_order]
        ax.bar(x + gi * width - 0.4 + width / 2, meds, width=width,
               label=GROUP_LABELS.get(group, group), color=GROUP_COLORS.get(group, None))
    ax.set_xticks(x)
    ax.set_xticklabels([FUNCTION_LABELS.get(f, f) for f in func_order], rotation=35, ha="right")
    ax.set_ylabel("Mean loss of access (%)")
    ax.set_title("(b) Loss of access by population group")
    ax.legend(loc="upper right", fontsize=11)

    # (c) equity gap -- pinpoint biggest group differences
    ax = axes[2]
    non_ref_groups = [g for g in groups if g != REFERENCE_GROUP]
    gap_rows = []
    for f in func_order:
        for g in non_ref_groups:
            sub = base[(base["function"] == f) & (base["population_group"] == g)]
            gap_rows.append((f, g, sub["equity_abs_gap"].median()))
    gap_df = pd.DataFrame(gap_rows, columns=["function", "population_group", "gap"])
    gap_df = gap_df.reindex(gap_df["gap"].abs().sort_values(ascending=False).index)
    labels = [f"{FUNCTION_LABELS.get(f, f)} ({GROUP_LABELS.get(g, g)})" for f, g in
              zip(gap_df["function"], gap_df["population_group"])]
    colors = [GROUP_COLORS.get(g, "grey") for g in gap_df["population_group"]]
    yy = np.arange(len(gap_df))
    ax.barh(yy, gap_df["gap"], color=colors)
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Equity gap (total % access - group % access)")
    ax.set_title("(c) Where the biggest group differences lie")

    fig.suptitle(
        f"Figure 2. Population-group-specific loss of access to societal functions during\n"
        f"pluvial flooding (baseline policy: {baseline_policy}, n=100 scenarios)",
        fontsize=15,
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "figure2_loss_of_access.png")
    plt.close(fig)

    # Underlying stats: median/p05/p95 loss (panels a/b) and median equity
    # gap (panel c), one row per function x population group.
    stats_rows = []
    for f in func_order:
        for g in groups:
            sub = base.loc[(base["function"] == f) & (base["population_group"] == g), "loss_pct"]
            gap_sub = base.loc[(base["function"] == f) & (base["population_group"] == g), "equity_abs_gap"]
            stats_rows.append({
                "function": f,
                "population_group": g,
                "loss_pct_median": sub.median(),
                "loss_pct_p05": sub.quantile(0.05),
                "loss_pct_p95": sub.quantile(0.95),
                "equity_abs_gap_median": gap_sub.median() if g != REFERENCE_GROUP else np.nan,
                "n_scenarios": sub.shape[0],
            })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(fig_dir / "figure2_loss_of_access_stats.csv", index=False)


# --------------------------------------------------------------------------
# Figure 3 -- adaptation benefits differentiated by population group
# --------------------------------------------------------------------------
def compute_benefit_frame(metric_df: pd.DataFrame, baseline_policy: str) -> pd.DataFrame:
    """Paired per-scenario benefit of every adaptation policy vs the
    baseline (same scenario => same uncertainty draw, so differencing
    cancels out scenario-level variation). Uses ALL adaptation policies,
    not just the plotted subsample."""
    baseline = metric_df[metric_df["policy"] == baseline_policy].set_index(
        ["scenario", "function", "population_group"]
    )[["mean_access_pct", "people_days_no_access", "equity_abs_gap"]]

    adaptation = metric_df[metric_df["policy_category"].isin(["L1-only", "L2-only", "L1+L2"])]
    merged = adaptation.join(
        baseline, on=["scenario", "function", "population_group"], rsuffix="_baseline"
    )
    merged["access_pct_gain"] = merged["mean_access_pct"] - merged["mean_access_pct_baseline"]
    merged["people_days_avoided"] = merged["people_days_no_access_baseline"] - merged["people_days_no_access"]
    merged["equity_gap_closed"] = merged["equity_abs_gap_baseline"] - merged["equity_abs_gap"]
    return merged


def figure3_adaptation_benefits(benefit_df: pd.DataFrame, groups: Sequence[str], fig_dir: Path) -> None:
    """3.2 -- who benefits from adaptation? All 168 adaptation policies
    contribute to each distribution; the plot subsamples by showing
    category-level distributions (L1-only / L2-only / L1+L2) rather than
    every individual policy, for legibility."""
    overall = benefit_df[benefit_df["function"] == "all_functions"]
    categories = ["L1-only", "L2-only", "L1+L2"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)

    # (a) relative benefit: percentage points of access gained
    ax = axes[0]
    _grouped_box(ax, overall, categories, groups, "access_pct_gain",
                 "Access gained (percentage points)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("(a) Access benefit by adaptation category and population group")

    # (b) equity-gap closed: does adaptation narrow or widen group gaps?
    non_ref = overall[overall["population_group"] != REFERENCE_GROUP]
    ax = axes[1]
    _grouped_box(ax, non_ref, categories, [g for g in groups if g != REFERENCE_GROUP],
                 "equity_gap_closed", "Equity gap closed (pp)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("(b) Change in group-vs-total equity gap")

    fig.suptitle(
        "Figure 3. Population-group-specific access benefits of adaptation measures\n"
        "(all 168 adaptation policies x 100 scenarios; distributions by category)",
        fontsize=15,
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "figure3_adaptation_benefits.png")
    plt.close(fig)

    # Underlying stats: per-policy (all 168, every function + the
    # all_functions aggregate) x population group median/p05/p95 benefit
    # across the 100 paired scenarios -- the full data behind the
    # category-level box plots above.
    policy_stats = benefit_df.groupby(
        ["policy", "policy_category", "function", "population_group"]
    ).agg(
        access_pct_gain_median=("access_pct_gain", "median"),
        access_pct_gain_p05=("access_pct_gain", lambda s: s.quantile(0.05)),
        access_pct_gain_p95=("access_pct_gain", lambda s: s.quantile(0.95)),
        people_days_avoided_median=("people_days_avoided", "median"),
        equity_gap_closed_median=("equity_gap_closed", "median"),
        n_scenarios=("access_pct_gain", "size"),
    ).reset_index()
    policy_stats.to_csv(fig_dir / "figure3_policy_benefit_stats.csv", index=False)


def _grouped_box(ax, df: pd.DataFrame, categories: Sequence[str], groups: Sequence[str],
                  value_col: str, ylabel: str) -> None:
    n_groups = len(groups)
    width = 0.8 / n_groups
    positions_base = np.arange(len(categories))
    for gi, group in enumerate(groups):
        data = [
            df.loc[(df["policy_category"] == cat) & (df["population_group"] == group), value_col].dropna()
            for cat in categories
        ]
        positions = positions_base + gi * width - 0.4 + width / 2
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True,
                         showfliers=False, medianprops={"color": "black"})
        for patch in bp["boxes"]:
            patch.set_facecolor(GROUP_COLORS.get(group, "grey"))
            patch.set_alpha(0.75)
    ax.set_xticks(positions_base)
    ax.set_xticklabels([CATEGORY_LABELS.get(c, c) for c in categories])
    ax.set_ylabel(ylabel)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS.get(g, "grey"), alpha=0.75)
               for g in groups]
    ax.legend(handles, [GROUP_LABELS.get(g, g) for g in groups], loc="best", fontsize=11)


# --------------------------------------------------------------------------
# Figure 4 -- consequences of group differentiation for adaptation preference
# --------------------------------------------------------------------------
def _short_policy_label(policy: str) -> str:
    """Abbreviate a (potentially long, L1xL2-combination) policy name for
    compact axis annotation, e.g.
    ``L1L2_policy_L1_area_population_top30_red_0.4_pop_subs_top30_1.0m`` ->
    ``L1-pop t30 r0.4 + L2-pop t30 1.0m``."""
    name = re.sub(r"^L1L2_", "", policy)
    name = re.sub(r"^L1_only_", "L1-only ", name)
    name = re.sub(r"^L2_only_", "L2-only ", name)
    name = re.sub(r"policy_L1_area_monetary_", "L1-mon ", name)
    name = re.sub(r"policy_L1_area_population_", "L1-pop ", name)
    name = re.sub(r"_mon_subs_", " + L2-mon ", name)
    name = re.sub(r"_pop_subs_", " + L2-pop ", name)
    name = name.replace("_red_", " r")
    name = name.replace("top", "t")
    name = name.replace("_", " ")
    return name


def figure4_preference_consequences(benefit_df: pd.DataFrame, groups: Sequence[str], fig_dir: Path,
                                     top_n: int = 10) -> None:
    """3.3 -- does distinguishing beneficiaries change which adaptation is
    preferred? Ranks ALL 168 adaptation policies, per group, by median
    cross-function access benefit (``all_functions`` access_pct_gain across
    the 100 scenarios). The plot subsamples to a bump chart of the union of
    each group's top-``top_n`` policies (a "clever subsample": every policy
    is ranked using the full dataset; only the top performers are drawn).
    """
    overall = benefit_df[benefit_df["function"] == "all_functions"]
    rank_table = overall.groupby(["population_group", "policy"])["access_pct_gain"].median().unstack(0)
    rank_table = rank_table[groups]  # fixed column order: reference group first
    ranks = rank_table.rank(ascending=False, method="min")

    top_policies = set()
    for group in groups:
        top_policies.update(ranks[group].sort_values().index[:top_n])
    top_policies = sorted(top_policies, key=lambda p: ranks.loc[p, REFERENCE_GROUP])

    fig, ax = plt.subplots(figsize=(max(14, 2.4 * len(groups)), 0.62 * len(top_policies) + 2.5))
    xs = np.arange(len(groups))
    for policy in top_policies:
        ys = ranks.loc[policy, groups].to_numpy(dtype=float)
        category = classify_policy(policy)
        color = CATEGORY_COLORS.get(category, "grey")
        is_top1_anywhere = (ys == 1).any()
        ax.plot(xs, ys, marker="o", color=color, linewidth=2.2 if is_top1_anywhere else 1.0,
                alpha=1.0 if is_top1_anywhere else 0.55)
        ax.annotate(_short_policy_label(policy), xy=(xs[0] - 0.12, ys[0]),
                    ha="right", va="center", fontsize=9, clip_on=False)

    ax.set_xlim(-1.9, len(groups) - 1 + 0.3)
    ax.set_xticks(xs)
    ax.set_xticklabels([GROUP_LABELS.get(g, g) for g in groups])
    ax.invert_yaxis()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Rank (1 = best, among all 168 adaptation policies)")
    ax.set_title(
        "Figure 4. Influence of beneficiary differentiation on adaptation preference\n"
        f"(union of each group's top-{top_n} policies, ranked on cross-function access benefit)"
    )

    handles = [plt.Line2D([0], [0], color=c, lw=2.2) for c in CATEGORY_COLORS.values()]
    ax.legend(handles, [CATEGORY_LABELS[c] for c in CATEGORY_COLORS], loc="upper center",
              bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=11)

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "figure4_preference_consequences.png")
    plt.close(fig)

    # Underlying stats: full ranking (all 168 adaptation policies, not just
    # the plotted top-n subsample) of median cross-function access benefit
    # per population group, plus whether each policy falls in the plotted
    # top-n union for any group.
    rank_long = ranks.reset_index().melt(id_vars="policy", value_name="rank", var_name="population_group")
    benefit_long = rank_table.reset_index().melt(
        id_vars="policy", value_name="access_pct_gain_median", var_name="population_group"
    )
    rank_stats = rank_long.merge(benefit_long, on=["policy", "population_group"])
    rank_stats["policy_category"] = rank_stats["policy"].map(classify_policy)
    rank_stats["in_plotted_top_n"] = rank_stats["policy"].isin(top_policies)
    rank_stats = rank_stats.sort_values(["population_group", "rank"])
    rank_stats.to_csv(fig_dir / "figure4_policy_rankings.csv", index=False)

    top1_per_group = {group: ranks[group].idxmin() for group in groups}
    print("Top-ranked (rank=1) adaptation policy per population group:")
    for group, policy in top1_per_group.items():
        print(f"  {GROUP_LABELS.get(group, group):15s}: {policy}")
    unique_top1 = set(top1_per_group.values())
    if len(unique_top1) == 1:
        print("=> Preferred adaptation is UNCHANGED across population groups.")
    else:
        print(f"=> Preferred adaptation CHANGES across population groups ({len(unique_top1)} distinct top choices).")

    summary_df = pd.DataFrame([
        {"population_group": group, "top1_policy": policy}
        for group, policy in top1_per_group.items()
    ])
    summary_df["preferred_adaptation_unchanged"] = len(unique_top1) == 1
    summary_df.to_csv(fig_dir / "figure4_top1_per_group.csv", index=False)
    return top1_per_group


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def resolve_baseline_policy(available_policies: Sequence[str]) -> str:
    for candidate in BASELINE_POLICY_CANDIDATES:
        if candidate in available_policies:
            return candidate
    raise ValueError(
        f"None of the baseline candidates {BASELINE_POLICY_CANDIDATES} found in "
        f"available policies: {sorted(available_policies)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--top-n", type=int, default=10, help="Policies per group shown in Figure 4.")
    args = parser.parse_args()

    plt.rcParams.update(PLOT_STYLE)

    print(f"Loading results from: {args.results_path}")
    experiments, outcomes = load_or_cache_results(args.results_path, refresh=args.refresh_cache)
    experiments_df = pd.DataFrame(experiments)
    print(f"Loaded {len(experiments_df)} experiments, {len(outcomes)} outcomes.")

    functions, groups = discover_functions_and_groups(outcomes.keys())
    print(f"Functions: {functions}")
    print(f"Population groups: {groups}")

    baseline_policy = resolve_baseline_policy(experiments_df["policy"].unique())
    print(f"Baseline (no-adaptation) policy: {baseline_policy}")

    metric_df = build_metric_frame(experiments_df, outcomes, functions, groups)

    print("Building Figure 2 (loss of access)...")
    figure2_loss_of_access(metric_df, baseline_policy, functions, groups, FIG_DIR)

    print("Computing paired adaptation benefits (all 168 policies)...")
    benefit_df = compute_benefit_frame(metric_df, baseline_policy)

    print("Building Figure 3 (adaptation benefits by group)...")
    figure3_adaptation_benefits(benefit_df, groups, FIG_DIR)

    print("Building Figure 4 (preference consequences)...")
    figure4_preference_consequences(benefit_df, groups, FIG_DIR, top_n=args.top_n)

    print(f"\nFigures + underlying-stats CSVs written to: {FIG_DIR}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data" / "evidence"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    mechanism = pd.read_csv(EVIDENCE / "publication_trend_2x2_decomposition.csv")
    concentration = json.loads((EVIDENCE / "publication_trend_concentration_summary.json").read_text())
    primary = pd.read_csv(EVIDENCE / "publication_final_primary_results.csv")
    comparison = pd.read_csv(EVIDENCE / "publication_corrected_vs_frozen_comparison.csv")
    attribution = pd.read_csv(EVIDENCE / "publication_step11_trend_attribution.csv")
    universe_ablation = pd.read_csv(EVIDENCE / "publication_step11_trend_universe_ablation_comparison.csv")

    # Principal article display: matched fold-level mechanism decomposition.
    mechanism.to_csv(TABLES / "table1_trend_mechanism_decomposition.csv", index=False)

    concentration_rows = [
        ("Mean total treatment effect", concentration["mean_total_universe_treatment_effect_nav_difference"]),
        ("Mean universe/composition component", concentration["mean_shapley_universe_component_nav_difference"]),
        ("Mean parameter-selection component", concentration["mean_shapley_parameter_selection_component_nav_difference"]),
        ("Assets", concentration["asset_count"]),
        ("Top-1 absolute contribution share", concentration["top_1_absolute_share"]),
        ("Top-5 absolute contribution share", concentration["top_5_absolute_share"]),
        ("Top-10 absolute contribution share", concentration["top_10_absolute_share"]),
        ("Assets to 50% absolute contribution", concentration["asset_count_to_50pct_absolute_share"]),
        ("Assets to 80% absolute contribution", concentration["asset_count_to_80pct_absolute_share"]),
        ("Absolute-contribution HHI", concentration["absolute_contribution_hhi"]),
    ]
    pd.DataFrame(concentration_rows, columns=["measure", "value"]).to_csv(
        TABLES / "table2_trend_concentration_summary.csv", index=False
    )

    # Supporting confirmatory results.
    table3_cols = [
        "strategy_family",
        "primary_metric",
        "effect_estimate",
        "ci_lower_95",
        "ci_upper_95",
        "raw_p_value",
        "holm_p_value",
        "reject_after_holm_0_05",
        "sample_size",
        "sample_unit",
    ]
    primary.loc[:, table3_cols].to_csv(TABLES / "table3_confirmatory_results.csv", index=False)
    comparison.to_csv(TABLES / "table4_frozen_vs_publication.csv", index=False)
    attribution.to_csv(TABLES / "table5_earlier_trend_diagnostic_attribution.csv", index=False)

    # Figure 1: exact fold-level decomposition. The lead visual is the mechanism result.
    x = list(range(1, len(mechanism) + 1))
    universe = mechanism["shapley_universe_component_nav_difference"].astype(float)
    parameter = mechanism["shapley_parameter_selection_component_nav_difference"].astype(float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, universe, label="Universe/composition component")
    ax.bar(x, parameter, bottom=universe, label="Parameter-selection component")
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x, mechanism["fold_id"].tolist())
    ax.set_xlabel("Matched evaluation fold")
    ax.set_ylabel("Benchmark-relative terminal NAV difference")
    ax.set_title("Exact decomposition of retrospective-universe treatment effect")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1_trend_mechanism_decomposition.png", dpi=300)
    plt.close(fig)

    # Figure 2: same-vendor PIT versus retrospective universe levels.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, mechanism["a_pit_pitparams_nav_difference"].astype(float), marker="o", label="PIT universe + PIT parameters")
    ax.plot(x, mechanism["b_retro_retroparams_nav_difference"].astype(float), marker="o", label="Retrospective universe + retrospective parameters")
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x, mechanism["fold_id"].tolist())
    ax.set_xlabel("Matched evaluation fold")
    ax.set_ylabel("Benchmark-relative terminal NAV difference")
    ax.set_title("Point-in-time versus retrospective constituent treatment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2_trend_pit_vs_retrospective.png", dpi=300)
    plt.close(fig)

    # Figure 3: supporting confirmatory effect estimates and 95% intervals.
    daily = primary.loc[primary["strategy_family"].ne("tax_loss_selling")].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    y = list(range(len(daily)))
    estimates = daily["effect_estimate"].astype(float)
    lower = estimates - daily["ci_lower_95"].astype(float)
    upper = daily["ci_upper_95"].astype(float) - estimates
    ax.errorbar(estimates, y, xerr=[lower, upper], fmt="o", capsize=4)
    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(y, daily["strategy_family"].str.replace("_", " "))
    ax.set_xlabel("Mean benchmark-relative NAV difference across evaluation folds")
    ax.set_title("Supporting confirmatory daily-strategy estimates (95% intervals)")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_confirmatory_daily_strategies.png", dpi=300)
    plt.close(fig)

    # Preserve the earlier same-vendor ablation as supporting diagnostic context.
    fig, ax = plt.subplots(figsize=(9, 5))
    xa = list(range(1, len(universe_ablation) + 1))
    ax.plot(xa, universe_ablation["pit_net_excess_nav_difference"].astype(float), marker="o", label="Point-in-time membership")
    ax.plot(xa, universe_ablation["retrospective_net_excess_nav_difference"].astype(float), marker="o", label="Retrospective-current membership")
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(xa, universe_ablation["fold_id"].tolist())
    ax.set_xlabel("Common-period evaluation fold")
    ax.set_ylabel("Benchmark-relative NAV difference")
    ax.set_title("Earlier trend universe-construction diagnostic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4_earlier_trend_membership_ablation.png", dpi=300)
    plt.close(fig)

    print(f"Wrote tables to {TABLES}")
    print(f"Wrote figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

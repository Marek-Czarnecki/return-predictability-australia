from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data" / "evidence"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    primary = pd.read_csv(EVIDENCE / "publication_final_primary_results.csv")
    comparison = pd.read_csv(EVIDENCE / "publication_corrected_vs_frozen_comparison.csv")
    attribution = pd.read_csv(EVIDENCE / "publication_step11_trend_attribution.csv")
    universe = pd.read_csv(EVIDENCE / "publication_step11_trend_universe_ablation_comparison.csv")

    # Manuscript Table 2: four confirmatory results.
    table2_cols = [
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
    primary.loc[:, table2_cols].to_csv(TABLES / "table2_confirmatory_results.csv", index=False)

    # Manuscript Table 3: frozen versus publication comparison.
    comparison.to_csv(TABLES / "table3_frozen_vs_publication.csv", index=False)

    # Manuscript Table 4: trend diagnostic attribution.
    attribution.to_csv(TABLES / "table4_trend_attribution.csv", index=False)

    # Figure 1: confirmatory effect estimates and 95% intervals. The tax-loss
    # estimate has a different inferential unit, so it is separated visually.
    daily = primary.loc[primary["strategy_family"].ne("tax_loss_selling")].copy()
    tax = primary.loc[primary["strategy_family"].eq("tax_loss_selling")].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    y = range(len(daily))
    x = daily["effect_estimate"].astype(float)
    lower = x - daily["ci_lower_95"].astype(float)
    upper = daily["ci_upper_95"].astype(float) - x
    ax.errorbar(x, list(y), xerr=[lower, upper], fmt="o", capsize=4)
    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(list(y), daily["strategy_family"].str.replace("_", " "))
    ax.set_xlabel("Mean benchmark-relative NAV difference across evaluation folds")
    ax.set_title("Confirmatory daily-strategy estimates (95% intervals)")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1a_confirmatory_daily_strategies.png", dpi=300)
    plt.close(fig)

    if not tax.empty:
        fig, ax = plt.subplots(figsize=(8, 2.8))
        x = float(tax["effect_estimate"].iloc[0])
        lower = x - float(tax["ci_lower_95"].iloc[0])
        upper = float(tax["ci_upper_95"].iloc[0]) - x
        ax.errorbar([x], [0], xerr=[[lower], [upper]], fmt="o", capsize=4)
        ax.axvline(0.0, linewidth=1)
        ax.set_yticks([0], ["tax loss selling"])
        ax.set_xlabel("Annual mean abnormal event-minus-control return")
        ax.set_title("Confirmatory tax-loss estimate (95% interval)")
        fig.tight_layout()
        fig.savefig(FIGURES / "figure1b_confirmatory_tax_loss.png", dpi=300)
        plt.close(fig)

    # Figure 2: same-vendor common-period PIT versus retrospective-current universe.
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(1, len(universe) + 1)
    ax.plot(list(x), universe["pit_net_excess_nav_difference"].astype(float), marker="o", label="Point-in-time membership")
    ax.plot(list(x), universe["retrospective_net_excess_nav_difference"].astype(float), marker="o", label="Retrospective-current membership")
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(list(x), universe["fold_id"].tolist())
    ax.set_xlabel("Common-period evaluation fold")
    ax.set_ylabel("Benchmark-relative NAV difference")
    ax.set_title("Trend-following universe-construction diagnostic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2_trend_membership_ablation.png", dpi=300)
    plt.close(fig)

    print(f"Wrote tables to {TABLES}")
    print(f"Wrote figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

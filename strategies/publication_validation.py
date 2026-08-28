from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WALK_FORWARD_STRATEGIES = ("trend_following", "mean_reversion", "pairs_trading")
EXPECTED_FOLD_COUNT = 24
EXPECTED_DAILY_ROW_COUNT = 5926

@dataclass(frozen=True)
class PublicationValidationResult:
    strategy_metrics: pd.DataFrame
    checks: pd.DataFrame
    risk_free_coverage: pd.DataFrame
    @property
    def passed(self) -> bool:
        return bool((self.checks["status"] == "passed").all()) if not self.checks.empty else False

def validate_publication_results(results_root: Path, risk_free_path: Path | None = None) -> PublicationValidationResult:
    results_root=Path(results_root); metric_frames=[]; checks=[]
    for strategy_name in WALK_FORWARD_STRATEGIES:
        prefix=f"publication_{strategy_name}_walk_forward"; daily=pd.read_csv(results_root/f"{prefix}_daily_returns.csv",parse_dates=["trade_date"]); summary=pd.read_csv(results_root/f"{prefix}_summary.csv"); folds=pd.read_csv(results_root/f"{prefix}_folds.csv")
        checks.extend(_walk_forward_structural_checks(strategy_name=strategy_name,daily=daily,summary=summary,folds=folds)); metrics=build_walk_forward_metric_audit(strategy_name,daily,summary); metric_frames.append(metrics); checks.extend(_metric_semantic_checks(strategy_name,metrics))
    checks.extend(_validate_tax_loss(results_root)); metric_table=pd.concat(metric_frames,ignore_index=True); risk_free_coverage=build_risk_free_coverage_audit(metric_table,risk_free_path)
    if risk_free_path is not None:
        row=risk_free_coverage.iloc[0]; checks.append(_check("risk_free_full_publication_coverage",row["coverage_status"]=="complete",f"risk_free={row['risk_free_start']}..{row['risk_free_end']}; publication={row['publication_start']}..{row['publication_end']}"))
    return PublicationValidationResult(metric_table,pd.DataFrame(checks),risk_free_coverage)

def build_walk_forward_metric_audit(strategy_name,daily,summary):
    required={"fold_id","trade_date","net_return","benchmark_return","excess_return","benchmark_observed"}; missing=required.difference(daily.columns)
    if missing: raise ValueError(f"{strategy_name} daily results missing columns: {', '.join(sorted(missing))}")
    if "fold_id" not in summary.columns or "total_net_excess_return" not in summary.columns: raise ValueError(f"{strategy_name} summary is missing fold or legacy excess metric columns.")
    summary_lookup=summary.set_index("fold_id"); rows=[]
    for fold_id,group in daily.groupby("fold_id",sort=False):
        comparable=group.loc[group["benchmark_observed"].astype(bool)].copy(); strategy_total=_compound_return(comparable["net_return"]); benchmark_total=_compound_return(comparable["benchmark_return"]); sum_daily_excess=float(comparable["excess_return"].sum()); compounded_excess_stream=_compound_return(comparable["excess_return"]); legacy=float(summary_lookup.loc[fold_id,"total_net_excess_return"])
        rows.append({"strategy_name":strategy_name,"fold_id":fold_id,"evaluation_start":pd.to_datetime(comparable["trade_date"]).min(),"evaluation_end":pd.to_datetime(comparable["trade_date"]).max(),"observation_count":int(len(comparable)),"strategy_total_return":strategy_total,"benchmark_total_return":benchmark_total,"net_excess_nav_difference":strategy_total-benchmark_total,"sum_daily_excess_return":sum_daily_excess,"compounded_daily_excess_stream":compounded_excess_stream,"legacy_total_net_excess_return":legacy,"legacy_metric_matches_sum_daily_excess":bool(np.isclose(legacy,sum_daily_excess,atol=1e-12,rtol=0.0))})
    return pd.DataFrame(rows)

def build_risk_free_coverage_audit(metric_table,risk_free_path):
    publication_start=pd.to_datetime(metric_table["evaluation_start"]).min(); publication_end=pd.to_datetime(metric_table["evaluation_end"]).max()
    if risk_free_path is None: return pd.DataFrame([{"risk_free_path":None,"risk_free_start":pd.NaT,"risk_free_end":pd.NaT,"publication_start":publication_start,"publication_end":publication_end,"coverage_status":"not_supplied"}])
    frame=pd.read_csv(risk_free_path,parse_dates=["trade_date"])
    if "risk_free_return" not in frame.columns: raise ValueError("Risk-free input must contain risk_free_return.")
    valid=frame.loc[frame["risk_free_return"].notna()].copy(); start=pd.NaT if valid.empty else valid["trade_date"].min(); end=pd.NaT if valid.empty else valid["trade_date"].max(); status="empty" if valid.empty else ("complete" if start<=publication_start and end>=publication_end else "incomplete")
    return pd.DataFrame([{"risk_free_path":str(risk_free_path),"risk_free_start":start,"risk_free_end":end,"publication_start":publication_start,"publication_end":publication_end,"coverage_status":status}])

def _walk_forward_structural_checks(*,strategy_name,daily,summary,folds):
    return [_check(f"{strategy_name}_fold_count",len(folds)==EXPECTED_FOLD_COUNT,f"observed={len(folds)} expected={EXPECTED_FOLD_COUNT}"),_check(f"{strategy_name}_summary_fold_count",len(summary)==EXPECTED_FOLD_COUNT,f"observed={len(summary)} expected={EXPECTED_FOLD_COUNT}"),_check(f"{strategy_name}_daily_row_count",len(daily)==EXPECTED_DAILY_ROW_COUNT,f"observed={len(daily)} expected={EXPECTED_DAILY_ROW_COUNT}"),_check(f"{strategy_name}_daily_null_returns",not daily[["net_return","benchmark_return","excess_return"]].isna().any().any(),"net_return, benchmark_return and excess_return must be complete"),_check(f"{strategy_name}_benchmark_observed",daily["benchmark_observed"].astype(bool).all(),f"observed={int(daily['benchmark_observed'].astype(bool).sum())}/{len(daily)}")]

def _metric_semantic_checks(strategy_name,metrics):
    return [_check(f"{strategy_name}_legacy_excess_metric_identified",bool(metrics["legacy_metric_matches_sum_daily_excess"].all()),"Existing total_net_excess_return is confirmed to be the arithmetic sum of daily excess returns, not a compounded relative return."),_check(f"{strategy_name}_metric_audit_fold_count",len(metrics)==EXPECTED_FOLD_COUNT,f"observed={len(metrics)} expected={EXPECTED_FOLD_COUNT}")]

def _validate_tax_loss(results_root):
    prefix="publication_tax_loss_selling"; events=pd.read_csv(results_root/f"{prefix}_event_study.csv"); summary=pd.read_csv(results_root/f"{prefix}_summary.csv"); robustness=pd.read_csv(results_root/f"{prefix}_year_robustness.csv"); row=summary.iloc[0]; complete=int(events["net_return_difference"].notna().sum())
    return [_check("tax_loss_event_count",len(events)==int(row["event_observation_count"]),f"events={len(events)} summary={int(row['event_observation_count'])}"),_check("tax_loss_complete_matched_count",complete==int(row["complete_matched_observation_count"]),f"complete={complete} summary={int(row['complete_matched_observation_count'])}"),_check("tax_loss_year_count",int(events.loc[events["net_return_difference"].notna(),"year"].nunique())==int(row["year_count"]),f"summary_years={int(row['year_count'])}"),_check("tax_loss_year_robustness_structure",(robustness["analysis_level"]=="year_summary").sum()==int(row["year_count"]) and (robustness["analysis_level"]=="leave_one_year_out").sum()==int(row["year_count"]) and (robustness["analysis_level"]=="year_clustered_sign_flip").sum()==1,"year summary, leave-one-year-out and clustered rows must reconcile")]

def _compound_return(series):
    values=pd.to_numeric(series,errors="coerce")
    if values.isna().any(): raise ValueError("Cannot compound a return stream containing missing observations.")
    return float((1.0+values).prod()-1.0)
def _check(name,passed,detail): return {"check":name,"status":"passed" if passed else "failed","detail":detail}

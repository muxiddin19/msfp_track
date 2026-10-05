"""
Statistical Validation for MSFP-Track Results

Computes p-values, confidence intervals, and effect sizes for comparing
MSFP-Track against baselines.

Usage:
    python -m litepp.utils.statistical_tests \
        --lite-runs "61.1,60.9,61.3" \
        --msfp-runs "63.2,63.0,63.4" \
        --metric HOTA
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict
import argparse


def compute_paired_ttest(
    baseline_runs: List[float],
    method_runs: List[float],
) -> Tuple[float, float]:
    """
    Compute paired t-test between baseline and method.

    Args:
        baseline_runs: List of metric values for baseline (e.g., LITE)
        method_runs: List of metric values for proposed method

    Returns:
        t_statistic: The t-statistic
        p_value: Two-tailed p-value
    """
    assert len(baseline_runs) == len(method_runs), "Must have equal number of runs"

    t_stat, p_value = stats.ttest_rel(method_runs, baseline_runs)
    return t_stat, p_value


def compute_cohens_d(
    baseline_runs: List[float],
    method_runs: List[float],
) -> float:
    """
    Compute Cohen's d effect size.

    Effect size interpretation:
        < 0.2: negligible
        0.2-0.5: small
        0.5-0.8: medium
        > 0.8: large

    Args:
        baseline_runs: Baseline metric values
        method_runs: Proposed method metric values

    Returns:
        cohens_d: Effect size
    """
    baseline_mean = np.mean(baseline_runs)
    method_mean = np.mean(method_runs)

    # Pooled standard deviation
    baseline_std = np.std(baseline_runs, ddof=1)
    method_std = np.std(method_runs, ddof=1)
    pooled_std = np.sqrt((baseline_std**2 + method_std**2) / 2)

    cohens_d = (method_mean - baseline_mean) / pooled_std
    return cohens_d


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for the mean.

    Args:
        values: List of observed values
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        mean: Sample mean
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
    """
    n = len(values)
    mean = np.mean(values)
    std_err = stats.sem(values)  # Standard error of the mean

    # t-distribution critical value for given confidence level
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)

    margin_of_error = t_critical * std_err
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error

    return mean, ci_lower, ci_upper


def format_result_with_ci(
    values: List[float],
    confidence: float = 0.95,
    decimal_places: int = 1,
) -> str:
    """
    Format result with confidence interval for LaTeX table.

    Args:
        values: List of metric values
        confidence: Confidence level
        decimal_places: Number of decimal places

    Returns:
        Formatted string like "63.2±0.3 (CI: [62.9, 63.5])"
    """
    mean, ci_lower, ci_upper = compute_confidence_interval(values, confidence)
    std = np.std(values, ddof=1)

    fmt = f"{{:.{decimal_places}f}}"

    # For LaTeX subscript notation
    result = (
        f"{fmt.format(mean)}$_{{\\pm{fmt.format(std)}}}$ "
        f"(CI: [{fmt.format(ci_lower)}, {fmt.format(ci_upper)}])"
    )

    return result


def interpret_effect_size(cohens_d: float) -> str:
    """Get verbal interpretation of Cohen's d."""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def full_comparison_report(
    baseline_runs: List[float],
    method_runs: List[float],
    baseline_name: str = "LITE",
    method_name: str = "MSFP-Track",
    metric_name: str = "HOTA",
) -> Dict[str, any]:
    """
    Generate complete statistical comparison report.

    Args:
        baseline_runs: Baseline metric values
        method_runs: Proposed method metric values
        baseline_name: Name of baseline method
        method_name: Name of proposed method
        metric_name: Name of the metric being compared

    Returns:
        Dictionary with all statistical measures
    """
    # Basic statistics
    baseline_mean = np.mean(baseline_runs)
    method_mean = np.mean(method_runs)
    improvement = method_mean - baseline_mean

    # Confidence intervals
    _, bl_ci_lower, bl_ci_upper = compute_confidence_interval(baseline_runs)
    _, mt_ci_lower, mt_ci_upper = compute_confidence_interval(method_runs)

    # Statistical tests
    t_stat, p_value = compute_paired_ttest(baseline_runs, method_runs)
    cohens_d = compute_cohens_d(baseline_runs, method_runs)

    # Compile report
    report = {
        "baseline_name": baseline_name,
        "method_name": method_name,
        "metric_name": metric_name,
        "baseline_mean": baseline_mean,
        "baseline_std": np.std(baseline_runs, ddof=1),
        "baseline_ci": (bl_ci_lower, bl_ci_upper),
        "method_mean": method_mean,
        "method_std": np.std(method_runs, ddof=1),
        "method_ci": (mt_ci_lower, mt_ci_upper),
        "improvement": improvement,
        "improvement_pct": (improvement / baseline_mean) * 100,
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "effect_size_interpretation": interpret_effect_size(cohens_d),
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
    }

    return report


def print_report(report: Dict) -> None:
    """Print formatted statistical report."""
    print("\n" + "="*60)
    print(f"Statistical Comparison: {report['baseline_name']} vs {report['method_name']}")
    print(f"Metric: {report['metric_name']}")
    print("="*60)

    print(f"\n{report['baseline_name']}:")
    print(f"  Mean: {report['baseline_mean']:.2f}")
    print(f"  Std:  {report['baseline_std']:.2f}")
    print(f"  95% CI: [{report['baseline_ci'][0]:.2f}, {report['baseline_ci'][1]:.2f}]")

    print(f"\n{report['method_name']}:")
    print(f"  Mean: {report['method_mean']:.2f}")
    print(f"  Std:  {report['method_std']:.2f}")
    print(f"  95% CI: [{report['method_ci'][0]:.2f}, {report['method_ci'][1]:.2f}]")

    print(f"\nImprovement:")
    print(f"  Absolute: +{report['improvement']:.2f} {report['metric_name']}")
    print(f"  Relative: +{report['improvement_pct']:.1f}%")

    print(f"\nStatistical Tests:")
    print(f"  t-statistic: {report['t_statistic']:.3f}")
    print(f"  p-value: {report['p_value']:.4f} {'***' if report['p_value'] < 0.001 else '**' if report['p_value'] < 0.01 else '*' if report['p_value'] < 0.05 else ''}")
    print(f"  Cohen's d: {report['cohens_d']:.2f} ({report['effect_size_interpretation']} effect)")

    print(f"\nSignificance:")
    print(f"  p < 0.05: {'YES ✓' if report['significant_at_05'] else 'NO ✗'}")
    print(f"  p < 0.01: {'YES ✓' if report['significant_at_01'] else 'NO ✗'}")

    print("\n" + "="*60)

    # LaTeX table row
    print(f"\nLaTeX Table Row:")
    print(f"{report['method_name']} & "
          f"{report['method_mean']:.1f}$_{{\\pm{report['method_std']:.2f}}}$ "
          f"(p={report['p_value']:.3f}) & ... \\\\")


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistical tests for MOT results"
    )
    parser.add_argument(
        "--lite-runs",
        type=str,
        default="61.1,60.9,61.3",
        help="Comma-separated LITE HOTA values from 3 runs"
    )
    parser.add_argument(
        "--msfp-runs",
        type=str,
        default="63.2,63.0,63.4",
        help="Comma-separated MSFP-Track HOTA values from 3 runs"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="HOTA",
        help="Name of the metric (for display)"
    )

    args = parser.parse_args()

    # Parse runs
    lite_runs = [float(x.strip()) for x in args.lite_runs.split(",")]
    msfp_runs = [float(x.strip()) for x in args.msfp_runs.split(",")]

    # Generate report
    report = full_comparison_report(
        baseline_runs=lite_runs,
        method_runs=msfp_runs,
        baseline_name="LITE",
        method_name="MSFP-Track",
        metric_name=args.metric,
    )

    # Print report
    print_report(report)

    # Additional examples for other metrics
    print("\n\n" + "="*60)
    print("Example Usage for Other Metrics:")
    print("="*60)

    # Example: AssA
    print("\nAssA (Association Accuracy):")
    assa_lite = [60.8, 60.6, 61.0]
    assa_msfp = [63.0, 62.8, 63.2]
    report_assa = full_comparison_report(
        assa_lite, assa_msfp,
        metric_name="AssA"
    )
    print(f"  p-value: {report_assa['p_value']:.4f}")
    print(f"  Cohen's d: {report_assa['cohens_d']:.2f}")

    # Example: IDSW (lower is better, so reverse comparison)
    print("\nIDSW (Identity Switches, lower is better):")
    idsw_lite = [1876, 1892, 1860]
    idsw_msfp = [1512, 1498, 1526]
    # For IDSW, we reverse the comparison
    t_stat, p_val = compute_paired_ttest(idsw_msfp, idsw_lite)
    print(f"  Reduction: {np.mean(idsw_lite) - np.mean(idsw_msfp):.0f} switches")
    print(f"  p-value: {p_val:.4f}")


if __name__ == "__main__":
    main()

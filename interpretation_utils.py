"""
interpretation_utils.py: Generate plain-language summaries of EDA results for non-technical users.
"""

def interpret_eda_results(eda_results: dict) -> str:
    """
    Given EDA results as a dict, return a user-friendly summary string.
    """
    summary = []
    stats = eda_results.get("summary_stats")
    missing = eda_results.get("missing_report")
    corr = eda_results.get("correlation_matrix")
    profile = eda_results.get("profile")

    if stats is not None:
        summary.append("Your data contains {} rows and {} columns. Here are some key statistics:".format(
            stats.shape[0], stats.shape[1]))
        summary.append("- The average values for each column are: {}".format(
            ", ".join(f"{col}: {stats[col].mean():.2f}" for col in stats.columns if stats[col].dtype != 'O')))
    if missing is not None:
        missing_total = missing.isnull().sum().sum() if hasattr(missing, 'isnull') else 0
        if missing_total > 0:
            summary.append(f"- There are {missing_total} missing values in your data. Consider cleaning these.")
        else:
            summary.append("- There are no missing values in your data.")
    if corr is not None:
        high_corr = [(a, b, corr[a][b]) for a in corr.columns for b in corr.columns if a != b and abs(corr[a][b]) > 0.7]
        if high_corr:
            summary.append("- Some columns are strongly related to each other (correlation > 0.7): " + ", ".join(f"{a} & {b}" for a, b, _ in high_corr))
        else:
            summary.append("- No strong relationships found between columns.")
    if profile is not None:
        summary.append("- The data profile suggests: {}".format(str(profile)[:120] + "..."))
    if not summary:
        return "No summary available."
    return "\n".join(summary)

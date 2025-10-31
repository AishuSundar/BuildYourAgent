"""EDA utility functions used by the Data Science Assistant demo.

These are small helpers that produce textual reports and save plots to temporary files
so a front-end (Streamlit) can display them.
"""
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
import matplotlib
# Use a non-interactive backend to avoid Tkinter / GUI calls when running under Flask
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import seaborn as sns
import tempfile
import os
import uuid


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    return df.describe(include='all').transpose()


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table of missing counts and percentages."""
    total = len(df)
    miss = df.isna().sum()
    pct = (miss / total * 100).round(2)
    return pd.DataFrame({'missing_count': miss, 'missing_pct': pct})


def correlation_matrix(df: pd.DataFrame, numeric_only: bool = True) -> pd.DataFrame:
    if numeric_only:
        dfc = df.select_dtypes(include=[np.number])
    else:
        dfc = df.copy()
    return dfc.corr()


def dataset_profile(df: pd.DataFrame, sample_values: int = 5) -> Dict[str, Any]:
    """Create a compact profile of the dataset to help produce actionable guidance.

    Returns a dict containing counts, dtypes, missing summary, cardinality, skewness,
    and simple outlier estimates per numeric column.
    """
    profile: Dict[str, Any] = {}
    profile['num_rows'] = len(df)
    profile['num_columns'] = len(df.columns)
    profile['columns'] = {}

    for col in df.columns:
        series = df[col]
        col_info: Dict[str, Any] = {}
        col_info['dtype'] = str(series.dtype)
        col_info['missing_count'] = int(series.isna().sum())
        col_info['missing_pct'] = float((series.isna().sum() / max(1, len(df))) * 100)
        try:
            col_info['unique'] = int(series.nunique(dropna=True))
        except Exception:
            col_info['unique'] = None
        # sample values
        col_info['sample_values'] = series.dropna().unique()[:sample_values].tolist()

        # numeric-specific metrics
        if pd.api.types.is_numeric_dtype(series):
            ser = series.dropna().astype(float)
            if len(ser) > 0:
                col_info['mean'] = float(ser.mean())
                col_info['std'] = float(ser.std())
                col_info['skew'] = float(ser.skew())
                # simple outlier estimate using IQR
                q1 = ser.quantile(0.25)
                q3 = ser.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_frac = float(((ser < lower) | (ser > upper)).mean())
                else:
                    outlier_frac = 0.0
                col_info['outlier_fraction'] = outlier_frac
            else:
                col_info['mean'] = None
                col_info['std'] = None
                col_info['skew'] = None
                col_info['outlier_fraction'] = 0.0

        profile['columns'][col] = col_info

    # quick missing overview
    miss = df.isna().sum()
    miss_pct = (miss / max(1, len(df)) * 100).round(2)
    profile['missing_summary'] = pd.DataFrame({'missing_count': miss, 'missing_pct': miss_pct})

    # correlation head for numeric
    profile['correlation'] = correlation_matrix(df)

    return profile


def _save_figure(fig) -> str:
    """
    Save a Matplotlib figure into the project's `static/plots` directory and
    return a path suitable for serving from Flask (relative path under static).
    Falls back to a temp file if the static directory is not writable.
    """
    plots_dir = os.path.join(os.getcwd(), 'static', 'plots')
    try:
        os.makedirs(plots_dir, exist_ok=True)
        fname = f'{uuid.uuid4().hex}.png'
        path = os.path.join(plots_dir, fname)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        # Return path relative to Flask static folder
        return os.path.join('static', 'plots', fname)
    except Exception:
        # Fallback to temp file if static isn't writable
        fd, path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        return path


def plot_histograms(df: pd.DataFrame, columns: list = None, bins: int = 30) -> Dict[str, str]:
    paths = {}
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax)
        ax.set_title(f'Histogram - {col}')
        paths[col] = _save_figure(fig)
    return paths


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    corr = correlation_matrix(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', ax=ax)
    ax.set_title('Correlation heatmap')
    return _save_figure(fig)


def plot_pairplot(df: pd.DataFrame, vars: list = None, sample: int = 500) -> str:
    # Pairplot can be heavy; sample for large datasets
    dfp = df.select_dtypes(include=[np.number]).copy()
    if vars:
        dfp = dfp[vars]
    if len(dfp) > sample:
        dfp = dfp.sample(sample, random_state=42)
    g = sns.pairplot(dfp)
    # Seaborn pairplot returns a PairGrid; save via its fig attribute
    return _save_figure(g.fig)


def cleanup_plots(plots_dir: str = None, older_than_seconds: int = 300) -> int:
    """Delete files in plots_dir older than older_than_seconds.

    Returns the number of files removed.
    """
    if plots_dir is None:
        plots_dir = os.path.join(os.getcwd(), 'static', 'plots')
    removed = 0
    try:
        if not os.path.isdir(plots_dir):
            return 0
        now = os.path.getmtime
        cutoff = __import__('time').time() - older_than_seconds
        for fname in os.listdir(plots_dir):
            fpath = os.path.join(plots_dir, fname)
            try:
                if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
                    removed += 1
            except Exception:
                # ignore transient file access errors
                continue
    except Exception:
        return removed
    return removed


def run_eda(df: pd.DataFrame) -> dict:
    """Run basic EDA and return results as a dict."""
    return {
        "summary_stats": summary_stats(df),
        "missing_report": missing_report(df),
        "correlation_matrix": correlation_matrix(df),
        "profile": dataset_profile(df)
    }


def generate_plots(df: pd.DataFrame) -> dict:
    """Generate main plots and return their paths."""
    histograms = plot_histograms(df)
    heatmap = plot_correlation_heatmap(df)
    pairplot = None
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        pairplot = plot_pairplot(df, vars=numeric_cols[:6])
    return {
        "histograms": histograms,
        "heatmap": heatmap,
        "pairplot": pairplot
    }

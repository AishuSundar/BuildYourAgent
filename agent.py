"""Agent wrapper that uses LangChain + Azure OpenAI to coordinate EDA tools.

This module provides a simple DataScienceAgent that runs a small EDA pipeline
and asks the LLM to produce recommendations. It is intentionally lightweight
so you can extend it with more tools, LangGraph planning, or a full agent framework.
"""
import os
import requests
import importlib
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from interpretation_utils import interpret_eda_results
from dotenv import load_dotenv

from eda_utils import (
    load_csv,
    summary_stats,
    missing_report,
    correlation_matrix,
    plot_histograms,
    plot_correlation_heatmap,
    plot_pairplot,
    dataset_profile,
    run_eda,
    generate_plots,
)

load_dotenv()

# Try to import LangChain Azure chat model; fall back gracefully if not available
AzureChatOpenAI = None
HumanMessage = None
try:
    # common location in many langchain versions
    from langchain.chat_models import AzureChatOpenAI
except Exception:
    # Some distributions don't expose the alternate module path; keep None and use stub LLM instead
    AzureChatOpenAI = None
# Try multiple locations for HumanMessage type using dynamic import to avoid static analyzer errors
HumanMessage = None
for _mod in ('langchain.schema', 'langchain_core.messages'):
    try:
        m = importlib.import_module(_mod)
        HumanMessage = getattr(m, 'HumanMessage', None)
        if HumanMessage is not None:
            break
    except Exception:
        continue
        HumanMessage = None

try:
    import langgraph
    HAS_LANGGRAPH = True
except Exception:
    HAS_LANGGRAPH = False


# A small stub LLM used when a proper AzureChatOpenAI isn't available.
class _StubLLM:
    def __init__(self, prefix: str = '(stub)'):
        self.prefix = prefix

    def generate(self, prompts):
        # mimic LangChain's generate returning a simple string-like result
        text = (
            "Dataset looks good for demo. Recommendations: 1) handle missing values;"
            " 2) consider scaling numeric features; 3) check class balance for target."
        )
        return text

    def __call__(self, messages=None, **kwargs):
        # mimic calling the chat model with messages and returning an object
        class _Resp:
            def __init__(self, content):
                self.content = content

        return _Resp(self.generate(messages))


class DataScienceAgent:
    def __init__(self, df: pd.DataFrame, temperature: float = 0.0):
        self.df = df
        self.temperature = temperature
        self.eda_results = None
        self.plots = []
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

    def analyze(self):
        self.eda_results = run_eda(self.df)
        self.plots = generate_plots(self.df)
        return self.eda_results, self.plots

    def get_recommendations(self):
        if not (self.azure_endpoint and self.azure_deployment and self.azure_api_key):
            return ["Azure OpenAI not configured. Set environment variables for recommendations."]
        prompt = (
            "Given the following EDA results: "
            f"{str(self.eda_results)}. Suggest prioritized next steps for a data scientist."
        )
        response = self._call_azure_openai(prompt)
        return response.split('\n')

    def _call_azure_openai(self, prompt: str) -> str:
        url = f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}/chat/completions?api-version=2023-05-15"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_api_key,
        }
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful data science assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": 512,
        }
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"]

    def _llm_recommendations(self, context: str) -> str:
        # Build a short prompt and call the chat LLM
        if HumanMessage is None:
            # Fallback simple call if schema unavailable
            resp = self.llm.generate([context])
            return str(resp)
        msgs = [HumanMessage(content=context)]
        resp = self.llm(messages=msgs)
        return resp.content

    def _rule_based_recommendations(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create simple, actionable recommendations from dataset profile.

        Returns a list of dicts with keys: title, detail, priority, type
        """
        recs: List[Dict[str, Any]] = []
        cols = profile['columns']

        # Missing values
        high_missing = []
        for c, info in cols.items():
            if info.get('missing_pct', 0) >= 30:
                high_missing.append(c)
        if high_missing:
            recs.append({
                'title': 'High missingness',
                'detail': f"Columns with >=30% missing: {', '.join(high_missing)}. Consider dropping or imputing (median/mode).",
                'priority': 'High',
                'type': 'data-cleaning',
                'action': {
                    'type': 'impute_or_drop',
                    'columns': high_missing
                }
            })

        # Outliers and skew
        skew_cols = []
        outlier_cols = []
        for c, info in cols.items():
            if info.get('skew') is not None and abs(info['skew']) > 1.0:
                skew_cols.append((c, round(info['skew'], 2)))
            if info.get('outlier_fraction', 0) > 0.05:
                outlier_cols.append((c, round(info['outlier_fraction'], 3)))

        if skew_cols:
            names = ', '.join([f"{c} (skew={s})" for c, s in skew_cols])
            recs.append({
                'title': 'Skewed numeric columns',
                'detail': f"Consider log or Box-Cox transforms for: {names} to improve model assumptions.",
                'priority': 'Medium',
                'type': 'feature-engineering'
                , 'action': {
                    'type': 'log_transform',
                    'columns': [c for c, _ in skew_cols]
                }
            })

        if outlier_cols:
            names = ', '.join([f"{c} (frac={f})" for c, f in outlier_cols])
            recs.append({
                'title': 'Outliers detected',
                'detail': f"Columns with >5% outliers: {names}. Consider robust scaling, clipping, or imputation.",
                'priority': 'Medium',
                'type': 'data-cleaning'
                , 'action': {
                    'type': 'clip_outliers',
                    'columns': [c for c, _ in outlier_cols]
                }
            })

        # High cardinality categorical
        high_card = []
        for c, info in cols.items():
            if info.get('dtype', '').startswith('object') and info.get('unique') and info['unique'] > 50:
                high_card.append(c)
        if high_card:
            recs.append({
                'title': 'High-cardinality categorical features',
                'detail': f"Consider target encoding or hashing for: {', '.join(high_card)}.",
                'priority': 'Medium',
                'type': 'feature-engineering'
                , 'action': {
                    'type': 'hash_encode',
                    'columns': high_card
                }
            })

        # Multicollinearity
        try:
            corr = profile.get('correlation')
            if corr is not None and not corr.empty:
                strong_pairs = []
                # find pairs with abs(corr) > 0.85
                numeric_cols = corr.columns.tolist()
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        a = numeric_cols[i]
                        b = numeric_cols[j]
                        val = corr.iloc[i, j]
                        if abs(val) > 0.85:
                            strong_pairs.append((a, b, round(val, 2)))
                if strong_pairs:
                    pairs_str = ', '.join([f"{a}-{b}({v})" for a, b, v in strong_pairs[:10]])
                    recs.append({
                        'title': 'High correlation between numeric features',
                        'detail': f"Consider removing or combining: {pairs_str}.",
                        'priority': 'Medium',
                        'type': 'feature-engineering'
                    })
        except Exception:
            pass

        # Basic modeling advice: look for likely target column
        target_candidates = [c for c, info in cols.items() if c.lower() in ('target', 'label', 'survived', 'y')]
        if not target_candidates:
            # heuristic: column with name 'quality' or small integer unique values
            for c, info in cols.items():
                if c.lower() in ('quality', 'class'):
                    target_candidates.append(c)
        if target_candidates:
            recs.append({
                'title': 'Target column detected',
                'detail': f"Detected possible target column(s): {', '.join(target_candidates)}. Check class balance and choose appropriate metric.",
                'priority': 'High',
                'type': 'modeling'
            })

        if not recs:
            recs.append({
                'title': 'No major issues detected',
                'detail': 'Dataset looks reasonably clean. Consider baseline models (logistic regression / random forest) and quick feature importance analysis.',
                'priority': 'Low',
                'type': 'note'
            })

        return recs

    def apply_action(self, action: Dict[str, Any], csv_path: str) -> str:
        """Apply a simple transformation to the CSV at csv_path and return the path to a new CSV.

        Supported actions: impute_or_drop, log_transform, clip_outliers, hash_encode
        Returns path to transformed CSV file.
        """
        df = load_csv(csv_path)
        a_type = action.get('type')
        cols = action.get('columns', []) or []

        # numeric/categorical split
        numeric = [c for c in cols if c in df.select_dtypes(include=['number']).columns]
        categorical = [c for c in cols if c in df.select_dtypes(include=['object', 'category']).columns]

        if a_type == 'impute_or_drop':
            # for numeric -> median, for categorical -> mode; if >70% missing drop
            for c in cols:
                pct = (df[c].isna().sum() / max(1, len(df))) * 100
                if pct > 70:
                    df = df.drop(columns=[c])
                else:
                    if c in numeric:
                        df[c] = df[c].fillna(df[c].median())
                    else:
                        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else '')

        elif a_type == 'log_transform':
            for c in numeric:
                # avoid negatives; shift if needed
                ser = df[c].copy()
                minv = ser.min()
                if pd.isna(minv):
                    continue
                shift = 0
                if minv <= 0:
                    shift = abs(minv) + 1
                df[c] = np.log1p(ser + shift)

        elif a_type == 'clip_outliers':
            for c in numeric:
                ser = df[c].dropna()
                if len(ser) == 0:
                    continue
                q1 = ser.quantile(0.25)
                q3 = ser.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                df[c] = df[c].clip(lower, upper)

        elif a_type == 'hash_encode':
            for c in categorical:
                df[c] = df[c].astype(str).apply(lambda x: str(hash(x) & 0xffffffff))

        else:
            raise ValueError(f'Unsupported action type: {a_type}')

        # Save transformed CSV
        import uuid
        outdir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, f'transformed_{uuid.uuid4().hex}.csv')
        df.to_csv(outpath, index=False)
        return outpath

    def run_eda(self, csv_path: str, pairplot_vars: List[str] = None) -> Dict[str, Any]:
        df = load_csv(csv_path)

        # Basic reports
        stats = summary_stats(df)
        missing = missing_report(df)
        corr = correlation_matrix(df)

        # Plots
        hist_paths = plot_histograms(df)
        heatmap_path = plot_correlation_heatmap(df)

        pairplot_path = None
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if pairplot_vars is None:
            # choose up to 6 numeric columns for pairplot
            take = numeric_cols[:6]
        else:
            take = [c for c in pairplot_vars if c in df.columns]
        if take:
            pairplot_path = plot_pairplot(df, vars=take)

        # Build a dataset profile for human-readable summaries and rule-based recommendations
        profile = dataset_profile(df)

        # Rule-based recommendations (always available) — actionable and prioritized
        actionable = self._rule_based_recommendations(profile)

        # Build a concise context for the LLM (if available)
        # Include a subset of the stats and top missing columns
        top_missing = missing.sort_values('missing_pct', ascending=False).head(10)
        ctx = (
            f"You are a data scientist assistant. We ran automated EDA on dataset '{os.path.basename(csv_path)}'.\n"
            f"Summary: rows={profile['num_rows']}, cols={profile['num_columns']}.\n"
            f"Top missing columns: {top_missing['missing_pct'].to_dict()}\n"
            f"Top rule-based recommendations: { [r['title'] for r in actionable][:6] }\n"
            "Given the profile above, provide a compact, prioritized list of recommended next steps for modeling, feature engineering, and data cleaning."
        )

        # Optionally use LangGraph to create a simple plan (if available)
        plan_text = None
        if HAS_LANGGRAPH:
            try:
                g = langgraph.Graph()
                g.add_node('load_data')
                g.add_node('summarize')
                g.add_node('plot')
                g.add_edge('load_data', 'summarize')
                g.add_edge('summarize', 'plot')
                plan_text = ' > '.join(n.name for n in g.nodes)
            except Exception as e:
                plan_text = f'(langgraph plan failed: {e})'

        # Get LLM recommendations if a real LLM is available; otherwise use stub's output
        recommendations = None
        try:
            recommendations = self._llm_recommendations(ctx)
        except Exception:
            # If LLM call fails, fall back to a concise textual summary of rule-based recs
            recommendations = '\n'.join([f"- {r['title']}: {r['detail']}" for r in actionable])

        result = {
            'stats': stats,
            'profile': profile,
            'missing': missing,
            'correlation': corr,
            'histograms': hist_paths,
            'heatmap': heatmap_path,
            'pairplot': pairplot_path,
            'recommendations': recommendations,
            'actionable': actionable,
            'langgraph_plan': plan_text,
            'interpretation': interpret_eda_results({
                "summary_stats": stats,
                "missing_report": missing,
                "correlation_matrix": corr,
                "profile": profile
            }),
        }

        # Run an agentic pipeline that produces a step-by-step plan and trace
        try:
            agentic_plan, agentic_trace = self.run_agentic_pipeline(df, csv_path, profile, out={
                'histograms': hist_paths,
                'heatmap': heatmap_path,
                'pairplot': pairplot_path,
            })
            result['agentic_plan'] = agentic_plan
            result['agentic_trace'] = agentic_trace
        except Exception as e:
            result['agentic_plan'] = None
            result['agentic_trace'] = [{'step': 'agentic_pipeline_failed', 'analysis': str(e)}]

        return result

    def run_agentic_pipeline(self, df: pd.DataFrame, csv_path: str, profile: Dict[str, Any], out: Dict[str, Any]):
        """Create and execute a small agentic plan. Each step produces an analysis string and optional artifacts.

        The planner is simple: it uses rule-based logic to pick important steps and optionally asks the LLM to
        expand on each step. Returns (plan, trace) where plan is a list of step names and trace is a list of
        {step, analysis, artifacts} entries.
        """
        plan = [
            'Profile dataset',
            'Missing values analysis',
            'Outlier & skew analysis',
            'Correlation & multicollinearity',
            'Feature engineering suggestions',
            'Modeling recommendations',
        ]

        trace = []

        # 1) Profile dataset
        ptext = f"Rows: {profile['num_rows']}, Columns: {profile['num_columns']}."
        ptext += f" Top columns by missing%: {', '.join(profile['missing_summary'].sort_values('missing_pct', ascending=False).head(3).index.tolist())}."
        trace.append({'step': plan[0], 'analysis': ptext, 'artifacts': {}})

        # 2) Missing values analysis
        miss_top = profile['missing_summary'].sort_values('missing_pct', ascending=False).head(10)
        miss_list = [f"{c}({pct}%)" for c, pct in zip(miss_top.index.tolist(), miss_top['missing_pct'].tolist()) if pct > 0]
        miss_text = 'No significant missing values.' if not miss_list else ('High missing columns: ' + ', '.join(miss_list))
        trace.append({'step': plan[1], 'analysis': miss_text, 'artifacts': {}})

        # 3) Outlier & skew analysis
        skew = []
        outliers = []
        for c, info in profile['columns'].items():
            if info.get('skew') is not None and abs(info['skew']) > 1.0:
                skew.append(f"{c}(skew={round(info['skew'],2)})")
            if info.get('outlier_fraction', 0) > 0.05:
                outliers.append(f"{c}(outlier_frac={round(info['outlier_fraction'],3)})")
        skew_text = 'No highly skewed numeric columns.' if not skew else ('Skewed columns: ' + ', '.join(skew))
        out_text = 'No major outlier issues.' if not outliers else ('Columns with notable outliers: ' + ', '.join(outliers))
        trace.append({'step': plan[2], 'analysis': f"{skew_text} {out_text}", 'artifacts': {}})

        # 4) Correlation & multicollinearity
        corr = profile.get('correlation')
        corr_text = 'Correlation analysis not available.'
        corr_artifacts = {}
        if corr is not None and not corr.empty:
            # find strong pairs
            strong = []
            cols = corr.columns.tolist()
            for i in range(len(cols)):
                for j in range(i+1, len(cols)):
                    v = corr.iloc[i, j]
                    if abs(v) > 0.85:
                        strong.append((cols[i], cols[j], round(v,2)))
            if strong:
                corr_text = 'Strongly correlated pairs: ' + ', '.join([f"{a}-{b}({v})" for a,b,v in strong[:10]])
            else:
                corr_text = 'No strong correlations detected.'
            # attach heatmap artifact if available
            if out.get('heatmap'):
                corr_artifacts['heatmap'] = out.get('heatmap')
        trace.append({'step': plan[3], 'analysis': corr_text, 'artifacts': corr_artifacts})

        # 5) Feature engineering suggestions (reuse actionable list)
        fe_suggestions = self._rule_based_recommendations(profile)
        fe_text = '\n'.join([f"{r['priority']}: {r['title']} - {r['detail']}" for r in fe_suggestions])
        # include sample histograms as artifacts
        fe_artifacts = {}
        # attach up to 4 histograms
        hist = out.get('histograms', {})
        if isinstance(hist, dict):
            keys = list(hist.keys())[:4]
            fe_artifacts['histograms'] = {k: hist[k] for k in keys}
        trace.append({'step': plan[4], 'analysis': fe_text, 'artifacts': fe_artifacts})

        # 6) Modeling recommendations
        model_text = 'Start with a baseline model: logistic regression for classification or linear regression for numeric targets. Use cross-validation and measure AUC/F1 for imbalanced tasks.'
        trace.append({'step': plan[5], 'analysis': model_text, 'artifacts': {}})

        # Optionally refine each step using the LLM (if available)
        if not getattr(self, '_using_stub', True):
            refined = []
            for entry in trace:
                try:
                    prompt = f"You are an expert data scientist. Given the dataset summary and previous findings, expand on this step:\nStep: {entry['step']}\nFindings: {entry['analysis']}\nProvide 2-4 concrete actions and expected impact."
                    text = self._llm_recommendations(prompt)
                    entry['analysis_refined'] = text
                except Exception:
                    entry['analysis_refined'] = None
                refined.append(entry)
            trace = refined

        return plan, trace


if __name__ == '__main__':
    # Quick sanity check if run directly (requires env vars)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='CSV file to analyze')
    args = parser.parse_args()
    agent = DataScienceAgent()
    out = agent.run_eda(args.csv)
    print('Recommendations:\n', out['recommendations'])

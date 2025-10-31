Data Science Assistant — Demo (Flask)

This repository is a lightweight, demoable Data Science Assistant that performs automated EDA (exploratory data analysis), creates visualizations, and asks an LLM (Azure OpenAI via LangChain) for recommendations. The UI uses Flask so you don't need Streamlit or conda.

What it does
- Upload or choose a sample CSV dataset
- Produce summary statistics, missing-value report, and correlation
- Generate plots (histograms, correlation heatmap, pairplot sample)
- Ask an LLM for prioritized recommendations (optional; requires Azure OpenAI credentials)

Quick start (Windows PowerShell) — no conda

1) Create and activate a venv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

3) Configure environment variables
- Copy `.env.example` -> `.env` and fill values if you plan to use Azure OpenAI. The variables are:
	- `AZURE_OPENAI_ENDPOINT` (e.g. https://your-resource.openai.azure.com)
	- `AZURE_OPENAI_KEY`
	- `AZURE_OPENAI_DEPLOYMENT` (model deployment name)

4) Run the Flask demo

```powershell
python flask_app.py
# then open http://127.0.0.1:5000 in your browser
```

Project layout
- `flask_app.py` — Flask web UI (upload/select sample, run EDA)
- `agent.py` — DataScienceAgent (wires LangChain / Azure OpenAI, runs EDA helpers)
- `eda_utils.py` — EDA helpers (pandas + seaborn/matplotlib) that save plots to `static/plots`
- `templates/` — Jinja2 HTML templates for the Flask UI
- `sample_data/` — small sample CSV files (iris, titanic_small, wine)
- `static/plots/` — generated images saved here at runtime

Notes about Azure/OpenAI
- The LLM recommendations are optional. If you don't set the `AZURE_*` environment variables, the agent may raise an error when trying to initialize the Azure chat client. If you want a quick demo without Azure, let me know and I can add a simple rule-based fallback for recommendations.

Troubleshooting common pip / wheel errors (Windows)
- If pip tries to build packages from source (e.g., pyarrow) you'll see "failed wheel build" errors. Typical remedies:
	- Make sure you're using a supported Python version (3.8–3.11 are safest for many binary wheels).
	- Upgrade pip/setuptools/wheel before installing (see step 2).
	- If a package truly requires compilation, installing a prebuilt wheel via conda-forge is easiest — but if you must avoid conda, consider using Python 3.10/3.11 where wheels are more widely available.
	- As a last resort, install the Visual C++ Build Tools to build from source (slow and more error-prone).

Security and credentials
- Do not commit your `.env` file. Use `.env.example` as a template.

Next steps / optional improvements
- Add a lightweight offline fallback for LLM recommendations so the UI works without Azure keys.
- Add tests for `eda_utils.py` and a small smoke test
- Add Dockerfile for reproducible environments

If you'd like, I can update the repo right now to add a non-LLM fallback so the Flask UI runs without Azure credentials — tell me if you want that.

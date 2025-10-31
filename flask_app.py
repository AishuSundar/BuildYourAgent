from flask import Flask, render_template, request, redirect, url_for
import os
import shutil
from werkzeug.utils import secure_filename

from agent import DataScienceAgent
from eda_utils import cleanup_plots

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# List of example datasets shipped with the repo
SAMPLE_DATA = {
    'iris': os.path.join('sample_data', 'iris.csv'),
    'titanic': os.path.join('sample_data', 'titanic_small.csv'),
    'wine': os.path.join('sample_data', 'wine.csv'),
}


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', samples=SAMPLE_DATA.keys())


@app.route('/analyze', methods=['POST'])
def analyze():
    # Choose sample or uploaded file
    sample = request.form.get('sample')
    file = request.files.get('file')

    # Auto-clean old plots before generating new ones (remove files older than 5 minutes)
    try:
        cleanup_plots(older_than_seconds=300)
    except Exception:
        pass

    if sample and sample in SAMPLE_DATA:
        csv_path = SAMPLE_DATA[sample]
    elif file and file.filename:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        csv_path = save_path
    else:
        return redirect(url_for('index'))

    # Initialize agent and run EDA (agent uses a stub LLM if real LLM unavailable)
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        agent = DataScienceAgent(df)
        out = agent.run_eda(csv_path)
    except Exception as e:
        return render_template('error.html', error=str(e))

    used_llm = 'stub' if getattr(agent, '_using_stub', False) else 'azure'

    # Convert tables to HTML
    stats_html = out['stats'].to_html(classes='table table-sm', escape=False)
    missing_html = out['missing'].to_html(classes='table table-sm', escape=False)

    # For images saved under static/plots, build URLs
    # Helper to convert a stored path into a Flask static URL when possible.
    def _to_static_url(pth: str):
        if not pth:
            return None
        # Normalize path separators
        pnorm = os.path.normpath(pth)
        static_dir = os.path.normpath(os.path.join(os.getcwd(), 'static'))

        # If the path is absolute and inside the static dir, make a relative path
        if os.path.isabs(pnorm):
            try:
                rel = os.path.relpath(pnorm, static_dir)
                # Flask expects POSIX-style paths for URLs
                return url_for('static', filename=rel.replace(os.path.sep, '/'))
            except Exception:
                return None

        # pnorm is relative; check if it starts with the 'static' dir name
        parts = pnorm.split(os.path.sep)
        if parts and parts[0] == 'static':
            rel = os.path.sep.join(parts[1:])
            return url_for('static', filename=rel.replace(os.path.sep, '/'))

        # Not under static; cannot serve via static route
        return None

    hist_imgs = []
    for k, p in out.get('histograms', {}).items():
        url = _to_static_url(p)
        hist_imgs.append((k, url or p))

    heatmap_url = _to_static_url(out.get('heatmap')) or out.get('heatmap')
    pairplot_url = _to_static_url(out.get('pairplot')) or out.get('pairplot')

    return render_template(
        'results.html',
        recommendations=out.get('recommendations'),
        stats_html=stats_html,
        missing_html=missing_html,
        hist_imgs=hist_imgs,
        heatmap_url=heatmap_url,
        pairplot_url=pairplot_url,
        plan=out.get('langgraph_plan'),
        actionable=out.get('actionable', []),
        used_llm=used_llm,
        csv_path=csv_path,
        interpretation=out.get('interpretation'),
    )


@app.route('/apply_action', methods=['POST'])
def apply_action():
    action_type = request.form.get('action_type')
    columns = request.form.get('columns', '')
    csv_path = request.form.get('csv_path')
    if not csv_path:
        return render_template('error.html', error='Missing csv_path for action')
    cols = [c.strip() for c in columns.split(',') if c.strip()]
    action = {'type': action_type, 'columns': cols}

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        agent = DataScienceAgent(df)
        outpath = agent.apply_action(action, csv_path)
    except Exception as e:
        return render_template('error.html', error=str(e))

    # Run EDA on transformed file and show results with download link
    try:
        out = agent.run_eda(outpath)
    except Exception as e:
        return render_template('error.html', error=str(e))

    stats_html = out['stats'].to_html(classes='table table-sm', escape=False)
    missing_html = out['missing'].to_html(classes='table table-sm', escape=False)

    # prepare images
    def _to_static_url(pth: str):
        if not pth:
            return None
        pnorm = os.path.normpath(pth)
        static_dir = os.path.normpath(os.path.join(os.getcwd(), 'static'))
        if os.path.isabs(pnorm):
            try:
                rel = os.path.relpath(pnorm, static_dir)
                return url_for('static', filename=rel.replace(os.path.sep, '/'))
            except Exception:
                return None
        parts = pnorm.split(os.path.sep)
        if parts and parts[0] == 'static':
            rel = os.path.sep.join(parts[1:])
            return url_for('static', filename=rel.replace(os.path.sep, '/'))
        return None

    hist_imgs = []
    for k, p in out.get('histograms', {}).items():
        url = _to_static_url(p)
        hist_imgs.append((k, url or p))

    heatmap_url = _to_static_url(out.get('heatmap')) or out.get('heatmap')
    pairplot_url = _to_static_url(out.get('pairplot')) or out.get('pairplot')

    return render_template(
        'results.html',
        recommendations=out.get('recommendations'),
        stats_html=stats_html,
        missing_html=missing_html,
        hist_imgs=hist_imgs,
        heatmap_url=heatmap_url,
        pairplot_url=pairplot_url,
        plan=out.get('langgraph_plan'),
        actionable=out.get('actionable', []),
        used_llm=('stub' if getattr(agent, '_using_stub', False) else 'azure'),
        csv_path=outpath,
        transformed_download=os.path.relpath(outpath, os.getcwd()),
    )


@app.route('/samples/<name>')
def sample_download(name):
    if name in SAMPLE_DATA:
        return redirect(url_for('static', filename=SAMPLE_DATA[name]))
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)

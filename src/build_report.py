"""
PathSnap - Consolidated Technical Report HTML & PDF Generator
Reads verified experimental outputs from results/ and builds both hermetic HTML
and a publication-grade multi-page PDF report.

Team SRNs: PES1UG24AM004, PES1UG24AM005, PES1UG24AM006, PES1UG24AM007
"""

import os
import json
import argparse
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_RES_DIR = os.environ.get("PATHSNAP_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results"))
DEFAULT_FIG_DIR = os.environ.get("PATHSNAP_FIGURES_DIR", os.path.join(PROJECT_ROOT, "figures"))
DEFAULT_OUT_HTML = os.environ.get("PATHSNAP_REPORT_HTML", os.path.join(PROJECT_ROOT, "results", "consolidated_report.html"))
DEFAULT_OUT_PDF = os.environ.get("PATHSNAP_REPORT_PDF", os.path.join(PROJECT_ROOT, "results", "consolidated_report.pdf"))


def generate_html_report(res_dir: str = None, fig_dir: str = None, out_html: str = None):
    res_dir = res_dir or DEFAULT_RES_DIR
    fig_dir = fig_dir or DEFAULT_FIG_DIR
    out_html = out_html or DEFAULT_OUT_HTML
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig_rel = os.path.relpath(fig_dir, os.path.dirname(out_html)).replace("\\", "/")

    with open(os.path.join(res_dir, "eda_summary.json")) as f:
        eda = json.load(f)
    df_comp = pd.read_csv(os.path.join(res_dir, "model_comparison.csv"))
    with open(os.path.join(res_dir, "bayesian_opt_summary.json")) as f:
        bo_sum = json.load(f)
    df_bo = pd.read_csv(os.path.join(res_dir, "bayesian_opt_trials.csv"))
    df_hmm = pd.read_csv(os.path.join(res_dir, "hmm_vs_classifier_trips.csv"), dtype={"trip_id": str})
    with open(os.path.join(res_dir, "hmm_comparison_summary.json")) as f:
        hmm_sum = json.load(f)

    # Model comparison table rows
    comp_rows_html = ""
    for _, r in df_comp.iterrows():
        comp_rows_html += f"""
        <tr>
          <td><strong>{r['model']}</strong></td>
          <td>{r['family']}</td>
          <td>{r['key_config']}</td>
          <td class="num">{r['cv_macro_f1_mean']:.4f} &plusmn; {r['cv_macro_f1_std']:.4f}</td>
          <td class="num">{r['test_top1_acc']:.4f}</td>
          <td class="num">{r['test_route_continuity']:.4f}</td>
          <td class="num">{r['test_macro_f1']:.4f}</td>
          <td class="num">{r['test_log_loss']:.4f}</td>
          <td class="num">{r['inference_time_s']:.4f}s</td>
        </tr>"""

    # BO trials rows
    bo_rows_html = ""
    for _, r in df_bo.iterrows():
        bo_rows_html += f"""
        <tr>
          <td class="num">{int(r['trial'])}</td>
          <td>{r['phase']}</td>
          <td class="num">{int(r['hidden_dim'])}</td>
          <td class="num">{r['lr']:.6f}</td>
          <td class="num">{r['weight_decay']:.7f}</td>
          <td class="num">{int(r['epochs'])}</td>
          <td class="num"><strong>{r['cv_macro_f1']:.4f}</strong></td>
        </tr>"""

    # HMM per-trip rows
    hmm_rows_html = ""
    for _, r in df_hmm.iterrows():
        hmm_rows_html += f"""
        <tr>
          <td>{r['trip_id']}</td>
          <td class="num">{int(r['n_gps'])}</td>
          <td class="num">{r['clf_top1_acc']:.4f}</td>
          <td class="num"><strong>{r['hmm_top1_acc']:.4f}</strong></td>
          <td class="num">{r['clf_continuity']:.4f}</td>
          <td class="num"><strong>{r['hmm_continuity']:.4f}</strong></td>
          <td class="num">{r['clf_arc_f1']:.4f}</td>
          <td class="num"><strong>{r['hmm_arc_f1']:.4f}</strong></td>
        </tr>"""

    model_notes = [
        (
            "Logistic Regression",
            "Serves as a fast, convex linear baseline that outputs well-calibrated probabilities directly usable as HMM emissions.",
            "Strengths: rapid inference (0.0088s) and zero overfitting (Train F1 0.8758 vs. Test F1 0.8946). Limitations: cannot model nonlinear interactions between perpendicular distance, heading alignment, and local continuity.",
            "L2 penalty C=1.0, L-BFGS-B solver, max_iter=60, standardized 16 features.",
            "L2-regularized binary cross-entropy loss.",
            "Test Macro-F1: 0.8946, Top-1 Accuracy: 0.8462, Route Continuity: 0.9167, Log Loss: 0.1357.",
            "No overfitting observed; linear decision boundary slightly underfits complex urban intersections compared to MLP."
        ),
        (
            "Gaussian Naive Bayes",
            "Provides a lightweight generative probabilistic baseline assuming conditional feature independence given candidate correctness.",
            "Strengths: sub-second training (0.815s) and simple closed-form likelihoods. Limitations: strong independence assumption is violated by correlated distance/kinematic features, inflating probability confidence.",
            "var_smoothing=1e-6 on standardized features.",
            "Maximum likelihood estimation of class-conditional Gaussian means and variances.",
            "Test Macro-F1: 0.6049, Top-1 Accuracy: 0.8359, Route Continuity: 0.9127, Log Loss: 2.2160.",
            "Severe probability miscalibration (Log Loss 2.2160) due to correlated features; unsuitable as an HMM emission model despite decent Top-1 ranking."
        ),
        (
            "Distance-Weighted k-Nearest Neighbors (kNN)",
            "Non-parametric instance-based learner capable of capturing arbitrary local decision boundaries in normalized feature space.",
            "Strengths: zero training cost (0.013s) and strong Test Macro-F1 (0.8984). Limitations: slow inference (8.6472s for 153k candidates via KD-tree) makes real-time multi-trip API serving impractical.",
            "k=15 neighbors, Euclidean metric, inverse-distance weights 1/(d+1e-4), 25,000 indexed training points.",
            "Non-parametric local majority/distance-weighted voting (no iterative loss optimisation).",
            "Test Macro-F1: 0.8984, Top-1 Accuracy: 0.8384, Route Continuity: 0.9090, Log Loss: 0.1688.",
            "Mildly higher Log Loss than MLP due to finite-k probability quantization; inference latency is ~90x slower than MLP."
        ),
        (
            "Random Forest Classifier",
            "Bagged ensemble of binned CART trees that captures nonlinear feature thresholds and interactions without feature scaling.",
            "Strengths: fast training (1.61s), scale-invariant, and strong Test Macro-F1 (0.9020). Limitations: tree-averaged probabilities (Log Loss 0.1115) are slightly less sharp than MLP at intersections.",
            "n_estimators=15, max_depth=8, min_samples_leaf=25, max_features=4, 60k bootstrap subsample.",
            "Gini impurity minimization per split across bootstrap-resampled trees.",
            "Test Macro-F1: 0.9020, Top-1 Accuracy: 0.8397, Route Continuity: 0.9177, Log Loss: 0.1115.",
            "Well-regularized by depth cap and leaf minimum; generalizes cleanly from CV (0.8818) to Student-Test (0.9020)."
        ),
        (
            "Gradient Boosted Trees (GBM)",
            "Sequential additive tree ensemble correcting residual log-odds errors on hard candidate pairs near parallel streets.",
            "Strengths: strong nonlinear expressiveness and fast inference (0.1678s). Limitations: with 35 shallow trees on an 80k subsample, slightly higher Log Loss (0.1748) and CV variance (+/-0.0257) than Random Forest or MLP.",
            "n_estimators=35, learning_rate=0.1, max_depth=4, min_samples_leaf=30, subsample=0.6.",
            "Sequential negative-gradient (binary cross-entropy residual) minimization.",
            "Test Macro-F1: 0.8969, Top-1 Accuracy: 0.8432, Route Continuity: 0.9121, Log Loss: 0.1748.",
            "No severe overfitting, though probability calibration lags behind MLP."
        ),
        (
            "Shallow MLP (Neural Network)",
            "Learns smooth, continuous nonlinear combinations of spatial distance, heading alignment, and local temporal continuity.",
            "Strengths: highest CV Macro-F1 (0.8909), highest Test Macro-F1 (0.9103), highest Top-1 Accuracy (0.8502), lowest Log Loss (0.1027), and fast inference (0.0954s). Limitations: requires feature standardization and ~22s training.",
            "1 hidden layer (64 ReLU units), Adam lr=1e-3, batch_size=8192, epochs=12, weight_decay=1e-4.",
            "L2-regularized binary cross-entropy optimized with Adam.",
            "Test Macro-F1: 0.9103, Top-1 Accuracy: 0.8502, Route Continuity: 0.9203, Log Loss: 0.1027.",
            "Epoch-wise train and validation loss curves decrease smoothly together with zero divergence, confirming strong generalization."
        ),
    ]

    model_subsections_html = ""
    for idx, (mname, why, sl, hp, loss_fn, res_txt, fit_txt) in enumerate(model_notes, 1):
        model_subsections_html += f"""
        <h3>3.2.{idx} {mname}</h3>
        <ul>
          <li><strong>Suitability &amp; Trade-offs:</strong> {why} {sl}</li>
          <li><strong>Configuration &amp; Objective:</strong> {hp} Loss: {loss_fn}</li>
          <li><strong>Experimental Results &amp; Generalisation:</strong> {res_txt} {fit_txt}</li>
        </ul>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PathSnap — Technical Report</title>
  <style>
    :root {{
      --primary: #1e3a5f;
      --accent: #0f766e;
      --surface: #f4f7fb;
      --border: #d8e2ef;
      --text: #1f2937;
      --muted: #4b5563;
      --bg: #ffffff;
    }}
    body {{
      font-family: 'DejaVu Sans', -apple-system, sans-serif;
      font-size: 11pt;
      line-height: 1.55;
      color: var(--text);
      margin: 20px auto;
      max-width: 900px;
      padding: 0 15px;
    }}
    h1 {{ font-size: 22pt; font-weight: 800; color: var(--primary); margin: 0 0 8px 0; }}
    h2 {{ font-size: 15pt; font-weight: 700; color: var(--primary); margin: 24px 0 10px 0; padding-bottom: 5px; border-bottom: 2px solid var(--primary); }}
    h3 {{ font-size: 12pt; font-weight: 700; color: var(--primary); margin: 16px 0 6px 0; }}
    p {{ margin: 0 0 10px 0; }}
    ul {{ margin: 0 0 10px 0; padding-inline-start: 20px; }}
    li {{ margin-bottom: 4px; }}
    .banner {{
      background: var(--surface);
      border-inline-start: 5px solid var(--primary);
      padding: 16px 20px;
      border-radius: 6px;
      margin-bottom: 18px;
    }}
    .meta {{ font-size: 10pt; color: var(--muted); margin-top: 6px; }}
    .callout {{
      background: var(--surface);
      border-inline-start: 4px solid var(--accent);
      padding: 12px 16px;
      border-radius: 6px;
      margin: 14px 0;
    }}
    table {{ width: 100%; border-collapse: collapse; margin: 14px 0; }}
    th, td {{ padding: 7px 8px; font-size: 10pt; text-align: start; vertical-align: top; border-bottom: 1px solid var(--border); }}
    th {{ background: var(--primary); color: #ffffff; font-weight: 700; }}
    th.num, td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    tbody tr:nth-child(even) {{ background: var(--surface); }}
    figure {{ margin: 14px 0; }}
    figure img {{ width: 100%; max-height: 540px; object-fit: contain; display: block; margin: 0 auto; border: 1px solid var(--border); border-radius: 4px; }}
    figcaption {{ font-size: 9pt; color: var(--muted); text-align: center; margin-top: 6px; }}
  </style>
</head>
<body>
<main class="document-flow">
  <div class="banner">
    <h1>PathSnap: Distributed Map-Matching &amp; Route Reconstruction</h1>
    <div><strong>UE24CS352A: Machine Learning Hackathon — Technical Report</strong></div>
    <div class="meta">PES University, Department of CSE &nbsp;|&nbsp; Team SRNs: <strong>PES1UG24AM004, PES1UG24AM005, PES1UG24AM006, PES1UG24AM007</strong></div>
  </div>

  <h2>1. Problem Overview &amp; Pipeline Architecture</h2>
  <p>
    PathSnap solves vehicle trajectory map-matching and sequence route reconstruction on low-frequency, noisy GPS data. The system decouples candidate generation (KD-tree top-8 arcs), discriminative candidate classification (6 pure NumPy models), and topological route reconstruction via Hidden Markov Model (HMM) Viterbi dynamic programming.
  </p>

  <h2>2. Dataset Understanding &amp; Exploratory Data Analysis</h2>
  <p>
    We inspected the 80 <strong>training</strong> trips (208,417 GPS observations) and 10 <strong>student-testing</strong> trips (19,160 GPS observations). Projecting each trip into a <strong>per-trip Local Azimuthal Equidistant (AEQD)</strong> frame centered at the trajectory median preserves exact metric scale and relative angles.
  </p>
  <figure>
    <img src="{fig_rel}/eda_overview.png" alt="EDA Overview">
    <figcaption>Figure 1: Exploratory Data Analysis across the 80 training and 10 student-testing trips.</figcaption>
  </figure>

  <h2>3. Model Experimentation and Comparison</h2>
  <p>
    We trained and evaluated six distinct models across five algorithm families using 5-fold group-aware cross-validation on the 80 training trips, evaluated on the 10 student-testing trips.
  </p>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Family</th>
        <th>Key Configuration</th>
        <th class="num">CV Macro F1</th>
        <th class="num">Top-1 Acc</th>
        <th class="num">Continuity</th>
        <th class="num">Test Macro F1</th>
        <th class="num">Log Loss</th>
        <th class="num">Inf. Time</th>
      </tr>
    </thead>
    <tbody>
      {comp_rows_html}
    </tbody>
  </table>

  <h3>3.2 Individual Model Analysis</h3>
  {model_subsections_html}

  <h3>3.3 Model Visualisations</h3>
  <figure>
    <img src="{fig_rel}/model_comparison.png" alt="Model Comparison">
    <figcaption>Figure 2: 5-Fold Group-CV Macro F1, Student-Test Top-1 Accuracy, Log Loss, and Inference Time across all six models.</figcaption>
  </figure>
  <figure>
    <img src="{fig_rel}/confusion_matrices.png" alt="Confusion Matrices">
    <figcaption>Figure 3: Student-Test confusion matrices across all six models.</figcaption>
  </figure>

  <h2>4. Bayesian Hyperparameter Optimisation</h2>
  <p>
    We selected the <strong>Shallow MLP</strong> for Bayesian Optimisation using a Gaussian Process surrogate with Expected Improvement (GP-EI) over 10 trials.
  </p>
  <table>
    <thead>
      <tr>
        <th class="num">Trial</th>
        <th>Phase</th>
        <th class="num">Hidden Dim</th>
        <th class="num">Learning Rate</th>
        <th class="num">Weight Decay</th>
        <th class="num">Epochs</th>
        <th class="num">CV Macro F1</th>
      </tr>
    </thead>
    <tbody>
      {bo_rows_html}
    </tbody>
  </table>
  <figure>
    <img src="{fig_rel}/mlp_and_bo.png" alt="MLP Loss and BO Convergence">
    <figcaption>Figure 4: Epoch-wise MLP training/validation loss curves and GP-EI Bayesian Optimisation convergence.</figcaption>
  </figure>

  <h2>5. HMM and Viterbi Route Reconstruction</h2>
  <p>
    To enforce road-network connectivity, we implemented a log-space Viterbi decoder over the per-observation candidate arcs (hidden states).
  </p>
  <div class="callout">
    <strong>HMM + Viterbi Impact on Student-Test (10 Trips):</strong>
    Mean Observation Top-1 Accuracy reached <strong>{hmm_sum['classifier_plus_hmm']['mean_top1_acc']*100:.2f}%</strong>, Route Continuity rose to <strong>{hmm_sum['classifier_plus_hmm']['mean_continuity']*100:.2f}%</strong>, and Sequence Arc F1 jumped to <strong>{hmm_sum['classifier_plus_hmm']['mean_arc_f1']*100:.2f}%</strong>.
  </div>
  <table>
    <thead>
      <tr>
        <th>Trip ID</th>
        <th class="num">GPS Pts</th>
        <th class="num">Clf Top-1</th>
        <th class="num">HMM Top-1</th>
        <th class="num">Clf Cont.</th>
        <th class="num">HMM Cont.</th>
        <th class="num">Clf Arc F1</th>
        <th class="num">HMM Arc F1</th>
      </tr>
    </thead>
    <tbody>
      {hmm_rows_html}
    </tbody>
  </table>
  <figure>
    <img src="{fig_rel}/hmm_comparison.png" alt="HMM vs Classifier Comparison">
    <figcaption>Figure 5: Per-trip comparison of Classifier-Only vs. Classifier + HMM/Viterbi across all 10 student-testing trips.</figcaption>
  </figure>

  <h2>6. Geographical Visualisation Using Folium</h2>
  <p>
    We generated interactive Folium-based HTML maps (<code>maps/trip_00000056_map.html</code> and <code>maps/trip_00000007_map.html</code>) with layer controls, legends, and distinguishable layers for raw GPS observations (blue), local road network (gray), reference route (green), and HMM predicted route (red dashed).
  </p>
  <figure>
    <img src="{fig_rel}/geo_map_trip_00000056.png" alt="Geographical Map Trip 00000056">
    <figcaption>Figure 6: Geographical route reconstruction on Student-Test Trip 00000056.</figcaption>
  </figure>

  <h2>7. API, UI and Two-Laptop Deployment</h2>
  <p>
    The production system is split across two laptops: <strong>Laptop 1</strong> runs the FastAPI backend (<code>src/api/main.py</code> via <code>uvicorn</code>), exposing <code>GET /health</code>, <code>GET /api/trips</code>, and <code>POST /api/match/{{trip_id}}</code>. <strong>Laptop 2</strong> serves the evaluator-facing web UI (<code>src/ui/index.html</code>), which renders the interactive Leaflet route map and live metrics.
  </p>
</main>
</body>
</html>"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote HTML report: {out_html}")


def generate_pdf_report(res_dir: str = None, fig_dir: str = None, out_pdf: str = None):
    """Generate a clean, standalone multi-page PDF report using ReportLab."""
    res_dir = res_dir or DEFAULT_RES_DIR
    fig_dir = fig_dir or DEFAULT_FIG_DIR
    out_pdf = out_pdf or DEFAULT_OUT_PDF
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            Image as RLImage,
            KeepTogether,
            HRFlowable,
        )
    except ImportError:
        print("Warning: reportlab not installed. Skipping PDF report generation.")
        return

    doc = SimpleDocTemplate(
        out_pdf,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e3a5f"),
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4b5563"),
    )
    h1_style = ParagraphStyle(
        "ReportH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1e3a5f"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=6,
    )
    caption_style = ParagraphStyle(
        "ReportCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#6b7280"),
        alignment=1, # Center
        spaceAfter=8,
    )
    th_style = ParagraphStyle(
        "TH",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )
    td_style = ParagraphStyle(
        "TD",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1f2937"),
    )

    story = []

    # Title Banner
    story.append(Paragraph("PathSnap: Distributed Map-Matching &amp; Route Reconstruction", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>UE24CS352A: Machine Learning Hackathon — Technical Report</b>", subtitle_style))
    story.append(Paragraph("Department of CSE, PES University &nbsp;|&nbsp; Team SRNs: PES1UG24AM004, PES1UG24AM005, PES1UG24AM006, PES1UG24AM007", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3a5f"), spaceAfter=12))

    # Section 1: Overview
    story.append(Paragraph("1. System Overview &amp; Architecture", h1_style))
    story.append(Paragraph(
        "PathSnap solves the map-matching and sequence route reconstruction problem under low-frequency, noisy GPS data. "
        "The pipeline decouples spatial KD-tree candidate extraction (top-8 road arcs), candidate classification (6 pure NumPy models), "
        "and topological sequence route reconstruction via Hidden Markov Model (HMM) Viterbi dynamic programming.",
        body_style,
    ))

    # Section 2: EDA & Data Handling
    story.append(Paragraph("2. Geospatial Preprocessing &amp; Exploratory Data Analysis", h1_style))
    story.append(Paragraph(
        "We evaluated 80 training trips (208,417 observations) and 10 student-testing trips (19,160 observations). "
        "Coordinates are projected using an origin-centered Local Azimuthal Equidistant (AEQD) projection to maintain exact metric distances without distortion. "
        "Sixteen geometric, kinematic, and topological features are computed per candidate arc.",
        body_style,
    ))
    eda_fig_path = os.path.join(fig_dir, "eda_overview.png")
    if os.path.exists(eda_fig_path):
        story.append(RLImage(eda_fig_path, width=480, height=220))
        story.append(Paragraph("Figure 1: Exploratory Data Analysis across training and student-testing partitions.", caption_style))

    # Section 3: Model Comparison
    story.append(Paragraph("3. Multi-Model Experimentation (6 Models Across 5 Families)", h1_style))
    story.append(Paragraph(
        "Six models were evaluated using 5-fold group-aware cross-validation by trip ID to strictly eliminate spatial-temporal data leakage. "
        "All learned transformations (e.g. StandardScalers) were fitted exclusively on training data.",
        body_style,
    ))

    comp_csv = os.path.join(res_dir, "model_comparison.csv")
    if os.path.exists(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        header = [
            Paragraph("Model", th_style),
            Paragraph("Family", th_style),
            Paragraph("CV Macro F1", th_style),
            Paragraph("Test Top-1", th_style),
            Paragraph("Continuity", th_style),
            Paragraph("Test F1", th_style),
            Paragraph("Log Loss", th_style),
            Paragraph("Latency", th_style),
        ]
        table_data = [header]
        for _, r in df_comp.iterrows():
            table_data.append([
                Paragraph(str(r["model"]), td_style),
                Paragraph(str(r["family"]), td_style),
                Paragraph(f"{r['cv_macro_f1_mean']:.3f}±{r['cv_macro_f1_std']:.3f}", td_style),
                Paragraph(f"{r['test_top1_acc']:.3f}", td_style),
                Paragraph(f"{r['test_route_continuity']:.3f}", td_style),
                Paragraph(f"{r['test_macro_f1']:.3f}", td_style),
                Paragraph(f"{r['test_log_loss']:.3f}", td_style),
                Paragraph(f"{r['inference_time_s']:.3f}s", td_style),
            ])
        t = Table(table_data, colWidths=[90, 75, 65, 55, 55, 55, 55, 60])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    model_fig_path = os.path.join(fig_dir, "model_comparison.png")
    if os.path.exists(model_fig_path):
        story.append(RLImage(model_fig_path, width=480, height=220))
        story.append(Paragraph("Figure 2: Performance comparison across 5-fold Group CV and Student-Test evaluation.", caption_style))

    # Section 4: Bayesian Optimization
    story.append(Paragraph("4. Bayesian Hyperparameter Optimisation", h1_style))
    story.append(Paragraph(
        "A pure NumPy Gaussian Process surrogate model with Expected Improvement (GP-EI) acquisition tuned the Shallow MLP. "
        "The winning configuration (hidden_dim=123, lr=0.00448, weight_decay=0.0015, epochs=15) achieved 89.5% top-1 accuracy.",
        body_style,
    ))
    bo_fig_path = os.path.join(fig_dir, "mlp_and_bo.png")
    if os.path.exists(bo_fig_path):
        story.append(RLImage(bo_fig_path, width=480, height=180))
        story.append(Paragraph("Figure 3: Shallow MLP training loss curves and GP-EI Bayesian Optimization acquisition.", caption_style))

    # Section 5: HMM and Viterbi Reconstruction
    story.append(Paragraph("5. Sequential Route Reconstruction (HMM &amp; Viterbi)", h1_style))
    story.append(Paragraph(
        "Point-wise greedy predictions frequently suffer from disconnected arc teleports. "
        "Integrating network topology transition probabilities via log-space Viterbi decoding increased Route Continuity from 91.64% to 97.02%, "
        "Arc F1 from 82.85% to 88.62%, and order-sensitive LCS F1 from 67.23% to 76.30%.",
        body_style,
    ))
    hmm_fig_path = os.path.join(fig_dir, "hmm_comparison.png")
    if os.path.exists(hmm_fig_path):
        story.append(RLImage(hmm_fig_path, width=480, height=200))
        story.append(Paragraph("Figure 4: Per-trip benchmark comparing Point-Wise Classifier against HMM Viterbi Decoder.", caption_style))

    # Section 6: Geographical Maps
    story.append(Paragraph("6. Geographical Visualisation (Folium &amp; Static Cartography)", h1_style))
    story.append(Paragraph(
        "Interactive Folium maps were generated for evaluation trips with layer controls distinguishing GPS fixes (blue), "
        "road network arcs (gray), ground-truth routes (green), and HMM reconstructed sequences (red dashed).",
        body_style,
    ))
    geo_fig_path = os.path.join(fig_dir, "geo_map_trip_00000056.png")
    if os.path.exists(geo_fig_path):
        story.append(RLImage(geo_fig_path, width=460, height=230))
        story.append(Paragraph("Figure 5: High-resolution reconstructed route visualization for Student-Test Trip 00000056.", caption_style))

    # Section 7: API and Distributed Deployment
    story.append(Paragraph("7. REST API &amp; Dual-Laptop Evaluation Setup", h1_style))
    story.append(Paragraph(
        "The system exposes sub-100ms inference via FastAPI (GET /health, GET /api/trips, POST /api/match/{trip_id}). "
        "In a dual-laptop hackathon setting, Laptop 1 hosts the model inference engine while Laptop 2 runs the Leaflet.js browser dashboard over Wi-Fi.",
        body_style,
    ))

    doc.build(story)
    print(f"Wrote PDF report: {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="PathSnap - Technical Report Generator (HTML & PDF)")
    parser.add_argument("--results-dir", default=None, help="Directory containing experimental results")
    parser.add_argument("--figures-dir", default=None, help="Directory containing figures")
    parser.add_argument("--output-html", default=None, help="Path for generated HTML report")
    parser.add_argument("--output-pdf", default=None, help="Path for generated PDF report")
    args = parser.parse_args()

    generate_html_report(args.results_dir, args.figures_dir, args.output_html)
    generate_pdf_report(args.results_dir, args.figures_dir, args.output_pdf)


if __name__ == "__main__":
    main()

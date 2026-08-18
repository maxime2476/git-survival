from jinja2 import Template
import pandas as pd
from typing import Dict, Any

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Git-Survival Analysis Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 2rem; color: #333; }
        h1, h2, h3 { color: #2c3e50; }
        .metric-box { background: #f8f9fa; border-radius: 8px; padding: 1rem; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .metric-box span { font-weight: bold; color: #e74c3c; font-size: 1.2em; }
        table { border-collapse: collapse; width: 100%; margin: 1rem 0; overflow-x: auto; display: block; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        .plot-container { margin: 2rem 0; }
    </style>
</head>
<body>
    <h1>Git-Survival Analysis Report</h1>
    <p>Generated on: {{ date }}</p>
    
    <h2>1. Overview</h2>
    <div class="metric-box">
        <p>Total Contributors Analyzed: <span>{{ total_contributors }}</span></p>
        <p>Observed Churn Events: <span>{{ churn_events }}</span> ({{ churn_rate }}%)</p>
        <p>Median Survival Time: <span>{{ median_survival }}</span> days</p>
    </div>

    <h2>2. Global Survival Analysis (Kaplan-Meier)</h2>
    <div class="plot-container">
        {{ km_plot }}
    </div>

    <h2>3. Contributor Risk Factors (Cox PH Model)</h2>
    {% if ph_violation %}
    <p style="color: #e67e22; font-weight: bold;">⚠️ Warning: The Proportional Hazards assumption may be violated for some variables. Consider AFT models below.</p>
    {% endif %}
    
    {% if cox_summary_html %}
        {{ cox_summary_html }}
        <div class="plot-container">
            {{ cox_plot }}
        </div>
    {% else %}
        <p>Cox model could not be fitted.</p>
    {% endif %}

    <h2>4. At-Risk Contributors (Imminent Churn Prediction)</h2>
    <p>Based on the Cox model, here are the most active contributors with the highest risk of imminent churn:</p>
    {% if risk_html %}
        {{ risk_html }}
    {% else %}
        <p>No active contributors found or risk prediction failed.</p>
    {% endif %}

    <h2>5. Accelerated Failure Time Models (AFT)</h2>
    <p>Best Model selected by AIC: <strong>{{ aft_best_model }}</strong></p>
    {% if aft_summary_html %}
        {{ aft_summary_html }}
    {% endif %}

    <h2>5. Stratified Analysis</h2>
    {% if stratified_plot %}
    <div class="plot-container">
        {{ stratified_plot }}
    </div>
    {% else %}
    <p>Not enough data to stratify by weekend contribution.</p>
    {% endif %}
</body>
</html>
"""

def generate_report(
    output_path: str,
    df_features: pd.DataFrame,
    km_metrics: Dict[str, Any],
    cox_summary: pd.DataFrame,
    ph_violation: bool,
    aft_info: Dict[str, Any],
    aft_summary: pd.DataFrame,
    km_plot_b64: str,
    cox_plot_b64: str,
    stratified_plot_b64: str,
    risk_df: pd.DataFrame
):
    total = len(df_features)
    churn = int(df_features['E'].sum()) if total > 0 else 0
    
    if total > 0:
        churn_rate = round(churn / total * 100, 1)
    else:
        churn_rate = 0
        
    median_surv = km_metrics.get('median_survival_time', 'N/A')
    
    # Format tables
    cox_html = cox_summary.to_html(classes='table') if not cox_summary.empty else ""
    aft_html = aft_summary.to_html(classes='table') if not aft_summary.empty else ""
    risk_html = risk_df.to_html(classes='table', index=False) if not risk_df.empty else ""
    
    template = Template(TEMPLATE)
    html = template.render(
        date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        total_contributors=total,
        churn_events=churn,
        churn_rate=churn_rate,
        median_survival=median_surv,
        km_plot=km_plot_b64,
        cox_summary_html=cox_html,
        cox_plot=cox_plot_b64,
        ph_violation=ph_violation,
        risk_html=risk_html,
        aft_best_model=aft_info.get('best_model', 'N/A'),
        aft_summary_html=aft_html,
        stratified_plot=stratified_plot_b64
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

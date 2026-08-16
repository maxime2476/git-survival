import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
import pandas as pd
import io
import base64

def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_kaplan_meier(kmf: KaplanMeierFitter) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf.plot_survival_function(ax=ax, at_risk_counts=True)
    ax.set_title("Kaplan-Meier Survival Curve (Global)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival Probability (Retention)")
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64

def plot_forest_cox(cph: CoxPHFitter) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    cph.plot(ax=ax)
    ax.set_title("Forest Plot - Cox Proportional Hazards")
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64

def plot_stratified_km(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    kmf = KaplanMeierFitter()
    
    # Stratify by weekend ratio (e.g. median split)
    if not df.empty and 'weekend_ratio' in df.columns:
        median_weekend = df['weekend_ratio'].median()
        high_weekend = df['weekend_ratio'] > median_weekend
        
        if sum(high_weekend) > 0 and sum(~high_weekend) > 0:
            kmf.fit(df[high_weekend]['T'], df[high_weekend]['E'], label='High Weekend Commits')
            kmf.plot_survival_function(ax=ax)
            
            kmf.fit(df[~high_weekend]['T'], df[~high_weekend]['E'], label='Low Weekend Commits')
            kmf.plot_survival_function(ax=ax)
            
            ax.set_title("Stratified Survival by Weekend Contribution")
            ax.set_xlabel("Days")
            ax.set_ylabel("Retention")
    
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64

import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
import plotly.graph_objects as go

def plot_kaplan_meier(kmf: KaplanMeierFitter) -> str:
    fig = go.Figure()
    
    surv_df = kmf.survival_function_
    fig.add_trace(go.Scatter(
        x=surv_df.index,
        y=surv_df.iloc[:, 0],
        mode='lines',
        name='Survival Probability',
        line=dict(shape='hv', color='blue')
    ))
    
    ci_df = kmf.confidence_interval_
    fig.add_trace(go.Scatter(
        x=list(ci_df.index) + list(ci_df.index)[::-1],
        y=list(ci_df.iloc[:, 1]) + list(ci_df.iloc[:, 0])[::-1],
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='95% CI'
    ))
    
    fig.update_layout(title="Kaplan-Meier Survival Curve (Global)", 
                      xaxis_title="Days", 
                      yaxis_title="Survival Probability (Retention)")
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def plot_forest_cox(cph: CoxPHFitter) -> str:
    summary = cph.summary
    fig = go.Figure()
    
    # Forest plot visualization
    fig.add_trace(go.Scatter(
        x=summary['exp(coef)'],
        y=summary.index,
        mode='markers',
        error_x=dict(
            type='data',
            symmetric=False,
            array=summary['exp(coef) upper 95%'] - summary['exp(coef)'],
            arrayminus=summary['exp(coef)'] - summary['exp(coef) lower 95%']
        ),
        marker=dict(size=10, color='red'),
        name='Hazard Ratio'
    ))
    
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Forest Plot - Cox Proportional Hazards (Hazard Ratios)",
                      xaxis_title="Hazard Ratio (log scale)",
                      xaxis_type="log")
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def plot_stratified_km(df: pd.DataFrame) -> str:
    fig = go.Figure()
    kmf = KaplanMeierFitter()
    
    if not df.empty and 'weekend_ratio' in df.columns:
        median_weekend = df['weekend_ratio'].median()
        high_weekend = df['weekend_ratio'] > median_weekend
        
        if sum(high_weekend) > 0 and sum(~high_weekend) > 0:
            # High Weekend
            kmf.fit(df[high_weekend]['T'], df[high_weekend]['E'])
            surv_high = kmf.survival_function_
            fig.add_trace(go.Scatter(x=surv_high.index, y=surv_high.iloc[:, 0], 
                                     mode='lines', line=dict(shape='hv', color='orange'), name='High Weekend Commits'))
            
            # Low Weekend
            kmf.fit(df[~high_weekend]['T'], df[~high_weekend]['E'])
            surv_low = kmf.survival_function_
            fig.add_trace(go.Scatter(x=surv_low.index, y=surv_low.iloc[:, 0], 
                                     mode='lines', line=dict(shape='hv', color='green'), name='Low Weekend Commits'))
            
            fig.update_layout(title="Stratified Survival by Weekend Contribution",
                              xaxis_title="Days",
                              yaxis_title="Retention")
    
    if len(fig.data) == 0:
        return ""
        
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

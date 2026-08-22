import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
import plotly.graph_objects as go

def plot_kaplan_meier(kmf: KaplanMeierFitter) -> go.Figure:
    if not hasattr(kmf, 'survival_function_'):
        return None
        
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
    return fig

def plot_forest_cox(cph: CoxPHFitter) -> go.Figure:
    if not hasattr(cph, 'summary') or cph.summary.empty:
        return None
        
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
    return fig

def plot_schoenfeld_residuals(cph: CoxPHFitter, df: pd.DataFrame) -> go.Figure:
    try:
        from lifelines.statistics import proportional_hazard_test
        results = proportional_hazard_test(cph, df, time_transform='rank')
        p_values = results.summary['p']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=p_values.values,
            y=p_values.index,
            orientation='h',
            marker_color=['red' if p < 0.05 else 'green' for p in p_values.values],
            name='P-value'
        ))
        
        # Ajouter une ligne verticale au seuil de significativité 0.05
        fig.add_vline(x=0.05, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title="Test de Schoenfeld (Stabilité de l'effet dans le temps)",
            xaxis_title="P-value (Rouge : Effet instable, p < 0.05 | Vert : Effet stable)",
            yaxis_title="Variables",
            showlegend=False
        )
        return fig
    except Exception:
        return None

from lifelines.statistics import logrank_test

def plot_stratified_km(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    kmf = KaplanMeierFitter()
    
    if not df.empty and 'weekend_ratio' in df.columns:
        median_weekend = df['weekend_ratio'].median()
        high_weekend = df['weekend_ratio'] > median_weekend
        
        if sum(high_weekend) > 0 and sum(~high_weekend) > 0:
            # High Weekend
            df_high = df[high_weekend]
            kmf.fit(df_high['T'], df_high['E'])
            surv_high = kmf.survival_function_
            fig.add_trace(go.Scatter(x=surv_high.index, y=surv_high.iloc[:, 0], 
                                     mode='lines', line=dict(shape='hv', color='orange'), name='High Weekend Commits'))
            
            # Low Weekend
            df_low = df[~high_weekend]
            kmf.fit(df_low['T'], df_low['E'])
            surv_low = kmf.survival_function_
            fig.add_trace(go.Scatter(x=surv_low.index, y=surv_low.iloc[:, 0], 
                                     mode='lines', line=dict(shape='hv', color='green'), name='Low Weekend Commits'))
            
            # Log-Rank Test
            results = logrank_test(df_high['T'], df_low['T'], event_observed_A=df_high['E'], event_observed_B=df_low['E'])
            p_value = results.p_value
            significance = "Significatif" if p_value < 0.05 else "Non significatif"
            
            fig.update_layout(
                title=f"Survie par activité le Week-end<br><sup>Test Log-Rank : p={p_value:.4f} ({significance})</sup>",
                xaxis_title="Jours",
                yaxis_title="Probabilité de survie (Rétention)"
            )
    
    if len(fig.data) == 0:
        return None
        
    return fig

import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from typing import Dict, Any, Tuple
import warnings

def fit_cox_model(df: pd.DataFrame) -> Tuple[CoxPHFitter, pd.DataFrame, bool, pd.DataFrame]:
    # Select columns
    cols = ['T', 'E', 'night_commit_ratio', 'weekend_ratio', 'fix_vs_feat_ratio', 
            'avg_churn_per_commit', 'code_dispersion', 'commit_frequency', 'ownership_index']
    
    df_model = df[cols].copy()
    
    # Standardize continuous covariates to help convergence with penalizer
    if not df_model.empty and len(df_model) > 1:
        covariates = [c for c in cols if c not in ['T', 'E']]
        std = df_model[covariates].std()
        std = std.replace(0, 1) # Avoid division by zero
        df_model[covariates] = (df_model[covariates] - df_model[covariates].mean()) / std
    
    # Regularization via penalizer=0.01
    cph = CoxPHFitter(penalizer=0.01)
    
    try:
        cph.fit(df_model, duration_col='T', event_col='E')
    except Exception as e:
        warnings.warn(f"Cox model failed to converge: {e}")
        return cph, pd.DataFrame(), False, pd.DataFrame()
        
    summary = cph.summary[['coef', 'exp(coef)', 'p', 'coef lower 95%', 'coef upper 95%']].copy()
    summary.rename(columns={'exp(coef)': 'hazard_ratio'}, inplace=True)
    
    # Check proportional hazards assumption
    try:
        results = proportional_hazard_test(cph, df_model, time_transform='rank')
        ph_violation = any(results.p_value < 0.05)
    except:
        ph_violation = False
        
    # Predict Churn Risk for active contributors
    active_mask = df['E'] == 0
    risk_df = pd.DataFrame()
    
    if active_mask.any() and not df_model.empty:
        try:
            active_covariates = df_model.loc[active_mask]
            hazard_scores = cph.predict_partial_hazard(active_covariates)
            risk_df = pd.DataFrame({
                'author_email': df.loc[active_mask, 'author_email'],
                'hazard_score': hazard_scores.round(3),
                'total_commits': df.loc[active_mask, 'total_commits'],
                'days_active': df.loc[active_mask, 'T']
            })
            risk_df = risk_df.sort_values(by='hazard_score', ascending=False).head(10)
        except Exception as e:
            warnings.warn(f"Failed to predict churn risk: {e}")
            
    return cph, summary, ph_violation, risk_df

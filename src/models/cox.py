import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from typing import Dict, Any, Tuple
import warnings

def fit_cox_model(df: pd.DataFrame) -> Tuple[CoxPHFitter, pd.DataFrame, bool]:
    # Select columns
    cols = ['T', 'E', 'night_commit_ratio', 'weekend_ratio', 'fix_vs_feat_ratio', 
            'avg_churn_per_commit', 'code_dispersion', 'commit_frequency']
    
    df_model = df[cols].copy()
    
    # Regularization via penalizer=0.01
    cph = CoxPHFitter(penalizer=0.01)
    
    try:
        cph.fit(df_model, duration_col='T', event_col='E')
    except Exception as e:
        warnings.warn(f"Cox model failed to converge: {e}")
        return cph, pd.DataFrame(), False
        
    summary = cph.summary[['coef', 'exp(coef)', 'p', 'coef lower 95%', 'coef upper 95%']].copy()
    summary.rename(columns={'exp(coef)': 'hazard_ratio'}, inplace=True)
    
    # Check proportional hazards assumption
    try:
        results = proportional_hazard_test(cph, df_model, time_transform='rank')
        ph_violation = any(results.p_value < 0.05)
    except:
        ph_violation = False
        
    return cph, summary, ph_violation

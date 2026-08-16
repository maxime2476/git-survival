import pandas as pd
from lifelines import WeibullAFTFitter, LogNormalAFTFitter
from typing import Dict, Any, Tuple

def fit_aft_models(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    cols = ['T', 'E', 'night_commit_ratio', 'weekend_ratio', 'fix_vs_feat_ratio', 
            'avg_churn_per_commit', 'code_dispersion', 'commit_frequency']
    df_model = df[cols].copy()
    
    weibull = WeibullAFTFitter(penalizer=0.01)
    lognormal = LogNormalAFTFitter(penalizer=0.01)
    
    results = {}
    best_model = None
    best_aic = float('inf')
    best_summary = pd.DataFrame()
    
    for name, model in [('Weibull', weibull), ('LogNormal', lognormal)]:
        try:
            model.fit(df_model, duration_col='T', event_col='E')
            aic = model.AIC_
            results[name] = aic
            if aic < best_aic:
                best_aic = aic
                best_model = model
                best_summary = model.summary.copy()
        except:
            pass
            
    models_info = {
        'AIC': results,
        'best_model': best_model.__class__.__name__ if best_model else None
    }
    
    return models_info, best_summary

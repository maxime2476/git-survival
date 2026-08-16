import pandas as pd
from lifelines import KaplanMeierFitter
from typing import Dict, Any, Tuple

def fit_kaplan_meier(df: pd.DataFrame) -> Tuple[KaplanMeierFitter, Dict[str, Any]]:
    kmf = KaplanMeierFitter()
    if df.empty:
        return kmf, {}
        
    kmf.fit(df['T'], event_observed=df['E'], label="All Contributors")
    
    metrics = {
        'median_survival_time': kmf.median_survival_time_,
    }
    return kmf, metrics

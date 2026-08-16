import pandas as pd
from git_survival.models.kaplan_meier import fit_kaplan_meier

def test_km_fit():
    df = pd.DataFrame({
        'T': [10, 20, 30, 40],
        'E': [1, 1, 0, 1]
    })
    kmf, metrics = fit_kaplan_meier(df)
    assert 'median_survival_time' in metrics

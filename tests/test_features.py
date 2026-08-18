import pandas as pd
from datetime import datetime, timezone, timedelta
from features import build_survival_matrix

def test_build_survival_matrix():
    now = datetime.now(timezone.utc)
    commits = []
    
    # Contributor 1: Churned (last commit 110 days ago)
    for i in range(5):
        commits.append({
            'author_email': 'churned@example.com',
            'timestamp': now - timedelta(days=110 + i*5),
            'local_hour': 12,
            'local_weekday': 0,
            'insertions': 10,
            'deletions': 5,
            'files_modified': 2,
            'is_fix': 1,
            'is_feat': 0,
            'num_dirs': 1
        })
        
    # Contributor 2: Active (last commit 10 days ago)
    for i in range(3):
        commits.append({
            'author_email': 'active@example.com',
            'timestamp': now - timedelta(days=10 + i*2),
            'local_hour': 12,
            'local_weekday': 0,
            'insertions': 20,
            'deletions': 0,
            'files_modified': 1,
            'is_fix': 0,
            'is_feat': 1,
            'num_dirs': 1
        })
        
    df = pd.DataFrame(commits)
    
    features = build_survival_matrix(df, inactivity_window_days=90, min_contributions=2)
    assert len(features) == 2
    
    churned = features[features['author_email'] == 'churned@example.com'].iloc[0]
    active = features[features['author_email'] == 'active@example.com'].iloc[0]
    
    assert churned['E'] == 1
    assert active['E'] == 0
    assert churned['T'] >= 20 # 5 commits * 5 days = 20 days duration
    assert active['T'] >= 4 # 3 commits * 2 days = 4 days duration

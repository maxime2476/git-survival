import pandas as pd
import numpy as np

def build_survival_matrix(df_commits: pd.DataFrame, inactivity_window_days: int = 90, min_contributions: int = 2, reference_date: pd.Timestamp = None) -> pd.DataFrame:
    if df_commits.empty:
        return pd.DataFrame()
        
    # Compute Ownership / Bus Factor Index
    author_ownership = {}
    if 'modified_paths' in df_commits.columns:
        df_files = df_commits[['author_email', 'modified_paths']].explode('modified_paths').dropna(subset=['modified_paths'])
        if not df_files.empty:
            file_author_counts = df_files.groupby('modified_paths')['author_email'].nunique()
            df_files['file_ownership'] = 1.0 / df_files['modified_paths'].map(file_author_counts)
            author_ownership = df_files.groupby('author_email')['file_ownership'].mean().to_dict()
    
    # Group by author_email
    grouped = df_commits.groupby('author_email')
    
    features = []
    
    if reference_date is not None:
        end_of_observation = reference_date
    else:
        end_of_observation = pd.Timestamp.now(tz='UTC')
    
    for email, group in grouped:
        num_commits = len(group)
        if num_commits < min_contributions:
            continue
            
        first_commit = group['timestamp'].min()
        last_commit = group['timestamp'].max()
        
        # Duration T in days
        time_since_first = (last_commit - first_commit).days
        # Handle 1-day or 0-day duration to avoid div by zero, minimum T=1.0
        T = max(time_since_first, 1.0)
        
        # Event E
        time_since_last = (end_of_observation - last_commit).days
        E = 1 if time_since_last > inactivity_window_days else 0
        if E == 1:
            # If churned, T is time between first and last commit
            pass
        else:
            # If censored, T is time between first commit and end of observation
            T = max((end_of_observation - first_commit).days, 1.0)
            
        # Covariates
        local_hours = group['local_hour']
        night_commits = sum((local_hours >= 22) | (local_hours < 6))
        night_commit_ratio = night_commits / num_commits
        
        weekdays = group['local_weekday']
        weekend_commits = sum(weekdays >= 5)
        weekend_ratio = weekend_commits / num_commits
        
        fixes = group['is_fix'].sum()
        feats = group['is_feat'].sum()
        fix_vs_feat_ratio = (feats + 1) / (fixes + 1)
        
        avg_churn_per_commit = np.log(1 + (group['insertions'] + group['deletions']).mean())
        code_dispersion = group['num_dirs'].mean()
        commit_frequency = num_commits / T
        
        ownership_index = author_ownership.get(email, 1.0)
        
        features.append({
            'author_email': email,
            'T': float(T),
            'E': int(E),
            'night_commit_ratio': float(night_commit_ratio),
            'weekend_ratio': float(weekend_ratio),
            'fix_vs_feat_ratio': float(fix_vs_feat_ratio),
            'avg_churn_per_commit': float(avg_churn_per_commit),
            'code_dispersion': float(code_dispersion),
            'commit_frequency': float(commit_frequency),
            'ownership_index': float(ownership_index),
            'total_commits': int(num_commits)
        })
        
    return pd.DataFrame(features)

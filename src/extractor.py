import pandas as pd
from typing import Optional, List, Dict, Any
from pydriller import Repository
from datetime import datetime, timezone
import re

BOT_PATTERNS = [
    r'bot', r'dependabot', r'renovate', r'github-actions', 
    r'snyk', r'travis', r'jenkins'
]

def is_bot(author_name: str, author_email: str) -> bool:
    name_email = f"{author_name} {author_email}".lower()
    for pattern in BOT_PATTERNS:
        if re.search(pattern, name_email):
            return True
    return False

import hashlib
import os

def get_cache_path(repo_url: str) -> str:
    repo_hash = hashlib.md5(repo_url.encode()).hexdigest()
    return f".git_survival_cache_{repo_hash}.pkl"

def extract_git_history(repo_path_or_url: str, max_commits: Optional[int] = None) -> pd.DataFrame:
    cache_path = get_cache_path(repo_path_or_url)
    cached_df = pd.DataFrame()
    
    if os.path.exists(cache_path):
        try:
            cached_df = pd.read_pickle(cache_path)
            print(f"Loaded {len(cached_df)} commits from cache.")
        except Exception:
            pass
            
    commits_data: List[Dict[str, Any]] = []
    
    kwargs = {}
    if not cached_df.empty:
        kwargs['since'] = cached_df['timestamp'].max()
        
    repo = Repository(repo_path_or_url, **kwargs)
    
    count = 0
    # Create a fast lookup set for existing hashes
    existing_hashes = set(cached_df['hash'].values) if not cached_df.empty else set()
    
    for commit in repo.traverse_commits():
        if commit.hash in existing_hashes:
            continue
            
        if max_commits and count >= max_commits:
            break
            
        if is_bot(commit.author.name, commit.author.email):
            continue
            
        # Detect tags
        msg_lower = commit.msg.lower()
        is_fix = 1 if re.search(r'\b(fix|bug|hotfix)\b', msg_lower) else 0
        is_feat = 1 if re.search(r'\b(feat|feature)\b', msg_lower) else 0
        
        # Modified dirs
        modified_dirs = set()
        modified_paths = set()
        for m in commit.modified_files:
            if m.old_path:
                modified_dirs.add(m.old_path.rsplit('/', 1)[0] if '/' in m.old_path else '')
                modified_paths.add(m.old_path)
            if m.new_path:
                modified_dirs.add(m.new_path.rsplit('/', 1)[0] if '/' in m.new_path else '')
                modified_paths.add(m.new_path)
                
        # Time processing
        dt_utc = commit.author_date.astimezone(timezone.utc)
        
        commits_data.append({
            'hash': commit.hash,
            'author_name': commit.author.name,
            'author_email': commit.author.email,
            'timestamp': dt_utc,
            'local_hour': commit.author_date.hour,
            'local_weekday': commit.author_date.weekday(),
            'insertions': commit.insertions,
            'deletions': commit.deletions,
            'files_modified': commit.files,
            'is_fix': is_fix,
            'is_feat': is_feat,
            'num_dirs': len(modified_dirs),
            'modified_paths': list(modified_paths)
        })
        count += 1
        
    new_df = pd.DataFrame(commits_data)
    if not new_df.empty:
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], utc=True)
        
    if not cached_df.empty and not new_df.empty:
        df_raw = pd.concat([cached_df, new_df], ignore_index=True)
    elif not cached_df.empty:
        df_raw = cached_df
    else:
        df_raw = new_df
        
    if not df_raw.empty:
        df_raw = df_raw.drop_duplicates(subset=['hash'])
        df_raw.to_pickle(cache_path)
        
        # We return a resolved copy so we don't corrupt the raw emails in cache
        df_resolved = df_raw.copy()
        df_resolved = resolve_identities(df_resolved)
        return df_resolved
        
    return df_raw
        


def resolve_identities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolves author identities to avoid false churn events caused by multiple emails.
    """
    if df.empty:
        return df
        
    df['author_name'] = df['author_name'].fillna('').astype(str).str.lower().str.strip()
    df['author_email'] = df['author_email'].fillna('').astype(str).str.lower().str.strip()
    
    # Clean GitHub noreply emails (12345+username@users.noreply.github.com -> username@github-noreply.local)
    mask = df['author_email'].str.contains('@users.noreply.github.com', na=False)
    extracted = df.loc[mask, 'author_email'].str.extract(r'(?:^\d+\+)?([^@]+)@', expand=False)
    df.loc[mask, 'author_email'] = extracted + '@github-noreply.local'
    
    # Deduplicate by name: map each name to its most frequently used email
    email_mapping = df.groupby('author_name')['author_email'].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else ""
    ).to_dict()
    
    df['author_email'] = df['author_name'].map(email_mapping)
    return df

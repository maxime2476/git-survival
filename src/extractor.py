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

def extract_git_history(repo_path_or_url: str, max_commits: Optional[int] = None) -> pd.DataFrame:
    commits_data: List[Dict[str, Any]] = []
    
    repo = Repository(repo_path_or_url)
    
    count = 0
    for commit in repo.traverse_commits():
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
        for m in commit.modified_files:
            if m.old_path:
                modified_dirs.add(m.old_path.rsplit('/', 1)[0] if '/' in m.old_path else '')
            if m.new_path:
                modified_dirs.add(m.new_path.rsplit('/', 1)[0] if '/' in m.new_path else '')
                
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
            'num_dirs': len(modified_dirs)
        })
        count += 1
        
    df = pd.DataFrame(commits_data)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df

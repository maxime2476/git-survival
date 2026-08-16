import pandas as pd
from git_survival.extractor import is_bot

def test_is_bot():
    assert is_bot("Dependabot", "support@github.com") == True
    assert is_bot("John Doe", "john@example.com") == False
    assert is_bot("github-actions", "github-actions[bot]@users.noreply.github.com") == True

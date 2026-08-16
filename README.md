# Git-Survival

A CLI tool for analyzing contributor churn in open-source Git repositories using survival analysis (Kaplan-Meier, Cox Proportional Hazards, Accelerated Failure Time models).

## Features
- Extracts contributor behavior directly from Git history.
- Filters out bots and calculates advanced covariates (night commits, weekend ratios, fix vs feat, etc.).
- Fits statistical survival models to estimate retention probabilities and risk factors.
- Generates a standalone HTML report with survival curves and hazard ratios.

## Usage

```bash
git-survival analyze <REPO_PATH_OR_URL> --threshold 90 --min-commits 2 --output ./report.html
```

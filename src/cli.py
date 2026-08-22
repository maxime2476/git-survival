import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from extractor import extract_git_history
from features import build_survival_matrix
from models.kaplan_meier import fit_kaplan_meier
from models.cox import fit_cox_model
from models.aft import fit_aft_models
from visualization import plot_kaplan_meier, plot_forest_cox, plot_stratified_km
from report import generate_report
import os
import warnings

warnings.filterwarnings("ignore")

app = typer.Typer(help="Git-Survival: Survival Analysis for Contributor Churn")
console = Console()

@app.command()
def analyze(
    repo_path_or_url: str = typer.Argument(..., help="Path or URL to the Git repository"),
    threshold: int = typer.Option(90, help="Inactivity window in days to consider a churn event"),
    min_commits: int = typer.Option(2, help="Minimum number of commits to include a contributor"),
    output: str = typer.Option("./report.html", help="Path to save the HTML report"),
    max_commits: int = typer.Option(None, help="Maximum number of commits to extract (for testing)"),
    reference_date: str = typer.Option(None, help="Reference date (YYYY-MM-DD) to calculate churn. Defaults to today.")
):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Phase 1: Extraction
        task_extract = progress.add_task("[cyan]Extracting Git history...", total=None)
        df_commits = extract_git_history(repo_path_or_url, max_commits)
        progress.update(task_extract, completed=1)
        
        if df_commits.empty:
            console.print("[red]No commits found or extracted![/red]")
            raise typer.Exit(1)
            
        console.print(f"[green]Extracted {len(df_commits)} commits.[/green]")
        
        # Phase 2: Features
        task_feat = progress.add_task("[cyan]Building survival matrix...", total=None)
        
        import pandas as pd
        ref_dt = pd.to_datetime(reference_date, utc=True) if reference_date else None
        
        df_features = build_survival_matrix(df_commits, threshold, min_commits, ref_dt)
        progress.update(task_feat, completed=1)
        
        if df_features.empty:
            console.print("[red]Not enough data after filtering (e.g. min_commits).[/red]")
            raise typer.Exit(1)
            
        if len(df_features) < 10:
            console.print("[yellow]WARNING: Less than 10 contributors found. Statistical models may not converge.[/yellow]")
            
        console.print(f"[green]Built matrix for {len(df_features)} contributors.[/green]")
        
        # Phase 3: Modeling
        task_model = progress.add_task("[cyan]Fitting statistical models...", total=None)
        kmf, km_metrics = fit_kaplan_meier(df_features)
        cph, cox_summary, ph_violation, risk_df, df_model = fit_cox_model(df_features)
        aft_info, aft_summary = fit_aft_models(df_features)
        progress.update(task_model, completed=1)
        
        # Phase 4: Visualization & Report
        task_report = progress.add_task("[cyan]Generating report...", total=None)
        km_fig = plot_kaplan_meier(kmf)
        cox_fig = plot_forest_cox(cph)
        strat_fig = plot_stratified_km(df_features)
        
        km_plot_b64 = km_fig.to_html(full_html=False, include_plotlyjs='cdn') if km_fig else ""
        cox_plot_b64 = cox_fig.to_html(full_html=False, include_plotlyjs='cdn') if cox_fig else ""
        strat_plot_b64 = strat_fig.to_html(full_html=False, include_plotlyjs='cdn') if strat_fig else ""
        
        generate_report(
            output, df_features, km_metrics, 
            cox_summary, ph_violation, 
            aft_info, aft_summary,
            km_plot_b64, cox_plot_b64, strat_plot_b64, risk_df
        )
        progress.update(task_report, completed=1)
        
    # Summary Table
    console.print("\n[bold]Summary:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=30)
    table.add_column("Value")
    
    table.add_row("Total Contributors Analyzed", str(len(df_features)))
    table.add_row("Observed Churn Events", str(int(df_features['E'].sum())))
    table.add_row("Median Survival Time (Days)", str(km_metrics.get('median_survival_time', 'N/A')))
    table.add_row("Proportional Hazards Violation", str(ph_violation))
    
    console.print(table)
    console.print(f"\n[bold green]Report successfully generated at: [/bold green]\n{os.path.abspath(output)}")

    # Write to GITHUB_STEP_SUMMARY if running in GitHub Actions
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
            f.write("# 📊 Git Survival Analysis\n\n")
            f.write(f"Analyzed **{len(df_features)}** contributors. Observed **{int(df_features['E'].sum())}** churn events.\n\n")
            if not risk_df.empty:
                f.write("### ⚠️ At-Risk Contributors\n")
                f.write(risk_df.to_markdown(index=False))
                f.write("\n\n")
            else:
                f.write("### ✅ No active contributors currently at risk.\n\n")
            f.write(f"Download the full HTML report from the artifact for detailed survival curves and metrics.")

@app.command()
def dashboard():
    """Launch the interactive Streamlit Dashboard."""
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py")
    console.print(f"[bold green]Launching Streamlit Dashboard from {dashboard_path}...[/bold green]")
    os.system(f"streamlit run {dashboard_path}")

if __name__ == "__main__":
    app()

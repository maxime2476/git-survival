import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from .extractor import extract_git_history
from .features import build_survival_matrix
from .models.kaplan_meier import fit_kaplan_meier
from .models.cox import fit_cox_model
from .models.aft import fit_aft_models
from .visualization import plot_kaplan_meier, plot_forest_cox, plot_stratified_km
from .report import generate_report
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
    max_commits: int = typer.Option(None, help="Maximum number of commits to extract (for testing)")
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
        df_features = build_survival_matrix(df_commits, threshold, min_commits)
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
        cph, cox_summary, ph_violation = fit_cox_model(df_features)
        aft_info, aft_summary = fit_aft_models(df_features)
        progress.update(task_model, completed=1)
        
        # Phase 4: Visualization & Report
        task_report = progress.add_task("[cyan]Generating report...", total=None)
        km_plot_b64 = plot_kaplan_meier(kmf) if kmf else ""
        cox_plot_b64 = plot_forest_cox(cph) if not cox_summary.empty else ""
        strat_plot_b64 = plot_stratified_km(df_features)
        
        generate_report(
            output, df_features, km_metrics, 
            cox_summary, ph_violation, 
            aft_info, aft_summary,
            km_plot_b64, cox_plot_b64, strat_plot_b64
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
    console.print(f"\n[green]Report successfully generated at: {os.path.abspath(output)}[/green]")

if __name__ == "__main__":
    app()

import streamlit as st
import pandas as pd
import os
import sys

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extractor import extract_git_history
from features import build_survival_matrix
from models.kaplan_meier import fit_kaplan_meier
from models.cox import fit_cox_model
from models.aft import fit_aft_models
from visualization import plot_kaplan_meier, plot_forest_cox, plot_stratified_km

st.set_page_config(page_title="Git Survival Dashboard", layout="wide")

st.title("Git Survival - Analyse de rétention")
st.markdown("""
Interface d'analyse de survie pour les dépôts Git. Permet d'identifier les facteurs 
qui poussent les contributeurs à quitter un projet open-source.
""")

repo_url = st.text_input("URL du repo ou chemin local", "https://github.com/tiangolo/typer")

if st.button("Lancer l'analyse"):
    with st.spinner("Parsing de l'historique Git..."):
        df_commits = extract_git_history(repo_url)
        
    if df_commits.empty:
        st.error("Impossible d'extraire les commits.")
        st.stop()
        
    st.success(f"{len(df_commits)} commits parsés.")
    
    with st.spinner("Construction de la matrice..."):
        df_features = build_survival_matrix(df_commits)
        
    if len(df_features) < 2:
        st.error("Pas assez de données pour faire tourner les modèles.")
        st.stop()
        
    st.info(f"Analyse sur {len(df_features)} développeurs (dont {int(df_features['E'].sum())} qui ont quitté le projet).")
    
    with st.spinner("Entraînement des modèles (Lifelines)..."):
        kmf, km_metrics = fit_kaplan_meier(df_features)
        cph, cox_summary, ph_violation, risk_df = fit_cox_model(df_features)
        aft_info, aft_summary = fit_aft_models(df_features)
        
    st.divider()
    
    # Ligne 1 : Kaplan-Meier et Cox
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Courbe globale de rétention (Kaplan-Meier)")
        st.markdown("Estimation basique du taux de survie des contributeurs dans le temps.")
        km_fig = plot_kaplan_meier(kmf)
        if km_fig:
            st.plotly_chart(km_fig, use_container_width=True)
            
    with col2:
        st.subheader("Régression de Cox (Facteurs de risque)")
        st.markdown("Un Hazard Ratio (HR) > 1 indique que la variable augmente les chances que le développeur arrête de contribuer.")
        if not cox_summary.empty:
            cox_fig = plot_forest_cox(cph)
            if cox_fig:
                st.plotly_chart(cox_fig, use_container_width=True)
            if ph_violation:
                st.warning("Note: l'hypothèse des risques proportionnels n'est pas totalement respectée.")
        else:
            st.warning("Impossible de faire converger le modèle de Cox.")
            
    st.divider()
    
    # Ligne 2 : Week-end et AFT
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Impact des commits le week-end")
        st.markdown("Comparaison entre les contributeurs qui codent beaucoup hors semaine et les autres.")
        strat_fig = plot_stratified_km(df_features)
        if strat_fig:
            st.plotly_chart(strat_fig, use_container_width=True)
            
    with col4:
        st.subheader("Modèle AFT (Estimation du temps restant)")
        st.markdown(f"Meilleure distribution : `{aft_info.get('best_model', 'N/A')}`")
        if not aft_summary.empty:
            st.dataframe(aft_summary, use_container_width=True)
            
    st.divider()
    st.subheader("Contributeurs actuels à surveiller")
    st.markdown("Liste des développeurs actifs triés par leur risque estimé de churn (calculé via le modèle de Cox).")
    if not risk_df.empty:
        st.dataframe(risk_df, use_container_width=True)
    else:
        st.success("Aucun contributeur actif n'est classé à risque.")

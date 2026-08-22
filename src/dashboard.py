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
        cph, cox_summary, ph_violation, risk_df, df_model = fit_cox_model(df_features)
        aft_info, aft_summary = fit_aft_models(df_features)
        
    st.divider()
    
    st.subheader("Synthèse des résultats")
    st.markdown("Analyse automatique des métriques extraites par les modèles :")
    
    insights = []
    
    # 1. Analyse Kaplan-Meier
    median_time = km_metrics.get("median_survival_time", float('inf'))
    if median_time != float('inf') and pd.notnull(median_time):
        insights.append(f"- **Durée de vie médiane** : {median_time:.0f} jours (la moitié des contributeurs abandonnent avant ce délai).")
    else:
        insights.append("- **Durée de vie médiane** : Non atteinte (la majorité des contributeurs restent actifs).")

    # 2. Analyse Cox (Facteurs de risque)
    if not cox_summary.empty:
        # P-value < 0.20 tolérée pour dégager des tendances
        sig_vars = cox_summary[cox_summary['p'] < 0.20]
        if not sig_vars.empty:
            max_risk = sig_vars['hazard_ratio'].idxmax()
            max_hr = sig_vars.loc[max_risk, 'hazard_ratio']
            if max_hr > 1.1:
                insights.append(f"- **Facteur aggravant principal** : La variable `{max_risk}` augmente significativement le risque de départ (HR = {max_hr:.2f}).")
                
            min_risk = sig_vars['hazard_ratio'].idxmin()
            min_hr = sig_vars.loc[min_risk, 'hazard_ratio']
            if min_hr < 0.9:
                insights.append(f"- **Facteur protecteur principal** : La variable `{min_risk}` est associée à une meilleure rétention (HR = {min_hr:.2f}).")

    # 3. Analyse des contributeurs à risque
    if not risk_df.empty:
        top_risk = risk_df.iloc[0]
        insights.append(f"- **Prédiction** : {len(risk_df)} contributeur(s) actif(s) identifié(s) à risque de churn élevé (notamment `{top_risk['author_email']}`).")
    else:
        insights.append("- **Prédiction** : Aucun contributeur actif ne présente un risque de départ imminent selon le modèle.")

    for insight in insights:
        st.markdown(insight)
        
    st.divider()
    
    # Ligne 1 : Kaplan-Meier et Cox
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Courbe globale de rétention (Kaplan-Meier)")
        st.markdown("Estimation non-paramétrique du taux de survie des contributeurs dans le temps.")
        km_fig = plot_kaplan_meier(kmf)
        if km_fig:
            st.plotly_chart(km_fig, use_container_width=True)
            
        with st.expander("Détails mathématiques (KM)"):
            st.markdown(r"L'estimateur de Kaplan-Meier calcule la probabilité de survie $S(t)$ en multipliant les probabilités conditionnelles à chaque instant $t_i$ où un départ est observé : $S(t) = \prod_{t_i \le t} (1 - \frac{d_i}{n_i})$ où $d_i$ est le nombre de départs et $n_i$ le nombre de devs encore actifs.")
            
    from visualization import plot_schoenfeld_residuals
    
    with col2:
        st.subheader("Régression de Cox (Facteurs de risque)")
        
        if not cox_summary.empty and hasattr(cph, 'concordance_index_'):
            c_index = cph.concordance_index_
            st.markdown(f"Modèle semi-paramétrique. **Indice de Concordance (C-Index) : {c_index:.3f}** (mesure de précision du modèle). Un Hazard Ratio (HR) > 1 indique que la variable augmente le risque.")
            cox_fig = plot_forest_cox(cph)
            if cox_fig:
                st.plotly_chart(cox_fig, use_container_width=True)
            
            with st.expander("Vérification des Résidus de Schoenfeld"):
                st.markdown("L'hypothèse des risques proportionnels stipule que l'effet d'une variable ne change pas avec le temps. Si une variable a une p-value < 0.05 (barre rouge), cela signifie que son effet diminue ou augmente avec le temps (hypothèse violée).")
                
                from lifelines.statistics import proportional_hazard_test
                try:
                    ph_results = proportional_hazard_test(cph, df_model, time_transform='rank')
                    violators = ph_results.summary[ph_results.summary['p'] < 0.05].index.tolist()
                    if violators:
                        st.warning(f"⚠️ Les variables suivantes voient leur effet changer avec le temps : **{', '.join(violators)}**.")
                    else:
                        st.success("✅ Toutes les variables ont un effet constant dans le temps (hypothèse parfaitement respectée).")
                except:
                    pass
                
                schoenfeld_fig = plot_schoenfeld_residuals(cph, df_model)
                if schoenfeld_fig:
                    st.plotly_chart(schoenfeld_fig, use_container_width=True)
        else:
            st.markdown("Modèle semi-paramétrique. Un Hazard Ratio (HR) > 1 indique que la variable augmente le risque.")
            st.warning("Impossible de faire converger le modèle de Cox.")
            
        with st.expander("Légende des variables étudiées"):
            st.markdown("""
            - **`avg_sentiment`** : Polarité moyenne des messages de commit (analyse NLP via TextBlob). Un score bas = frustration.
            - **`contagion_score`** : Modèle de contagion (graphe). Ratio de collaborateurs directs (touchant les mêmes fichiers) ayant déjà quitté le projet.
            - **`ownership_index`** : Bus Factor. Part des fichiers modifiés où le développeur est le seul intervenant.
            - **`weekend_ratio` / `night_commit_ratio`** : Proportion du travail effectué hors horaires standards (risque de burnout).
            - **`fix_vs_feat_ratio`** : Proportion de correctifs (bugfixes) par rapport aux ajouts de features.
            """)
            
    st.divider()
    
    # Ligne 2 : Week-end et AFT
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Impact des commits le week-end")
        st.markdown("Stratification de Kaplan-Meier pour observer l'effet du travail sur le temps libre.")
        strat_fig = plot_stratified_km(df_features)
        if strat_fig:
            st.plotly_chart(strat_fig, use_container_width=True)
            
    with col4:
        st.subheader("Modèle AFT (Estimation du temps restant)")
        st.markdown(f"Modèle paramétrique strict. Meilleure distribution trouvée : `{aft_info.get('best_model', 'N/A')}`")
        if not aft_summary.empty:
            st.dataframe(aft_summary, use_container_width=True)
            
        with st.expander("Détails mathématiques (AFT)"):
            st.markdown(r"Contrairement à Cox qui modélise le risque (Hazard), l'Accelerated Failure Time (AFT) modélise directement le logarithme du temps de survie : $\log(T) = \beta_0 + \beta_1 X_1 + ... + \sigma \epsilon$. Cela permet d'étirer ou de contracter le temps restant estimé.")
            
    st.divider()
    st.subheader("Contributeurs actuels à surveiller")
    st.markdown("Estimation de la probabilité de churn pour les contributeurs n'ayant pas encore quitté le projet, basée sur la fonction de survie partielle de Cox.")
    if not risk_df.empty:
        st.dataframe(risk_df, use_container_width=True)
    else:
        st.success("Aucun contributeur actif n'est classé à risque.")

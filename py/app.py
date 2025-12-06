import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
import os

# IMPORT DU MODULE LOGIQUE (Votre fichier partie1.py)
import partie1 as p1

# ==============================================================================
# 1. CONFIGURATION DE LA PAGE
# ==============================================================================
st.set_page_config(
    page_title="Optimisation de Portefeuille - Markowitz",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Optimisation de Portefeuille & Front de Pareto")
st.markdown("""
Cette application permet de visualiser la frontière efficiente (Markowitz), 
de définir un rendement cible et d'analyser la composition sectorielle du portefeuille optimal.
""")

# ==============================================================================
# 2. FONCTIONS UTILITAIRES SPÉCIFIQUES À L'APP
# ==============================================================================

@st.cache_data
def charger_donnees_app():
    """
    Gère les chemins de fichiers et appelle partie1 pour le chargement CSV.
    Charge également le JSON des secteurs.
    """
    # 1. Gestion flexible des chemins (local ou dossier parent)
    csv_filename = "returns_final.csv"
    json_filename = "tick.json"
    possible_paths = ["../data/", "data/", "./"]
    
    csv_path = None
    json_path = None

    # Recherche du CSV
    for path in possible_paths:
        full_path = os.path.join(path, csv_filename)
        if os.path.exists(full_path):
            csv_path = full_path
            break
            
    # Recherche du JSON
    for path in possible_paths:
        full_path = os.path.join(path, json_filename)
        if os.path.exists(full_path):
            json_path = full_path
            break

    if csv_path is None:
        st.error(f"Le fichier '{csv_filename}' est introuvable dans {possible_paths}.")
        return None, None, None, None, None

    # 2. Appel à partie1 pour la logique financière
    try:
        mu, Sigma, asset_names, df = p1.charger_et_preparer_donnees(csv_path)
    except Exception as e:
        st.error(f"Erreur lors du chargement des données financières : {e}")
        return None, None, None, None, None

    # 3. Chargement du JSON (Spécifique à l'UI)
    secteurs_dict = {}
    if json_path:
        try:
            with open(json_path, "r") as f:
                secteurs_dict = json.load(f)
        except Exception as e:
            st.warning(f"Impossible de lire le fichier JSON : {e}")
    else:
        st.warning(f"Le fichier '{json_filename}' est introuvable. L'analyse sectorielle sera désactivée.")

    return df, mu, Sigma, asset_names, secteurs_dict

@st.cache_data
def executer_optimisation(mu, Sigma):
    """Wrapper pour mettre en cache le résultat du calcul intensif de partie1."""
    return p1.calculer_frontiere_efficiente(mu, Sigma, nb_points=100, nb_iterations=300)

def map_secteurs(weights, asset_names, secteurs_dict):
    """
    Convertit un vecteur de poids (par actif) en poids par secteur.
    """
    if not secteurs_dict:
        return {}

    # Inversion du dict JSON : Ticker -> Secteur
    ticker_to_sector = {}
    for sector, tickers in secteurs_dict.items():
        for t in tickers:
            ticker_to_sector[t] = sector
            
    sector_weights = {}
    
    for w, asset in zip(weights, asset_names):
        if w > 0.001:  # On ignore les poids négligeables (< 0.1%)
            sec = ticker_to_sector.get(asset, "Autre / Inconnu")
            sector_weights[sec] = sector_weights.get(sec, 0) + w
            
    return sector_weights

# ==============================================================================
# 3. LOGIQUE PRINCIPALE
# ==============================================================================

# A. Chargement
df_rendements, mu, Sigma, asset_names, secteurs_data = charger_donnees_app()

if df_rendements is not None:
    
    # B. Calcul (Exécuté via partie1)
    with st.spinner('Calcul de la frontière efficiente en cours (Descente de Gradient)...'):
        risques, rendements, historiques_poids = executer_optimisation(mu, Sigma)

    # C. Interface Utilisateur - Sidebar
    st.sidebar.header("Paramètres du Portefeuille")
    
    min_ret_dispo = float(np.min(rendements))
    max_ret_dispo = float(np.max(rendements))
    
    # Sélection du R_min par l'utilisateur
    r_min_user = st.sidebar.slider(
        "Rendement annuel minimal souhaité ($r_{min}$)",
        min_value=min_ret_dispo,
        max_value=max_ret_dispo,
        value=(min_ret_dispo + max_ret_dispo) / 2,
        format="%.2f"
    )

    # D. Sélection du Portefeuille Optimal
    # On cherche le premier point où Rendement >= r_min
    valid_indices = np.where(rendements >= r_min_user)[0]
    
    if len(valid_indices) > 0:
        # Comme la frontière est triée (alpha 0->1 correspond généralement à risque min->max),
        # le premier index valide est celui qui minimise le risque pour ce niveau de rendement.
        best_idx = valid_indices[0] 
        
        opt_risk = risques[best_idx]
        opt_ret = rendements[best_idx]
        opt_weights = historiques_poids[best_idx]
        
        # E. Visualisation
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Frontière Efficiente (Pareto)")
            
            # Création du graphique Plotly
            fig = go.Figure()

            # Ligne de la frontière
            fig.add_trace(go.Scatter(
                x=risques, y=rendements,
                mode='lines',
                name='Frontière Efficiente',
                line=dict(color='royalblue', width=3)
            ))

            # Point sélectionné
            fig.add_trace(go.Scatter(
                x=[opt_risk], y=[opt_ret],
                mode='markers',
                name='Portefeuille Sélectionné',
                marker=dict(color='red', size=14, symbol='star'),
                text=[f"Risque: {opt_risk:.2f}<br>Rendement: {opt_ret:.2f}"],
                hoverinfo='text'
            ))
            
            # Nuage de points des actifs individuels (Optionnel)
            vol_actifs = np.sqrt(np.diag(Sigma))
            mu_actifs = mu.values
            fig.add_trace(go.Scatter(
                x=vol_actifs, y=mu_actifs,
                mode='markers',
                name='Actifs Individuels',
                marker=dict(color='lightgray', size=5, opacity=0.5),
                hovertext=asset_names
            ))

            fig.update_layout(
                xaxis_title="Risque (Volatilité Annuelle)",
                yaxis_title="Rendement Espéré Annuel",
                hovermode="closest",
                height=500,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Détails du Portefeuille")
            
            # Affichage des métriques
            col_metrics1, col_metrics2 = st.columns(2)
            with col_metrics1:
                 st.metric(label="Rendement", value=f"{opt_ret:.2%}")
            with col_metrics2:
                 st.metric(label="Risque", value=f"{opt_risk:.2%}")
            
            ratio_sharpe = opt_ret / opt_risk if opt_risk > 0 else 0
            st.metric(label="Ratio de Sharpe", value=f"{ratio_sharpe:.2f}", help="Hypothèse taux sans risque = 0")

            st.markdown("---")
            st.info(f"Ce portefeuille respecte la contrainte : $R \\geq {r_min_user:.2f}$")

        # F. Analyse Sectorielle
        st.subheader("Répartition Sectorielle & Macro-économique")
        
        # Mapping des poids vers les secteurs
        weights_by_sector = map_secteurs(opt_weights, asset_names, secteurs_data)
        
        col_pie, col_table = st.columns([1, 1])
        
        with col_pie:
            if weights_by_sector:
                df_sector = pd.DataFrame(list(weights_by_sector.items()), columns=['Secteur', 'Poids'])
                
                # Graphique en camembert
                fig_pie = px.pie(
                    df_sector, 
                    values='Poids', 
                    names='Secteur', 
                    title='Exposition par Industrie',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.write("Pas de données sectorielles disponibles pour générer le graphique.")

        with col_table:
            st.write("**Top 10 des Actifs**")
            df_assets = pd.DataFrame({'Ticker': asset_names, 'Poids': opt_weights})
            df_assets = df_assets.sort_values(by='Poids', ascending=False).head(10)
            
            # Formattage pour l'affichage
            df_assets['Poids'] = df_assets['Poids'].apply(lambda x: f"{x:.2%}")
            st.table(df_assets.reset_index(drop=True))

    else:
        st.warning("Aucun portefeuille ne satisfait ce niveau de rendement (au-dessus du maximum possible sur la frontière).")
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px

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
# 2. FONCTIONS MATHÉMATIQUES (Issues de votre Notebook)
# ==============================================================================

def calculer_gradient(poids, mu, Sigma, alpha):
    """
    Calcule la direction de la pente (le gradient).
    """
    grad_rendement = -mu
    grad_variance = 2 * np.dot(Sigma, poids)
    return alpha * grad_rendement + (1 - alpha) * grad_variance

def projection_simplex(v):
    """
    Projection exacte sur le simplexe (Somme = 1, x >= 0).
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    indices = np.arange(n) + 1
    cond = u - (cssv - 1) / indices > 0
    if not np.any(cond):
        # Fallback de sécurité si la condition n'est jamais remplie
        return np.ones(n) / n
    rho = indices[cond][-1]
    theta = (cssv[rho - 1] - 1) / rho
    w = np.maximum(v - theta, 0)
    return w

@st.cache_data
def charger_donnees():
    """Charge les fichiers CSV et JSON."""
    # 1. Chargement des rendements
    try:
        df = pd.read_csv("../data/returns_final.csv", index_col="Date", parse_dates=True)
    except FileNotFoundError:
        st.error("Le fichier 'returns_final.csv' est introuvable.")
        return None, None, None

    # 2. Chargement des secteurs
    try:
        with open("../data/tick.json", "r") as f:
            secteurs_dict = json.load(f)
    except FileNotFoundError:
        st.error("Le fichier 'tick.json' est introuvable.")
        secteurs_dict = {}

    return df, secteurs_dict

@st.cache_data
def calculer_frontiere_efficiente(mu, Sigma, asset_names):
    """
    Calcule la frontière efficiente en utilisant la Descente de Gradient avec Warm Start.
    Retourne les risques, rendements et les poids pour chaque point de la frontière.
    """
    nombre_actifs = len(mu)
    alphas = np.linspace(0, 1, 100)  # Discrétisation du paramètre alpha
    
    # Paramètres d'optimisation
    NB_ITERATIONS = 300
    TAUX_APPRENTISSAGE = 0.05
    
    # Listes pour stocker les résultats
    resultats_risque = []
    resultats_rendement = []
    resultats_poids = []

    # Initialisation (1/N)
    poids_courants = np.array([1/nombre_actifs] * nombre_actifs)

    # Barre de progression dans Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, alpha in enumerate(alphas):
        # Mise à jour UI
        if idx % 10 == 0:
            status_text.text(f"Optimisation du point {idx+1}/{len(alphas)}...")
            progress_bar.progress((idx + 1) / len(alphas))

        # Descente de gradient
        for _ in range(NB_ITERATIONS):
            grad = calculer_gradient(poids_courants, mu, Sigma, alpha)
            poids_courants = poids_courants - TAUX_APPRENTISSAGE * grad
            poids_courants = projection_simplex(poids_courants)
        
        # Calcul métriques finales pour ce point
        ret = np.dot(poids_courants, mu)
        risk = np.sqrt(np.dot(poids_courants.T, np.dot(Sigma, poids_courants)))
        
        resultats_rendement.append(ret)
        resultats_risque.append(risk)
        resultats_poids.append(poids_courants.copy())

    progress_bar.empty()
    status_text.empty()
    
    return np.array(resultats_risque), np.array(resultats_rendement), np.array(resultats_poids)

def map_secteurs(weights, asset_names, secteurs_dict):
    """
    Convertit un vecteur de poids (par actif) en poids par secteur.
    """
    # Inversion du dict JSON : Ticker -> Secteur
    ticker_to_sector = {}
    for sector, tickers in secteurs_dict.items():
        for t in tickers:
            ticker_to_sector[t] = sector
            
    sector_weights = {}
    
    for w, asset in zip(weights, asset_names):
        if w > 0.001:  # On ignore les poids négligeables
            sec = ticker_to_sector.get(asset, "Autre / Inconnu")
            sector_weights[sec] = sector_weights.get(sec, 0) + w
            
    return sector_weights

# ==============================================================================
# 3. LOGIQUE PRINCIPALE
# ==============================================================================

# A. Chargement
df_rendements, secteurs_data = charger_donnees()

if df_rendements is not None:
    # Paramètres financiers
    JOURS_DE_BOURSE = 252
    mu = df_rendements.mean() * JOURS_DE_BOURSE
    Sigma = df_rendements.cov() * JOURS_DE_BOURSE
    asset_names = df_rendements.columns.tolist()

    # B. Calcul (Mis en cache pour ne pas recalculer à chaque interaction)
    with st.spinner('Calcul de la frontière efficiente en cours...'):
        risques, rendements, historiques_poids = calculer_frontiere_efficiente(mu, Sigma, asset_names)

    # C. Interface Utilisateur - Sidebar
    st.sidebar.header("Paramètres du Portefeuille")
    
    min_ret = float(np.min(rendements))
    max_ret = float(np.max(rendements))
    
    # Sélection du R_min par l'utilisateur
    r_min_user = st.sidebar.slider(
        "Rendement annuel minimal souhaité ($r_{min}$)",
        min_value=min_ret,
        max_value=max_ret,
        value=(min_ret + max_ret) / 2,
        format="%.2f"
    )

    # D. Sélection du Portefeuille Optimal
    # On cherche le portefeuille sur la frontière qui respecte Return >= r_min
    # Comme la frontière est triée (généralement) par risque croissant pour rendement croissant,
    # le premier point qui satisfait la condition est celui qui minimise le risque (ou le coût).
    
    # Filtrer les indices qui satisfont la contrainte
    valid_indices = np.where(rendements >= r_min_user)[0]
    
    if len(valid_indices) > 0:
        # On prend le point avec le rendement le plus proche (ou le plus petit risque parmi les valides)
        # Ici, comme la frontière monte, le premier index valide est le risque min.
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
                marker=dict(color='red', size=12, symbol='star'),
                text=[f"Risque: {opt_risk:.2f}<br>Rendement: {opt_ret:.2f}"],
                hoverinfo='text'
            ))
            
            # Nuage de points des actifs individuels (Optionnel, pour contexte)
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
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Détails du Portefeuille")
            
            # Affichage des métriques
            st.metric(label="Rendement Espéré", value=f"{opt_ret:.2%}")
            st.metric(label="Risque (Volatilité)", value=f"{opt_risk:.2%}")
            ratio_sharpe = opt_ret / opt_risk if opt_risk > 0 else 0
            st.metric(label="Ratio de Sharpe (sans risque=0)", value=f"{ratio_sharpe:.2f}")

            st.markdown("---")
            st.write(f"**Contrainte :** $R \\geq {r_min_user:.2f}$")

        # F. Analyse Sectorielle
        st.subheader("Répartition Sectorielle & Macro-économique")
        
        # Mapping des poids vers les secteurs
        weights_by_sector = map_secteurs(opt_weights, asset_names, secteurs_data)
        
        if weights_by_sector:
            df_sector = pd.DataFrame(list(weights_by_sector.items()), columns=['Secteur', 'Poids'])
            
            # Graphique en camembert (Pie Chart)
            fig_pie = px.pie(
                df_sector, 
                values='Poids', 
                names='Secteur', 
                title='Exposition par Industrie',
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Tableau des Top Actifs
            with st.expander("Voir le détail des actifs individuels (Top 10)"):
                df_assets = pd.DataFrame({'Ticker': asset_names, 'Poids': opt_weights})
                df_assets = df_assets.sort_values(by='Poids', ascending=False).head(10)
                df_assets['Poids'] = df_assets['Poids'].apply(lambda x: f"{x:.2%}")
                st.table(df_assets)
        else:
            st.warning("Impossible de mapper les secteurs (vérifiez la correspondance entre CSV et JSON).")

    else:
        st.warning("Aucun portefeuille ne satisfait ce niveau de rendement (au-dessus du maximum possible).")
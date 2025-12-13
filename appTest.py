import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
import os

# ==============================================================================
# 0. IMPORT DES MODULES (Gestion des erreurs)
# ==============================================================================

# 1. MARKOWITZ (Niveau 1)
try:
    import src.partie1_markowitz as p1
except ImportError:
    st.error("Le module 'partie1_markowitz.py' est introuvable.")

# 2. MONTE CARLO (Niveau 2 - Approche Naïve)
try:
    import src.partie2_montecarlo as p2_mc
except ImportError:
    st.warning("Le fichier 'partie2_montecarlo.py' est introuvable. La méthode Monte Carlo sera désactivée.")

# 3. NSGA-II (Niveau 2 - Approche Métaheuristique)
try:
    import py.nsga2_optimizer as nsga2
except ImportError:
    # Fallback si le fichier est à la racine
    try:
        import nsga2_optimizer as nsga2
    except ImportError:
        st.warning("Le fichier 'nsga2_optimizer.py' est introuvable. La méthode NSGA-II sera désactivée.")

# ==============================================================================
# 1. CONFIGURATION DE LA PAGE
# ==============================================================================
st.set_page_config(
    page_title="Optimisation de Portefeuille - Comparaison",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Optimisation de Portefeuille & Front de Pareto")
st.markdown("""
Cette application compare différentes approches d'optimisation :
1.  **Markowitz (Niveau 1)** : Optimisation convexe classique (Rendement / Risque).
2.  **Multi-Critère (Niveau 2)** : Gestion de la cardinalité et des coûts via **Monte Carlo** ou **NSGA-II**.
""")

# Sélection du niveau principal
niveau = st.sidebar.selectbox(
    "Choisir le module",
    ["Niveau 1 : Markowitz Classique", "Niveau 2 : Multi-Critère (K & Coûts)"],
    index=0
)

# ==============================================================================
# 2. CHARGEMENT DES DONNÉES
# ==============================================================================

@st.cache_data
def charger_donnees_app():
    """Charge les données financières et sectorielles."""
    # Chemins possibles
    csv_candidates = ["returns_final.csv", "data/returns_final.csv", "../data/returns_final.csv"]
    json_candidates = ["tick.json", "data/tick.json", "../data/tick.json"]
    
    csv_path = None
    for path in csv_candidates:
        if os.path.exists(path):
            csv_path = path
            break
            
    if csv_path is None:
        st.error(f"Fichier CSV introuvable. Cherché dans : {csv_candidates}")
        return None, None, None, None, None

    # Chargement via partie1
    try:
        mu, Sigma, asset_names, df = p1.charger_et_preparer_donnees(csv_path)
    except Exception as e:
        st.error(f"Erreur de chargement des données : {e}")
        return None, None, None, None, None

    # Chargement JSON (Optionnel)
    secteurs_dict = {}
    for path in json_candidates:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    secteurs_dict = json.load(f)
                break
            except:
                pass

    return df, mu, Sigma, asset_names, secteurs_dict

def map_secteurs(weights, asset_names, secteurs_dict):
    """Agrège les poids par secteur pour le graphique camembert."""
    if not secteurs_dict:
        return {}
    
    ticker_to_sector = {}
    for sector, tickers in secteurs_dict.items():
        for t in tickers:
            ticker_to_sector[t] = sector
            
    sector_weights = {}
    for w, asset in zip(weights, asset_names):
        if w > 0.001: 
            sec = ticker_to_sector.get(asset, "Autre / Inconnu")
            sector_weights[sec] = sector_weights.get(sec, 0) + w
            
    return sector_weights

# Chargement
df_rendements, mu, Sigma, asset_names, secteurs_data = charger_donnees_app()

# ==============================================================================
# 3. MOTEUR DE L'APPLICATION
# ==============================================================================

if df_rendements is not None:

    # --------------------------------------------------------------------------
    # NIVEAU 1 : MARKOWITZ
    # --------------------------------------------------------------------------
    if "Niveau 1" in niveau:
        st.sidebar.header("Paramètres Markowitz")
        
        @st.cache_data
        def run_markowitz(mu_in, Sigma_in):
            return p1.calculer_frontiere_efficiente(mu_in, Sigma_in, nb_points=50, nb_iterations=300)

        with st.spinner('Optimisation par Descente de Gradient...'):
            risques, rendements, historiques_poids = run_markowitz(mu, Sigma)

        # Slider Rendement Cible
        min_r, max_r = float(min(rendements)), float(max(rendements))
        r_cible = st.sidebar.slider("Rendement Cible Annuel", min_r, max_r, (min_r+max_r)/2, format="%.2f")

        # Sélection
        idx_valid = np.where(rendements >= r_cible)[0]
        if len(idx_valid) > 0:
            best_idx = idx_valid[0] # Premier élément = risque minimal car trié
            
            opt_ret = rendements[best_idx]
            opt_risk = risques[best_idx]
            opt_weights = historiques_poids[best_idx]

            # Affichage Graphique 2D
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Frontière Efficiente")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=risques, y=rendements, mode='lines', name='Frontière', line=dict(color='blue', width=3)))
                fig.add_trace(go.Scatter(x=[opt_risk], y=[opt_ret], mode='markers', name='Sélection', marker=dict(color='red', size=15, symbol='star')))
                fig.update_layout(xaxis_title="Risque (Volatilité)", yaxis_title="Rendement Espéré", height=500)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Performance")
                st.metric("Rendement", f"{opt_ret:.2%}")
                st.metric("Risque", f"{opt_risk:.2%}")
                st.metric("Sharpe", f"{(opt_ret/opt_risk):.2f}")

            # Secteurs
            st.divider()
            c_pie, c_tab = st.columns(2)
            with c_pie:
                if secteurs_data:
                    w_sec = map_secteurs(opt_weights, asset_names, secteurs_data)
                    df_sec = pd.DataFrame(list(w_sec.items()), columns=['Secteur', 'Poids'])
                    fig_pie = px.pie(df_sec, values='Poids', names='Secteur', title='Allocation Sectorielle', hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
            with c_tab:
                st.write("**Top 10 Actifs**")
                df_w = pd.DataFrame({'Ticker': asset_names, 'Poids': opt_weights})
                df_w = df_w.sort_values('Poids', ascending=False).head(10)
                df_w['Poids'] = df_w['Poids'].apply(lambda x: f"{x:.2%}")
                st.dataframe(df_w, use_container_width=True, hide_index=True)
        else:
            st.warning("Rendement cible inatteignable.")

    # --------------------------------------------------------------------------
    # NIVEAU 2 : MULTI-CRITÈRE (Monte Carlo & NSGA-II)
    # --------------------------------------------------------------------------
    elif "Niveau 2" in niveau:
        
        # Choix de la méthode
        methode_n2 = st.sidebar.radio("Méthode de Résolution", ["Monte Carlo (Aléatoire)", "NSGA-II (Génétique)"])
        
        st.sidebar.divider()
        st.sidebar.header("Contraintes & Objectifs")
        
        # Paramètres communs
        K = st.sidebar.slider("Cardinalité (Nb Actifs)", 2, 20, 10)
        c_prop = st.sidebar.slider("Coûts de Transaction (%)", 0.0, 2.0, 0.5, 0.1) / 100.0
        
        # Portefeuille de départ (Cash ou 1/N) pour calculer les coûts
        #w_current = np.ones(len(asset_names)) / len(asset_names)

        # Avant (C'était uniforme 1/N -> Coûts constants) :
        # w_current = np.ones(len(asset_names)) / len(asset_names)

        # Après (On simule qu'on détient déjà 100% du premier actif -> Coûts variables) :
        #w_current = np.zeros(len(asset_names))
        #w_current[0] = 1.0

        st.sidebar.divider()
        st.sidebar.subheader("3. Portefeuille Actuel (w_current)")
        
        # Initialiser un portefeuille aléatoire fixe dans la session si n'existe pas
        if 'w_current_fixe' not in st.session_state:
            # On génère un portefeuille valide (somme = 1)
            w_rnd = np.random.random(len(asset_names))
            w_rnd = w_rnd / np.sum(w_rnd)
            st.session_state['w_current_fixe'] = w_rnd

        # Bouton pour changer de scénario (Générer un nouveau portefeuille de départ)
        if st.sidebar.button("🎲 Générer un nouveau portefeuille actuel"):
            w_rnd = np.random.random(len(asset_names))
            w_rnd = w_rnd / np.sum(w_rnd)
            st.session_state['w_current_fixe'] = w_rnd
            st.rerun() # Force le rafraichissement immédiat

        # Récupération
        w_current = st.session_state['w_current_fixe']

        # Affichage visuel pour confirmer qu'il n'est pas vide
        st.sidebar.info(f"Portefeuille de départ défini.\nInvesti sur {np.sum(w_current > 0.01)} actifs.")
        
        # Visualisation optionnelle de w_current (petit expanander)
        with st.sidebar.expander("Voir composition w_current"):
             df_curr = pd.DataFrame({"Actif": asset_names, "Poids": w_current})
             st.dataframe(df_curr.sort_values("Poids", ascending=False).head(5), hide_index=True)


        # ---------------- A. MONTE CARLO ----------------
        if "Monte Carlo" in methode_n2:
            nb_sims = st.sidebar.number_input("Nombre de simulations", 1000, 100000, 20000, step=5000)
            
            if st.sidebar.button("Lancer Simulation"):
                with st.spinner(f"Génération de {nb_sims} portefeuilles aléatoires..."):
                    # results est supposé être un array (3, N) : [Risque, Rendement, Coûts]
                    results_mc, weights_mc = p2_mc.simulation_monte_carlo(
                        mu=mu.values,
                        Sigma=Sigma.values,
                        w_initial=w_current,
                        K=K,
                        c_prop=c_prop,
                        nb_simulations=nb_sims
                    )
                    
                    # Sauvegarde Session
                    st.session_state['mc_results'] = results_mc
                    st.session_state['mc_weights'] = weights_mc
                    st.session_state['mc_params'] = {'K': K, 'c': c_prop}
                    st.success("Simulation terminée !")

            # Affichage Résultats MC
            if 'mc_results' in st.session_state:
                res_mc = st.session_state['mc_results']
                w_mc_list = st.session_state['mc_weights']
                
                # Extraction
                risks = res_mc[0]
                returns = res_mc[1]
                costs = res_mc[2]
                
                st.subheader("Nuage de Points Monte Carlo (3D)")
                
                # Visualisation 3D
                fig_3d = go.Figure(data=[go.Scatter3d(
                    x=risks, y=returns, z=costs,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=returns,
                        colorscale='Jet',
                        opacity=0.6
                    ),
                    hovertemplate='Risque: %{x:.2%}<br>Rendement: %{y:.2%}<br>Coût: %{z:.2%}'
                )])
                fig_3d.update_layout(
                    scene=dict(xaxis_title='Risque', yaxis_title='Rendement', zaxis_title='Coûts', aspectmode='cube'),
                    height=600, margin=dict(l=0, r=0, b=0, t=0)
                )
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # --- Filtrage (Sélection du meilleur parmi le nuage) ---
                st.divider()
                st.subheader("Sélection du Portefeuille")
                
                min_r_mc, max_r_mc = float(returns.min()), float(returns.max())
                target_r = st.slider("Rendement Minimal", min_r_mc, max_r_mc, (min_r_mc+max_r_mc)/2)
                
                # Trouver les indices qui respectent le rendement min
                valid_indices = np.where(returns >= target_r)[0]
                
                if len(valid_indices) > 0:
                    # Parmi ceux valides, on prend celui qui minimise le risque
                    # (On pourrait aussi minimiser le ratio risque/coût, mais restons simple)
                    sub_risks = risks[valid_indices]
                    best_sub_idx = np.argmin(sub_risks)
                    final_idx = valid_indices[best_sub_idx]
                    
                    best_w = w_mc_list[final_idx]
                    best_r = returns[final_idx]
                    best_risk = risks[final_idx]
                    best_cost = costs[final_idx]
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Rendement", f"{best_r:.2%}")
                    c2.metric("Risque", f"{best_risk:.2%}")
                    c3.metric("Coûts", f"{best_cost:.2%}")
                    
                    # Affichage composition
                    st.write(f"**Composition (Top {K})**")
                    df_w = pd.DataFrame({'Actif': asset_names, 'Poids': best_w})
                    df_w = df_w.sort_values('Poids', ascending=False).head(K)
                    df_w['Poids'] = df_w['Poids'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(df_w, hide_index=True)
                    
                    # Graphique Sectoriel (si dispo)
                    if secteurs_data:
                        w_sec = map_secteurs(best_w, asset_names, secteurs_data)
                        fig_p = px.pie(names=list(w_sec.keys()), values=list(w_sec.values()), title="Répartition Sectorielle")
                        st.plotly_chart(fig_p, use_container_width=True)
                else:
                    st.warning("Aucun portefeuille trouvé pour ce rendement cible.")

        # ---------------- B. NSGA-II ----------------
        elif "NSGA-II" in methode_n2:
            pop_size = st.sidebar.number_input("Taille Population", 50, 500, 100, step=50)
            n_gen = st.sidebar.number_input("Générations", 50, 1000, 200, step=50)

            if st.sidebar.button("Lancer Optimisation NSGA-II"):
                with st.spinner("Evolution génétique en cours..."):
                    try:
                        res = nsga2.optimize_portfolio_nsga2(
                            mu=mu.values, Sigma=Sigma.values, w_current=w_current,
                            K=K, pop_size=pop_size, n_gen=n_gen, c_prop=c_prop
                        )
                        st.session_state['nsga_res'] = res
                        st.session_state['nsga_params'] = {'K': K, 'c': c_prop}
                        st.success("Optimisation terminée !")
                    except Exception as e:
                        st.error(f"Erreur NSGA-II : {e}")

            if 'nsga_res' in st.session_state:
                res = st.session_state['nsga_res']
                params = st.session_state['nsga_params']
                
                # Extraction (pymoo: F1 minimized (-Ret), F2 minimized (Var), F3 minimized (Cost))
                returns_n = -res.F[:, 0]
                risks_n = np.sqrt(res.F[:, 1])
                costs_n = res.F[:, 2]

                st.subheader(f"Front de Pareto NSGA-II (K={params['K']})")
                
                fig_3d = go.Figure(data=[go.Scatter3d(
                    x=risks_n, y=returns_n, z=costs_n,
                    mode='markers',
                    marker=dict(size=6, color=returns_n, colorscale='Viridis', showscale=True),
                    hovertemplate='Risque: %{x:.2%}<br>Rendement: %{y:.2%}<br>Coût: %{z:.2%}'
                )])
                fig_3d.update_layout(scene=dict(xaxis_title='Risque', yaxis_title='Rendement', zaxis_title='Coûts', aspectmode='cube'), height=600)
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Filtrage NSGA-II
                st.divider()
                min_r_n, max_r_n = float(returns_n.min()), float(returns_n.max())
                r_min_user = st.slider("Rendement Minimal", min_r_n, max_r_n, (min_r_n+max_r_n)/2)
                
                filtered_X, filtered_F, indices = nsga2.filter_pareto_by_min_return(res, r_min_user)
                
                if filtered_X is not None:
                    best_w, best_r, best_risk, best_cost = nsga2.select_min_risk_portfolio(filtered_X, filtered_F)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Rendement", f"{best_r:.2%}")
                    c2.metric("Risque", f"{best_risk:.2%}")
                    c3.metric("Coûts", f"{best_cost:.2%}")
                    
                    df_top = nsga2.get_top_assets(best_w, asset_names, top_n=params['K'])
                    df_top['Weight'] = df_top['Weight'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(df_top, hide_index=True)
                else:
                    st.warning("Aucun portefeuille trouvé.")
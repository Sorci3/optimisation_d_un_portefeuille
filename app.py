import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
import os

# --- IMPORTS LOGIQUES ---
# On essaie d'importer selon la structure probable (src ou racine)
try:
    import src.partie1_markowitz as p1
except ImportError:
    try:
        import partie1 as p1
    except ImportError:
        st.error("Le module 'partie1' ou 'src.partie1_markowitz' est introuvable.")

# Import du nouveau moteur NSGA-II (pymoo)
try:
    import py.nsga2_optimizer as nsga2
except ImportError:
    st.error("Le fichier 'nsga2_optimizer.py' est introuvable. Assurez-vous qu'il est dans le même dossier.")

# ==============================================================================
# 1. CONFIGURATION DE LA PAGE
# ==============================================================================
st.set_page_config(
    page_title="Optimisation de Portefeuille - Markowitz & NSGA-II",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Optimisation de Portefeuille & Front de Pareto")
st.markdown("""
Cette application compare deux approches d'optimisation :
1.  **Markowitz Classique (Niveau 1)** : Optimisation quadratique (Rendement / Risque) sans contraintes complexes.
2.  **NSGA-II (Niveau 2)** : Algorithme génétique gérant la **cardinalité** (nombre d'actifs) et les **coûts de transaction**.
""")

# Sélection du niveau
niveau = st.sidebar.selectbox(
    "Choisir le module d'optimisation",
    ["Niveau 1 : Markowitz Classique (Gradient)", "Niveau 2 : Cardinalité & Coûts (NSGA-II)"],
    index=0
)

# ==============================================================================
# 2. CHARGEMENT DES DONNÉES
# ==============================================================================

@st.cache_data
def charger_donnees_app():
    """Charge les données financières et sectorielles."""
    # Liste de chemins possibles pour gérer le déploiement local/cloud
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
    """Agrège les poids par secteur."""
    if not secteurs_dict:
        return {}
    
    # Inversion : Ticker -> Secteur
    ticker_to_sector = {}
    for sector, tickers in secteurs_dict.items():
        for t in tickers:
            ticker_to_sector[t] = sector
            
    sector_weights = {}
    for w, asset in zip(weights, asset_names):
        if w > 0.001: # Filtre petits poids
            sec = ticker_to_sector.get(asset, "Autre / Inconnu")
            sector_weights[sec] = sector_weights.get(sec, 0) + w
            
    return sector_weights

# ==============================================================================
# 3. MOTEUR DE L'APPLICATION
# ==============================================================================

df_rendements, mu, Sigma, asset_names, secteurs_data = charger_donnees_app()

if df_rendements is not None:

    # --------------------------------------------------------------------------
    # NIVEAU 1 : MARKOWITZ (Code existant inchangé)
    # --------------------------------------------------------------------------
    if "Niveau 1" in niveau:
        st.sidebar.header("Paramètres Markowitz")
        
        # Calcul (mis en cache)
        @st.cache_data
        def run_markowitz(mu_in, Sigma_in):
            return p1.calculer_frontiere_efficiente(mu_in, Sigma_in, nb_points=50, nb_iterations=300)

        with st.spinner('Optimisation par Descente de Gradient en cours...'):
            risques, rendements, historiques_poids = run_markowitz(mu, Sigma)

        # Slider Rendement Cible
        min_r, max_r = float(min(rendements)), float(max(rendements))
        r_cible = st.sidebar.slider("Rendement Cible Annuel", min_r, max_r, (min_r+max_r)/2, format="%.2f")

        # Sélection du meilleur portefeuille pour ce rendement
        # On cherche l'index où le rendement est >= cible avec le risque min
        idx_valid = np.where(rendements >= r_cible)[0]
        if len(idx_valid) > 0:
            best_idx = idx_valid[0] # Comme c'est trié par le gradient, le premier est souvent le moins risqué pour ce niveau
            
            opt_ret = rendements[best_idx]
            opt_risk = risques[best_idx]
            opt_weights = historiques_poids[best_idx]

            # Affichage
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
                st.metric("Ratio de Sharpe", f"{(opt_ret/opt_risk):.2f}")

            # Secteurs et Actifs
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
    # NIVEAU 2 : NSGA-II (Intégration de votre module)
    # --------------------------------------------------------------------------
    elif "Niveau 2" in niveau:
        st.sidebar.header("Paramètres NSGA-II")
        
        # Paramètres utilisateur
        K = st.sidebar.slider("Cardinalité (Nombre d'actifs K)", 2, 20, 10)
        c_prop = st.sidebar.slider("Coûts de Transaction (%)", 0.0, 2.0, 0.5, 0.1) / 100.0
        
        with st.sidebar.expander("Paramètres Avancés"):
            pop_size = st.number_input("Taille Population", 50, 500, 100, step=50)
            n_gen = st.number_input("Nombre Générations", 50, 1000, 200, step=50)

        # Bouton d'exécution
        if st.sidebar.button("Lancer l'Optimisation"):
            with st.spinner("Optimisation Multi-Objectifs (NSGA-II) en cours..."):
                # Portefeuille initial (Cash ou 1/N pour calculer les coûts de réallocation)
                w_current = np.ones(len(asset_names)) / len(asset_names)
                
                try:
                    # Appel à votre module nsga2_optimizer
                    res = nsga2.optimize_portfolio_nsga2(
                        mu=mu.values,
                        Sigma=Sigma.values,
                        w_current=w_current,
                        K=K,
                        pop_size=pop_size,
                        n_gen=n_gen,
                        c_prop=c_prop,
                        verbose=False
                    )
                    
                    # Sauvegarde dans la session pour ne pas recalculer si on bouge un slider
                    st.session_state['n2_res'] = res
                    st.session_state['n2_params'] = {'K': K, 'c': c_prop}
                    st.success("Optimisation terminée !")
                    
                except Exception as e:
                    st.error(f"Erreur durant l'optimisation : {e}")

        # Affichage des résultats
        if 'n2_res' in st.session_state:
            res = st.session_state['n2_res']
            params = st.session_state['n2_params']
            
            # Extraction (pymoo renvoie F pour les objectifs)
            # F1 = -Rendement, F2 = Variance (Risque), F3 = Coûts
            returns = -res.F[:, 0]
            risks = np.sqrt(res.F[:, 1]) # On passe en volatilité pour l'affichage
            costs = res.F[:, 2]
            
            # --- VISUALISATION 3D CORRIGÉE ---
            st.subheader(f"Front de Pareto 3D (K={params['K']}, Coûts={params['c']*100:.1f}%)")
            
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=risks,    # X: Risque
                y=returns,  # Y: Rendement
                z=costs,    # Z: Coûts
                mode='markers',
                marker=dict(
                    size=6,
                    color=returns,          # Couleur = Rendement
                    colorscale='Viridis',
                    opacity=0.9,
                    showscale=True,
                    colorbar=dict(title="Rendement")
                ),
                hovertemplate='<b>Risque</b>: %{x:.2%}<br><b>Rendement</b>: %{y:.2%}<br><b>Coût</b>: %{z:.2%}<extra></extra>'
            )])

            # C'EST ICI QUE LA MAGIE OPÈRE : aspectmode='cube'
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='Risque (Volatilité)',
                    yaxis_title='Rendement',
                    zaxis_title='Coûts Transaction',
                    # 'cube' force les 3 axes à avoir la même longueur visuelle à l'écran,
                    # peu importe si les coûts sont 0.01 et le rendement 0.20.
                    aspectmode='cube' 
                ),
                height=700,
                margin=dict(l=0, r=0, b=0, t=0)
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            
            st.divider()
            
            # --- SÉLECTION ET ANALYSE ---
            col_sel, col_res = st.columns([1, 2])
            
            with col_sel:
                st.subheader("Filtrage")
                min_r, max_r = float(returns.min()), float(returns.max())
                
                if min_r < max_r:
                    r_min_user = st.slider(
                        "Rendement Minimal Souhaité", 
                        min_r, max_r, (min_r+max_r)/2, 
                        format="%.3f"
                    )
                else:
                    st.warning("Un seul niveau de rendement trouvé.")
                    r_min_user = min_r

            # Utilisation des fonctions utilitaires de votre fichier nsga2_optimizer.py
            filtered_X, filtered_F, indices = nsga2.filter_pareto_by_min_return(res, r_min_user)
            
            with col_res:
                if filtered_X is not None:
                    # Sélection du meilleur (celui avec le risque min parmi ceux filtrés)
                    best_w, best_r, best_risk, best_cost = nsga2.select_min_risk_portfolio(filtered_X, filtered_F)
                    
                    st.success("✅ Portefeuille Retenu (Min Risque pour R > R_min)")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Rendement", f"{best_r:.2%}")
                    k2.metric("Risque", f"{best_risk:.2%}")
                    k3.metric("Coûts", f"{best_cost:.2%}")
                    # Ratio Sharpe (Hypothèse taux sans risque = 0)
                    k4.metric("Sharpe", f"{(best_r/best_risk):.2f}")
                    
                    # Composition
                    st.write("---")
                    st.write(f"**Composition du Portefeuille (Top {params['K']} actifs)**")
                    
                    # Utilisation de votre fonction get_top_assets
                    df_top = nsga2.get_top_assets(best_w, asset_names, top_n=params['K'])
                    
                    # Petit nettoyage pour l'affichage
                    df_display = df_top.copy()
                    df_display.columns = ["Actif", "Poids"]
                    df_display["Poids"] = df_display["Poids"].apply(lambda x: f"{x:.2%}")
                    
                    # Affichage graphique de la composition
                    c_tab, c_chart = st.columns(2)
                    with c_tab:
                        st.dataframe(df_display, use_container_width=True, hide_index=True)
                    with c_chart:
                        # Si on a les données sectorielles
                        if secteurs_data:
                            w_sec = map_secteurs(best_w, asset_names, secteurs_data)
                            df_sec = pd.DataFrame(list(w_sec.items()), columns=['Secteur', 'Poids'])
                            fig_p = px.pie(df_sec, values='Poids', names='Secteur', hole=0.3)
                            fig_p.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
                            st.plotly_chart(fig_p, use_container_width=True)
                        else:
                            st.bar_chart(df_top.set_index('Asset')['Weight'])

                else:
                    st.warning("Aucun portefeuille ne satisfait ce critère de rendement.")

            # Stats globales (Debug / Info)
            with st.expander("Voir les statistiques globales de l'optimisation"):
                stats = nsga2.get_portfolio_statistics(res)
                st.write(stats)
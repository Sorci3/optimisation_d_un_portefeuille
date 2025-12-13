import numpy as np

# ==============================================================================
# GESTION DES IMPORTS
# ==============================================================================
try:
    import partie2_constraint as cs
except ImportError:
    try:
        import src.partie2_constraint as cs
    except ImportError:
        raise ImportError(
            "CRITIQUE : Le fichier 'partie2_constraint.py' est introuvable. "
            "Assurez-vous qu'il est dans le même dossier que 'partie2_montecarlo.py'."
        )

# ==============================================================================
# SIMULATION MONTE CARLO
# ==============================================================================

def simulation_monte_carlo(mu, Sigma, w_initial, K=10, c_prop=0.005, nb_simulations=50000):
    """
    Exécute une simulation Monte Carlo pour générer des portefeuilles aléatoires
    respectant une contrainte de cardinalité.
    """
    n_assets = len(mu)
    results = np.zeros((3, nb_simulations)) 
    weights_record = []

    for i in range(nb_simulations):
        w = np.random.random(n_assets)
        w /= np.sum(w)
        w = cs.imposer_cardinalite(w, K) 
        ret = np.dot(w, mu)
        risk = np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
        cost = cs.calculer_couts_transaction(w, w_initial, c_prop)
        
        results[0,i] = risk
        results[1,i] = ret
        results[2,i] = cost
        weights_record.append(w)
        
    return results, weights_record
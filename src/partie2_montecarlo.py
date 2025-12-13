import numpy as np

# ==============================================================================
# GESTION DES IMPORTS
# ==============================================================================
# On tente d'importer le module de contraintes. 
# Si cela échoue, on essaie de regarder dans un dossier 'src' (cas courant).
try:
    import partie2_constraint as cs
except ImportError:
    try:
        import src.partie2_constraint as cs
    except ImportError:
        # Si on ne le trouve toujours pas, on arrête tout avec une erreur claire
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
    results = np.zeros((3, nb_simulations)) # 0: Risk, 1: Return, 2: Costs
    weights_record = []

    # Pour éviter de recalculer np.sum(np.abs(...)) à chaque boucle, on peut optimiser,
    # mais gardons le code lisible pour l'instant.

    for i in range(nb_simulations):
        # 1. Génération aléatoire
        w = np.random.random(n_assets)
        w /= np.sum(w)
        
        # 2. Application Cardinalité (Top K) via le module importé 'cs'
        w = cs.imposer_cardinalite(w, K) 
        
        # 3. Calculs Métriques
        # Rendement attendu
        ret = np.dot(w, mu)
        
        # Risque (Volatilité)
        risk = np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
        
        # Coûts de transaction (Hypothèse: w_initial est le portefeuille actuel)
        # On utilise la fonction du module cs, ou on le fait directement ici.
        # Utilisons le module pour la cohérence :
        cost = cs.calculer_couts_transaction(w, w_initial, c_prop)
        
        # Stockage
        results[0,i] = risk
        results[1,i] = ret
        results[2,i] = cost
        weights_record.append(w)
        
    return results, weights_record
    return results, weights_record
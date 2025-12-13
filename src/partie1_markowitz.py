import pandas as pd
import numpy as np

# ==============================================================================
# 1. GESTION DES DONNÉES
# ==============================================================================

def charger_et_preparer_donnees(chemin_fichier, jours_bourse=252):
    """
    Charge le CSV des rendements et calcule mu (moyenne) et Sigma (covariance).
    
    Args:
        chemin_fichier (str): Le chemin vers le fichier CSV.
        jours_bourse (int): Facteur d'annualisation (défaut 252).
        
    Returns:
        tuple: (mu, Sigma, asset_names, df_rendements)
    """
    try:
        df = pd.read_csv(chemin_fichier, index_col="Date", parse_dates=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Le fichier {chemin_fichier} est introuvable.")

    # Calculs annuels
    mu = df.mean() * jours_bourse
    Sigma = df.cov() * jours_bourse
    asset_names = df.columns.tolist()
    
    return mu, Sigma, asset_names, df

# ==============================================================================
# 2. FONCTIONS MATHÉMATIQUES (Outils)
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
    Algorithme de Duchi et al. (2008).
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    indices = np.arange(n) + 1
    cond = u - (cssv - 1) / indices > 0
    
    if not np.any(cond):
        return np.ones(n) / n  
        
    rho = indices[cond][-1]
    theta = (cssv[rho - 1] - 1) / rho
    w = np.maximum(v - theta, 0)
    return w

# ==============================================================================
# 3. ALGORITHME D'OPTIMISATION (Warm Start)
# ==============================================================================

def calculer_frontiere_efficiente(mu, Sigma, nb_points=100, nb_iterations=300, lr=0.05):
    """
    Exécute l'optimisation par descente de gradient avec 'Warm Start'.
    
    Args:
        mu (Series/array): Rendements espérés.
        Sigma (DataFrame/array): Matrice de covariance.
        nb_points (int): Nombre de points sur la frontière (résolution).
        nb_iterations (int): Itérations de gradient par point.
        lr (float): Taux d'apprentissage (Learning Rate).
        
    Returns:
        tuple: (risques, rendements, poids_historique) sous forme de numpy arrays.
    """
    nombre_actifs = len(mu)
    alphas = np.linspace(0, 1, nb_points)
    
    resultats_risque = []
    resultats_rendement = []
    resultats_poids = []

    poids_courants = np.array([1/nombre_actifs] * nombre_actifs)

    for alpha in alphas:
        for _ in range(nb_iterations):
            grad = calculer_gradient(poids_courants, mu, Sigma, alpha)
            poids_courants = poids_courants - lr * grad
            poids_courants = projection_simplex(poids_courants)
        
        ret = np.dot(poids_courants, mu)
        risk = np.sqrt(np.dot(poids_courants.T, np.dot(Sigma, poids_courants)))
        
        resultats_rendement.append(ret)
        resultats_risque.append(risk)
        resultats_poids.append(poids_courants.copy())
    
    return np.array(resultats_risque), np.array(resultats_rendement), np.array(resultats_poids)

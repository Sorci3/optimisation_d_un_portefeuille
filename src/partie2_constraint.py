import numpy as np

def calculer_couts_transaction(w_new, w_old, c_prop=0.005):
    """
    Calcule f3(w) : Coûts de transaction.
    w_new : Nouveaux poids proposés
    w_old : Poids actuels (avant réallocation)
    c_prop : Coût unitaire (ex: 0.5%)
    """
    # Formule : somme(c * |w_i - w_t_i|)
    diff_absolue = np.abs(w_new - w_old)
    return np.sum(diff_absolue) * c_prop

def imposer_cardinalite(poids, K):
    """
    Garde seulement les K plus grandes valeurs et remet à l'échelle.
    """
    poids_significatifs = poids.copy()
    
    # Trouver les indices des (N-K) plus petites valeurs
    indices_a_zeroter = np.argsort(poids_significatifs)[:-K]
    
    # Mettre à zéro
    poids_significatifs[indices_a_zeroter] = 0.0
    
    # Re-normaliser pour que la somme fasse 1
    somme = np.sum(poids_significatifs)
    if somme > 0:
        poids_significatifs /= somme
    else:
        # Fallback si tout est à 0 (improbable)
        poids_significatifs[-K:] = 1.0 / K
        
    return poids_significatifs
import numpy as np
import time
from functools import wraps
from pymoo.indicators.hv import HV

def measure_execution_time(func):
    """
    Décorateur pour mesurer le temps d'exécution d'une fonction

    Usage:
        @measure_execution_time
        def my_optimization():
            ...

        result, exec_time = my_optimization()
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        return result, execution_time

    return wrapper


def calculate_hypervolume(pareto_front, reference_point=None):
    """
    Calcule l'hypervolume. Robuste aux dimensions et aux échelles.
    Utilise pymoo obligatoirement pour > 2 dimensions.
    """
    # Si aucun point de ref n'est donné, on prend le Nadir + 10%
    if reference_point is None:
        nadir = np.max(pareto_front, axis=0)
        # Marge de sécurité (gestion des valeurs nulles/négatives)
        delta = np.abs(nadir) * 0.1
        delta[delta < 1e-6] = 1e-6 # Évite d'ajouter 0
        reference_point = nadir + delta

    try:
        ind = HV(ref_point=reference_point)
        hv = ind(pareto_front)
        return hv
    except Exception as e:
        print(f"Erreur calcul HV (pymoo): {e}")
        return 0.0


def _hypervolume_2d_simple(front, ref_point):
    """Approximation simple de l'hypervolume en 2D (Markowitz)"""
    # Trier par premier objectif
    sorted_indices = np.argsort(front[:, 0])
    sorted_front = front[sorted_indices]

    hv = 0.0
    prev_x = 0.0

    for point in sorted_front:
        if point[0] >= ref_point[0] or point[1] >= ref_point[1]:
            continue

        width = ref_point[0] - point[0]
        height = ref_point[1] - prev_x
        hv += width * height
        prev_x = point[1]

    return hv







def calculate_sharpe_ratios(returns, risks, risk_free_rate=0.0):
    """
    Calcule les ratios de Sharpe pour un ensemble de portefeuilles

    Args:
        returns: array des rendements
        risks: array des risques (volatilités)
        risk_free_rate: taux sans risque (défaut: 0)

    Returns:
        sharpe_ratios: array des ratios de Sharpe
    """
    # Éviter division par zéro
    risks_safe = np.where(risks > 1e-10, risks, 1e-10)
    sharpe = (returns - risk_free_rate) / risks_safe
    return sharpe


def calculate_front_sharpe_metrics(returns, risks, risk_free_rate=0.0):
    """
    Calcule les statistiques du ratio de Sharpe pour un front de Pareto

    Args:
        returns: array des rendements du front
        risks: array des risques du front
        risk_free_rate: taux sans risque

    Returns:
        dict: {
            'mean_sharpe': float,
            'max_sharpe': float,
            'min_sharpe': float,
            'std_sharpe': float,
            'median_sharpe': float
        }
    """
    sharpe_ratios = calculate_sharpe_ratios(returns, risks, risk_free_rate)

    # Filtrer les valeurs infinies/NaN
    valid_sharpe = sharpe_ratios[np.isfinite(sharpe_ratios)]

    if len(valid_sharpe) == 0:
        return {
            'mean_sharpe': 0.0,
            'max_sharpe': 0.0,
            'min_sharpe': 0.0,
            'std_sharpe': 0.0,
            'median_sharpe': 0.0
        }

    return {
        'mean_sharpe': np.mean(valid_sharpe),
        'max_sharpe': np.max(valid_sharpe),
        'min_sharpe': np.min(valid_sharpe),
        'std_sharpe': np.std(valid_sharpe),
        'median_sharpe': np.median(valid_sharpe)
    }


def evaluate_optimization_quality(
        pareto_front_objectives,
        portfolio_weights,
        execution_time,
        K=None,
        risk_free_rate=0.0,
        reference_point=None
):
    """
    Évalue la qualité avec normalisation pour l'Hypervolume.
    """
    # 1. Extraction des données
    # Conversion explicite en float64 pour éviter les erreurs de précision
    front = np.array(pareto_front_objectives, dtype=np.float64)

    # 2. Calcul des métriques financières (Sharpe)
    returns = -front[:, 0]  # On remet le rendement en positif
    risks = np.sqrt(front[:, 1])  # Racine de la variance pour avoir l'écart-type
    sharpe_metrics = calculate_front_sharpe_metrics(returns, risks, risk_free_rate)

    # 3. Calcul de l'Hypervolume NORMALISÉ
    # L'HV brut est minuscule (1e-8), on normalise le front entre 0 et 1 pour avoir un score lisible.

    # a. Trouver les bornes idéales (Min) et Nadir (Max) du front actuel
    ideal_point = np.min(front, axis=0)
    nadir_point = np.max(front, axis=0)

    # b. Éviter la division par zéro si tous les points sont identiques
    denom = nadir_point - ideal_point
    denom[denom < 1e-9] = 1.0

    # c. Normalisation Min-Max : (x - min) / (max - min)
    normalized_front = (front - ideal_point) / denom

    # d. Point de référence pour l'espace normalisé (toujours > 1.0)
    # Le point (1.1, 1.1, 1.1) est standard pour des données normalisées [0,1]
    ref_point_norm = np.ones(front.shape[1]) * 1.1

    hv_value = calculate_hypervolume(normalized_front, ref_point_norm)

    return {
        'hypervolume': hv_value,  # Score entre 0 et ~1.33 (lisible !)
        'hypervolume_raw': 0.0,  # On ignore le brut illisible
        'execution_time': execution_time,
        'sharpe_metrics': sharpe_metrics,
        'n_solutions': len(front),
        'reference_point_used': ref_point_norm  # Pour info
    }


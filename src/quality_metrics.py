import numpy as np
import time
from functools import wraps
from scipy.spatial.distance import cdist

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


def calculate_hypervolume(pareto_front, reference_point):
    """
    Calcule l'hypervolume du front de Pareto
    Plus l'hypervolume est GRAND, meilleur est le front

    Args:
        pareto_front: array (n_solutions, n_objectives) - objectifs à MINIMISER
        reference_point: array (n_objectives,) - point de référence (nadir)

    Returns:
        hv: valeur de l'hypervolume

    Exemple:
        # Front 2D : [(risque, -rendement), ...]
        ref_point = [max_risk * 1.1, max_neg_return * 1.1]
        hv = calculate_hypervolume(front, ref_point)
    """
    try:
        from pymoo.indicators.hv import HV
        ind = HV(ref_point=reference_point)
        hv = ind(pareto_front)
        return hv
    except ImportError:
        # Fallback : approximation simple pour 2D
        if pareto_front.shape[1] == 2:
            return _hypervolume_2d_simple(pareto_front, reference_point)
        else:
            return None


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
    Évalue la qualité globale d'une optimisation avec les 5 indicateurs

    Args:
        pareto_front_objectives: array (n_solutions, n_objectives)
            Format attendu: [f1, f2, ...] où f1 = -rendement, f2 = variance
        portfolio_weights: array (n_solutions, n_assets)
        execution_time: float (secondes)
        K: cardinalité cible (optionnel)
        risk_free_rate: taux sans risque
        reference_point: point de référence pour hypervolume (optionnel)

    Returns:
        dict: dictionnaire complet des 5 indicateurs
    """
    # Extraction des objectifs
    n_obj = pareto_front_objectives.shape[1]

    # Pour Markowitz (2 objectifs) : f1 = -rendement, f2 = variance
    # Pour NSGA-II (3 objectifs) : f1 = -rendement, f2 = variance, f3 = coûts

    returns = -pareto_front_objectives[:, 0]  # Convertir en rendement positif
    risks = np.sqrt(pareto_front_objectives[:, 1])  # Volatilité

    # Hypervolume
    if reference_point is None:
        # Point de référence par défaut : nadir + 10%
        reference_point = np.max(pareto_front_objectives, axis=0) * 1.1

    try:
        hv = calculate_hypervolume(pareto_front_objectives, reference_point)
    except:
        hv = None

    # Temps d'exécution (déjà fourni)
    exec_time = execution_time

    # Sharpe moyen
    sharpe_metrics = calculate_front_sharpe_metrics(returns, risks, risk_free_rate)

    # Résultat global
    results = {
        'hypervolume': hv,
        'execution_time': exec_time,
        'sharpe_metrics': sharpe_metrics,
        'n_solutions': len(pareto_front_objectives),
        'reference_point': reference_point
    }

    return results

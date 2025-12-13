

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination


class PortfolioOptimizationProblem(Problem):
    """
    Problème d'optimisation de portefeuille tri-objectif
    Objectifs: Minimiser (-rendement, risque, coûts de transaction)
    """

    def __init__(self, mu, Sigma, w_current, K, c_prop=0.005, delta_tol=1e-4):
        """
        Args:
            mu: vecteur des rendements moyens (N,)
            Sigma: matrice de covariance (N, N)
            w_current: portefeuille actuel w_t (N,)
            K: cardinalité exacte (nombre d'actifs)
            c_prop: coût proportionnel de transaction
            delta_tol: seuil minimal pour considérer un poids actif
        """
        self.mu = mu if isinstance(mu, np.ndarray) else mu.values
        self.Sigma = Sigma if isinstance(Sigma, np.ndarray) else Sigma.values
        self.w_current = w_current
        self.K = K
        self.c_prop = c_prop
        self.delta_tol = delta_tol
        self.N = len(self.mu)

        super().__init__(
            n_var=self.N,
            n_obj=3,
            n_constr=2,
            xl=0.0,
            xu=1.0
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """Évaluation des objectifs et contraintes"""
        # Objectif 1: -rendement (minimiser = maximiser rendement)
        f1 = -X @ self.mu

        # Objectif 2: risque (variance)
        f2 = np.sum(X @ self.Sigma * X, axis=1)

        # Objectif 3: coûts de transaction
        f3 = self.c_prop * np.sum(np.abs(X - self.w_current), axis=1)

        # Contrainte 1: somme des poids = 1
        g1 = np.abs(X.sum(axis=1) - 1.0) - 1e-6

        # Contrainte 2: cardinalité exacte K
        n_active = np.sum(X > self.delta_tol, axis=1)
        g2 = np.abs(n_active - self.K) - 0.5

        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2])


class CustomSampling(FloatRandomSampling):
    """Échantillonnage initial respectant la contrainte de cardinalité"""

    def __init__(self, K, delta_tol=1e-4):
        super().__init__()
        self.K = K
        self.delta_tol = delta_tol

    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.N))

        for i in range(n_samples):
            # Sélectionner K actifs aléatoirement
            active_assets = np.random.choice(problem.N, self.K, replace=False)

            # Générer des poids aléatoires pour ces actifs
            weights = np.random.dirichlet(np.ones(self.K))

            # Assigner les poids
            X[i, active_assets] = weights

        return X


def optimize_portfolio_nsga2(mu, Sigma, w_current, K,
                             pop_size=100, n_gen=200,
                             c_prop=0.005, delta_tol=1e-4,
                             verbose=False):
    """
    Optimisation du portefeuille avec NSGA-II

    Args:
        mu: rendements moyens
        Sigma: matrice de covariance
        w_current: portefeuille actuel
        K: cardinalité cible
        pop_size: taille de la population
        n_gen: nombre de générations
        c_prop: coût proportionnel
        delta_tol: tolérance pour poids actifs
        verbose: afficher la progression

    Returns:
        res: objet résultat de pymoo contenant le front de Pareto
    """

    # Définir le problème
    problem = PortfolioOptimizationProblem(
        mu=mu,
        Sigma=Sigma,
        w_current=w_current,
        K=K,
        c_prop=c_prop,
        delta_tol=delta_tol
    )

    # Configurer l'algorithme NSGA-II
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=CustomSampling(K=K, delta_tol=delta_tol),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=1.0 / problem.N, eta=20),
        eliminate_duplicates=True
    )

    # Critère d'arrêt
    termination = get_termination("n_gen", n_gen)

    # Optimisation
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=verbose,
        save_history=False
    )

    return res


def filter_pareto_by_min_return(res, r_min):
    """
    Filtre les solutions du front de Pareto avec rendement >= r_min

    Args:
        res: résultat de l'optimisation NSGA-II
        r_min: rendement minimal souhaité

    Returns:
        filtered_X: poids des portefeuilles filtrés
        filtered_F: objectifs des portefeuilles filtrés
        indices: indices des solutions retenues
    """
    # Le rendement est -F[:, 0] (car on minimise -rendement)
    returns = -res.F[:, 0]

    # Filtrer les solutions
    mask = returns >= r_min
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return None, None, None

    filtered_X = res.X[mask]
    filtered_F = res.F[mask]

    return filtered_X, filtered_F, indices


def select_min_risk_portfolio(filtered_X, filtered_F):
    """
    Sélectionne le portefeuille à risque minimal parmi les solutions filtrées

    Args:
        filtered_X: poids des portefeuilles filtrés
        filtered_F: objectifs des portefeuilles filtrés

    Returns:
        best_weights: poids du meilleur portefeuille
        best_return: rendement du meilleur portefeuille
        best_risk: risque du meilleur portefeuille
        best_cost: coût du meilleur portefeuille
    """
    if filtered_X is None or len(filtered_X) == 0:
        return None, None, None, None

    # Trouver l'indice du portefeuille à risque minimal (objectif 2)
    idx_min_risk = np.argmin(filtered_F[:, 1])

    best_weights = filtered_X[idx_min_risk]
    best_return = -filtered_F[idx_min_risk, 0]  # Conversion en rendement positif
    best_risk = np.sqrt(filtered_F[idx_min_risk, 1])  # Volatilité (racine carrée de variance)
    best_cost = filtered_F[idx_min_risk, 2]

    return best_weights, best_return, best_risk, best_cost


def get_portfolio_statistics(res):
    """
    Calcule des statistiques sur le front de Pareto

    Args:
        res: résultat de l'optimisation NSGA-II

    Returns:
        dict: statistiques du front de Pareto
    """
    F = res.F
    returns = -F[:, 0]
    risks = np.sqrt(F[:, 1])
    costs = F[:, 2]

    stats = {
        'n_solutions': len(F),
        'return_min': returns.min(),
        'return_max': returns.max(),
        'return_mean': returns.mean(),
        'risk_min': risks.min(),
        'risk_max': risks.max(),
        'risk_mean': risks.mean(),
        'cost_min': costs.min(),
        'cost_max': costs.max(),
        'cost_mean': costs.mean()
    }

    return stats


def get_top_assets(weights, asset_names, top_n=10):
    """
    Retourne les top N actifs par poids

    Args:
        weights: vecteur de poids
        asset_names: noms des actifs
        top_n: nombre d'actifs à retourner

    Returns:
        DataFrame avec les top actifs et leurs poids
    """
    df = pd.DataFrame({
        'Asset': asset_names,
        'Weight': weights
    })

    df = df[df['Weight'] > 1e-4].sort_values('Weight', ascending=False).head(top_n)

    return df

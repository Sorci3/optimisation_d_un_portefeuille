

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
        """
        Évaluation des objectifs et contraintes
        - Objectif 1: -rendement (minimiser = maximiser rendement)
        - Objectif 2: risque (variance)
        - Objectif 3: coûts de transaction

        - Contrainte 1: somme des poids = 1
        - Contrainte 2: cardinalité exacte K
        """
        f1 = -X @ self.mu

        f2 = np.sum(X @ self.Sigma * X, axis=1)


        f3 = self.c_prop * np.sum(np.abs(X - self.w_current), axis=1)
        
        g1 = np.abs(X.sum(axis=1) - 1.0) - 1e-6

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
            active_assets = np.random.choice(problem.N, self.K, replace=False)
            weights = np.random.dirichlet(np.ones(self.K))
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

    problem = PortfolioOptimizationProblem(
        mu=mu,
        Sigma=Sigma,
        w_current=w_current,
        K=K,
        c_prop=c_prop,
        delta_tol=delta_tol
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=CustomSampling(K=K, delta_tol=delta_tol),
        crossover=SBX(prob=0.9, eta=10),
        mutation=PM(prob=1.0 / problem.N, eta=5),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", n_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=verbose,
        save_history=False
    )

    return res


def filter_step1_return(res, r_min):
    """
        Filtre les résultats d'optimisation pour ne conserver que les solutions
        dépassant un seuil de rendement minimum.

        Parameters
        ----------
        res : object
            L'objet résultat de l'optimisation (ex: objet Result de pymoo).
            Il doit posséder les attributs suivants :
            - res.F : np.ndarray, matrice des valeurs objectives.
            - res.X : np.ndarray, matrice des variables de décision.
        r_min : float
            Le rendement minimum acceptable (seuil).

        Returns
        -------
        tuple
            Un tuple (X_filtered, F_filtered) contenant :
            - X_filtered (np.ndarray) : Les variables de décision satisfaisant le critère.
            - F_filtered (np.ndarray) : Les valeurs objectives correspondantes.

            Retourne (None, None) si aucune solution ne satisfait le critère `r_min`.
        """

    # Extraction des données
    returns = -res.F[:, 0]  # Rappel : on maximise le rendement (qui est minimisé en négatif)

    # Création du masque
    mask = returns >= r_min
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return None, None

    return res.X[mask], res.F[mask]


def filter_step2_risk(X, F, tolerance=0.20):
    """
        Filtre les solutions en conservant celles dont le risque est proche
        du risque minimal absolu (dans une marge de tolérance)..

        Parameters
        ----------
        X : np.ndarray or None
            Variables de décision (filtrées à l'étape précédente).
            Si None, la fonction retourne immédiatement (None, None).
        F : np.ndarray or None
            Valeurs objectives correspondantes.
            Hypothèse : F[:, 1] représente la variance (on applique sqrt pour avoir l'écart-type).
        tolerance : float, optional
            La marge de tolérance en pourcentage par rapport au risque minimum.
            Par défaut 0.20 (soit +20% au-dessus du risque minimal).

        Returns
        -------
        tuple
            Un tuple (X_filtered, F_filtered) contenant les solutions sous le seuil de risque.
            Retourne (None, None) si l'entrée X est None.
        """

    if X is None:
        return None, None

    # Calcul des risques
    risks = np.sqrt(F[:, 1])

    # On identifie le "champion" du risque (le plus bas absolu)
    min_risk_val = np.min(risks)

    # On définit la barre à ne pas franchir (Risque Min + 10%)
    risk_threshold = min_risk_val * (1.0 + tolerance)

    # On garde tous ceux qui sont sous cette barre
    mask = risks <= risk_threshold

    return X[mask], F[mask]


def select_step3_cost(X, F):
    """
        Sélectionne la solution finale en minimisant le coût (3ème objectif) parmi les candidats restants.

        Parameters
        ----------
        X : np.ndarray or None
            Les variables de décision (poids) des solutions ayant passé les filtres précédents.
            Si None ou vide, la fonction renvoie des None.
        F : np.ndarray or None
            La matrice des objectifs correspondante.
            Structure attendue :
            - F[:, 0] : -Rendement (à inverser).
            - F[:, 1] : Variance (à passer à la racine).
            - F[:, 2] : Coût (critère de sélection ici).

        Returns
        -------
        tuple
            Un tuple (best_weights, best_return, best_risk, best_cost) contenant :
            - best_weights (np.ndarray) : Le vecteur de poids final du portefeuille choisi.
            - best_return (float) : Le rendement attendu (positif).
            - best_risk (float) : La volatilité (écart-type).
            - best_cost (float) : Le coût de transaction ou de gestion.

            Retourne (None, None, None, None) si aucune solution n'est disponible en entrée.
        """

    if X is None or len(X) == 0:
        return None, None, None, None

    costs = F[:, 2]

    # On prend l'indice du coût minimal parmi les survivants
    idx_best = np.argmin(costs)

    # Récupération des valeurs finales
    best_weights = X[idx_best]
    best_return = -F[idx_best, 0]
    best_risk = np.sqrt(F[idx_best, 1])
    best_cost = F[idx_best, 2]

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

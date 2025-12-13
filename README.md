# Contexte et objectif
Dans un environnement financier moderne caractérisé par l'incertitude des marchés et la complexité des corrélations entre actifs, la gestion de portefeuille ne se résume plus à la simple maximisation des gains. Un investisseur rationnel doit arbitrer en permanence entre la recherche de performance et la maîtrise du risque.

Cependant, l'approche théorique classique se heurte souvent à la réalité opérationnelle. En pratique, la construction d'un portefeuille est soumise à des frictions de marché inévitables : les coûts de transaction qui érodent la rentabilité lors des réallocations, les contraintes de cardinalité limitant le nombre d'actifs gérables, et l'instabilité statistique des données historiques. Ces éléments transforment le problème d'allocation en un défi d'optimisation multicritère, souvent non linéaire et non convexe.

L'objectif principal de ce projet est de modéliser et de résoudre ce problème d'optimisation complexe en adoptant une démarche progressive. Il s'agit de développer un outil d'aide à la décision capable de proposer des allocations optimales en tenant compte de multiples dimensions antagonistes :
- Maximiser le rendement attendu du portefeuille.
- Minimiser le risque global, mesuré par la variance.
- Intégrer les contraintes réalistes liées aux coûts et à la structure du portefeuille.


# Lancer le projet
**Assurez-vous d'être à la racine du projet.**

Installation des dépendances :
```Bash
pip install -r requirements.txt
```

Lancement de l'application :
```Bash
streamlit run app.py
```

# Structure du projet
```
OPTIMISATION_MULTICRITERE/
│
├── consigne/ # Dossier contenant les instructions du projet
│   └── Projet_Final_Optimisation.pdf
│
├── data/ # Jeux de données financiers (Secteurs, retours, tickers)
│   ├── Communication_Services.csv
│   ├── ... (Autres fichiers sectoriels CSV)
│   ├── returns_final.csv
│   └── tick.json
│
├── notebook/ # Expérimentations et analyses interactives
│   ├── data_cleaning.ipynb # Nettoyage et pré-traitement des données
│   ├── partie1_markowitz.ipynb # Analyse via modèle de Markowitz
│   └── partie2_nsga2.ipynb # Application de l'algorithme NSGA-II
│
├── src/ # Code source et modules logiques
│   ├── nsga2_optimizer.py # Implémentation de l'optimiseur NSGA-II
│   ├── partie1_markowitz.py # Fonctions liées à l'optimisation Markowitz
│   ├── partie2_constraint.py # Gestion des contraintes du problème
│   └── partie2_montecarlo.py # Simulations de Monte Carlo
│
├── app.py # Point d'entrée de l'application
├── Rapport_optimisationMulticritere.pdf # Rapport final du projet
├── requirements.txt # Liste des dépendances Python
└── README.md # Documentation du projet
```


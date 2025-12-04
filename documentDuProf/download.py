"""
Téléchargement des historiques de prix Yahoo Finance
à partir d'un fichier JSON de tickers catégorisés par secteur.

Prérequis :
    pip install yfinance pandas tqdm
"""

import json
import os
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import sys

# === Chemins de base ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))     # Dossier du script
json_path = os.path.join(SCRIPT_DIR, "tick.json")           # Fichier des tickers
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")               # Dossier de sortie

# === Paramètres généraux ===
START_DATE = "2020-01-01"   # Date de début de l’historique
END_DATE = None             # None => jusqu’à aujourd’hui

# === Création du dossier de sortie s’il n’existe pas ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Chargement du fichier JSON ===
with open(json_path, "r", encoding="utf-8") as f:
    sectors = json.load(f)

total_tickers = sum(len(ticks) for ticks in sectors.values())
print(f"Nombre total d'actions à télécharger : {total_tickers}")

# === Fonction utilitaire ===
def download_sector(sector_name, tickers):
    """Télécharge les prix ajustés pour un secteur complet."""
    print(f"Secteur : {sector_name} ({len(tickers)} tickers)")
    all_data = {}

    # Crée un index de dates journalières de 2015 à 2024
    index = pd.date_range(start="2015-01-01", end="2024-12-31", freq="B")  # "B" = business days

    # Crée un DataFrame vide avec cet index
    df = pd.DataFrame(index=index)

    for ticker in tqdm(tickers):
        # try:
        data = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            progress=False,
            auto_adjust=True,
            threads=True
        )
        if not data.empty:
            all_data[ticker] = data["Close"].to_numpy()[:, 0]
            df[ticker] = data["Close"].reindex(index)  # alignement automatique sur l’index complet

    # Combinaison en DataFrame unique (colonnes = tickers)
    if all_data:
        csv_path = os.path.join(OUTPUT_DIR, f"{sector_name.replace(' ', '_')}.csv")
        df.to_csv(csv_path, index_label="Date")
        print(f" Données sauvegardées : {csv_path}")
    else:
        print(f" Aucun téléchargement réussi pour {sector_name}")

# === Téléchargement par secteur ===
for sector, tickers in sectors.items():
    download_sector(sector, tickers)
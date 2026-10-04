# Mini-projet Séries Temporelles : NVDA vs SMH (VAR · cointégration · GARCH · LSTM/GRU)

## Exécution (dans cet ordre, depuis ce dossier)
```
pip install -r requirements.txt
python 00_download_data.py        # prix Yahoo -> data/prices.csv   (--synthetic = test du pipeline UNIQUEMENT)
python 02_classique.py            # exploration, ARIMA, VAR, Johansen/VECM, GARCH, prévisions glissantes
python 03_deep_learning.py        # LSTM/GRU + CV walk-forward  (--fast pour un test rapide)
Rscript 01_classique.R            # même socle classique en R (package list ci-dessous)
python 04_comparaison.py          # tableau unique + Diebold-Mariano (Python ET R)
```
Packages R : forecast, tseries, urca, vars, rugarch (+ quantmod en repli).
Changer d'actif : modifier `TARGET`/`PARTNER` dans `common.py` (et en tête de `01_classique.R`).

## Correspondance avec l'énoncé
| Énoncé | Où |
|---|---|
| 1 Données, split 90/10 chronologique | `00`, `common.py` |
| 2 Exploration, ACF/PACF, STL, outliers, ADF+KPSS | `02` §1-2, figures 01-03 |
| 3.1 ARIMA, Ljung-Box / JB / ARCH-LM | `02` §3 et §6 |
| 3.2 VAR, Granger, Johansen, VECM | `02` §4-5, figures 04-06 |
| 3.3 GARCH, résidus standardisés, volatilité vs événements | `02` §6, figure 07 |
| 4.1 LSTM/GRU, walk-forward, courbes de perte | `03`, figure 08 |
| 6 Tableau MSE/RMSE/MAE/MAPE + Diebold-Mariano | `04`, figures 09-10 |

## Protocole d'évaluation (à décrire dans le rapport)
Prévisions du **prix** de la cible à h = 5 et 10 jours, origine glissante (une origine par jour de test), mêmes observations pour tous les modèles. Benchmark = marche aléatoire (« Naif »).
ARIMA et VAR : paramètres fixés sur le train. VECM : ré-estimé à chaque origine. LSTM/GRU : entraînés une fois sur le train (moyenne de 3 graines).

## Ce qui reste à faire par vous
1. Lancer sur les vraies données et **lire les résultats** : `ARIMA_MANUAL` (02) à fixer après les ACF/PACF ; rang de Johansen ; si rang 0, rapporter « pas de cointégration » (valide).
2. Section LLM (4 pts) : utiliser `results/stats_descriptives.csv`, `stationnarite.csv`, etc. comme entrée du prompt ; rédiger VOTRE interprétation avant ; garder prompts/réponses ; lister les erreurs du LLM.
3. Bonus : hybride (ARIMA/VAR + LSTM sur résidus), Prophet, DCC-GARCH.
4. Rapport 15-25 pages + notebook (Rmd/ipynb) qui appelle ces scripts.
5. Vérifier que `01_classique.R` s'exécute chez vous (voir limites).
# Mini-projet-s-ries-temporelles

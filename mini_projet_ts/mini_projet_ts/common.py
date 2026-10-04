"""Paramètres et fonctions partagés par tous les scripts Python du projet.

Choix de conception : tous les modèles (R et Python) produisent le MÊME objet,
une prévision du prix de la série cible à l'horizon h pour chaque origine de test,
sauvegardée dans results/forecasts_<modele>.csv. Le script 04 compare ensuite tout
sur exactement les mêmes observations.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- configuration
SEED = 42
TARGET, PARTNER = "NVDA", "SMH"   # actif cible + actif corrélé (modifiable ici seulement)
START, END = "2015-01-01", None   # END=None -> jusqu'à aujourd'hui
TRAIN_FRAC = 0.90                 # découpage chronologique 90 % / 10 % (exigé par l'énoncé)
HORIZONS = (5, 10)                # horizons de prévision h (énoncé : h = 5 à 10)
HMAX = max(HORIZONS)

ROOT = Path(__file__).resolve().parent
DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
for _d in (DATA, RES, FIG):
    _d.mkdir(exist_ok=True)

# Événements à annoter sur le graphique de volatilité (à adapter à l'actif choisi)
EVENTS = {
    "2020-03-16": "Krach COVID-19",
    "2022-06-13": "Bear market 2022 (taux Fed)",
    "2025-01-27": "DeepSeek",
    "2025-04-04": "Droits de douane US",
}


# ---------------------------------------------------------------------- données
def load_frame() -> pd.DataFrame:
    """Prix ajustés -> log-prix, rendements logarithmiques (en %), indicateurs techniques."""
    px = pd.read_csv(DATA / "prices.csv", index_col=0, parse_dates=True).sort_index()
    df = pd.DataFrame({"p_t": px[f"{TARGET}_Close"], "p_p": px[f"{PARTNER}_Close"]}).dropna()
    df["lp_t"], df["lp_p"] = np.log(df.p_t), np.log(df.p_p)
    df["r_t"], df["r_p"] = 100 * df.lp_t.diff(), 100 * df.lp_p.diff()   # rendements log, en %
    df["rv_t"] = df.r_t.rolling(21).std()        # volatilité réalisée 21 jours (indicateur dérivé)
    df["ma50_t"] = df.p_t.rolling(50).mean()     # moyenne mobile 50 jours (indicateur dérivé)
    return df.dropna(subset=["r_t", "r_p", "rv_t"])


def n_train_of(df: pd.DataFrame) -> int:
    return int(len(df) * TRAIN_FRAC)


def test_origins(n: int, n_train: int) -> list:
    """Origines de prévision (dernier indice observé). Même liste pour TOUS les modèles.
    t+HMAX <= n-1 garantit que tous les horizons ont une valeur réelle observée."""
    return list(range(n_train - 1, n - HMAX))


# ------------------------------------------------------------------ évaluation
def save_forecasts(model: str, df: pd.DataFrame, origins: list, lp_hat: np.ndarray) -> None:
    """lp_hat : (n_origines, HMAX) prévisions du LOG-prix cible aux horizons 1..HMAX."""
    rows = []
    for i, t in enumerate(origins):
        for h in HORIZONS:
            rows.append(dict(model=model, origin=df.index[t], h=h, target_date=df.index[t + h],
                             y_true=df.p_t.iloc[t + h], y_pred=float(np.exp(lp_hat[i, h - 1]))))
    pd.DataFrame(rows).to_csv(RES / f"forecasts_{model}.csv", index=False)


def metrics(y, yhat) -> dict:
    e = np.asarray(y) - np.asarray(yhat)
    return dict(MSE=np.mean(e**2), RMSE=np.sqrt(np.mean(e**2)), MAE=np.mean(np.abs(e)),
                MAPE=100 * np.mean(np.abs(e / np.asarray(y))))


def dm_test(e1, e2, lag: int, power: int = 2):
    """Test de Diebold-Mariano (H0 : même précision), variance de Newey-West (poids de
    Bartlett, `lag` = h-1 car les erreurs à h pas se chevauchent) + correction de
    Harvey-Leybourne-Newbold, loi de Student. Statistique > 0 : le modèle 2 est meilleur."""
    d = np.abs(np.asarray(e1)) ** power - np.abs(np.asarray(e2)) ** power
    n, dbar = len(d), d.mean()
    dc = d - dbar
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = dbar / math.sqrt(max(var, 1e-12) / n)
    h = lag + 1
    stat *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, 2 * (1 - stats.t.cdf(abs(stat), n - 1))

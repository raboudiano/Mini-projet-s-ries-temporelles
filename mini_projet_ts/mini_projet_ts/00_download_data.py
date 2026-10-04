"""Étape 0 - Télécharge les prix Yahoo Finance (prix ajustés des splits/dividendes).

    python 00_download_data.py              # vraies données (internet requis)
    python 00_download_data.py --synthetic  # données SIMULÉES, uniquement pour tester le pipeline

Source : finance.yahoo.com via yfinance ; auto_adjust=True => 'Close' est ajusté des splits
et dividendes (à documenter dans le rapport : NVDA a eu des splits en 2021 et 2024).
"""
import sys

import numpy as np
import pandas as pd

from common import DATA, END, PARTNER, SEED, START, TARGET


def synthetic(n=2900):
    """Deux log-prix cointégrés avec chocs GARCH(1,1) : sert à vérifier que le code tourne."""
    rng = np.random.default_rng(SEED)
    def garch(n, omega=0.05, a=0.08, b=0.90):
        s2, e = np.empty(n), np.empty(n)
        s2[0] = omega / (1 - a - b)
        for t in range(n):
            e[t] = np.sqrt(s2[t]) * rng.standard_normal()
            if t + 1 < n:
                s2[t + 1] = omega + a * e[t] ** 2 + b * s2[t]
        return e
    lp_p = 4 + np.cumsum(0.04 + garch(n)) / 100
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.97 * u[t - 1] + 0.004 * rng.standard_normal()
    lp_t = 1.0 + 1.4 * lp_p + u + np.cumsum(0.3 * garch(n, 0.02, 0.05, 0.9)) / 100
    idx = pd.bdate_range("2015-01-02", periods=n)
    return pd.DataFrame({f"{TARGET}_Close": np.exp(lp_t), f"{PARTNER}_Close": np.exp(lp_p)}, index=idx)


if "--synthetic" in sys.argv:
    print("!" * 70, "\n DONNÉES SIMULÉES : ne JAMAIS les utiliser pour le rapport.\n" + "!" * 70)
    out = synthetic()
else:
    import yfinance as yf
    print("yfinance", yf.__version__)
    raw = yf.download([TARGET, PARTNER], start=START, end=END, auto_adjust=True, progress=False)
    raw.columns = [f"{tkr}_{field}" for field, tkr in raw.columns]   # (Price, Ticker) -> NVDA_Close
    out = raw.dropna(how="all")
out.index.name = "Date"
out.to_csv(DATA / "prices.csv")
print(out.shape, out.index.min().date(), "->", out.index.max().date())

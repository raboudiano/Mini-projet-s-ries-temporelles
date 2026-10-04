"""Étape 6 - Tableau unique MSE/RMSE/MAE/MAPE + test de Diebold-Mariano (tous modèles, mêmes observations)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import *

allf = pd.concat([pd.read_csv(f, parse_dates=["origin", "target_date"]) for f in sorted(RES.glob("forecasts_*.csv"))])
# Même jeu d'observations pour tous : on garde les (origine, h) présents chez TOUS les modèles.
models = allf.model.unique()
common_keys = allf.groupby(["origin", "h"]).model.nunique()
common_keys = common_keys[common_keys == len(models)].index
allf = allf.set_index(["origin", "h"]).loc[common_keys].reset_index()
print("Modèles :", list(models), "| observations communes par horizon :", len(allf) // len(models) // len(HORIZONS))

tabs, dm_rows = [], []
for h in HORIZONS:
    sub = allf[allf.h == h]
    E = {m: (g.sort_values("origin").y_true.values - g.sort_values("origin").y_pred.values) for m, g in sub.groupby("model")}
    for m, g in sub.groupby("model"):
        g = g.sort_values("origin"); tabs.append(dict(h=h, modele=m, **metrics(g.y_true, g.y_pred)))
    for m in E:
        if m != "Naif":
            s, p = dm_test(E["Naif"], E[m], lag=h - 1)     # s > 0 : m meilleur que le naïf
            dm_rows.append(dict(h=h, modele=m, DM_vs_naif=s, p_value=p, verdict="significativement meilleur" if (s > 0 and p < .05)
                                else "significativement pire" if (s < 0 and p < .05) else "indiscernable du naïf"))
tab = pd.DataFrame(tabs).sort_values(["h", "RMSE"]); dm = pd.DataFrame(dm_rows)
tab.to_csv(RES / "table_comparaison.csv", index=False); dm.to_csv(RES / "diebold_mariano.csv", index=False)
print(tab.round(3).to_string(index=False)); print(dm.round(4).to_string(index=False))

fig, ax = plt.subplots(1, len(HORIZONS), figsize=(11, 4))
for a, h in zip(np.atleast_1d(ax), HORIZONS):
    t = tab[tab.h == h].sort_values("RMSE"); a.barh(t.modele, t.RMSE); a.set_title(f"RMSE (USD), h={h}"); a.invert_yaxis()
plt.tight_layout(); plt.savefig(FIG / "09_rmse_modeles.png", dpi=140); plt.close()

h0 = HORIZONS[0]; sub = allf[allf.h == h0].sort_values("origin"); last = sub.origin.unique()[-120:]
plt.figure(figsize=(11, 4))
plt.plot(sub[sub.model == sub.model.iloc[0]].set_index("origin").loc[last].target_date, sub[sub.model == sub.model.iloc[0]].set_index("origin").loc[last].y_true, "k", lw=2, label="observé")
for m, g in sub.groupby("model"):
    g = g.set_index("origin").loc[last]; plt.plot(g.target_date, g.y_pred, lw=.9, label=m)
plt.title(f"Prévisions à h={h0} jours vs observations (120 dernières origines du test)"); plt.ylabel("Prix (USD)"); plt.legend(ncol=3, fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "10_previsions_vs_obs.png", dpi=140); plt.close()

"""Étape 4 - LSTM et GRU (PyTorch) : fenêtres glissantes, validation walk-forward, courbes de perte.

    python 03_deep_learning.py          # grille d'hyperparamètres réduite + modèle final
    python 03_deep_learning.py --fast   # test rapide (peu d'époques)

Cible : les HMAX prochains rendements log de la cible (sortie vectorielle, prévision directe),
convertis ensuite en prix pour être comparables aux modèles économétriques (même table, même test).
Anti-fuite : normalisation ajustée sur le train seul ; échantillons d'apprentissage dont TOUTES
les cibles sont dans le train ; un intervalle de HMAX jours sépare train et validation.
"""
import itertools
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import *

FAST = "--fast" in sys.argv
FEATURES = ["r_t", "r_p", "rv_t"]          # rendement cible, rendement partenaire, vol. réalisée 21j
EPOCHS, PATIENCE, LR, BATCH, DROPOUT = (8 if FAST else 80), 10, 1e-3, 64, 0.2
GRID = dict(W=[20, 40], HID=[16, 32])      # taille de fenêtre, taille de l'état caché
N_FOLDS, N_SEEDS = 4, (1 if FAST else 3)
torch.set_num_threads(1)

df = load_frame()
n, n_tr = len(df), n_train_of(df)
origins = test_origins(n, n_tr)
Xraw, rt = df[FEATURES].values, df.r_t.values
mu, sd = Xraw[:n_tr].mean(0), Xraw[:n_tr].std(0)         # statistiques du TRAIN uniquement
Xs, ysd = (Xraw - mu) / sd, rt[:n_tr].std()
T0 = max(GRID["W"]) - 1                                  # première origine commune à toutes les fenêtres


def build(W, idx):
    X = np.stack([Xs[t - W + 1: t + 1] for t in idx]).astype("float32")
    Y = np.stack([rt[t + 1: t + 1 + HMAX] / ysd for t in idx]).astype("float32")
    return torch.tensor(X), torch.tensor(Y)


class Net(nn.Module):
    def __init__(self, kind, hid):
        super().__init__()
        self.rnn = (nn.LSTM if kind == "LSTM" else nn.GRU)(len(FEATURES), hid, batch_first=True)
        self.drop, self.fc = nn.Dropout(DROPOUT), nn.Linear(hid, HMAX)
    def forward(self, x):
        return self.fc(self.drop(self.rnn(x)[0][:, -1]))


def fit(kind, W, hid, tr_idx, es_idx, seed):
    """Entraîne avec arrêt anticipé sur un bloc de validation postérieur (chronologique)."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    Xt, Yt = build(W, tr_idx); Xe, Ye = build(W, es_idx)
    net, opt, loss_fn = Net(kind, hid), None, nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(EPOCHS):
        net.train()
        for b in torch.randperm(len(Xt), generator=g).split(BATCH):
            opt.zero_grad(); loss_fn(net(Xt[b]), Yt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            tl, vl = loss_fn(net(Xt), Yt).item(), loss_fn(net(Xe), Ye).item()
        hist.append((tl, vl))
        if vl < best - 1e-5:
            best, best_state, wait = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(best_state)
    return net.eval(), hist


def predict(net, W, idx):
    X, _ = build(W, idx)
    with torch.no_grad():
        return net(X).numpy() * ysd            # rendements prévus (%), (n, HMAX)


# ---- origines d'apprentissage : toutes les cibles (t+1..t+HMAX) tombent dans le train
train_origins = np.arange(T0, n_tr - HMAX)

# ---- validation croisée temporelle (walk-forward, fenêtre extensible) sur le TRAIN uniquement
def split_fold(i):
    e = np.linspace(.5, 1., N_FOLDS + 1)
    va = train_origins[int(e[i] * len(train_origins)): int(e[i + 1] * len(train_origins))]
    pool = train_origins[train_origins <= va[0] - HMAX]          # trou de HMAX jours : pas de chevauchement
    k = int(.85 * len(pool))
    return pool[: k - HMAX], pool[k:], va                         # entraînement, arrêt anticipé, score

cv_rows = []
for kind, (W, hid) in itertools.product(["LSTM", "GRU"], itertools.product(GRID["W"], GRID["HID"])):
    for i in range(N_FOLDS):
        a, b, va = split_fold(i)
        net, _ = fit(kind, W, hid, a, b, SEED)
        yhat5 = predict(net, W, va)[:, :5].sum(1); ytrue5 = np.array([rt[t + 1: t + 6].sum() for t in va])
        cv_rows.append(dict(modele=kind, W=W, HID=hid, fold=i, RMSE_cum5=np.sqrt(np.mean((yhat5 - ytrue5) ** 2)),
                            RMSE_zero=np.sqrt(np.mean(ytrue5 ** 2))))     # zéro = prévision naïve
    print(kind, W, hid, "ok", flush=True)
cv = pd.DataFrame(cv_rows); cv.to_csv(RES / "dl_cv_walkforward.csv", index=False)
summ = cv.groupby(["modele", "W", "HID"])[["RMSE_cum5", "RMSE_zero"]].mean().reset_index()
summ["ratio_vs_naif"] = summ.RMSE_cum5 / summ.RMSE_zero
summ.to_csv(RES / "dl_cv_resume.csv", index=False); print(summ.round(4).to_string(index=False))

# ---- modèle final : meilleure config de CV par architecture, réentraîné sur tout le train
a_idx = train_origins[: int(.85 * len(train_origins)) - HMAX]; b_idx = train_origins[int(.85 * len(train_origins)):]
test_idx = np.array(origins)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for j, kind in enumerate(["LSTM", "GRU"]):
    best = summ[summ.modele == kind].sort_values("RMSE_cum5").iloc[0]; W, hid = int(best.W), int(best.HID)
    print(f"{kind} retenu : fenêtre={W}, état caché={hid}")
    preds = []
    for s in range(N_SEEDS):                                      # moyenne de plusieurs graines = résultat plus stable
        net, hist = fit(kind, W, hid, a_idx, b_idx, SEED + s)
        preds.append(predict(net, W, test_idx))
        if s == 0:
            ax[j].plot([h[0] for h in hist], label="train"); ax[j].plot([h[1] for h in hist], label="validation")
            ax[j].set_title(f"{kind} (W={W}, HID={hid}) : courbes de perte"); ax[j].set_xlabel("époque"); ax[j].set_ylabel("MSE (échelle normalisée)"); ax[j].legend()
    cum = np.cumsum(np.mean(preds, 0), axis=1) / 100
    save_forecasts(kind, df, origins, df.lp_t.values[test_idx][:, None] + cum)
plt.tight_layout(); plt.savefig(FIG / "08_dl_loss.png", dpi=140); plt.close()
print("OK -> results/forecasts_LSTM.csv, forecasts_GRU.csv")

"""Étapes 1-3 (Python) - exploration, ARIMA, VAR, cointégration/VECM, GARCH + prévisions glissantes.

Règle d'or : TOUT (identification, choix des ordres, tests) est fait sur l'échantillon
d'apprentissage uniquement. Le test (10 %) ne sert qu'à évaluer les prévisions, une seule fois.
Chaque bloc est précédé d'un commentaire qui explique POURQUOI cette étape existe.
"""
import itertools
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_coint_rank, select_order
from arch import arch_model

from common import *

warnings.filterwarnings("ignore")
np.random.seed(SEED)
ARIMA_MANUAL = (0, 1, 0)      # <- à fixer APRÈS lecture des ACF/PACF (figure 03), puis comparer à la grille
VECM_RANK_OVERRIDE = None     # None = rang choisi par le test de Johansen ; sinon forcer 0/1 (à justifier)

df = load_frame()
n, n_tr = len(df), n_train_of(df)
tr = df.iloc[:n_tr]
origins = test_origins(n, n_tr)
print(f"{TARGET}/{PARTNER}: {n} jours ({df.index[0].date()} -> {df.index[-1].date()}), "
      f"train={n_tr}, test={n - n_tr}, origines de prévision={len(origins)}")


def save_fig(name):
    plt.tight_layout(); plt.savefig(FIG / f"{name}.png", dpi=140); plt.close()


# ============================================================ 1. EXPLORATION
# Pourquoi : on regarde les données AVANT de modéliser (tendance, ruptures, volatilité groupée).
def describe(x):
    return pd.Series(dict(n=len(x), moyenne=x.mean(), ecart_type=x.std(), skewness=stats.skew(x),
                          kurtosis_exces=stats.kurtosis(x), min=x.min(), max=x.max(),
                          JB_pvalue=stats.jarque_bera(x).pvalue))

desc = pd.DataFrame({f"r_{TARGET} (%)": describe(tr.r_t), f"r_{PARTNER} (%)": describe(tr.r_p)}).T
desc.to_csv(RES / "stats_descriptives.csv"); print(desc.round(3))

fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
ax[0].plot(df.p_t, label=TARGET); ax[0].plot(df.p_t.rolling(50).mean(), label="MM 50j", lw=.8); ax[0].set_ylabel("Prix ajusté (USD)"); ax[0].legend()
ax[1].plot(df.p_p, color="C1", label=PARTNER); ax[1].set_ylabel("Prix ajusté (USD)"); ax[1].legend()
ax[2].plot(df.r_t, lw=.5, label=f"r {TARGET}"); ax[2].plot(df.rv_t, color="C3", lw=1, label="Vol. réalisée 21j")
ax[2].axvline(df.index[n_tr], color="k", ls="--"); ax[2].set_ylabel("%"); ax[2].legend()
ax[0].set_title("Prix, rendements log et volatilité réalisée (tirets : début du test)"); save_fig("01_series")

# Valeurs aberrantes (IQR sur rendements). On CONSERVE : ce sont de vrais événements de marché ;
# les supprimer fausserait la volatilité. On les traite via des lois à queues épaisses (Student).
q1, q3 = tr.r_t.quantile([.25, .75]); iqr = q3 - q1
out = tr.r_t[(tr.r_t < q1 - 3 * iqr) | (tr.r_t > q3 + 3 * iqr)]
print(f"Aberrations (>3 IQR) sur r_{TARGET}: {len(out)} ({100 * len(out) / n_tr:.1f} %)")
out.to_csv(RES / "outliers.csv")

# Décomposition STL (période 252 = 1 an de bourse). On s'attend à une saisonnalité quasi nulle.
stl = STL(tr.lp_t.values, period=252, robust=True).fit()
fig, ax = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
for a, s, t in zip(ax, [tr.lp_t.values, stl.trend, stl.seasonal, stl.resid], ["log-prix", "tendance", "saisonnalité (252j)", "reste"]):
    a.plot(s, lw=.7); a.set_ylabel(t)
ax[0].set_title(f"Décomposition STL du log-prix {TARGET}"); save_fig("02_stl")
print(f"STL: part de variance saisonnière = {np.var(stl.seasonal) / np.var(stl.trend + stl.seasonal + stl.resid):.4f}")

fig, ax = plt.subplots(2, 3, figsize=(13, 6))
for j, (x, nm) in enumerate([(tr.lp_t, "log-prix"), (tr.r_t, "rendements"), (tr.r_t ** 2, "rendements²")]):
    plot_acf(x, lags=30, ax=ax[0, j], title=f"ACF {nm}"); plot_pacf(x, lags=30, ax=ax[1, j], title=f"PACF {nm}", method="ywm")
save_fig("03_acf_pacf")

# ============================================================ 2. STATIONNARITÉ
# Pourquoi ADF ET KPSS : H0 opposées (ADF : racine unitaire ; KPSS : stationnarité).
def station(name, x, reg):
    a, k = adfuller(x, regression=reg, autolag="AIC"), kpss(x, regression=reg, nlags="auto")
    verdict = ("stationnaire" if a[1] < .05 and k[1] >= .05 else
               "I(1) (racine unitaire)" if a[1] >= .05 and k[1] < .05 else "ambigu")
    return dict(serie=name, adf_stat=a[0], adf_p=a[1], kpss_stat=k[0], kpss_p=k[1], conclusion=verdict)

st = pd.DataFrame([station(f"log-prix {TARGET}", tr.lp_t, "ct"), station(f"log-prix {PARTNER}", tr.lp_p, "ct"),
                   station(f"rendement {TARGET}", tr.r_t, "c"), station(f"rendement {PARTNER}", tr.r_p, "c")])
st.to_csv(RES / "stationnarite.csv", index=False); print(st.round(4).to_string(index=False))

# ============================================================ 3. ARIMA (base de comparaison)
# d=1 d'après le tableau ci-dessus. Grille (p,q) <= 3 notée par BIC, comparée à l'ordre manuel.
grid = []
for p, q in itertools.product(range(4), range(4)):
    try:
        m = ARIMA(tr.lp_t.values, order=(p, 1, q), trend="t").fit()
        grid.append(dict(order=(p, 1, q), aic=m.aic, bic=m.bic))
    except Exception:
        pass
grid = pd.DataFrame(grid).sort_values("bic"); grid.to_csv(RES / "arima_grille.csv", index=False)
auto_order = grid.iloc[0]["order"]
print("Top-3 grille ARIMA (BIC):\n", grid.head(3).to_string(index=False), f"\nOrdre manuel: {ARIMA_MANUAL}")
ORDER = auto_order     # modèle retenu pour les prévisions (changez en ARIMA_MANUAL si vous le justifiez)
arima_tr = ARIMA(tr.lp_t.values, order=ORDER, trend="t").fit()

def diagnostics(resid, nm, lags=10):
    resid = np.asarray(resid)
    return dict(modele=nm, LjungBox_p=acorr_ljungbox(resid, lags=[lags]).lb_pvalue.iloc[0],
                JarqueBera_p=stats.jarque_bera(resid).pvalue, ARCH_LM_p=het_arch(resid, nlags=5)[1])

# ============================================================ 4. VAR sur les rendements
# Pourquoi des rendements : les log-prix sont I(1) ; un VAR sur niveaux non cointégrés est invalide.
# Ordre de Cholesky : [partenaire, cible] = on suppose que l'indice/ETF réagit en premier
# (hypothèse d'identification à justifier et à tester en sensibilité en inversant l'ordre).
VAR_COLS = ["r_p", "r_t"]
var_m = VAR(tr[VAR_COLS])
sel = var_m.select_order(maxlags=10)
p_var = max(1, sel.selected_orders["bic"])
var_res = var_m.fit(p_var, trend="c")
print(f"VAR: p(BIC)={p_var} (AIC={sel.selected_orders['aic']}), stable={var_res.is_stable()}")
gr = pd.DataFrame([
    dict(hypothese=f"{PARTNER} -> {TARGET}", p_value=var_res.test_causality("r_t", ["r_p"], kind="f").pvalue),
    dict(hypothese=f"{TARGET} -> {PARTNER}", p_value=var_res.test_causality("r_p", ["r_t"], kind="f").pvalue)])
gr.to_csv(RES / "var_granger.csv", index=False); print(gr.round(4).to_string(index=False))
print("Blancheur résidus VAR (Portmanteau) p =", round(var_res.test_whiteness(nlags=max(12, p_var + 5)).pvalue, 4))
irf = var_res.irf(10); irf.plot(orth=True, impulse="r_p", response="r_t"); plt.suptitle(f"IRF orthogonalisée : choc {PARTNER} -> {TARGET}"); save_fig("04_irf")
var_res.fevd(10).plot(); save_fig("05_fevd")

# ============================================================ 5. COINTÉGRATION / VECM
# Pourquoi : deux log-prix I(1) peuvent partager un équilibre de long terme (spread stationnaire).
lp2 = tr[["lp_t", "lp_p"]]
k_diff = max(1, select_order(lp2, maxlags=10, deterministic="co").bic)
rows = []
for det, lab in [(0, "constante"), (1, "tendance linéaire")]:   # le résultat dépend de la spéc. déterministe
    rk = select_coint_rank(lp2, det_order=det, k_ar_diff=k_diff, method="trace", signif=.05)
    rows.append(dict(spec=lab, k_ar_diff=k_diff, rang_trace=rk.rank))
eg = coint(tr.lp_t, tr.lp_p, trend="c")
rows.append(dict(spec="Engle-Granger (asymétrique)", k_ar_diff=np.nan, rang_trace=f"stat={eg[0]:.2f}, p={eg[1]:.3f}"))
joh = pd.DataFrame(rows); joh.to_csv(RES / "johansen.csv", index=False); print(joh.to_string(index=False))
print(select_coint_rank(lp2, det_order=0, k_ar_diff=k_diff, method="trace", signif=.05).summary())
rank = select_coint_rank(lp2, det_order=0, k_ar_diff=k_diff, method="trace", signif=.05).rank
if VECM_RANK_OVERRIDE is not None:
    rank = VECM_RANK_OVERRIDE
print("Rang de cointégration retenu:", rank)
USE_VECM = rank >= 1
if USE_VECM:
    vecm_res = VECM(lp2, k_ar_diff=k_diff, coint_rank=rank, deterministic="co").fit()
    alpha, beta = vecm_res.alpha, vecm_res.beta
    print("beta (relation de long terme, normalisée):\n", beta.round(3), "\nalpha (vitesses d'ajustement):\n", alpha.round(4))
    if rank == 1:
        rho = 1 + float(beta[:, 0] @ alpha[:, 0])
        print(f"rho={rho:.4f}", f"-> demi-vie ~ {np.log(.5) / np.log(rho):.1f} jours" if 0 < rho < 1 else "(pas de retour à l'équilibre)")
        spread = lp2.lp_t - beta[1, 0] * lp2.lp_p
        spread.plot(figsize=(10, 3), title=f"Écart à l'équilibre (log {TARGET} − β·log {PARTNER})"); save_fig("06_spread")
else:
    print("Pas de cointégration au seuil 5 % : on le rapporte tel quel (résultat valable) ; pas de VECM.")

# ============================================================ 6. ARCH / GARCH
# Pourquoi : ARIMA/VAR modélisent la moyenne ; la variance conditionnelle est une autre question.
d_arima, d_var = diagnostics(arima_tr.resid[1:], "ARIMA"), diagnostics(var_res.resid["r_t"], "VAR (éq. cible)")
print(pd.DataFrame([d_arima, d_var]).round(4).to_string(index=False))
pd.DataFrame([d_arima, d_var]).to_csv(RES / "diagnostics_residus.csv", index=False)

specs = {"GARCH(1,1)-normal": dict(vol="GARCH", p=1, q=1, dist="normal"),
         "GARCH(1,1)-Student": dict(vol="GARCH", p=1, q=1, dist="t"),
         "GJR-GARCH(1,1)-Student": dict(vol="GARCH", p=1, o=1, q=1, dist="t"),
         "EGARCH(1,1)-Student": dict(vol="EGARCH", p=1, o=1, q=1, dist="t")}
fits, g_rows = {}, []
for nm, kw in specs.items():
    f = arch_model(tr.r_t, mean="Constant", **kw).fit(disp="off"); fits[nm] = f
    z = f.std_resid.dropna()
    g_rows.append(dict(modele=nm, loglik=f.loglikelihood, aic=f.aic, bic=f.bic,
                       persistance=(f.params.get("beta[1]", np.nan) if "EGARCH" in nm else   # EGARCH : persistance = beta
                                    f.params.get("alpha[1]", np.nan) + f.params.get("gamma[1]", 0) / 2 + f.params.get("beta[1]", np.nan)),
                       LB_z_p=acorr_ljungbox(z, lags=[10]).lb_pvalue.iloc[0],
                       LB_z2_p=acorr_ljungbox(z ** 2, lags=[10]).lb_pvalue.iloc[0], ARCHLM_z_p=het_arch(z, nlags=5)[1]))
garch_tab = pd.DataFrame(g_rows); garch_tab.to_csv(RES / "garch_comparaison.csv", index=False)
print(garch_tab.round(4).to_string(index=False))
best_g = garch_tab.sort_values("bic").iloc[0]["modele"]; print("GARCH retenu (BIC):", best_g)
print(fits[best_g].summary())

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(tr.index, tr.r_t.abs(), lw=.4, color="C0", alpha=.6, label=f"|r| {TARGET}")
ax.plot(tr.index, fits[best_g].conditional_volatility, color="C3", lw=1.1, label=f"σ conditionnel ({best_g})")
for d, lab in EVENTS.items():
    if pd.Timestamp(d) <= tr.index[-1]:
        ax.axvline(pd.Timestamp(d), color="k", ls=":", lw=.8); ax.text(pd.Timestamp(d), ax.get_ylim()[1] * .9, lab, rotation=90, fontsize=7, va="top")
ax.set_ylabel("%"); ax.set_title("Volatilité conditionnelle estimée et événements"); ax.legend(); save_fig("07_garch_vol")

# Évaluation hors échantillon de la volatilité : paramètres du train, filtre appliqué à tout l'échantillon.
spec_kw = specs[best_g]
sig_g = arch_model(df.r_t, mean="Constant", **spec_kw).fix(fits[best_g].params).conditional_volatility
sig_naive = df.r_t.rolling(21).std().shift(1)      # benchmark : vol. réalisée 21j connue la veille
te = df.index[n_tr:]
def qlike(s2, r2): return np.mean(np.log(s2) + r2 / s2)
vol_tab = pd.DataFrame([dict(modele=best_g, QLIKE=qlike(sig_g[te] ** 2, df.r_t[te] ** 2), MSE_var=np.mean((sig_g[te] ** 2 - df.r_t[te] ** 2) ** 2)),
                        dict(modele="Vol. réalisée 21j (retard 1)", QLIKE=qlike(sig_naive[te] ** 2, df.r_t[te] ** 2), MSE_var=np.mean((sig_naive[te] ** 2 - df.r_t[te] ** 2) ** 2))])
vol_tab.to_csv(RES / "garch_hors_echantillon.csv", index=False); print(vol_tab.round(4).to_string(index=False))

# ============================================================ 7. PRÉVISIONS GLISSANTES (origine glissante)
# Pour chaque origine t du test : prévision du log-prix cible à t+1..t+10, puis conversion en prix.
#   Naïf (marche aléatoire) : prix(t+h) = prix(t)                       <- benchmark obligatoire
#   ARIMA  : paramètres fixés sur le train, état mis à jour avec l'historique (apply)
#   VAR    : paramètres fixés sur le train, rendements cumulés
#   VECM   : ré-estimé à chaque origine (statsmodels ne permet pas de prévoir depuis une nouvelle histoire)
lp = df.lp_t.values; Y = df[VAR_COLS].values; LP2 = df[["lp_t", "lp_p"]].values
F = {k: np.zeros((len(origins), HMAX)) for k in ["Naif", "ARIMA", "VAR"] + (["VECM"] if USE_VECM else [])}
for i, t in enumerate(origins):
    F["Naif"][i] = lp[t]
    F["ARIMA"][i] = arima_tr.apply(lp[: t + 1], refit=False).forecast(HMAX)
    F["VAR"][i] = lp[t] + np.cumsum(var_res.forecast(Y[t + 1 - p_var: t + 1], steps=HMAX)[:, 1]) / 100
    if USE_VECM:
        F["VECM"][i] = VECM(LP2[: t + 1], k_ar_diff=k_diff, coint_rank=rank, deterministic="co").fit().predict(steps=HMAX)[:, 0]
for k, v in F.items():
    save_forecasts(k, df, origins, v)
    yt = np.array([df.p_t.iloc[t + 5] for t in origins]); yh = np.exp(v[:, 4])
    print(f"{k:6s} h=5  RMSE={metrics(yt, yh)['RMSE']:.3f}  MAPE={metrics(yt, yh)['MAPE']:.2f}%")
print("OK -> results/*.csv et figures/*.png")

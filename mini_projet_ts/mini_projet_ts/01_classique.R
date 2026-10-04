# ==============================================================================
# 01_classique.R - Socle classique en R : exploration, ARIMA, VAR, Johansen/VECM, GARCH
# + prévisions glissantes (mêmes origines que Python) -> results/forecasts_R_*.csv
# Lancer depuis la racine du projet :  source("01_classique.R")
# Prérequis : python 00_download_data.py (crée data/prices.csv ; sinon repli quantmod ci-dessous)
# ==============================================================================
set.seed(42)
suppressPackageStartupMessages({library(forecast); library(tseries); library(urca); library(vars); library(rugarch)})
TARGET <- "NVDA"; PARTNER <- "SMH"; START <- "2015-01-01"
TRAIN_FRAC <- 0.90; HORIZONS <- c(5, 10); HMAX <- max(HORIZONS)   # identiques à common.py
dir.create("results", showWarnings = FALSE); dir.create("figures", showWarnings = FALSE)

# ---- Données : on réutilise le CSV Python pour que R et Python travaillent sur les MÊMES prix
if (!file.exists("data/prices.csv")) {                     # repli : téléchargement direct via quantmod
  library(quantmod); dir.create("data", showWarnings = FALSE)
  a <- getSymbols(TARGET, from = START, auto.adjust = TRUE, auto.assign = FALSE)
  b <- getSymbols(PARTNER, from = START, auto.adjust = TRUE, auto.assign = FALSE)
  m <- na.omit(merge(Cl(a), Cl(b)))
  write.csv(data.frame(Date = index(m), setNames(as.data.frame(m), paste0(c(TARGET, PARTNER), "_Close"))), "data/prices.csv", row.names = FALSE)
}
px <- read.csv("data/prices.csv"); px$Date <- as.Date(px[[1]])
df <- na.omit(data.frame(Date = px$Date, p_t = px[[paste0(TARGET, "_Close")]], p_p = px[[paste0(PARTNER, "_Close")]]))
df$lp_t <- log(df$p_t); df$lp_p <- log(df$p_p)
df$r_t <- c(NA, 100 * diff(df$lp_t)); df$r_p <- c(NA, 100 * diff(df$lp_p))
df$rv_t <- sapply(seq_len(nrow(df)), function(i) if (i >= 22) sd(df$r_t[(i - 20):i]) else NA)   # vol. réalisée 21j
df$ma50_t <- as.numeric(stats::filter(df$p_t, rep(1 / 50, 50), sides = 1))                                   # moyenne mobile 50j
df <- df[!is.na(df$rv_t), ]; rownames(df) <- NULL
n <- nrow(df); n_tr <- floor(n * TRAIN_FRAC); tr <- df[1:n_tr, ]
origins <- (n_tr):(n - HMAX)            # indices R (base 1) : même liste d'origines que Python (n_tr-1 en base 0)
cat(sprintf("%s/%s: n=%d, train=%d, test=%d, origines=%d\n", TARGET, PARTNER, n, n_tr, n - n_tr, length(origins)))

# ---- Outils : test ARCH-LM (régression de x² sur ses retards, stat = n*R²) ------
arch_lm <- function(x, lags = 5) {
  Z <- embed(x^2, lags + 1); R2 <- summary(lm(Z[, 1] ~ Z[, -1]))$r.squared
  c(stat = nrow(Z) * R2, p = pchisq(nrow(Z) * R2, lags, lower.tail = FALSE))
}
diag_resid <- function(e, fitdf = 0) c(LjungBox_p = Box.test(e, 10, "Ljung-Box", fitdf = fitdf)$p.value,
                                       JarqueBera_p = jarque.bera.test(e)$p.value, ARCH_LM_p = arch_lm(e)["p"])

# ==== 1. EXPLORATION ==========================================================
# Pourquoi : voir tendance, ruptures et volatilité groupée avant de choisir un modèle.
png("figures/R01_series.png", 1100, 800, res = 120); par(mfrow = c(3, 1))
plot(df$Date, df$p_t, type = "l", main = paste("Prix ajusté", TARGET), xlab = "", ylab = "USD")
plot(df$Date, df$r_t, type = "l", lwd = .5, main = "Rendements log (%) et vol. réalisée 21j", xlab = "", ylab = "%"); lines(df$Date, df$rv_t, col = "red")
abline(v = df$Date[n_tr], lty = 2)
plot(df$Date, df$p_p, type = "l", col = "darkorange", main = paste("Prix ajusté", PARTNER), xlab = "", ylab = "USD"); dev.off()
desc <- function(x) c(n = length(x), moy = mean(x), sd = sd(x), skew = skewness_(x), kurt_exces = kurt_(x), min = min(x), max = max(x), JB_p = jarque.bera.test(x)$p.value)
skewness_ <- function(x) mean((x - mean(x))^3) / sd(x)^3; kurt_ <- function(x) mean((x - mean(x))^4) / sd(x)^4 - 3
print(round(rbind(r_target = desc(tr$r_t), r_partner = desc(tr$r_p)), 3))
iqr <- IQR(tr$r_t); q <- quantile(tr$r_t, c(.25, .75))
cat("Valeurs aberrantes (>3 IQR), conservées car ce sont de vrais chocs de marché :", sum(tr$r_t < q[1] - 3 * iqr | tr$r_t > q[2] + 3 * iqr), "\n")
png("figures/R02_acf_pacf.png", 1200, 700, res = 120); par(mfrow = c(2, 3))
for (v in list(list(tr$lp_t, "log-prix"), list(tr$r_t, "rendements"), list(tr$r_t^2, "rendements²"))) acf(v[[1]], 30, main = paste("ACF", v[[2]]))
for (v in list(list(tr$lp_t, "log-prix"), list(tr$r_t, "rendements"), list(tr$r_t^2, "rendements²"))) pacf(v[[1]], 30, main = paste("PACF", v[[2]])); dev.off()
stl_fit <- stl(ts(tr$lp_t, frequency = 252), s.window = "periodic", robust = TRUE)
png("figures/R03_stl.png", 1000, 700, res = 120); plot(stl_fit, main = paste("STL du log-prix", TARGET)); dev.off()

# ==== 2. STATIONNARITÉ : ADF + KPSS (hypothèses nulles opposées) ===============
st <- function(x, nm, lvl) {
  a <- ur.df(x, type = if (lvl) "trend" else "drift", selectlags = "AIC"); k <- kpss.test(x, null = if (lvl) "Trend" else "Level")
  adf_stat <- a@teststat[1, 1]; adf_5 <- a@cval[1, "5pct"]                       # rejet de H0 (racine unitaire) si stat < valeur critique
  data.frame(serie = nm, adf_stat = adf_stat, adf_crit5 = adf_5, kpss_p = k$p.value,
             conclusion = ifelse(adf_stat < adf_5 & k$p.value >= .05, "stationnaire", ifelse(adf_stat >= adf_5 & k$p.value < .05, "I(1)", "ambigu")))
}
print(rbind(st(tr$lp_t, "log-prix cible", TRUE), st(tr$lp_p, "log-prix partenaire", TRUE), st(tr$r_t, "rendement cible", FALSE), st(tr$r_p, "rendement partenaire", FALSE)))

# ==== 3. ARIMA : identification manuelle (ACF/PACF) puis auto.arima, comparées =
ARIMA_MANUAL <- c(0, 1, 0)                                                      # à fixer après lecture des ACF/PACF
fit_auto <- auto.arima(tr$lp_t, d = 1, ic = "bic", stepwise = FALSE, approximation = FALSE, allowdrift = TRUE)
fit_man <- Arima(tr$lp_t, order = ARIMA_MANUAL, include.drift = TRUE)
print(data.frame(modele = c("auto", "manuel"), AIC = c(AIC(fit_auto), AIC(fit_man)), BIC = c(BIC(fit_auto), BIC(fit_man))))
arma_n <- sum(arimaorder(fit_auto)[c("p", "q")]); print(round(diag_resid(residuals(fit_auto), arma_n), 4))

# ==== 4. VAR sur rendements : VARselect, stabilité, Granger, IRF, FEVD ==========
y <- as.matrix(tr[, c("r_p", "r_t")])                          # ordre de Cholesky : partenaire d'abord (hypothèse à justifier)
p_var <- max(1, VARselect(y, lag.max = 10, type = "const")$selection["SC(n)"])
var_fit <- VAR(y, p = p_var, type = "const"); cat("VAR p =", p_var, "\n")
print(serial.test(var_fit, lags.pt = 12 + p_var, type = "PT.asymptotic")); print(roots(var_fit))
print(causality(var_fit, cause = "r_p")$Granger); print(causality(var_fit, cause = "r_t")$Granger)
print(arch.test(var_fit, lags.multi = 5))                     # effets ARCH multivariés dans les résidus du VAR
png("figures/R04_irf.png", 900, 600, res = 120); plot(irf(var_fit, impulse = "r_p", response = "r_t", n.ahead = 10, ortho = TRUE, boot = TRUE)); dev.off()
png("figures/R05_fevd.png", 900, 600, res = 120); plot(fevd(var_fit, n.ahead = 10)); dev.off()

# ==== 5. COINTÉGRATION (Johansen) et VECM ======================================
# Rang par la procédure séquentielle (r = 0 d'abord). La spécification déterministe change le résultat : on compare deux.
lp2 <- as.matrix(tr[, c("lp_t", "lp_p")]); K <- max(2, VARselect(lp2, lag.max = 10, type = "const")$selection["SC(n)"])
get_rank <- function(jo) { s <- rev(jo@teststat); cv <- rev(jo@cval[, "5pct"]); r <- 0
  for (i in seq_along(s)) { if (s[i] > cv[i]) r <- i else break }; r }
jo_c <- ca.jo(lp2, type = "trace", ecdet = "const", K = K); jo_t <- ca.jo(lp2, type = "trace", ecdet = "trend", K = K)
cat("Rang Johansen : constante =", get_rank(jo_c), "| tendance =", get_rank(jo_t), "\n"); print(summary(jo_c))
cat("Engle-Granger (test ADF sur résidus, valeurs critiques non standard !) :\n")
print(summary(ur.df(residuals(lm(lp_t ~ lp_p, data = tr)), type = "none", selectlags = "AIC")))
rank_r <- get_rank(jo_c); USE_VECM <- rank_r >= 1
if (USE_VECM) { vecm_ols <- cajorls(jo_c, r = rank_r); print(vecm_ols$beta); print(vecm_ols$rlm$coefficients[grep("ect", rownames(vecm_ols$rlm$coefficients)), , drop = FALSE])   # alpha = vitesse d'ajustement
  
} else cat("Pas de cointégration à 5 % : résultat à rapporter, pas de VECM.\n")

# ==== 6. ARCH / GARCH (rugarch) =================================================
cat("ARCH-LM sur résidus ARIMA :", round(arch_lm(residuals(fit_auto))["p"], 4), "\n")
mk <- function(model, dist, o = 0) ugarchspec(variance.model = list(model = model, garchOrder = c(1, 1)), mean.model = list(armaOrder = c(0, 0)), distribution.model = dist)
specs <- list("GARCH(1,1)-norm" = mk("sGARCH", "norm"), "GARCH(1,1)-std" = mk("sGARCH", "std"), "GJR-GARCH(1,1)-std" = mk("gjrGARCH", "std"), "EGARCH(1,1)-std" = mk("eGARCH", "std"))
gfits <- lapply(specs, ugarchfit, data = tr$r_t)
gtab <- do.call(rbind, lapply(names(gfits), function(nm) { z <- as.numeric(residuals(gfits[[nm]], standardize = TRUE)); ic <- infocriteria(gfits[[nm]])
  data.frame(modele = nm, AIC = ic[1], BIC = ic[2], LB_z_p = Box.test(z, 10, "Ljung-Box")$p.value, LB_z2_p = Box.test(z^2, 10, "Ljung-Box")$p.value, ARCHLM_z_p = arch_lm(z)["p"]) }))
print(gtab); best <- gtab$modele[which.min(gtab$BIC)]; show(gfits[[best]])
png("figures/R06_garch_vol.png", 1100, 450, res = 120)
plot(tr$Date, abs(tr$r_t), type = "l", col = "grey70", ylab = "%", xlab = "", main = paste("σ conditionnel -", best)); lines(tr$Date, as.numeric(sigma(gfits[[best]])), col = "red")
ev <- as.Date(c("2020-03-16", "2022-06-13", "2025-01-27", "2025-04-04")); abline(v = ev[ev <= tr$Date[n_tr]], lty = 3); dev.off()

# ==== 7. PRÉVISIONS GLISSANTES (même protocole que Python) =====================
# Naïf ; ARIMA (paramètres du train) ; VAR et VECM ré-estimés à chaque origine (rapide en R).
write_fc <- function(name, LPH) {
  out <- do.call(rbind, lapply(seq_along(origins), function(i) { t <- origins[i]
    data.frame(model = paste0("R_", name), origin = df$Date[t], h = HORIZONS, target_date = df$Date[t + HORIZONS], y_true = df$p_t[t + HORIZONS], y_pred = exp(LPH[i, HORIZONS])) }))
  write.csv(out, sprintf("results/forecasts_R_%s.csv", name), row.names = FALSE)
  cat(sprintf("%-6s h=5 RMSE=%.3f\n", name, sqrt(mean((out$y_true[out$h == 5] - out$y_pred[out$h == 5])^2))))
}
lp <- df$lp_t; k <- length(origins)
F_naif <- F_arima <- F_var <- F_vecm <- matrix(NA, k, HMAX)
for (i in seq_len(k)) { t <- origins[i]
  F_naif[i, ] <- lp[t]
  F_arima[i, ] <- as.numeric(forecast(Arima(lp[1:t], model = fit_auto), h = HMAX)$mean)
  v <- VAR(as.matrix(df[1:t, c("r_p", "r_t")]), p = p_var, type = "const")
  F_var[i, ] <- lp[t] + cumsum(predict(v, n.ahead = HMAX)$fcst$r_t[, 1]) / 100
  if (USE_VECM) F_vecm[i, ] <- predict(vec2var(ca.jo(as.matrix(df[1:t, c("lp_t", "lp_p")]), type = "trace", ecdet = "const", K = K), r = rank_r), n.ahead = HMAX)$fcst$lp_t[, 1]
}
write_fc("naif", F_naif); write_fc("arima", F_arima); write_fc("var", F_var); if (USE_VECM) write_fc("vecm", F_vecm)
cat("OK -> results/forecasts_R_*.csv (à comparer avec Python via 04_comparaison.py)\n")
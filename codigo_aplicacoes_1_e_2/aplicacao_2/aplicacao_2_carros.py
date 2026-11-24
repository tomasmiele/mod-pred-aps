"""Aplicação 2 - Preço de Mercedes usados: Regressão Linear Múltipla, Árvore de Regressão, Random Forest e Boosting."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

SEED = 42
d = pd.read_csv(Path(__file__).parent / "used_cars.csv")
y = d.price
X = pd.get_dummies(d.drop(columns="price"), columns=["trim", "isOneOwner", "color", "fuel", "soundSystem", "wheelType"], drop_first=True).astype(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=SEED)
print("treino/teste:", len(Xtr), len(Xte), "| preço médio treino/teste: %.3f %.3f (unidade do arquivo)" % (ytr.mean(), yte.mean()))
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
preds = {}

lm = LinearRegression().fit(Xtr, ytr); preds["Regressão linear"] = lm.predict(Xte)
print("R2 treino da linear: %.4f" % lm.score(Xtr, ytr), "| previsões negativas no teste:", int((preds["Regressão linear"] < 0).sum()))
print(pd.Series(lm.coef_, index=X.columns).round(3).sort_values().to_string())

path = DecisionTreeRegressor(random_state=SEED).cost_complexity_pruning_path(Xtr, ytr)
alphas = np.unique(path.ccp_alphas); alphas = alphas[:: max(1, len(alphas) // 60)]
cv = KFold(10, shuffle=True, random_state=SEED)
cv_rmse = [np.sqrt(-cross_val_score(DecisionTreeRegressor(ccp_alpha=a, random_state=SEED), Xtr, ytr, cv=cv, scoring="neg_mean_squared_error").mean()) for a in alphas]
best = alphas[int(np.argmin(cv_rmse))]
tree = DecisionTreeRegressor(ccp_alpha=best, random_state=SEED).fit(Xtr, ytr); preds["Árvore de regressão"] = tree.predict(Xte)
print("poda: alpha=%.6f folhas=%d profundidade=%d RMSE CV=%.4f" % (best, tree.get_n_leaves(), tree.get_depth(), min(cv_rmse)))

rf = RandomForestRegressor(n_estimators=500, max_features=1/3, oob_score=True, random_state=SEED, n_jobs=2).fit(Xtr, ytr); preds["Random Forest"] = rf.predict(Xte)
print("RF RMSE OOB: %.4f" % rmse(ytr, rf.oob_prediction_))

grid = {"n_estimators": [300, 600, 1000], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]}
gb = GridSearchCV(GradientBoostingRegressor(random_state=SEED, subsample=0.8), grid, cv=5, scoring="neg_mean_squared_error", n_jobs=2).fit(Xtr, ytr)
print("boosting escolhido:", gb.best_params_); preds["Boosting"] = gb.best_estimator_.predict(Xte)

out = pd.DataFrame([dict(Modelo=k, RMSE_teste=rmse(yte, p), R2_teste=1 - np.sum((yte - p) ** 2) / np.sum((yte - yte.mean()) ** 2)) for k, p in preds.items()]).sort_values("RMSE_teste")
print("\nDesvio-padrão do preço no teste: %.4f" % yte.std()); print(out.round(4).to_string(index=False)); out.to_csv("resultados_carros.csv", index=False)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False); print("\n", imp.head(8).round(4))

fig, axs = plt.subplots(2, 2, figsize=(9.5, 9), sharex=True, sharey=True)
lim = [0, max(yte.max(), max(p.max() for p in preds.values())) * 1.02]
for ax, (nome, p) in zip(axs.ravel(), preds.items()):
    ax.scatter(yte, p, s=4, alpha=.35); ax.plot(lim, lim, "r--", lw=1); ax.set_title("%s (RMSE = %.3f)" % (nome, rmse(yte, p)), fontsize=10); ax.grid(alpha=.3)
for ax in axs[1]: ax.set_xlabel("Preço observado (teste)")
for ax in axs[:, 0]: ax.set_ylabel("Preço previsto")
fig.tight_layout(); fig.savefig("previsto_vs_observado.png", dpi=200)

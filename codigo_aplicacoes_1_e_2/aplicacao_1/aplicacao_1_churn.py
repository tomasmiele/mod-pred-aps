"""Aplicação 1 - Churn: k-NN, Regressão Logística, Árvore de Classificação, Random Forest e Boosting."""
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

SEED = 42
d = pd.read_csv(Path(__file__).parent / "churn.csv")
d["y"] = (d.Exited == "Yes").astype(int)                 # classe positiva: Exited = Yes
X = pd.get_dummies(d.drop(columns=["Exited", "y"]), columns=["Geography", "Gender"], drop_first=True).astype(float)
y = d.y
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=SEED, stratify=y)
print("treino/teste:", len(Xtr), len(Xte), "| prop. churn treino/teste: %.4f %.4f" % (ytr.mean(), yte.mean()))
res, probs = {}, {}

# ---- k-NN: regra de bolso k ~ raiz(n_treino); padronização das preditoras (k-NN usa distâncias)
k = int(round(np.sqrt(len(Xtr)))); k += (k % 2 == 0)          # força k ímpar
sc = StandardScaler().fit(Xtr)
knn = KNeighborsClassifier(n_neighbors=k).fit(sc.transform(Xtr), ytr)
probs["k-NN"] = knn.predict_proba(sc.transform(Xte))[:, 1]
print("k do k-NN:", k)

# ---- Regressão logística
lg = sm.Logit(ytr, sm.add_constant(Xtr)).fit(disp=0)
print(lg.summary2().tables[1].round(4))
probs["Regressão logística"] = lg.predict(sm.add_constant(Xte)).values

# ---- Árvore de classificação com poda por custo-complexidade (CV 10 lotes no treino)
path = DecisionTreeClassifier(random_state=SEED).cost_complexity_pruning_path(Xtr, ytr)
alphas = np.unique(np.round(path.ccp_alphas, 8)); alphas = alphas[:: max(1, len(alphas) // 60)]
cv = StratifiedKFold(10, shuffle=True, random_state=SEED)
cv_err = [1 - cross_val_score(DecisionTreeClassifier(ccp_alpha=a, random_state=SEED), Xtr, ytr, cv=cv).mean() for a in alphas]
best = alphas[int(np.argmin(cv_err))]
tree = DecisionTreeClassifier(ccp_alpha=best, random_state=SEED).fit(Xtr, ytr)
probs["Árvore"] = tree.predict_proba(Xte)[:, 1]
print("poda: alpha=%.6f folhas=%d profundidade=%d erro CV=%.4f" % (best, tree.get_n_leaves(), tree.get_depth(), min(cv_err)))

# ---- Random Forest (500 árvores, m = raiz(p) por divisão)
rf = RandomForestClassifier(n_estimators=500, max_features="sqrt", oob_score=True, random_state=SEED, n_jobs=2).fit(Xtr, ytr)
probs["Random Forest"] = rf.predict_proba(Xte)[:, 1]
print("RF acurácia OOB: %.4f" % rf.oob_score_)

# ---- Boosting (gradient boosting de árvores) - hiperparâmetros por CV 5 lotes no treino
grid = {"n_estimators": [100, 300, 500], "max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1]}
gb = GridSearchCV(GradientBoostingClassifier(random_state=SEED, subsample=0.8), grid, cv=5, scoring="accuracy", n_jobs=2).fit(Xtr, ytr)
print("boosting escolhido:", gb.best_params_)
probs["Boosting"] = gb.best_estimator_.predict_proba(Xte)[:, 1]

# ---- Avaliação no teste (corte de 50%)
base = max(yte.mean(), 1 - yte.mean())
rows = []
for nome, p in probs.items():
    pred = (p >= 0.5).astype(int); tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    rows.append(dict(Modelo=nome, Acuracia=(pred == yte).mean(), Sensibilidade=tp / (tp + fn), Especificidade=tn / (tn + fp), AUC=roc_auc_score(yte, p), VP=tp, FN=fn, FP=fp, VN=tn))
out = pd.DataFrame(rows).sort_values("AUC", ascending=False)
print("\nReferência (sempre 'No'): acurácia = %.4f" % base)
print(out.round(4).to_string(index=False))
out.to_csv("resultados_churn.csv", index=False)

# ---- Importância das variáveis (RF, impureza) para a discussão
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False); print("\n", imp.round(4))

# ---- Curvas ROC
fig, ax = plt.subplots(figsize=(6.2, 5.6))
for nome, p in probs.items():
    f, t, _ = roc_curve(yte, p); ax.plot(f, t, lw=1.8, label="%s (AUC = %.3f)" % (nome, roc_auc_score(yte, p)))
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("1 - especificidade (taxa de falsos positivos)"); ax.set_ylabel("Sensibilidade (taxa de verdadeiros positivos)")
ax.set_title("Curvas ROC no conjunto de teste - churn"); ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("roc_churn.png", dpi=200)

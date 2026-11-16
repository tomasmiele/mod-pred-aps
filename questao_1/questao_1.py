"""APS de Modelagem Preditiva — somente a questão 1 da parte teórica.

Uso, a partir da pasta do projeto: python questao_1/questao_1.py
Dependências: python -m pip install -r questao_1/requirements.txt

Referências locais: aulas 02 e 03, preservadas em referencias/.
O teste fornecido é usado somente para a avaliação final. Não é redividido.
"""

from pathlib import Path
import argparse
import json
import platform

import matplotlib
matplotlib.use("Agg")  # Salva figuras mesmo sem uma janela gráfica aberta.
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import sklearn
import statsmodels
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text


BASE = Path(__file__).resolve().parent
SEMENTE = 42
VARIAVEIS = ["x1", "x2"]
COR_LOG = "#B04B32"
COR_ARV = "#205D83"


def ler_dados(caminho):
    """Lê um CSV e valida as colunas, os valores e as classes da resposta y."""
    dados = pd.read_csv(caminho)
    if set(dados.columns) != {"x1", "x2", "y"}:
        raise ValueError(f"{caminho}: são esperadas exatamente as colunas x1, x2 e y.")
    if dados.empty or not np.isfinite(dados.to_numpy(dtype=float)).all():
        raise ValueError(f"{caminho}: há valores ausentes, não finitos ou dados vazios.")
    if set(dados.y.unique()) != {0, 1}:
        raise ValueError(f"{caminho}: y deve conter as classes 0 e 1.")
    if dados.duplicated().any():
        raise ValueError(f"{caminho}: existem observações duplicadas; revise a origem.")
    return dados


def criar_arvore(alpha=0.0):
    """Cria uma árvore com configurações fixas e a intensidade de poda informada."""
    # Entropia corresponde ao critério de deviance usado no crescimento em tree.
    # min_samples_split=10 e min_samples_leaf=5 aproximam minsize/mincut de tree.
    # A poda ccp_alpha usa impureza; não é uma tradução numérica de prune.misclass.
    return DecisionTreeClassifier(
        criterion="entropy", min_samples_split=10, min_samples_leaf=5,
        ccp_alpha=float(alpha), random_state=SEMENTE,
    )


def selecionar_poda(X, y):
    """Compara podas no treino e retorna a tabela CV, a árvore ajustada e a escolha."""
    # Grade fixada antes de avaliar o teste. Todos os candidatos usam os mesmos
    # dez lotes de treino. Em empate, escolhemos o maior alpha (poda mais forte).
    alphas = np.r_[0.0, np.logspace(-4, -1, 41)]
    lotes = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEMENTE)
    linhas = []
    for alpha in alphas:
        modelo = criar_arvore(alpha)
        erros = 1 - cross_val_score(modelo, X, y, cv=lotes,
                                     scoring="accuracy", n_jobs=1)
        modelo.fit(X, y)
        linhas.append({"alpha": float(alpha), "erro_cv": float(erros.mean()),
                       "erro_padrao_cv": float(erros.std(ddof=1) / np.sqrt(10)),
                       "folhas_no_treino": modelo.get_n_leaves()})
    tabela = pd.DataFrame(linhas)
    minimo = tabela.erro_cv.min()
    escolhido = tabela.loc[np.isclose(tabela.erro_cv, minimo, rtol=0, atol=1e-12)].iloc[-1]
    return tabela, criar_arvore(escolhido.alpha).fit(X, y), escolhido


def avaliar(nome, y, probabilidades):
    """Converte probabilidades em classes e retorna métricas e previsões no corte 0,50."""
    # Como no ifelse(prob >= threshold, ...) do professor, classe positiva = 1.
    previsto = (np.asarray(probabilidades) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, previsto, labels=[0, 1]).ravel()
    return {"modelo": nome, "acuracia": float((tn + tp) / len(y)),
            "erro": float((fp + fn) / len(y)),
            "sensibilidade": float(tp / (tp + fn)),
            "especificidade": float(tn / (tn + fp)),
            "auc": float(roc_auc_score(y, probabilidades)),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}, previsto


def salvar_figura(fig, destino):
    """Salva uma figura em PNG e fecha a imagem para liberar memória."""
    fig.savefig(destino, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figuras(treino, teste, logistica, arvore, probs, saida):
    """Gera as duas figuras do relatório: dados/regiões previstas e curvas ROC."""
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"})
    # O gráfico geométrico usa somente os dados de treinamento.
    limites = [(-3.6, 3.8), (-3.2, 3.9)]
    xx, yy = np.meshgrid(np.linspace(*limites[0], 250), np.linspace(*limites[1], 250))
    grade = pd.DataFrame({"x1": xx.ravel(), "x2": yy.ravel()})
    p_log = np.asarray(logistica.predict(sm.add_constant(grade, has_constant="add"))).reshape(xx.shape)
    p_arv = arvore.predict_proba(grade)[:, 1].reshape(xx.shape)
    fig, eixos = plt.subplots(1, 3, figsize=(10, 3.8), constrained_layout=True)
    cores = ListedColormap(["#2673A1", "#C35D3E"])
    for eixo, titulo, prob in zip(eixos, ["Dados de treinamento", "Regressão logística", "Árvore de classificação"], [None, p_log, p_arv]):
        if prob is not None:
            eixo.contourf(xx, yy, (prob >= .5).astype(int), levels=[-.5, .5, 1.5], cmap=cores, alpha=.15)
            if prob.min() < .5 < prob.max():
                eixo.contour(xx, yy, prob, levels=[.5], colors="black", linewidths=1)
            else:
                eixo.text(.04, .95, "Todo o plano exibido: classe 1", transform=eixo.transAxes,
                          va="top", fontsize=9)
        for classe, marcador in [(0, "o"), (1, "^")]:
            dados = treino[treino.y == classe]
            eixo.scatter(dados.x1, dados.x2, s=9, alpha=.5, marker=marcador,
                         color=cores(classe), label=f"y = {classe}")
        eixo.set(xlim=limites[0], ylim=limites[1], xlabel="x1", ylabel="x2", title=titulo)
    eixos[0].legend(loc="upper left", fontsize=9)
    salvar_figura(fig, saida / "01_dados_e_regioes.png")

    fig, eixo = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    for nome, prob, cor in zip(probs, probs.values(), [COR_LOG, COR_ARV]):
        fpr, tpr, _ = roc_curve(teste.y, prob, drop_intermediate=False)
        eixo.plot(fpr, tpr, color=cor, lw=2,
                  label=f"{nome} (AUC = {roc_auc_score(teste.y, prob):.4f})")
        tn, fp, fn, tp = confusion_matrix(teste.y, prob >= .5, labels=[0, 1]).ravel()
        eixo.scatter(fp / (fp + tn), tp / (tp + fn), color=cor, zorder=3)
    eixo.plot([0, 1], [0, 1], "--", color="gray", label="Referência aleatória")
    eixo.set(xlabel="FPR = 1 − especificidade", ylabel="TPR = sensibilidade",
             title="Curvas ROC no teste", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    eixo.text(.04, .06, "Pontos: corte de probabilidade 0,50", transform=eixo.transAxes, fontsize=9)
    eixo.legend(loc="lower right", fontsize=9); eixo.grid(alpha=.2)
    salvar_figura(fig, saida / "02_curvas_roc.png")


def main():
    """Coordena a leitura, o treinamento, a avaliação e a gravação dos resultados."""
    # 1. Define as pastas e lê os dois conjuntos fornecidos na questão.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=BASE / "dados")
    parser.add_argument("--saida", type=Path, default=BASE / "resultados")
    args = parser.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    treino = ler_dados(args.dados / "Q1_training.csv")
    teste = ler_dados(args.dados / "Q1_test.csv")
    if not treino.merge(teste, on=VARIAVEIS, how="inner").empty:
        raise ValueError("Há preditores idênticos nos arquivos de treino e teste.")
    X, y = treino[VARIAVEIS], treino.y
    # 2. Ajusta a regressão logística e seleciona/ajusta a árvore usando só o treino.
    # GLM binomial com intercepto e sem penalização: paralelo ao glm(...,
    # family=binomial) de R. Não criamos termos quadráticos ou interações.
    logistica = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit()
    tabela_cv, arvore, escolhido = selecionar_poda(X, y)
    # 3. Estima probabilidades no teste e calcula as métricas de cada modelo.
    probs = {"Regressão logística": np.asarray(logistica.predict(sm.add_constant(teste[VARIAVEIS]))),
             "Árvore de classificação": arvore.predict_proba(teste[VARIAVEIS])[:, 1]}
    metricas = []
    previsoes = teste.copy()
    for nome, prob in probs.items():
        linha, previsto = avaliar(nome, teste.y, prob)
        metricas.append(linha)
        sufixo = "logistica" if nome == "Regressão logística" else "arvore"
        previsoes[f"prob_{sufixo}"] = prob
        previsoes[f"prev_{sufixo}"] = previsto
    metricas = pd.DataFrame(metricas)
    # 4. Exporta previsões, métricas e informações dos modelos para consulta.
    metricas.to_csv(args.saida / "metricas_teste.csv", index=False)
    previsoes.to_csv(args.saida / "previsoes_teste.csv", index=False)
    tabela_cv.to_csv(args.saida / "validacao_cruzada.csv", index=False)
    coef = pd.DataFrame({"coeficiente": logistica.params, "erro_padrao": logistica.bse,
                         "valor_p": logistica.pvalues})
    coef.to_csv(args.saida / "coeficientes_logistica.csv", index_label="termo")
    # Matrizes na orientação das aulas: linhas previstas e colunas observadas.
    for nome in ["logistica", "arvore"]:
        matriz = pd.crosstab(previsoes[f"prev_{nome}"], teste.y,
                              rownames=["Previsto"], colnames=["Observado"]).reindex(
                                  index=[0, 1], columns=[0, 1], fill_value=0)
        matriz.to_csv(args.saida / f"matriz_confusao_{nome}.csv")
    diagnostico = {
        "n_treino": len(treino), "n_teste": len(teste),
        "classes_treino": {str(k): int(v) for k, v in treino.y.value_counts().items()},
        "classes_teste": {str(k): int(v) for k, v in teste.y.value_counts().items()},
        "classe_majoritaria_treino": int(y.mode().iloc[0]),
        "acuracia_baseline_teste": float((teste.y == y.mode().iloc[0]).mean()),
        "alpha": float(escolhido.alpha), "erro_cv": float(escolhido.erro_cv),
        "folhas": int(arvore.get_n_leaves()), "profundidade": int(arvore.get_depth()),
        "prob_logistica_min_teste": float(probs["Regressão logística"].min()),
        "prob_logistica_max_teste": float(probs["Regressão logística"].max()),
        "metricas": metricas.to_dict(orient="records"),
        "versoes": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "statsmodels": statsmodels.__version__, "matplotlib": matplotlib.__version__},
    }
    (args.saida / "resumo.json").write_text(json.dumps(diagnostico, ensure_ascii=False, indent=2), encoding="utf-8")
    regras = export_text(arvore, feature_names=VARIAVEIS, decimals=6)
    (args.saida / "regras_arvore.txt").write_text(regras, encoding="utf-8")
    (args.saida / "resumo_logistica.txt").write_text(str(logistica.summary()), encoding="utf-8")
    # 5. Gera somente os dois gráficos utilizados no relatório e exibe o resumo.
    figuras(treino, teste, logistica, arvore, probs, args.saida)
    print("\nMétricas de teste (classe positiva = 1; corte = 0,50):")
    print(metricas.to_string(index=False))
    print(f"\nÁrvore: {arvore.get_n_leaves()} folhas; alpha={escolhido.alpha:.6f}; erro CV={escolhido.erro_cv:.3f}")
    print(f"Baseline majoritário no teste: {diagnostico['acuracia_baseline_teste']:.3f}")
    print(f"Resultados salvos em: {args.saida.resolve()}")


if __name__ == "__main__":
    main()

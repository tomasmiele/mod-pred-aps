"""Questão 2 — bootstrap e bagging manual de árvores de regressão.

Uso, na pasta principal: python questao_2/questao_2.py
Dependências: python -m pip install -r questao_2/requirements.txt

O único estimador utilizado é DecisionTreeRegressor. Os sorteios bootstrap,
os subespaços por árvore e a agregação das previsões são programados no laço.
"""

from pathlib import Path
import argparse
import hashlib
import json
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon
import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeRegressor


BASE = Path(__file__).resolve().parent
ALVO = "median_house_value"
NUMERICAS = ["longitude", "latitude", "housing_median_age", "total_rooms",
             "total_bedrooms", "population", "households", "median_income"]
PREDITORAS = NUMERICAS + ["ocean_proximity"]
SEMENTE_DIVISAO = 42
SEMENTE_INTERNA = 314
SEMENTE_VALIDACAO = 2026
SEMENTE_FINAL = 42
CANDIDATOS_M = (3, 5, 7, 8)


def carregar_dados(caminho):
    """Lê e valida a versão de California Housing fornecida para a questão."""
    dados = pd.read_csv(caminho)
    if set(dados.columns) != set(PREDITORAS + [ALVO]):
        raise ValueError("As colunas do CSV não correspondem às esperadas para esta base.")
    if dados.empty or dados.isna().any().any():
        raise ValueError("Há dados vazios ou ausentes; revise a base antes do experimento.")
    if not np.isfinite(dados[NUMERICAS + [ALVO]].to_numpy(dtype=float)).all():
        raise ValueError("Há valores numéricos não finitos.")
    if dados.duplicated().any():
        raise ValueError("Há linhas duplicadas; revise a base antes da separação.")
    return dados


def preparar_dados(treino, avaliacao):
    """Codifica a categoria com níveis do treino e mantém as nove variáveis em grupos.

    Ao sortear ocean_proximity para uma árvore, entram todas as suas colunas
    indicadoras. Assim, m conta variáveis originais e não colunas dummy.
    """
    categorias = sorted(treino.ocean_proximity.unique().tolist())
    desconhecidas = set(avaliacao.ocean_proximity.unique()) - set(categorias)
    if desconhecidas:
        raise ValueError(f"Categorias ausentes do treino: {desconhecidas}")
    def transformar(dados):
        colunas = [dados[NUMERICAS].to_numpy(dtype=float)]
        colunas.extend((dados.ocean_proximity == nivel).to_numpy(dtype=float)[:, None]
                       for nivel in categorias)
        return np.column_stack(colunas)
    grupos = [np.array([i]) for i in range(len(NUMERICAS))]
    grupos.append(np.arange(len(NUMERICAS), len(NUMERICAS) + len(categorias)))
    nomes = NUMERICAS + [f"ocean_proximity={c}" for c in categorias]
    return transformar(treino), transformar(avaliacao), grupos, nomes


def criar_arvore_regressao(semente):
    """Cria uma árvore profunda e sem poda, com erro quadrático nos cortes."""
    return DecisionTreeRegressor(
        criterion="squared_error", max_depth=None, min_samples_split=2,
        min_samples_leaf=1, max_features=None, ccp_alpha=0.0,
        random_state=int(semente),
    )


def calcular_rmse(observado, previsto):
    """Calcula a raiz do erro quadrático médio na unidade da variável resposta."""
    return float(np.sqrt(np.mean((np.asarray(observado) - previsto) ** 2)))


def treinar_bagging(X, y, X_avaliacao, y_avaliacao, grupos, B, m, semente, nome):
    """Executa o bagging manual e devolve previsões, curva e registro dos sorteios.

    Linhas são sorteadas com reposição; variáveis, sem reposição e uma única vez
    por árvore. Dentro da árvore todos os preditores do subespaço ficam disponíveis.
    Fluxos de sorteio separados garantem os mesmos bootstraps nas comparações de m.
    Mantemos uma soma acumulada, sem armazenar todas as árvores ou uma matriz B x n.
    """
    fluxos = np.random.SeedSequence(semente).spawn(3)
    rng_linhas, rng_variaveis, rng_arvores = [np.random.default_rng(f) for f in fluxos]
    n = len(y)
    soma = np.zeros(len(X_avaliacao), dtype=float)
    pontos = {b for b in (1, 10, 25, 50, 100, 200, 500, B) if b <= B}
    curva, registro = [], []
    for b in range(1, B + 1):
        # Bootstrap de pares (X_i, y_i): a mesma linha é usada em X e y.
        indices = rng_linhas.integers(0, n, size=n)
        if m == len(grupos):
            variaveis = np.arange(len(grupos))
        else:
            variaveis = np.sort(rng_variaveis.choice(len(grupos), size=m, replace=False))
        colunas = np.concatenate([grupos[j] for j in variaveis])
        arvore = criar_arvore_regressao(rng_arvores.integers(0, 2**31 - 1))
        arvore.fit(X[indices][:, colunas], y[indices])
        soma += arvore.predict(X_avaliacao[:, colunas])
        registro.append({
            "arvore": b, "n_sorteios": n,
            "n_observacoes_distintas": int(np.unique(indices).size),
            "variaveis": ";".join(PREDITORAS[j] for j in variaveis),
            "sha256_bootstrap": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        })
        if b in pontos:
            erro = calcular_rmse(y_avaliacao, soma / b)
            curva.append({"modelo": nome, "arvores": b, "rmse": erro})
            print(f"{nome}: {b}/{B} árvores; RMSE = {erro:,.2f}", flush=True)
    return soma / B, pd.DataFrame(curva), pd.DataFrame(registro)


def selecionar_subespaco(treino, B):
    """Escolhe m usando somente uma separação interna 80/20 do treino externo."""
    ordem = np.random.default_rng(SEMENTE_INTERNA).permutation(len(treino))
    n_ajuste = int(.8 * len(treino))
    ajuste, validacao = treino.iloc[ordem[:n_ajuste]], treino.iloc[ordem[n_ajuste:]]
    X, X_val, grupos, _ = preparar_dados(ajuste, validacao)
    y, y_val = ajuste[ALVO].to_numpy(), validacao[ALVO].to_numpy()
    linhas = []
    for m in (*CANDIDATOS_M, len(PREDITORAS)):
        nome = f"Validação m={m}"
        previsto, _, _ = treinar_bagging(X, y, X_val, y_val, grupos, B, m,
                                        SEMENTE_VALIDACAO, nome)
        linhas.append({"m": m, "arvores": B, "n_ajuste": len(ajuste),
                       "n_validacao": len(validacao), "rmse_validacao": calcular_rmse(y_val, previsto)})
    tabela = pd.DataFrame(linhas)
    # O candidato de bagging puro (m=p) é referência; m é escolhido entre m<p.
    reduzidos = tabela[tabela.m < len(PREDITORAS)]
    escolhido = reduzidos.sort_values(["rmse_validacao", "m"], ascending=[True, False]).iloc[0]
    return int(escolhido.m), tabela


def gerar_exemplo_bootstrap():
    """Treina três árvores num exemplo de cinco casas com reamostragens ilustrativas."""
    original = pd.DataFrame({"casa": list("ABCDE"), "area_m2": [50, 70, 90, 110, 130],
                             "preco_mil_dolares": [100, 150, 200, 250, 300]})
    sorteios = np.array([[0, 2, 2, 4, 0], [1, 1, 3, 0, 4], [4, 2, 3, 3, 1]])
    previsoes = []
    for i, indices in enumerate(sorteios):
        arvore = criar_arvore_regressao(i)
        arvore.fit(original.loc[indices, ["area_m2"]].to_numpy(),
                   original.loc[indices, "preco_mil_dolares"].to_numpy())
        previsoes.append(float(arvore.predict(np.array([[95]]))[0]))
    return original, sorteios, np.array(previsoes)


def gerar_figuras(original, sorteios, previsoes_exemplo, curvas, saida):
    """Cria dois esquemas digitais para desenho manual e um gráfico do experimento."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    cores = ["#CBE4F1", "#DFEBCB", "#F4DFBE", "#E6D6ED", "#F1CED0"]
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    ax.set(xlim=(0, 10), ylim=(0, 5.1)); ax.axis("off")
    ax.text(.1, 4.8, "Cinco casas observadas", weight="bold", fontsize=13)
    for j, row in enumerate(original.itertuples()):
        x = .1 + j * 1.95
        ax.add_patch(Rectangle((x, 3.55), 1.8, .95, facecolor=cores[j], edgecolor="#777777"))
        ax.text(x+.9, 4.03, f"{row.casa}\n{row.area_m2} m² | US$ {row.preco_mil_dolares} mil",
                ha="center", va="center", fontsize=10)
    for i, indices in enumerate(sorteios):
        y = 2.55 - i * .85
        ax.text(.1, y+.25, f"Bootstrap {i+1}", weight="bold", va="center")
        for j, k in enumerate(indices):
            x=2.2+j*1.05
            ax.add_patch(Rectangle((x,y),.85,.55,facecolor=cores[k],edgecolor="#777777"))
            ax.text(x+.425,y+.275,original.casa.iloc[k],ha="center",va="center",weight="bold")
        ax.text(7.65,y+.25,f"{len(set(indices))} casas distintas",va="center",fontsize=10)
    ax.text(.1,.15,"Cada bootstrap contém cinco sorteios com reposição de linhas completas.",fontsize=11)
    fig.savefig(saida/"01_exemplo_bootstrap.png",dpi=190,bbox_inches="tight");plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    ax.set(xlim=(0,10),ylim=(0,5));ax.axis("off")
    ax.text(.1,4.65,"Previsão para uma nova casa de 95 m²",weight="bold",fontsize=13)
    for i, (indices, pred) in enumerate(zip(sorteios,previsoes_exemplo)):
        y=3.75-i*1.05
        labels=" ".join(original.casa.iloc[indices])
        ax.text(.1,y,f"Bootstrap {i+1}\n{labels}",va="center",fontsize=11)
        ax.add_patch(FancyArrowPatch((2,y),(2.8,y),arrowstyle="->",mutation_scale=14,color="#555555"))
        ax.add_patch(Polygon([[3,y-.4],[4.8,y-.4],[3.9,y+.4]],closed=True,facecolor="#CBE4F1",edgecolor="#555555"))
        ax.text(3.9,y-.12,f"Árvore {i+1}",ha="center",va="center",fontsize=10)
        ax.text(5.25,y,f"{pred:.0f}",ha="center",va="center",weight="bold",fontsize=13)
        ax.add_patch(FancyArrowPatch((5.7,y),(7.15,2.7),arrowstyle="->",mutation_scale=14,color="#555555"))
    ax.add_patch(Rectangle((7.2,2.05),2.55,1.3,facecolor="#DFEBCB",edgecolor="#555555"))
    ax.text(8.48,2.7,f"Média das previsões\n{previsoes_exemplo.mean():.1f} mil dólares".replace(".",","),ha="center",va="center")
    ax.text(.1,.3,"(200 + 250 + 200) / 3 = 216,7 mil dólares. Dados fictícios, distintos da base California.",fontsize=10)
    fig.savefig(saida/"02_exemplo_agregacao.png",dpi=190,bbox_inches="tight");plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2,4.2),constrained_layout=True)
    for (nome, dados), cor in zip(curvas.groupby("modelo", sort=False),["#205D83","#B04B32"]):
        ax.plot(dados.arvores,dados.rmse/1000,"o-",color=cor,label=nome)
    ax.set(xlabel="Número de árvores agregadas",ylabel="RMSE de teste (mil dólares)",
           title="Bagging e bagging com subespaço aleatório")
    ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.savefig(saida/"03_rmse_por_numero_arvores.png",dpi=190,bbox_inches="tight");plt.close(fig)


def main():
    """Coordena a divisão 50/50, a seleção interna e a comparação final no teste."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados",type=Path,default=BASE/"dados/california.csv")
    parser.add_argument("--saida",type=Path,default=BASE/"resultados")
    parser.add_argument("--arvores",type=int,default=1000)
    parser.add_argument("--arvores-validacao",type=int,default=200)
    args=parser.parse_args()
    if args.arvores<1 or args.arvores_validacao<1:raise ValueError("Use ao menos uma árvore.")
    args.saida.mkdir(parents=True,exist_ok=True)
    dados=carregar_dados(args.dados)
    ordem=np.random.default_rng(SEMENTE_DIVISAO).permutation(len(dados))
    n_treino=len(dados)//2
    idx_trn,idx_tst=ordem[:n_treino],ordem[n_treino:]
    treino,teste=dados.iloc[idx_trn],dados.iloc[idx_tst]
    pd.DataFrame({"linha_dados":np.arange(len(dados)),
                  "conjunto":np.where(np.isin(np.arange(len(dados)),idx_trn),"treino","teste")}).to_csv(args.saida/"divisao_dados.csv",index=False)
    print(f"Treino: {len(treino)}; teste: {len(teste)}; {len(PREDITORAS)} preditoras.",flush=True)
    m, validacao=selecionar_subespaco(treino,args.arvores_validacao)
    validacao.to_csv(args.saida/"selecao_subespaco_validacao.csv",index=False)
    print(f"Subespaço escolhido sem consultar o teste: m={m}.",flush=True)
    X,X_tst,grupos,nomes_codificados=preparar_dados(treino,teste)
    y,y_tst=treino[ALVO].to_numpy(),teste[ALVO].to_numpy()
    simples=criar_arvore_regressao(SEMENTE_FINAL).fit(X,y)
    pred_simples=simples.predict(X_tst)
    pred_bag,curva_bag,registro_bag=treinar_bagging(X,y,X_tst,y_tst,grupos,args.arvores,len(PREDITORAS),SEMENTE_FINAL,"Bagging")
    pred_sub,curva_sub,registro_sub=treinar_bagging(X,y,X_tst,y_tst,grupos,args.arvores,m,SEMENTE_FINAL,"Bagging + subespaço")
    curvas=pd.concat([curva_bag,curva_sub],ignore_index=True)
    curvas.to_csv(args.saida/"rmse_por_numero_arvores.csv",index=False)
    registro_bag.to_csv(args.saida/"sorteios_bagging.csv",index=False)
    registro_sub.to_csv(args.saida/"sorteios_subespaco.csv",index=False)
    previsoes=pd.DataFrame({"linha_dados":idx_tst,"observado":y_tst,"arvore_unica":pred_simples,
                            "bagging":pred_bag,"bagging_subespaco":pred_sub})
    previsoes.to_csv(args.saida/"previsoes_teste.csv",index=False)
    metricas=pd.DataFrame([
        {"modelo":"Árvore única","m":len(PREDITORAS),"arvores":1,"rmse_teste":calcular_rmse(y_tst,pred_simples)},
        {"modelo":"Bagging","m":len(PREDITORAS),"arvores":args.arvores,"rmse_teste":calcular_rmse(y_tst,pred_bag)},
        {"modelo":"Bagging + subespaço","m":m,"arvores":args.arvores,"rmse_teste":calcular_rmse(y_tst,pred_sub)},
    ])
    metricas.to_csv(args.saida/"metricas_teste.csv",index=False)
    original,sorteios,pred_exemplo=gerar_exemplo_bootstrap()
    original.to_csv(args.saida/"exemplo_casas.csv",index=False)
    pd.DataFrame({"arvore":[1,2,3],"casas_sorteadas":[" ".join(original.casa.iloc[i]) for i in sorteios],
                  "previsao_mil_dolares":pred_exemplo}).to_csv(args.saida/"exemplo_previsoes.csv",index=False)
    gerar_figuras(original,sorteios,pred_exemplo,curvas,args.saida)
    erro_bag,erro_sub=metricas.rmse_teste.iloc[1:3]
    resumo={"n_total":len(dados),"n_treino":len(treino),"n_teste":len(teste),
            "alvo":ALVO,"p":len(PREDITORAS),"colunas_codificadas":nomes_codificados,
            "m_escolhido":m,"B_final":args.arvores,"B_validacao":args.arvores_validacao,
            "sementes":{"divisao":SEMENTE_DIVISAO,"interna":SEMENTE_INTERNA,
                         "validacao":SEMENTE_VALIDACAO,"final":SEMENTE_FINAL},
            "rmse_metricas":metricas.to_dict(orient="records"),
            "ganho_subespaco_percentual":float(100*(erro_bag-erro_sub)/erro_bag),
            "fracao_unicos_media_bagging":float(registro_bag.n_observacoes_distintas.mean()/len(treino)),
            "fracao_unicos_teorica":float(1-(1-1/len(treino))**len(treino)),
            "mesmos_bootstraps":bool(registro_bag.sha256_bootstrap.equals(registro_sub.sha256_bootstrap)),
            "n_no_teto":int((dados[ALVO]==500001).sum()),
            "versoes":{"python":platform.python_version(),"numpy":np.__version__,
                        "pandas":pd.__version__,"scikit-learn":sklearn.__version__,"matplotlib":matplotlib.__version__}}
    (args.saida/"resumo.json").write_text(json.dumps(resumo,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\nComparação final no teste:",flush=True);print(metricas.to_string(index=False),flush=True)
    print(f"Ganho relativo do subespaço frente ao bagging: {resumo['ganho_subespaco_percentual']:.2f}%",flush=True)
    print(f"Resultados: {args.saida.resolve()}",flush=True)


if __name__=="__main__":
    main()

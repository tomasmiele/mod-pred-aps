# Questão 1 da parte teórica

Comparação de regressão logística e árvore de classificação nos arquivos fornecidos.
As demais questões da APS não foram implementadas.

## Executar no VS Code

Abra a pasta `mod-pred-aps` no VS Code. No terminal integrado:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r questao_1/requirements.txt
python questao_1/questao_1.py
```

Selecione o interpretador `.venv/bin/python` no VS Code. O programa localiza os CSVs
relativamente ao próprio arquivo, independentemente da pasta atual do terminal.
Recomendado: Python 3.12 ou superior. Execução verificada em Python 3.12.

## Arquivos

- `questao_1.py`: código comentado da análise, sem dependência de R.
- `dados/`: cópias intactas dos dois CSVs enviados.
- `referencias/`: cópias dos scripts do professor.
- `resultados/`: métricas, previsões individuais, matrizes de confusão,
  coeficientes, resumo, regras da árvore e quatro figuras.
- `relatorio_questao_1.docx`: resposta com resultados e interpretação, sem código no corpo.

## Correspondência com as aulas

1. `glm(y ~ x1 + x2, family = binomial)` → `statsmodels.GLM` binomial com
   intercepto, ligação logit e sem regularização.
2. `tree(...)` → árvore de classificação de `scikit-learn`, com entropia.
3. `cv.tree(..., FUN = prune.misclass)` e `prune.misclass(...)` → escolha da
   intensidade de poda `ccp_alpha` por erro de classificação em validação cruzada
   estratificada com dez lotes, somente no treino.
4. `predict(..., type = "response"/"vector")` → probabilidades da classe 1.
5. `ifelse(prob >= 0.5, ...)`, `table`, `roc` e `auc` → corte 0,50,
   matrizes de confusão, ROC e AUC com a classe positiva fixada em 1.

A adaptação da árvore é metodológica, não uma reprodução numérica do pacote
`tree`: a poda de `scikit-learn` usa impureza, enquanto `prune.misclass` usa erros
de classificação. Também não copiamos o número fixo de seis folhas do exemplo
com spam: a complexidade é escolhida para os dados Q1, sem consultar o teste.

As matrizes CSV têm linhas previstas e colunas observadas, como nas aulas.
AUC é calculada com probabilidades, sem inverter a orientação após olhar o teste.
O script não escolhe um corte no teste nem acrescenta transformações aos modelos.

O relatório inclui os gráficos calculados pelos modelos em Python. O enunciado
exige desenhos manuais para figuras conceituais e menciona R na exceção para
saídas de modelos; a aplicação dessa exceção aos gráficos de Python precisa ser
considerada na montagem da entrega final. Nenhum desenho manual foi simulado.

## Documentação consultada

- https://www.statsmodels.org/stable/glm.html
- https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

Os resultados dependem das versões das bibliotecas. As versões verificadas estão
fixadas em `requirements.txt` e registradas em `resultados/resumo.json`.

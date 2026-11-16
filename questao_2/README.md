# Questão 2 Bootstrap e bagging de árvores de regressão

Este código implementa o bagging manualmente: sorteia as linhas com reposição,
treina cada árvore e soma as previsões para obter a média. O único estimador é
`DecisionTreeRegressor`. Não utiliza uma implementação pronta de ensemble.

## Executar

Na pasta principal `mod-pred-aps`, com o ambiente da questão 1:

```bash
source .venv/bin/activate
python questao_2/questao_2.py
```

Para instalar as dependências num ambiente novo, use Python 3.12 ou superior:

```bash
python -m pip install -r questao_2/requirements.txt
```

O programa ajusta 3.000 árvores no total: 200 para cada uma das cinco
configurações de validação e 1.000 para cada um dos dois ensembles finais.
Também treina a árvore de referência e as três árvores do exemplo didático.
O terminal mostra o andamento; o tempo depende do computador.

É possível mudar os orçamentos, sabendo que isso altera os resultados:

```bash
python questao_2/questao_2.py --arvores 1000 --arvores-validacao 200
```

## Protocolo

- Base enviada: 20.433 linhas completas e sem duplicatas, sem remoções adicionais.
- Resposta: `median_house_value`, na escala original do CSV.
- Divisão externa: 10.216 linhas de treino e 10.217 de teste, semente 42.
  A diferença de uma linha decorre do total ímpar.
- Dentro do treino externo: 8.172 linhas de ajuste e 2.044 de validação, semente 314.
- Seleção do subespaço: m = 3, 5, 7 ou 8 entre nove preditoras originais,
  com 200 árvores por configuração. m = 9 serve como referência de bagging puro.
- Escolha: menor RMSE de validação entre os subespaços reduzidos; empate favorece
  o maior m. O teste externo não seleciona m nem o número de árvores.
- Comparação final: uma árvore única, bagging com nove variáveis e bagging com
  o m selecionado. Os dois ensembles finais têm 1.000 árvores e usam exatamente
  os mesmos índices bootstrap e as mesmas sementes para as árvores.
- Cada árvore cresce sem poda: profundidade sem limite, folha mínima de um caso,
  `ccp_alpha=0`, critério de erro quadrático.
- As amostras bootstrap contêm tantas linhas quanto o treino, com reposição.
  As linhas completas mantêm o vínculo entre preditoras e resposta.
- O subespaço é sorteado uma vez por árvore, sem reposição. Todos os cortes dessa
  árvore usam somente as variáveis desse subespaço.
- `ocean_proximity` é codificada em cinco indicadores usando os níveis do treino.
  As cinco colunas entram juntas quando a variável é sorteada. Assim, m conta as
  nove variáveis originais, embora a matriz completa tenha 13 colunas.

Como Python foi autorizado para a APS, a implementação usa a árvore de regressão
de scikit-learn como base e programa os demais passos explicitamente. A biblioteca
`tree` de R trata categorias diretamente; aqui a codificação em indicadores é uma
adaptação necessária à biblioteca utilizada. Não se promete igualdade numérica
com uma execução em R.

## Resultados verificados

O subespaço selecionado foi m = 8. No teste:

| Método | Árvores | RMSE |
|---|---:|---:|
| Árvore única | 1 | 70.118,41 |
| Bagging | 1.000 | 49.833,12 |
| Bagging com subespaço | 1.000 | 48.580,31 |

O ganho relativo do subespaço sobre o bagging foi de 2,51% no RMSE desse teste.
Trata-se de um resultado condicionado à divisão, às sementes e ao protocolo
descritos, sem afirmar ganho em qualquer conjunto de dados.

As métricas foram recalculadas a partir das previsões salvas. Também foram
conferidos a separação dos dados, a igualdade dos sorteios bootstrap entre métodos,
os tamanhos dos subespaços e a média das previsões do exemplo didático.

## Estrutura

As nove funções estão explicadas também na seção correspondente do relatório:

- `carregar_dados`: leitura e validação da base.
- `preparar_dados`: codificação da variável categórica e agrupamento das colunas.
- `criar_arvore_regressao`: configuração de cada árvore sem poda.
- `calcular_rmse`: cálculo do erro na escala da resposta.
- `treinar_bagging`: reamostragem, seleção de variáveis por árvore e média das previsões.
- `selecionar_subespaco`: escolha de m somente na validação interna.
- `gerar_exemplo_bootstrap`: exemplo fictício com cinco casas e três árvores.
- `gerar_figuras`: dois esquemas didáticos e o gráfico de RMSE por número de árvores.
- `main`: coordenação do experimento e gravação dos resultados.

## Arquivos

- `dados/california.csv`: cópia intacta da base enviada.
- `referencias/`: PDF e scripts R enviados pelo usuário.
- `resultados/metricas_teste.csv`: comparação final.
- `resultados/previsoes_teste.csv`: observações e previsões dos três métodos.
- `resultados/divisao_dados.csv`: identificação do treino e teste, com índice de linha
  do DataFrame iniciando em zero (a primeira observação do CSV tem índice zero).
- `resultados/selecao_subespaco_validacao.csv`: comparação interna de m.
- `resultados/sorteios_*.csv`: variáveis sorteadas, número de casos distintos e
  hash dos índices bootstrap de cada árvore.
- `resultados/rmse_por_numero_arvores.csv`: resultados nos pontos de acompanhamento.
- `resultados/resumo.json`: protocolo, versões das bibliotecas e resultados.
- `guia_figuras_manuais.md`: orientações para reproduzir as Figuras 3 e 4 à mão.
- `../relatorio.docx`: relatório principal, com a questão 2 acrescentada.

Os modelos digitais das Figuras 3 e 4 são provisórios, conforme combinado.
Eles devem ser substituídos pelas fotografias dos desenhos manuais antes da entrega.
O gráfico de RMSE é uma saída computacional do experimento.

## Referências técnicas

- Material da aula: `referencias/Bagging.pdf` e scripts Bootstrap e Bagging.
- Breiman, L. (1996). Bagging Predictors. https://doi.org/10.1007/BF00058655
- Ho, T. K. (1998). The Random Subspace Method for Constructing Decision Forests.
  https://doi.org/10.1109/34.709601
- Árvore de regressão: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

# Questão 1 da parte teórica

Comparação de regressão logística e árvore de classificação nos arquivos fornecidos.
As demais questões da APS não foram implementadas.

## Executar no VS Code

Abra a pasta `mod-pred-aps` no VS Code. As dependências requerem **Python 3.12 ou
superior**. O ambiente `base` do Anaconda com Python 3.9 não atende esse requisito.

O ambiente `.venv` do projeto já foi preparado com Python 3.13. Para usá-lo, a partir
da pasta principal, execute:

```bash
source .venv/bin/activate
python questao_1/questao_1.py
```

Se o terminal já estiver dentro da pasta `questao_1`, os comandos são:

```bash
source ../.venv/bin/activate
python questao_1.py
```

Para recriar o ambiente em outro computador, use um Python compatível. Exemplo
com Python 3.13, a partir da pasta principal:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r questao_1/requirements.txt
```

Selecione o interpretador `.venv/bin/python` no VS Code. O programa localiza os CSVs
relativamente ao próprio arquivo, independentemente da pasta atual do terminal.
Use `python --version` para conferir o interpretador ativo. Se aparecer Python 3.9,
ative `.venv` antes de instalar os pacotes ou executar o programa.

## Arquivos

- `questao_1.py`: código comentado da análise, sem dependência de R.
- `dados/`: cópias intactas dos dois CSVs enviados.
- `referencias/`: cópias dos scripts do professor.
- `resultados/`: métricas, previsões individuais, matrizes de confusão,
  coeficientes, resumo, regras da árvore e duas figuras.
- `../relatorio.docx`: relatório principal da APS, com os resultados, a interpretação
  e a explicação da estrutura do código.

## Estrutura do código

O arquivo começa com as bibliotecas e as configurações gerais: pastas, semente 42,
variáveis explicativas e cores dos gráficos. Depois define sete funções:

- `ler_dados`: lê e verifica cada CSV antes do treinamento.
- `criar_arvore`: cria uma árvore com a intensidade de poda recebida.
- `selecionar_poda`: compara os valores de poda por validação cruzada no treino
  e ajusta a árvore escolhida ao treino completo.
- `avaliar`: aplica o corte de 50% e calcula as métricas, incluindo a AUC a partir
  das probabilidades.
- `salvar_figura`: grava uma imagem PNG e fecha a figura.
- `figuras`: produz os dados com regiões previstas e as curvas ROC.
- `main`: coordena a execução e grava os resultados.

A chamada `if __name__ == "__main__"` inicia `main` quando o arquivo é executado.
A pasta `resultados/` recebe somente dois gráficos: `01_dados_e_regioes.png` e
`02_curvas_roc.png`. A validação cruzada continua escolhendo a poda da árvore e
seus resultados continuam registrados no CSV, sem gerar um gráfico separado.

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

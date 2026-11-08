# Análise e Classificação de Ocorrências por Período do Dia

Projeto acadêmico de aprendizado de máquina voltado à classificação do período do dia de ocorrências registradas em diferentes municípios, com foco na avaliação da capacidade de generalização dos modelos entre domínios geográficos.

## Objetivo

Investigar o desempenho de modelos de classificação na previsão do período do dia de uma ocorrência, utilizando dados de diferentes municípios.

O projeto também explora o problema de **Domain Adaptation**, avaliando como diferenças geográficas entre os municípios influenciam a capacidade de generalização dos modelos.

## Problema de classificação

A variável-alvo é `DESC_PERIODO`, composta por quatro classes:

- `a noite`
- `a tarde`
- `de madrugada`
- `pela manha`

Entre as variáveis utilizadas nos experimentos estão:

- `MES`: mês da ocorrência;
- `DIA_SEM`: dia da semana;
- `FERIADO`: indicação de feriado;
- `RUBRICA`: classificação da ocorrência;
- `DESCR_TIPOLOCAL`: descrição do tipo de local.

## Municípios analisados

Os experimentos consideram três municípios:

- Birigui;
- Sertãozinho;
- Itupeva.

A proposta é investigar se padrões aprendidos em um município podem ser utilizados para classificar ocorrências em outro.

## Metodologia

### Divisão temporal

Os experimentos utilizam uma divisão temporal dos dados:

| Conjunto | Período |
|---|---|
| Treinamento | Janeiro a setembro |
| Teste | Outubro a dezembro |

Essa estratégia permite avaliar os modelos em registros posteriores aos utilizados durante o treinamento.

### Transferência entre municípios

São investigadas diferentes configurações de treinamento e aplicação, incluindo:

- Treinamento em Sertãozinho e aplicação em Birigui;
- Treinamento em Sertãozinho e aplicação em Itupeva;
- Treinamento conjunto em Sertãozinho e Birigui, com aplicação em Itupeva.

### Expansão progressiva de dados

Também é investigada uma estratégia de expansão progressiva do conjunto de treinamento:

1. Treinamento inicial utilizando dados de Sertãozinho.
2. Expansão do conjunto de treinamento com dados de Birigui.
3. Nova avaliação utilizando dados de Itupeva.

O objetivo é verificar se a inclusão de dados de um segundo município contribui para melhorar a generalização em um terceiro.

## Modelos

Os experimentos utilizam modelos de aprendizado de máquina para classificação:

- **Random Forest:** modelo baseado em conjuntos de árvores de decisão.
- **TabPFN:** modelo baseado em aprendizado pré-treinado para problemas de classificação tabular.

## Avaliação

O desempenho dos modelos é analisado por meio de métricas de classificação:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- Macro F1-score.

Também são utilizados relatórios de classificação e matrizes de confusão para analisar o desempenho individual em cada classe.

## Técnicas de Domain Adaptation

Além dos modelos de classificação, o projeto explora técnicas de adaptação de domínio para investigar e reduzir diferenças na distribuição dos dados entre municípios.

### CORAL (Correlation Alignment)

O CORAL é utilizado para alinhar estatisticamente os dados de origem (*source domain*) e de destino (*target domain*).

A técnica busca reduzir diferenças entre as distribuições dos domínios por meio do alinhamento das matrizes de covariância dos atributos.

No contexto do projeto:

- **Source domain:** município ou conjunto de municípios utilizados no treinamento.
- **Target domain:** município para o qual se deseja transferir o modelo.
- **Dados não rotulados do domínio de destino:** utilizados para estimar sua distribuição, sem utilizar os rótulos do conjunto de teste.

O objetivo é avaliar se o alinhamento estatístico melhora a capacidade de generalização dos classificadores entre municípios.

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

O t-SNE é utilizado para visualizar a distribuição dos dados em espaços de menor dimensionalidade, permitindo investigar semelhanças e diferenças entre os municípios.

A implementação considera:

1. Pré-processamento das variáveis categóricas por meio de One-Hot Encoding.
2. Transformação dos atributos em uma representação numérica.
3. Aplicação do t-SNE para redução da dimensionalidade.
4. Visualização dos registros em um espaço bidimensional.

As visualizações permitem observar:
- Sobreposição entre registros de diferentes municípios;
- Formação de agrupamentos;
- Diferenças na distribuição dos domínios;
- Possíveis mudanças na distribuição das classes entre municípios.

O t-SNE também pode ser utilizado para visualizar as distribuições antes e depois da aplicação de técnicas de adaptação de domínio.

**Importante:** o t-SNE é uma ferramenta exploratória de visualização, não um método de classificação ou uma técnica de alinhamento de domínios por si só.

## Resumo das técnicas utilizadas

| Técnica | Finalidade |
|---|---|
| Random Forest | Classificação |
| TabPFN | Classificação tabular |
| CORAL | Alinhamento estatístico entre domínios |
| t-SNE | Visualização e análise da distribuição dos domínios |
| One-Hot Encoding | Codificação de variáveis categóricas |

## Estrutura do projeto

```text
PT2-Local/
├── code/
│   ├── ... scripts dos experimentos
│
├── results/
│   ├── ... resultados e saídas dos experimentos
│
└── README.md
```

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-learn
- TabPFN
- PyTorch
- Matplotlib
- TSNE
- CORAL - CORrelation ALignment

## Observações

- Os conjuntos de treinamento e teste são separados temporalmente.
- Os dados de teste de Itupeva não devem ser utilizados para ajustar os modelos nos experimentos de transferência.
- As comparações entre modelos devem manter as mesmas divisões de dados, variáveis e métricas sempre que possível.
- A inclusão de dados de outro município deve ser avaliada considerando tanto o aumento da quantidade de registros quanto a diversidade geográfica introduzida.




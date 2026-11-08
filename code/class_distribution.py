import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("../datasets/normalized_dataset.csv", sep=",")

# 1. Selecionar os domínios
birigui = df[df['NOME_MUNICIPIO'] == 'birigui'].copy()
sertaozinho = df[df['NOME_MUNICIPIO'] == 'sertaozinho'].copy()
itupeva = df[df['NOME_MUNICIPIO'] == 'itupeva'].copy()

# 2. Balancear quantidade entre as três cidades
n = min(len(birigui), len(sertaozinho), len(itupeva))

birigui = birigui.sample(n=n, random_state=42)
sertaozinho = sertaozinho.sample(n=n, random_state=42)
itupeva = itupeva.sample(n=n, random_state=42)

# 3. Juntar os dados para calcular uma única projeção
features = ['MES', 'DIA_SEM', 'FERIADO', 'RUBRICA', 'DESCR_TIPOLOCAL']

dados = pd.concat([
    birigui,
    sertaozinho,
    itupeva
], ignore_index=True)

X_all = dados[features]
y_all = dados['DESC_PERIODO']

domain = dados['NOME_MUNICIPIO'].values

# 4. Encoding
categorical_cols = [
    'DIA_SEM',
    'FERIADO',
    'RUBRICA',
    'DESCR_TIPOLOCAL'
]

numeric_cols = ['MES']

preprocessor = ColumnTransformer([
    (
        'cat',
        OneHotEncoder(handle_unknown='ignore', sparse_output=False),
        categorical_cols
    ),
    ('num', 'passthrough', numeric_cols)
])

X_encoded = preprocessor.fit_transform(X_all)

print("Shape após encoding:", X_encoded.shape)

# 5. t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    init='pca',
    random_state=42
)

X_tsne = tsne.fit_transform(X_encoded)

# 6. Gerar um gráfico separado para cada cidade
cidades = ['birigui', 'sertaozinho', 'itupeva']
classes = y_all.unique()

for cidade in cidades:

    fig, ax = plt.subplots(figsize=(10, 7))

    mask_cidade = domain == cidade

    for classe in classes:

        mask = mask_cidade & (y_all.values == classe)

        ax.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            s=12,
            alpha=0.5,
            label=classe
        )

    ax.set_title(f't-SNE por classe - {cidade.capitalize()}')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(title='DESC_PERIODO')
    ax.grid(alpha=0.15)

    plt.tight_layout()
    plt.show()
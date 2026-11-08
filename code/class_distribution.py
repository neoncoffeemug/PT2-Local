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

# 2. Balancear quantidade

n = min(len(birigui), len(sertaozinho))

birigui = birigui.sample(n=n, random_state=42)
sertaozinho = sertaozinho.sample(n=n, random_state=42)

# 3. Juntar os domínios

features = ['MES', 'DIA_SEM', 'FERIADO', 'RUBRICA', 'DESCR_TIPOLOCAL']

# 1. Selecionar as 10 RUBRICAS mais frequentes
top_10_rubricas = (
    df['DESC_PERIODO']
    .value_counts()
    .head(5)
    .index
)

# 2. Filtrar os dois domínios
birigui = birigui[
    birigui['DESC_PERIODO'].isin(top_10_rubricas)
].copy()

sertaozinho = sertaozinho[
    sertaozinho['DESC_PERIODO'].isin(top_10_rubricas)
].copy()


X_all = pd.concat([
    birigui[features],
    sertaozinho[features]
], ignore_index=True)

y_all = pd.concat([
    birigui['DESC_PERIODO'],
    sertaozinho['DESC_PERIODO']
], ignore_index=True)

domain = np.array(
    ['Birigui'] * len(birigui) +
    ['Sertaozinho'] * len(sertaozinho)
)

domain = np.array(['Birigui'] * len(birigui) + ['Sertaozinho'] * len(sertaozinho))

# 4. Encoding

categorical_cols = ['DIA_SEM', 'FERIADO', 'RUBRICA', 'DESCR_TIPOLOCAL']
numeric_cols = ['MES']

preprocessor = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols), ('num', 'passthrough', numeric_cols)])
X_encoded = preprocessor.fit_transform(X_all)

print("Shape após encoding:", X_encoded.shape)

# 5. t-SNE

tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
X_tsne = tsne.fit_transform(X_encoded)

plt.figure(figsize=(10, 7))

for classe in y_all.unique():
    mask = y_all == classe

    plt.scatter(
        X_tsne[mask, 0],
        X_tsne[mask, 1],
        alpha=0.5,
        label=classe
    )

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE por classe - DESC_PERIODO')
plt.legend()
plt.show()


# =========================
# 6. Visualização
# =========================

# =========================
# Visualização por classe
# =========================

import matplotlib.pyplot as plt
import numpy as np

classes = y_all.unique()

fig, axes = plt.subplots(
    1, 2,
    figsize=(16, 7),
    sharex=True,
    sharey=True
)

cidades = ['Birigui', 'Sertaozinho']

for ax, cidade in zip(axes, cidades):

    mask_cidade = domain == cidade

    for classe in classes:

        mask = mask_cidade & (y_all == classe)

        ax.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            s=12,
            alpha=0.35,
            label=classe
        )

    ax.set_title(cidade, fontsize=15)
    ax.set_xlabel("t-SNE 1")
    ax.grid(alpha=0.15)

axes[0].set_ylabel("t-SNE 2")

# Uma única legenda para os dois gráficos
handles, labels = axes[1].get_legend_handles_labels()

plt.legend(
    title='DESC_PERIODO',
    bbox_to_anchor=(1.02, 1),
    loc='upper left'
)

fig.suptitle(
    "Distribuição das classes por domínio",
    fontsize=17
)

plt.tight_layout(rect=[0, 0, 0.88, 0.95])
plt.subplots_adjust(bottom=0.35)
plt.show()
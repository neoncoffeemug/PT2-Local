import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# 1. CARREGAR DATASET
# ============================================================

df = pd.read_csv("../datasets/normalized_dataset.csv", sep=",")

features = [
    'MES',
    'DIA_SEM',
    'FERIADO',
    'RUBRICA',
    'DESCR_TIPOLOCAL'
]

cidades = ['sertaozinho', 'birigui', 'itupeva']

classes = [
    'a noite',
    'a tarde',
    'de madrugada',
    'pela manha'
]

# ============================================================
# 2. SELECIONAR OS TRÊS DOMÍNIOS
# ============================================================

df = df[
    df['NOME_MUNICIPIO'].isin(cidades)
].copy()

# Remover registros sem período válido
df['DESC_PERIODO'] = (
    df['DESC_PERIODO']
    .astype('string')
    .str.strip()
    .str.lower()
)

df = df[
    df['DESC_PERIODO'].isin(classes)
].copy()

# ============================================================
# 3. BALANCEAR QUANTIDADE DE REGISTROS POR MUNICÍPIO
# ============================================================

n = df['NOME_MUNICIPIO'].value_counts().min()

dados = pd.concat([
    df[df['NOME_MUNICIPIO'] == cidade].sample(
        n=n,
        random_state=42
    )
    for cidade in cidades
], ignore_index=True)

print("Registros por município:")
print(dados['NOME_MUNICIPIO'].value_counts())

print("\nDistribuição das classes:")
print(
    pd.crosstab(
        dados['NOME_MUNICIPIO'],
        dados['DESC_PERIODO']
    )
)

# ============================================================
# 4. ENCODING DAS FEATURES
# ============================================================

# Todas as variáveis tratadas como categóricas.
# Isso evita que MES (1-12) domine as distâncias do t-SNE.

X = dados[features].copy()

X = X.fillna('ausente').astype(str)

encoder = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

X_encoded = encoder.fit_transform(X)

print("\nShape após encoding:", X_encoded.shape)

# ============================================================
# 5. CALCULAR UMA ÚNICA PROJEÇÃO t-SNE
# ============================================================

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    init='pca',
    random_state=42
)

X_tsne = tsne.fit_transform(X_encoded)

# Adicionar as coordenadas ao dataframe
dados['TSNE_1'] = X_tsne[:, 0]
dados['TSNE_2'] = X_tsne[:, 1]

# ============================================================
# 6. DEFINIR CORES
# ============================================================

cores_cidades = {
    'sertaozinho': 'red',
    'birigui': 'blue',
    'itupeva': 'green'
}

cores_classes = {
    'a noite': 'purple',
    'a tarde': 'blue',
    'de madrugada': 'red',
    'pela manha': 'orange'
}

# Limites iguais para todos os gráficos
xlim = (
    dados['TSNE_1'].min() - 2,
    dados['TSNE_1'].max() + 2
)

ylim = (
    dados['TSNE_2'].min() - 2,
    dados['TSNE_2'].max() + 2
)

# ============================================================
# 7. GRÁFICO 1 — t-SNE COLORIDO POR MUNICÍPIO
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))

for cidade in cidades:

    subset = dados[
        dados['NOME_MUNICIPIO'] == cidade
    ]

    ax.scatter(
        subset['TSNE_1'],
        subset['TSNE_2'],
        s=25,
        alpha=0.5,
        color=cores_cidades[cidade],
        label=cidade.capitalize()
    )

ax.set_title(
    't-SNE — Distribuição dos Domínios',
    fontsize=15
)

ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')

ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.legend(title='Município')
ax.grid(alpha=0.15)

plt.tight_layout()
plt.show()

# ============================================================
# 8. GRÁFICO 2 — t-SNE COLORIDO POR CLASSE
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))

for classe in classes:

    subset = dados[
        dados['DESC_PERIODO'] == classe
    ]

    ax.scatter(
        subset['TSNE_1'],
        subset['TSNE_2'],
        s=25,
        alpha=0.5,
        color=cores_classes[classe],
        label=classe
    )

ax.set_title(
    't-SNE — Distribuição das Classes',
    fontsize=15
)

ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')

ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.legend(title='DESC_PERIODO')
ax.grid(alpha=0.15)

plt.tight_layout()
plt.show()

## ============================================================
# t-SNE OTIMIZADO PARA O PÔSTER
# ============================================================

# Amostragem apenas para visualização
# A projeção t-SNE já foi calculada com os dados completos

dados_plot = pd.concat([
    dados[dados['NOME_MUNICIPIO'] == cidade].sample(
        n=min(250, len(dados[dados['NOME_MUNICIPIO'] == cidade])),
        random_state=42
    )
    for cidade in cidades
])

# ============================================================
# t-SNE POR MUNICÍPIO — PÔSTER
# ============================================================

cores_cidades = {
    'sertaozinho': 'red',
    'birigui': 'blue',
    'itupeva': 'green'
}

fig, ax = plt.subplots(figsize=(4.5, 4))

for cidade in cidades:

    subset = dados_plot[
        dados_plot['NOME_MUNICIPIO'] == cidade
    ]

    ax.scatter(
        subset['TSNE_1'],
        subset['TSNE_2'],
        s=60,
        alpha=0.50,
        color=cores_cidades[cidade],
        label=cidade.capitalize(),
        edgecolors='none'
    )

ax.set_title(
    'Distribuição dos Domínios',
    fontsize=13,
    fontweight='bold',
    pad=12
)

# Mesmos limites da projeção completa
ax.set_xlim(dados['TSNE_1'].min() - 3, dados['TSNE_1'].max() + 3)
ax.set_ylim(dados['TSNE_2'].min() - 3, dados['TSNE_2'].max() + 3)

# Manter grid, mas ocultar números dos eixos
ax.tick_params(
    axis='both',
    which='both',
    length=0,
    labelbottom=False,
    labelleft=False
)

ax.grid(
    True,
    alpha=0.15,
    linestyle='--',
    linewidth=0.5
)

# Bordas discretas
for spine in ax.spines.values():
    spine.set_visible(False)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.02),
    ncol=3,
    fontsize=8,
    frameon=False
)

plt.tight_layout()

plt.savefig(
    'tsne_municipios_poster.svg',
    bbox_inches='tight'
)

plt.savefig(
    'tsne_municipios_poster.png',
    dpi=400,
    bbox_inches='tight'
)

plt.show()


fig, ax = plt.subplots(figsize=(4.5, 4))

for classe in classes:

    subset = dados_plot[
        dados_plot['DESC_PERIODO'] == classe
    ]

    ax.scatter(
        subset['TSNE_1'],
        subset['TSNE_2'],
        s=50,
        alpha=0.50,
        color=cores_classes[classe],
        label=classe,
        edgecolors='none'
    )

ax.set_title(
    'Distribuição das Classes',
    fontsize=13,
    fontweight='bold',
    pad=12
)

# Os eixos do t-SNE não possuem interpretação física
# Manter as linhas da grade, mas ocultar os valores dos eixos
ax.tick_params(
    axis='both',
    which='both',
    length=0,
    labelbottom=False,
    labelleft=False
)

ax.set_xlabel('')
ax.set_ylabel('')

ax.grid(
    True,
    alpha=0.30,
    linestyle='--',
    linewidth=1
)

# Remover bordas desnecessárias
for spine in ax.spines.values():
    spine.set_visible(False)

# Legenda abaixo da figura
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.02),
    ncol=2,
    fontsize=9,
    frameon=False
)

plt.tight_layout()

# Exportação vetorial
plt.savefig(
    'tsne_poster.svg',
    bbox_inches='tight'
)

plt.savefig(
    'tsne_poster.pdf',
    bbox_inches='tight'
)

plt.show()
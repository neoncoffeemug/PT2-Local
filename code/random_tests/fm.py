import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tabfm import TabFMClassifier

from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
model = tabfm_v1_0_0.load(model_type="classification")

df_sertaozinho = pd.read_csv("../../datasets/normalized_dataset.csv", sep=",")

df_sertaozinho = df_sertaozinho[df_sertaozinho['NOME_MUNICIPIO'] == 'sertaozinho'].copy()
features = [ 'DIA_SEM', 'FERIADO', 'RUBRICA', 'DESCR_TIPOLOCAL', 'MES']

# LIDAR COM NULLS
df_sertaozinho.dropna(subset = ['DESC_PERIODO'], inplace=True)

X = df_sertaozinho[features].copy()
y = df_sertaozinho['DESC_PERIODO'].copy()

categorical_cols = ['DIA_SEM', 'RUBRICA', 'DESCR_TIPOLOCAL']


for col in categorical_cols:
  X[col] = X[col].astype(str)

X['MES'] = X['MES'].astype(int)
X['FERIADO'] = X['FERIADO'].astype(int)

train_mask = df_sertaozinho['MES'] <= 9
test_mask = df_sertaozinho['MES'] > 9

Xtrain = X.loc[train_mask].copy()
Xtest = X.loc[test_mask].copy()
# Xtrain = pd.get_dummies(Xtrain, columns=categorical_cols, dtype=int)
# Xtest = pd.get_dummies(Xtest, columns=categorical_cols, dtype=int)
# Xtest = Xtest.reindex(columns=Xtrain.columns, fill_value=0)

ytrain = y.loc[train_mask].copy()
ytest = y.loc[test_mask].copy()

clf = TabFMClassifier(model=model)

print("Training: ", Xtrain.shape)
print("Test: ", Xtest.shape)

print("\n",ytrain.value_counts())
print("\n",ytest.value_counts())

clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)
print("\n", classification_report(ytest, ypred, zero_division=0))

print("Município de treinamento: Sertãozinho")
print("Município de teste: Sertãozinho")

print("Meses de treinamento:", sorted(df_sertaozinho.loc[train_mask, 'MES'].unique()))
print("Meses de teste:", sorted(df_sertaozinho.loc[test_mask, 'MES'].unique()))

print("\nVariáveis utilizadas:", features)

print("\nParâmetros do TABPFN:")
for parametro, valor in tab.get_params().items():
    print(f"{parametro}: {valor}")

print("\nRegistros de treinamento:", len(Xtrain))
print("Registros de teste:", len(Xtest))

labels = ['a noite', 'a tarde', 'de madrugada', 'pela manha']

cm = confusion_matrix(ytest, ypred, labels=labels)
ConfusionMatrixDisplay.from_predictions(ytest, ypred, xticks_rotation=45)

plt.show()

features = [
    'DIA_SEM',
    'FERIADO',
    'RUBRICA',
    'DESCR_TIPOLOCAL',
    'MES'
]

df = df_sertaozinho.copy()

df['DESC_PERIODO'] = df['DESC_PERIODO'].replace(
    ['nan', 'NaN', 'None', ''], pd.NA
)

df = df.dropna(subset=['DESC_PERIODO'])

groups = df.groupby(features, dropna=False)['DESC_PERIODO']

# Number of distinct target classes for each feature combination
unique_targets = groups.nunique()

# Number of observations for each feature combination
group_sizes = groups.size()

# Percentage of observations belonging to ambiguous combinations
ambiguous_rows = group_sizes[unique_targets > 1].sum()

print("Observações totais:", len(df))
print("Observações em combinações ambíguas:", ambiguous_rows)

print(
    "Percentual de observações ambíguas:",
    round(ambiguous_rows / len(df) * 100, 2)
)

# Empirical maximum accuracy from assigning the most common
# label to every distinct feature combination
counts = (
    df.groupby(features + ['DESC_PERIODO'], dropna=False)
      .size()
)

max_correct = (
    counts.groupby(level=list(range(len(features))))
          .max()
          .sum()
)

print(
    "Acurácia máxima empírica por combinação:",
    round(max_correct / len(df) * 100, 2)
)
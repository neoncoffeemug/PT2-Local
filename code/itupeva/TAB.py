import pandas as pd
import matplotlib.pyplot as plt
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

clf = TabPFNClassifier(device="cpu", random_state=42, balance_probabilities=True)
df_itupeva = pd.read_csv("../../datasets/normalized_dataset.csv", sep=",")

df_itupeva = df_itupeva[df_itupeva['NOME_MUNICIPIO'] == 'itupeva'].copy()
features = [ 'DIA_SEM', 'FERIADO', 'RUBRICA', 'DESCR_TIPOLOCAL', 'MES']

# LIDAR COM NULLS
df_itupeva.dropna(subset = ['DESC_PERIODO'], inplace=True)

X = df_itupeva[features].copy()
y = df_itupeva['DESC_PERIODO'].copy()

categorical_cols = ['DIA_SEM', 'RUBRICA', 'DESCR_TIPOLOCAL']

for col in categorical_cols:
  X[col] = X[col].astype(str)

X['MES'] = X['MES'].astype(int)
X['FERIADO'] = X['FERIADO'].astype(int)

train_mask = df_itupeva['MES'] <= 9
test_mask = df_itupeva['MES'] > 9

Xtrain = X.loc[train_mask].copy()
Xtest = X.loc[test_mask].copy()
Xtrain = pd.get_dummies(Xtrain, columns=categorical_cols, dtype=int)
Xtest = pd.get_dummies(Xtest, columns=categorical_cols, dtype=int)
Xtest = Xtest.reindex(columns=Xtrain.columns, fill_value=0)

ytrain = y.loc[train_mask].copy()
ytest = y.loc[test_mask].copy()

print("Training: ", Xtrain.shape)
print("Test: ", Xtest.shape)

print("\n",ytrain.value_counts())
print("\n",ytest.value_counts())

clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)
print("\n", classification_report(ytest, ypred, zero_division=0))

print("Município de treinamento: Itupeva")
print("Município de teste: Itupeva")

print("Meses de treinamento:", sorted(df_itupeva.loc[train_mask, 'MES'].unique()))
print("Meses de teste:", sorted(df_itupeva.loc[test_mask, 'MES'].unique()))

print("\nVariáveis utilizadas:", features)

print("\nParâmetros do TABPFN:")
for parametro, valor in clf.get_params().items():
    print(f"{parametro}: {valor}")

print("\nRegistros de treinamento:", len(Xtrain))
print("Registros de teste:", len(Xtest))

labels = ['a noite', 'a tarde', 'de madrugada', 'pela manha']

cm = confusion_matrix(ytest, ypred, labels=labels)
ConfusionMatrixDisplay.from_predictions(ytest, ypred, xticks_rotation=45)

plt.show()


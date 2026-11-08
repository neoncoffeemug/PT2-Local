import pandas as pd
import matplotlib.pyplot as plt
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


tab = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST, device="cpu", random_state=42, balance_probabilities=True)

df = pd.read_csv("../../datasets/normalized_dataset.csv", sep=",")
df_sertaozinho = df[df['NOME_MUNICIPIO'] == 'sertaozinho'].copy()
df_birigui = df[df['NOME_MUNICIPIO'] == 'birigui'].copy()

features = ['MES', 'DIA_SEM', 'FERIADO', 'RUBRICA', 'DESCR_TIPOLOCAL']

df_sertaozinho.dropna(subset = ['DESC_PERIODO'], inplace=True)
df_birigui.dropna(subset = ['DESC_PERIODO'], inplace=True)

categorical_cols = ['DIA_SEM', 'RUBRICA', 'DESCR_TIPOLOCAL']

train_mask = df_sertaozinho['MES'] <= 9
test_mask = df_birigui['MES'] > 9

Xtrain = df_sertaozinho.loc[train_mask, features].copy()
ytrain = df_sertaozinho.loc[train_mask, 'DESC_PERIODO'].copy()

Xtest = df_birigui.loc[test_mask, features].copy()
ytest = df_birigui.loc[test_mask, 'DESC_PERIODO'].copy()

for col in categorical_cols:
    Xtrain[col] = Xtrain[col].astype(str)
    Xtest[col] = Xtest[col].astype(str)

Xtrain['MES'] = Xtrain['MES'].astype(int)
Xtest['MES'] = Xtest['MES'].astype(int)

Xtrain = pd.get_dummies(Xtrain, columns=categorical_cols, dtype=int)
Xtest = pd.get_dummies(Xtest, columns=categorical_cols, dtype=int)
Xtest = Xtest.reindex(columns=Xtrain.columns, fill_value=0)

print("Treino: ", Xtrain.shape)
print("Teste: ", Xtest.shape)

print("\nTreino:")
print(ytrain.value_counts())

print("\nTeste:")
print(ytest.value_counts())

tab.fit(Xtrain, ytrain)
ypred = tab.predict(Xtest)

print(classification_report(ytest, ypred, zero_division=0))

print("Município de treinamento: Sertãozinho")
print("Município de teste: Birigui")

print("Meses de treinamento:", sorted(df_sertaozinho.loc[train_mask, 'MES'].unique()))
print("Meses de teste:", sorted(df_birigui.loc[test_mask, 'MES'].unique()))

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
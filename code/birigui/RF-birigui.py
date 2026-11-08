import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1, class_weight="balanced")
df_birigui = pd.read_csv("../../datasets/normalized_dataset.csv", sep=",")

df_birigui = df_birigui[df_birigui['NOME_MUNICIPIO'] == 'birigui'].copy()
features = [ 'DIA_SEM', 'FERIADO', 'RUBRICA', 'DESCR_TIPOLOCAL', 'MES']

# LIDAR COM NULLS
df_birigui.dropna(subset = ['DESC_PERIODO'], inplace=True)

X = df_birigui[features].copy()
y = df_birigui['DESC_PERIODO'].copy()

categorical_cols = ['DIA_SEM', 'RUBRICA', 'DESCR_TIPOLOCAL']

for col in categorical_cols:
  X[col] = X[col].astype(str)

X['MES'] = X['MES'].astype(int)
X['FERIADO'] = X['FERIADO'].astype(int)

train_mask = df_birigui['MES'] <= 9
test_mask = df_birigui['MES'] > 9

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

rf.fit(Xtrain, ytrain)
ypred = rf.predict(Xtest)
print("\n", classification_report(ytest, ypred, zero_division=0))

print("Município de treinamento: Sertãozinho")
print("Município de teste: Sertãozinho")

print("Meses de treinamento:", sorted(df_birigui.loc[train_mask, 'MES'].unique()))
print("Meses de teste:", sorted(df_birigui.loc[test_mask, 'MES'].unique()))

print("\nVariáveis utilizadas:", features)

print("\nParâmetros do Random Forest:")
for parametro, valor in rf.get_params().items():
    print(f"{parametro}: {valor}")

print("\nRegistros de treinamento:", len(Xtrain))
print("Registros de teste:", len(Xtest))

labels = ['a noite', 'a tarde', 'de madrugada', 'pela manha']

cm = confusion_matrix(ytest, ypred, labels=labels)
ConfusionMatrixDisplay.from_predictions(ytest, ypred, xticks_rotation=45)

plt.show()


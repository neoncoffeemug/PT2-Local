import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, class_weight="balanced")

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

rf.fit(Xtrain, ytrain)
ypred = rf.predict(Xtest)

print(classification_report(ytest, ypred, zero_division=0))

print("Município de treinamento: Sertãozinho")
print("Município de teste: Birigui")

print("Meses de treinamento:", sorted(df_sertaozinho.loc[train_mask, 'MES'].unique()))
print("Meses de teste:", sorted(df_birigui.loc[test_mask, 'MES'].unique()))

print("\nVariáveis utilizadas:", features)

print("\nParâmetros do Random Forest:")
for parametro, valor in rf.get_params().items():
    print(f"{parametro}: {valor}")

print("\nRegistros de treinamento:", len(Xtrain))
print("Registros de teste:", len(Xtest))
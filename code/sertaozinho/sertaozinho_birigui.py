import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, class_weight="balanced")

df = pd.read_csv("../../datasets/normalized_dataset.csv", sep=",")
df_sertaozinho = df[df['NOME_MUNICIPIO'] == 'sertaozinho'].copy()
df_birigui = df[df['NOME_MUNICIPIO'] == 'birigui'].copy()

features = ['MES', 'DIA_SEM', 'FERIADO', 'RUBRICA', 'DESCR_TIPOLOCAL', 'BAIRRO']

df_sertaozinho.dropna(subset = ['DESC_PERIODO'], inplace=True)
df_birigui.dropna(subset = ['DESC_PERIODO'], inplace=True)

categorical_cols = ['DIA_SEM', 'RUBRICA', 'DESCR_TIPOLOCAL', 'BAIRRO']

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

print("Treino: ", Xtrain.shape)
print("Teste: ", Xtest.shape)



print("ok")
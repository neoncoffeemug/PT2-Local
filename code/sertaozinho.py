import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1, class_weight="balanced")
df_sertaozinho = pd.read_csv("../datasets/normalized_dataset.csv", sep=",")

df_sertaozinho = df_sertaozinho[df_sertaozinho['NOME_MUNICIPIO'] == 'sertaozinho'].copy()
features = ['MES', 'DIA_SEM', 'FERIADO', 'RUBRICA', 'DESCR_TIPOLOCAL']

# LIDAR COM NULLS
df_sertaozinho.dropna(subset = ['DESC_PERIODO'], inplace=True)

X = df_sertaozinho[features].copy()
y = df_sertaozinho['DESC_PERIODO'].copy()

categorical_cols = ['DIA_SEM', 'RUBRICA', 'DESCR_TIPOLOCAL']

for col in categorical_cols:
  X[col] = X[col].astype(str)

X['MES'] = X['MES'].astype(int)
X['FERIADO'] = X['FERIADO'].astype(int)

train_mask = df_sertaozinho['MES'] <= 8
test_mask = df_sertaozinho['MES'] > 8

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

print("\n INBALANCED, DROP NULLS, RF")
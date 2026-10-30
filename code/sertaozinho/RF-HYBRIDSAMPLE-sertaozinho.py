import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1)
rus = RandomUnderSampler(sampling_strategy={'pela manha': 350, 'a noite': 350, 'a tarde': 350}, random_state=42)
ros = RandomOverSampler(sampling_strategy={'de madrugada': 300}, random_state=42)

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
test_mask = df_sertaozinho['MES']  > 9

Xtrain = X.loc[train_mask].copy()
Xtest = X.loc[test_mask].copy()
Xtrain = pd.get_dummies(Xtrain, columns=categorical_cols, dtype=int)
Xtest = pd.get_dummies(Xtest, columns=categorical_cols, dtype=int)
Xtest = Xtest.reindex(columns=Xtrain.columns, fill_value=0)

ytrain = y.loc[train_mask].copy()
ytest = y.loc[test_mask].copy()

X_under, y_under = rus.fit_resample(Xtrain, ytrain)
X_res, y_res = ros.fit_resample(X_under, y_under)


print("\n===== VERIFICAÇÃO DOS CONJUNTOS =====")

print("Treinamento original:", Xtrain.shape)
print("Treinamento balanceado:", X_res.shape)
print("Teste:", Xtest.shape)

print("\nDistribuição original:")
print(ytrain.value_counts())

print("\nDistribuição balanceada:")
print(y_res.value_counts())

print("\nMeses presentes no treinamento:")
print(Xtrain['MES'].unique())

print("\nMeses presentes no teste:")
print(Xtest['MES'].unique())


rf.fit(X_res, y_res)
ypred = rf.predict(Xtest)
print("\n", classification_report(ytest, ypred, zero_division=0))

print("Município de treinamento: Sertãozinho")
print("Município de teste: Sertãozinho")

print("Meses de treinamento:", sorted(df_sertaozinho.loc[train_mask, 'MES'].unique()))
print("Meses de teste:", sorted(df_sertaozinho.loc[test_mask, 'MES'].unique()))

print("\nVariáveis utilizadas:", features)

print("\nParâmetros do Random Forest:")
for parametro, valor in rf.get_params().items():
    print(f"{parametro}: {valor}")

print("\nRegistros de treinamento:", len(Xtrain))
print("Registros de teste:", len(Xtest))
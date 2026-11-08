
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

df = pd.read_csv("../../datasets/normalized_dataset.csv", sep=",")

# Municípios
df_sertaozinho = df[df['NOME_MUNICIPIO'] == 'sertaozinho'].copy()
df_birigui = df[df['NOME_MUNICIPIO'] == 'birigui'].copy()
df_itupeva = df[df['NOME_MUNICIPIO'] == 'itupeva'].copy()

# Configuração
features = [
    'MES',
    'DIA_SEM',
    'FERIADO',
    'RUBRICA',
    'DESCR_TIPOLOCAL'
]

categorical_cols = [
    'DIA_SEM',
    'RUBRICA',
    'DESCR_TIPOLOCAL'
]

target = 'DESC_PERIODO'

# Remover registros sem target
for municipio in [df_sertaozinho, df_birigui, df_itupeva]:
    municipio.dropna(subset=[target], inplace=True)


# Divisão temporal
sertaozinho_train = df_sertaozinho[df_sertaozinho['MES'] <= 9]
birigui_train = df_birigui[df_birigui['MES'] <= 9]
itupeva_test = df_itupeva[df_itupeva['MES'] > 9]


# Preparação dos dados
def preparar_dados(df_train, df_test):

    Xtrain = df_train[features].copy()
    ytrain = df_train[target].copy()

    Xtest = df_test[features].copy()
    ytest = df_test[target].copy()

    for col in categorical_cols:
        Xtrain[col] = Xtrain[col].fillna('__MISSING__').astype(str)
        Xtest[col] = Xtest[col].fillna('__MISSING__').astype(str)

    Xtrain['FERIADO'] = Xtrain['FERIADO'].fillna('__MISSING__').astype(str)
    Xtest['FERIADO'] = Xtest['FERIADO'].fillna('__MISSING__').astype(str)

    Xtrain['MES'] = Xtrain['MES'].fillna(Xtrain['MES'].median())
    Xtest['MES'] = Xtest['MES'].fillna(Xtrain['MES'].median())

    Xtrain = pd.get_dummies(Xtrain, columns=categorical_cols + ['FERIADO'], dtype=int)
    Xtest = pd.get_dummies(Xtest, columns=categorical_cols + ['FERIADO'], dtype=int)

    Xtest = Xtest.reindex(columns=Xtrain.columns, fill_value=0)

    return Xtrain, ytrain, Xtest, ytest


# Execução dos experimentos
def executar_experimento(nome, df_train, df_test):

    Xtrain, ytrain, Xtest, ytest = preparar_dados(df_train, df_test)

    clf = TabPFNClassifier(device="cpu", random_state=42, balance_probabilities=True)

    clf.fit(Xtrain, ytrain)

    ypred = clf.predict(Xtest)

    print("\n" + "=" * 55)
    print(nome)
    print("=" * 55)

    print("Treino:", Xtrain.shape)
    print("Teste:", Xtest.shape)

    print("\nDistribuição do treino:")
    print(ytrain.value_counts())

    print("\nDistribuição do teste:")
    print(ytest.value_counts())

    print("\nClassification report:")
    print(classification_report(ytest, ypred, zero_division=0))

    accuracy = accuracy_score(ytest, ypred)
    macro_f1 = f1_score(ytest, ypred, average='macro')

    print("Accuracy:", accuracy)
    print("Macro F1:", macro_f1)

    return {
        'Experimento': nome,
        'Accuracy': accuracy,
        'Macro F1': macro_f1,
        'Registros treino': len(Xtrain),
        'Registros teste': len(Xtest)
    }


# EXPERIMENTO 1
# Sertãozinho -> Itupeva

resultado_1 = executar_experimento(
    "Modelo 1: Sertãozinho -> Itupeva",
    df_train=sertaozinho_train,
    df_test=itupeva_test
)


# EXPERIMENTO 2
# Sertãozinho + Birigui -> Itupeva

df_train_expandido = pd.concat(
    [sertaozinho_train, birigui_train],
    ignore_index=True
)

resultado_2 = executar_experimento(
    "Modelo 2: Sertãozinho + Birigui -> Itupeva",
    df_train_expandido,
    itupeva_test
)


# Comparação
comparacao = pd.DataFrame([resultado_1, resultado_2])

print("\nCOMPARAÇÃO FINAL")
print(comparacao.to_string(index=False))

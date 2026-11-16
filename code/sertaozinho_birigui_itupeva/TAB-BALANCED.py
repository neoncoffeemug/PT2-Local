import pandas as pd
import numpy as np

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from sklearn.metrics import classification_report, accuracy_score, f1_score
from adapt.feature_based import CORAL


# ============================================================
# 1. CARREGAR DATASET
# ============================================================

df = pd.read_csv("../../datasets/normalized_dataset.csv", sep=",")

df_sertaozinho = df[
    df['NOME_MUNICIPIO'] == 'sertaozinho'
].copy()

df_birigui = df[
    df['NOME_MUNICIPIO'] == 'birigui'
].copy()

df_itupeva = df[
    df['NOME_MUNICIPIO'] == 'itupeva'
].copy()


# ============================================================
# 2. CONFIGURAÇÕES
# ============================================================

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


# ============================================================
# 3. REMOVER TARGETS INVÁLIDOS
# ============================================================

for municipio in [
    df_sertaozinho,
    df_birigui,
    df_itupeva
]:
    municipio.dropna(subset=[target], inplace=True)


# ============================================================
# 4. DIVISÃO TEMPORAL
# ============================================================

sertaozinho_train = df_sertaozinho[
    df_sertaozinho['MES'] <= 12
].copy()

birigui_train = df_birigui[
    df_birigui['MES'] <= 12
].copy()

itupeva_test = df_itupeva[
    df_itupeva['MES'] <= 12
].copy()


# ============================================================
# 5. PREPARAÇÃO DOS DADOS
# ============================================================

def preparar_dados(df_train, df_test):

    Xtrain = df_train[features].copy()
    ytrain = df_train[target].copy()

    Xtest = df_test[features].copy()
    ytest = df_test[target].copy()

    for col in categorical_cols:

        Xtrain[col] = (
            Xtrain[col]
            .fillna('__MISSING__')
            .astype(str)
        )

        Xtest[col] = (
            Xtest[col]
            .fillna('__MISSING__')
            .astype(str)
        )

    Xtrain['FERIADO'] = (
        Xtrain['FERIADO']
        .fillna('__MISSING__')
        .astype(str)
    )

    Xtest['FERIADO'] = (
        Xtest['FERIADO']
        .fillna('__MISSING__')
        .astype(str)
    )

    Xtrain['MES'] = Xtrain['MES'].fillna(
        Xtrain['MES'].median()
    )

    Xtest['MES'] = Xtest['MES'].fillna(
        Xtrain['MES'].median()
    )

    Xtrain = pd.get_dummies(
        Xtrain,
        columns=categorical_cols + ['FERIADO'],
        dtype=int
    )

    Xtest = pd.get_dummies(
        Xtest,
        columns=categorical_cols + ['FERIADO'],
        dtype=int
    )

    Xtest = Xtest.reindex(
        columns=Xtrain.columns,
        fill_value=0
    )

    return Xtrain, ytrain, Xtest, ytest


# ============================================================
# 7. EXECUÇÃO DOS EXPERIMENTOS
# ============================================================

def executar_experimento(nome, df_train, df_test, usar_coral=False):

    Xtrain, ytrain, Xtest, ytest = preparar_dados(df_train, df_test)

    print("\n" + "=" * 60)
    print(nome)
    print("=" * 60)

    print("Treino:", Xtrain.shape)
    print("Teste:", Xtest.shape)

    # Aplicar CORAL quando solicitado
    if usar_coral:

        coral = CORAL(
            lambda_=1e-3,
            verbose=1
        )

        # Alinhar origem à distribuição de destino
        Xtrain_alinhado = coral.fit_transform(
            Xtrain.to_numpy(dtype=float),
            Xtest.to_numpy(dtype=float)
        )

        Xtrain = pd.DataFrame(
            Xtrain_alinhado,
            columns=Xtrain.columns,
            index=Xtrain.index
        )

    # Modelo TabPFN
    clf = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3_5_FAST,
        device="cpu",
        random_state=42,
        balance_probabilities=True
    )

    clf.fit(Xtrain, ytrain)

    ypred = clf.predict(Xtest)

    print("\nDistribuição do treino:")
    print(ytrain.value_counts())

    print("\nDistribuição do teste:")
    print(ytest.value_counts())

    print("\nClassification report:")
    print(classification_report(ytest, ypred, zero_division=0))

    accuracy = accuracy_score(ytest, ypred)
    macro_f1 = f1_score(ytest, ypred, average='macro')

    print("\nAccuracy:", accuracy)
    print("Macro F1:", macro_f1)

    return {
        'Experimento': nome,
        'CORAL': usar_coral,
        'Accuracy': accuracy,
        'Macro F1': macro_f1,
        'Registros treino': len(Xtrain),
        'Registros teste': len(Xtest)
    }


# ============================================================
# 8. PREPARAR TREINAMENTO EXPANDIDO
# ============================================================

df_train_expandido = pd.concat(
    [
        sertaozinho_train,
        birigui_train
    ],
    ignore_index=True
)


# ============================================================
# 9. EXPERIMENTO 1 - SEM CORAL
# Sertãozinho -> Itupeva
# ============================================================

resultado_1 = executar_experimento(
    nome="Modelo 1: Sertãozinho -> Itupeva",
    df_train=sertaozinho_train,
    df_test=itupeva_test,
    usar_coral=False
)


# ============================================================
# 10. EXPERIMENTO 2 - SEM CORAL
# Sertãozinho + Birigui -> Itupeva
# ============================================================

resultado_2 = executar_experimento(
    nome="Modelo 2: Sertãozinho + Birigui -> Itupeva",
    df_train=df_train_expandido,
    df_test=itupeva_test,
    usar_coral=False
)


# ============================================================
# 11. EXPERIMENTO 3 - COM CORAL
# Sertãozinho -> Itupeva
# ============================================================

resultado_3 = executar_experimento(
    nome="Modelo 3: Sertãozinho -> Itupeva + CORAL",
    df_train=sertaozinho_train,
    df_test=itupeva_test,
    usar_coral=True
)


# ============================================================
# 12. EXPERIMENTO 4 - COM CORAL
# Sertãozinho + Birigui -> Itupeva
# ============================================================

resultado_4 = executar_experimento(
    nome="Modelo 4: Sertãozinho + Birigui -> Itupeva + CORAL",
    df_train=df_train_expandido,
    df_test=itupeva_test,
    usar_coral=True
)


# ============================================================
# 13. COMPARAÇÃO FINAL
# ============================================================

comparacao = pd.DataFrame([
    resultado_1,
    resultado_2,
    resultado_3,
    resultado_4
])

print("\n" + "=" * 75)
print("COMPARAÇÃO FINAL")
print("=" * 75)

print(
    comparacao.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# Exportar resultados
comparacao.to_csv(
    "comparacao_tabpfn_coral.csv",
    index=False
)
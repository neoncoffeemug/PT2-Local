import pandas as pd 
import unidecode as unidecode

df_1 = pd.read_csv("../datasets/JAN_JUN_2025.csv", sep=";")
df_2 = pd.read_csv("../datasets/JUL_DEZ_2025.csv", sep=";")

full_df = pd.concat([df_1, df_2], ignore_index=True)

holidays_sp = {
  (1,1),   # Ano novo
  (3,3),   # Carnaval
  (3,4),   # Carnaval
  (3,5),   # CarnavaL
  (4,18),  # Sexta-feira santa
  (4,20),  # Páscoa
  (4,21),  # Tiradentes
  (5,1),   # Dia do trabalho
  (5,11),  # Dia das mãeslap
  (6,12),  # Dia dos namorados
  (6,19),  # Corpus Chisti
  (7,9),   # Revolução Constitucionalista
  (8,10),  # Dia dos pais
  (9,7),   # Independência do Brasil
  (10,12), # Nossa Senhora Aparecida
  (10,15), # Dia do Professor
  (10,17), # Dia do Comércio
  (10,28), # Dia do Servidor Público
  (11,2),  # Finados
  (11,15), # Proclamação da República
  (11,20), # Consciência Negra
  (12,25), # Natal
}

df_lowercase = full_df.copy()
text_columns = df_lowercase.select_dtypes(include=["object", "string"]).columns

for col in text_columns:
  df_lowercase[col] = df_lowercase[col].map(lambda x: unidecode.unidecode(x.lower())
  if isinstance(x, str) 
  else x   
)

# Converte e cria features temporais
def create_time_features(df_lowercase, time_feature='DATA_OCORRENCIA_BO'):
  df_lowercase[time_feature] = pd.to_datetime(df_lowercase[time_feature], dayfirst=True, errors='coerce')
  df_lowercase['ANO'] = df_lowercase[time_feature].dt.year.astype('Int64')
  df_lowercase['MES'] = df_lowercase[time_feature].dt.month.astype('Int64')
  df_lowercase['DIA'] = df_lowercase[time_feature].dt.day.astype('Int64')
  df_lowercase['DIA_SEM'] = df_lowercase[time_feature].dt.day_name().str.lower()
  df_lowercase['FERIADO'] = False

  return df_lowercase
df_lowercase = create_time_features(df_lowercase)

# Remoção de nulos
df_nulls = (df_lowercase.dropna(subset=['DESC_PERIODO', 'HORA_OCORRENCIA_BO'], how='all').drop(columns=['DATA_OCORRENCIA_BO']).dropna(subset=['DESCR_TIPOLOCAL', 'BAIRRO', 'RUBRICA', 'ANO', 'MES', 'DIA_SEM', 'DIA']))
df_scope = df_nulls[(df_nulls['DESC_PERIODO'] != 'em hora incerta') & (df_nulls['ANO'] == 2025)]

# Masks que preenchem valores nulos da coluna 'DESC_PERIODO' com base em horários de 'HORA_OCORRENCIA_BO'
df_masked = df_scope.copy()

hours = pd.to_datetime(df_masked['HORA_OCORRENCIA_BO'].astype('string').str.replace(r'\.\d+$', '', regex=True), format='%H:%M:%S', errors='coerce').dt.hour
df_masked['DESC_PERIODO'] = pd.cut(hours, bins=[-1, 5, 11, 17, 23], labels=['de madrugada', 'pela manha', 'a tarde', 'a noite'])

# Associa feriados com dias
def holiday(dia, mes):
  return (mes, dia) in holidays_sp

df_masked['FERIADO'] = df_masked.apply(lambda row: holiday(row['DIA'], row['MES']), axis = 1)

# Cria um dataset novo com as mudanças
df_masked.to_csv("../datasets/normalized_dataset.csv", index=False, encoding="utf-8")

print("ok")
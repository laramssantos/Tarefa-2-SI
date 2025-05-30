import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, mean_absolute_error
from tensorflow import keras
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('/Users/larasantos/Downloads/treino_sinais_vitais_com_label.txt', header=None)
dataset.columns = ['id', 'pressao_sistolica', 'pressao_diastolica', 'qualidade_pressao', 
                  'pulso', 'respiracao', 'gravidade', 'classe']

X = dataset[['qualidade_pressao', 'pulso', 'respiracao']]
y_gravidade = dataset['gravidade']
y_classe = dataset['classe']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n----- MODELO DE REGRESSÃO PARA GRAVIDADE -----")
X_train, X_test, y_gravidade_train, y_gravidade_test = train_test_split(
    X_scaled, y_gravidade, test_size=0.1, random_state=42)

modelo_gravidade = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='linear')  
])

modelo_gravidade.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae', 'mse']
)

print("Arquitetura do modelo de regressão:")
modelo_gravidade.summary()

history_gravidade = modelo_gravidade.fit(
    X_train, y_gravidade_train,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

y_gravidade_pred = modelo_gravidade.predict(X_test)
mse = mean_squared_error(y_gravidade_test, y_gravidade_pred)
mae = mean_absolute_error(y_gravidade_test, y_gravidade_pred)
r2 = r2_score(y_gravidade_test, y_gravidade_pred)

print(f"\nResultados da Regressão (Gravidade):")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

print("\n----- MODELO DE CLASSIFICAÇÃO PARA CLASSE -----")
label_encoder = LabelEncoder()
y_classe_encoded = label_encoder.fit_transform(y_classe)
num_classes = len(np.unique(y_classe_encoded))
y_classe_categorical = to_categorical(y_classe_encoded, num_classes)

X_train_clf, X_test_clf, y_classe_train, y_classe_test = train_test_split(
    X_scaled, y_classe_categorical, test_size=0.2, random_state=42)

clf_direto = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_clf.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  
])

clf_direto.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nArquitetura do modelo de classificação (apenas sinais vitais):")
clf_direto.summary()

history_clf_direto = clf_direto.fit(
    X_train_clf, y_classe_train,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

y_classe_pred_direto_prob = clf_direto.predict(X_test_clf)
y_classe_pred_direto = np.argmax(y_classe_pred_direto_prob, axis=1)
y_classe_test_labels = np.argmax(y_classe_test, axis=1)
acc_direto = accuracy_score(y_classe_test_labels, y_classe_pred_direto)

print(f"\nAcurácia da classificação (apenas sinais vitais): {acc_direto:.4f}")

y_gravidade_pred_train = modelo_gravidade.predict(X_train_clf)
y_gravidade_pred_test = modelo_gravidade.predict(X_test_clf)

X_train_com_gravidade = np.column_stack([X_train_clf, y_gravidade_pred_train.flatten()])
X_test_com_gravidade = np.column_stack([X_test_clf, y_gravidade_pred_test.flatten()])

clf_com_gravidade = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_com_gravidade.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

clf_com_gravidade.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nArquitetura do modelo de classificação (com gravidade prevista):")
clf_com_gravidade.summary()

history_clf_gravidade = clf_com_gravidade.fit(
    X_train_com_gravidade, y_classe_train,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

y_classe_pred_com_gravidade_prob = clf_com_gravidade.predict(X_test_com_gravidade)
y_classe_pred_com_gravidade = np.argmax(y_classe_pred_com_gravidade_prob, axis=1)
acc_com_gravidade = accuracy_score(y_classe_test_labels, y_classe_pred_com_gravidade)

print(f"\nAcurácia da classificação (com gravidade prevista): {acc_com_gravidade:.4f}")

y_classe_pred_original = label_encoder.inverse_transform(y_classe_pred_com_gravidade)
y_classe_test_original = label_encoder.inverse_transform(y_classe_test_labels)

print("\nRelatório detalhado da classificação (com gravidade):")
print(classification_report(y_classe_test_original, y_classe_pred_original))

print("\n----- PREVISÃO NOS DADOS SEM LABEL -----")
try:
    dados_sem_rotulos = pd.read_csv('/Users/larasantos/Downloads/treino_sinais_vitais_sem_label.txt', header=None)
    if len(dados_sem_rotulos.columns) >= 5: 
        dados_sem_rotulos.columns = ['id', 'pressao_sistolica', 'pressao_diastolica', 
                                    'qualidade_pressao', 'pulso', 'respiracao'] + \
                                    [f'extra_{i}' for i in range(len(dados_sem_rotulos.columns)-6)]
    else: 
        dados_sem_rotulos.columns = ['id', 'qualidade_pressao', 'pulso', 'respiracao'] + \
                                   [f'extra_{i}' for i in range(len(dados_sem_rotulos.columns)-4)]
    
    teste_semlabel = dados_sem_rotulos[['qualidade_pressao', 'pulso', 'respiracao']]
    
    print(f"\nCarregados {len(teste_semlabel)} registros para previsão.")
    print("\nPrimeiros registros:")
    print(teste_semlabel.head())
    
    teste_semlabel_scaled = scaler.transform(teste_semlabel)
    
    gravidade_prevista = modelo_gravidade.predict(teste_semlabel_scaled).flatten()
    
    teste_semlabel_com_gravidade = np.column_stack([teste_semlabel_scaled, gravidade_prevista])
    
    classe_prevista_prob = clf_com_gravidade.predict(teste_semlabel_com_gravidade)
    classe_prevista_encoded = np.argmax(classe_prevista_prob, axis=1)
    classe_prevista = label_encoder.inverse_transform(classe_prevista_encoded)
    
    mapa_classes = {
        1: 'Crítico',
        2: 'Instável',
        3: 'Potencialmente Estável',
        4: 'Estável'
    }
    classe_descritiva = [mapa_classes.get(c, f'Classe {c}') for c in classe_prevista]
    
    confianca_previsao = np.max(classe_prevista_prob, axis=1)
    
    resultados = pd.DataFrame({
        'ID': dados_sem_rotulos['id'],
        'Qualidade Pressão': teste_semlabel['qualidade_pressao'],
        'Pulso': teste_semlabel['pulso'],
        'Respiração': teste_semlabel['respiracao'],
        'Gravidade Prevista': gravidade_prevista.round(2),
        'Classe Prevista': classe_prevista,
        'Estado': classe_descritiva,
        'Confiança': confianca_previsao.round(3)
    })
    
    print("\nResultados previstos (primeiros 10 registros):")
    print(resultados.head(10))
    
    print("\nDistribuição das classes previstas:")
    print(resultados['Classe Prevista'].value_counts())
    
    print(f"\nConfiança média das previsões: {confianca_previsao.mean():.3f}")
    print(f"Confiança mínima: {confianca_previsao.min():.3f}")
    print(f"Confiança máxima: {confianca_previsao.max():.3f}")
    
    resultados.to_csv('/Users/larasantos/Downloads/resultados_previsoes_keras.csv', index=False)
    print("\nResultados completos salvos em 'resultados_previsoes_keras.csv'")
    
except FileNotFoundError:
    print("Arquivo 'treino_sinais_vitais_sem_label.txt' não encontrado!")
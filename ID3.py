import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('/Users/larasantos/Downloads/treino_sinais_vitais_com_label.txt', header=None)
dataset.columns = ['id', 'pressao_sistolica', 'pressao_diastolica', 'qualidade_pressao', 
                  'pulso', 'respiracao', 'gravidade', 'classe']

X = dataset[['qualidade_pressao', 'pulso', 'respiracao']]
y_gravidade = dataset['gravidade']
y_classe = dataset['classe']

print("\n----- REGRESSÃO PARA GRAVIDADE -----")
X_train, X_test, y_gravidade_train, y_gravidade_test = train_test_split(
    X, y_gravidade, test_size=0.3, random_state=42)

modelo_gravidade = DecisionTreeRegressor(
    criterion='squared_error',  
    max_depth=10,             
    min_samples_split=2,       
    random_state=42
)
modelo_gravidade.fit(X_train, y_gravidade_train)

y_gravidade_pred = modelo_gravidade.predict(X_test)
mse = mean_squared_error(y_gravidade_test, y_gravidade_pred)
mae = mean_absolute_error(y_gravidade_test, y_gravidade_pred)
r2 = r2_score(y_gravidade_test, y_gravidade_pred)

print(f"Resultados da Regressão (Gravidade):")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

plt.figure(figsize=(20, 10))
plot_tree(modelo_gravidade, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title("Árvore de Decisão para Gravidade")
plt.savefig('/Users/larasantos/Downloads/arvore_gravidade.png')
print("\nÁrvore de regressão salva em 'arvore_gravidade.png'")

print("\n----- CLASSE -----")

clf_direto = DecisionTreeClassifier(
    criterion='entropy',       
    max_depth=10,               
    min_samples_split=2,      
    random_state=42
)
clf_direto.fit(X_train, y_classe.loc[X_train.index])
y_classe_pred_direto = clf_direto.predict(X_test)
acc_direto = accuracy_score(y_classe.loc[X_test.index], y_classe_pred_direto)

X_train_com_gravidade = X_train.copy()
X_test_com_gravidade = X_test.copy()

X_train_com_gravidade['gravidade_prevista'] = modelo_gravidade.predict(X_train)
X_test_com_gravidade['gravidade_prevista'] = y_gravidade_pred

clf_com_gravidade = DecisionTreeClassifier(
    criterion='entropy',       
    max_depth=10,               
    min_samples_split=2,     
    random_state=42
)
clf_com_gravidade.fit(X_train_com_gravidade, y_classe.loc[X_train.index])
y_classe_pred_com_gravidade = clf_com_gravidade.predict(X_test_com_gravidade)
acc_com_gravidade = accuracy_score(y_classe.loc[X_test.index], y_classe_pred_com_gravidade)

print(f"\nAcurácia da classificação (apenas sinais vitais): {acc_direto:.4f}")
print(f"Acurácia da classificação (com gravidade prevista): {acc_com_gravidade:.4f}")

print("\nRelatório detalhado da classificação (com gravidade):")
print(classification_report(y_classe.loc[X_test.index], y_classe_pred_com_gravidade))

plt.figure(figsize=(20, 10))
plot_tree(clf_com_gravidade, feature_names=X_train_com_gravidade.columns, 
          class_names=[str(i) for i in clf_com_gravidade.classes_], 
          filled=True, rounded=True, fontsize=10)
plt.title("Árvore de Decisão para Classificação")
plt.savefig('/Users/larasantos/Downloads/arvore_classificacao.png')
print("\nÁrvore de classificação salva em 'arvore_classificacao.png'")

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
    
    gravidade_prevista = modelo_gravidade.predict(teste_semlabel)
    
    teste_semlabel_com_gravidade = teste_semlabel.copy()
    teste_semlabel_com_gravidade['gravidade_prevista'] = gravidade_prevista
    
    classe_prevista = clf_com_gravidade.predict(teste_semlabel_com_gravidade)
    
    mapa_classes = {
        1: 'Crítico',
        2: 'Instável',
        3: 'Potencialmente Estável',
        4: 'Estável'
    }
    classe_descritiva = [mapa_classes[c] for c in classe_prevista]
    
    resultados = pd.DataFrame({
        'ID': dados_sem_rotulos['id'],
        'Qualidade Pressão': teste_semlabel['qualidade_pressao'],
        'Pulso': teste_semlabel['pulso'],
        'Respiração': teste_semlabel['respiracao'],
        'Gravidade Prevista': gravidade_prevista.round(2),
        'Classe Prevista': classe_prevista,
        'Estado': classe_descritiva
    })
    
    print("\nResultados previstos (primeiros 10 registros):")
    print(resultados.head(10))
    
    print("\nDistribuição das classes previstas:")
    print(resultados['Classe Prevista'].value_counts())
    
    resultados.to_csv('/Users/larasantos/Downloads/resultados_previsoes_id3.csv', index=False)
    print("\nResultados completos salvos em 'resultados_previsoes_id3.csv'")
    
except FileNotFoundError:
    print("Arquivo 'treino_sinais_vitais_sem_label.txt' não encontrado!")



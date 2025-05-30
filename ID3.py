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

print("\n----- CLASSIFICAÇÃO -----")
X_train, X_test, y_train, y_test = train_test_split(X, y_classe, test_size=0.1, random_state=42)

modelo_id3 = DecisionTreeClassifier(
    criterion='entropy',  
    max_depth=12, 
    min_samples_split=2, 
    random_state=42
)
modelo_id3.fit(X_train, y_train)

y_pred = modelo_id3.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAcurácia da classificação (ID3): {acc:.4f}")

print("\nRelatório detalhado da classificação (ID3):")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(20, 10))
plot_tree(modelo_id3, feature_names=X.columns, 
          class_names=[str(c) for c in modelo_id3.classes_],
          filled=True, rounded=True, fontsize=10)
plt.title("Árvore de Decisão (ID3)")
plt.savefig('/Users/larasantos/Downloads/arvore_id3.png')
print("\nÁrvore de decisão salva em 'arvore_id3.png'")

print("\n----- CLASSIFICAÇÃO NOS DADOS SEM LABEL -----")
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

    classe_prevista = modelo_id3.predict(teste_semlabel)

    mapa_classes = {
        1: 'Crítico',
        2: 'Instável',
        3: 'Potencialmente Estável',
        4: 'Estável'
    }
    classe_descritiva = [mapa_classes.get(c, 'Desconhecido') for c in classe_prevista]

    resultados = pd.DataFrame({
        'ID': dados_sem_rotulos['id'],
        'Qualidade Pressão': teste_semlabel['qualidade_pressao'],
        'Pulso': teste_semlabel['pulso'],
        'Respiração': teste_semlabel['respiracao'],
        'Classe Prevista': classe_prevista,
        'Estado': classe_descritiva
    })

    print("\nResultados previstos (primeiros 10 registros):")
    print(resultados.head(10))

    print("\nDistribuição das classes previstas:")
    print(resultados['Classe Prevista'].value_counts())

    resultados.to_csv('/Users/larasantos/Downloads/resultados_previsoes_id3_puro.csv', index=False)
    print("\nResultados completos salvos em 'resultados_previsoes_id3.csv'")

except FileNotFoundError:
    print("Arquivo 'treino_sinais_vitais_sem_label.txt' não encontrado!")
